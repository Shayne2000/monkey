#!/usr/bin/env python3
"""Robust RTSP input -> RTSP output relay for Jetson Nano.

Design goals:
- Read RTSP with OpenCV FFmpeg.
- Keep only the newest frame so latency does not build up.
- Keep output stream alive by repeating the last frame or black frame.
"""

import argparse
import logging
import threading
import time
from typing import Optional, Tuple

import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")
gi.require_version("GstRtspServer", "1.0")
from gi.repository import GLib, Gst, GstRtspServer  # noqa: E402


log = logging.getLogger("robust_rtsp_relay")


class LatestFrame:
    """Single-slot frame store. New writes replace old frames."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._frame: Optional[np.ndarray] = None
        self._updated_at = 0.0
        self._seq = 0

    def put(self, frame: np.ndarray) -> None:
        with self._lock:
            self._frame = frame
            self._updated_at = time.monotonic()
            self._seq += 1

    def get(self) -> Tuple[Optional[np.ndarray], float, int]:
        with self._lock:
            if self._frame is None:
                return None, self._updated_at, self._seq
            return self._frame.copy(), self._updated_at, self._seq


def open_capture(
    url: str,
) -> cv2.VideoCapture:
    log.info("opening input RTSP with OpenCV FFmpeg")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def camera_loop(
    *,
    rtsp_url: str,
    raw_frame: LatestFrame,
    stop_event: threading.Event,
) -> None:
    cap: Optional[cv2.VideoCapture] = None
    fail_count = 0
    backoff_s = 0.5
    frames_read = 0

    while not stop_event.is_set():
        if cap is None or not cap.isOpened():
            cap = open_capture(rtsp_url)
            if cap is None or not cap.isOpened():
                log.warning("input open failed; retrying in %.1fs", backoff_s)
                if cap is not None:
                    cap.release()
                    cap = None
                stop_event.wait(backoff_s)
                backoff_s = min(backoff_s * 2, 5.0)
                continue
            log.info("input RTSP connected")
            fail_count = 0
            backoff_s = 0.5

        ok, frame = cap.read()
        if not ok or frame is None:
            fail_count += 1
            if fail_count % 10 == 0:
                log.warning("input frame read failed count=%d", fail_count)
            if fail_count >= 30:
                log.warning("too many input failures; reconnecting")
                cap.release()
                cap = None
                fail_count = 0
            else:
                stop_event.wait(0.03)
            continue

        fail_count = 0
        frames_read += 1
        if frames_read == 1 or frames_read % 100 == 0:
            log.info("input frames read=%d shape=%s", frames_read, frame.shape)
        raw_frame.put(frame)

    if cap is not None:
        cap.release()
    log.info("camera loop stopped")


class RelayFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self, latest_frame: LatestFrame, width: int, height: int, fps: int) -> None:
        super().__init__()
        self.latest_frame = latest_frame
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_id = 0
        self.push_count = 0
        self.empty_count = 0
        self.last_output = np.zeros((height, width, 3), dtype=np.uint8)
        self.set_shared(True)
        self.set_launch(
            "appsrc name=source is-live=true block=false format=time do-timestamp=false "
            f"caps=video/x-raw,format=BGR,width={width},height={height},framerate={fps}/1 ! "
            "queue max-size-buffers=2 leaky=downstream ! "
            "videoconvert ! "
            "video/x-raw,format=NV12 ! "
            "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
            "nvv4l2h264enc bitrate=2000000 insert-sps-pps=true maxperf-enable=1 ! "
            "h264parse ! "
            "rtph264pay name=pay0 pt=96 config-interval=1"
        )

    def do_configure(self, media):
        appsrc = media.get_element().get_child_by_name("source")
        appsrc.connect("need-data", self.on_need_data)

    def on_need_data(self, src, _length):
        frame, _, _ = self.latest_frame.get()
        if frame is not None:
            self.last_output = cv2.resize(frame, (self.width, self.height))
        else:
            self.empty_count += 1
            if self.empty_count == 1 or self.empty_count % self.fps == 0:
                log.warning("output requested data but no input frame is available yet")

        out = np.ascontiguousarray(self.last_output)
        buf = Gst.Buffer.new_allocate(None, out.nbytes, None)
        buf.fill(0, out.tobytes())
        buf.pts = buf.dts = int(self.frame_id * Gst.SECOND / self.fps)
        buf.duration = int(Gst.SECOND / self.fps)
        buf.offset = self.frame_id
        self.frame_id += 1

        ret = src.emit("push-buffer", buf)
        self.push_count += 1
        if self.push_count == 1 or self.push_count % (self.fps * 5) == 0:
            log.info("output frames pushed=%d flow=%s", self.push_count, ret)
        if ret != Gst.FlowReturn.OK:
            log.warning("push-buffer returned %s", ret)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="rtsp://10.0.11.153:8554/cctv02")
    parser.add_argument("--port", default="8554")
    parser.add_argument("--mount", default="/jetson")
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=270)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    Gst.init(None)
    stop_event = threading.Event()
    raw_frame = LatestFrame()

    threading.Thread(
        target=camera_loop,
        kwargs={
            "rtsp_url": args.input,
            "raw_frame": raw_frame,
            "stop_event": stop_event,
        },
        name="camera_loop",
        daemon=True,
    ).start()

    server = GstRtspServer.RTSPServer()
    server.set_service(args.port)
    factory = RelayFactory(raw_frame, args.width, args.height, args.fps)
    server.get_mount_points().add_factory(args.mount, factory)
    server.attach(None)

    log.info("RTSP output: rtsp://<JETSON_IP>:%s%s", args.port, args.mount)
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        log.info("stopping")
        stop_event.set()
        loop.quit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
