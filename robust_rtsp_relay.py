#!/usr/bin/env python3
"""Robust DeepStream-style input reader with 2-frame motion logging.

Design goals:
- Read RTSP with GStreamer, preferring DeepStream/NVIDIA decode elements.
- Reconnect when the stream stalls.
- Detect motion with 2-frame diff.
- Write vehicle_log-shaped JSONL events for downstream ingestion.
- Print live FPS and event counts in the terminal.
"""

import argparse
import json
import logging
import math
import os
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import gi
import numpy as np

gi.require_version("Gst", "1.0")
from gi.repository import Gst


log = logging.getLogger("robust_rtsp_relay")
Gst.init(None)


def gst_element_exists(factory_name: str) -> bool:
    return Gst.ElementFactory.find(factory_name) is not None


class GstFrameReader:
    """Latest-frame reader based on the DeepStream sample's GStreamer source path."""

    def __init__(self, uri: str, latency_ms: int) -> None:
        self.uri = uri
        self.latency_ms = latency_ms
        self.pipeline = None
        self.appsink = None
        self._lock = threading.Condition()
        self._frame = None
        self._seq = 0

    def start(self) -> bool:
        for source in self._source_candidates():
            for converter in self._converter_candidates():
                try:
                    self._build(source, converter)
                    ret = self.pipeline.set_state(Gst.State.PLAYING)
                    if ret == Gst.StateChangeReturn.FAILURE:
                        raise RuntimeError("set_state(PLAYING) failed")
                    log.info("input RTSP connected via %s + %s", source, converter)
                    return True
                except Exception as exc:
                    log.warning(
                        "GStreamer input open failed via %s + %s: %r",
                        source,
                        converter,
                        exc,
                    )
                    self.close()
        return False

    def read(self, timeout_s: float = 1.0):
        deadline = time.monotonic() + timeout_s
        with self._lock:
            start_seq = self._seq
            while self._seq == start_seq:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False, None
                self._lock.wait(remaining)
            return True, self._frame.copy()

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
        self.pipeline = None
        self.appsink = None

    def _source_candidates(self) -> List[str]:
        candidates = []
        if gst_element_exists("nvurisrcbin"):
            candidates.append("nvurisrcbin")
        candidates.append("uridecodebin")
        return candidates

    def _converter_candidates(self) -> List[str]:
        candidates = []
        if gst_element_exists("nvvideoconvert"):
            candidates.append("nvvideoconvert")
        if gst_element_exists("nvvidconv"):
            candidates.append("nvvidconv")
        candidates.append("videoconvert")
        return candidates

    def _build(self, source: str, converter: str) -> None:
        if source == "nvurisrcbin":
            source_part = (
                f'nvurisrcbin uri="{self.uri}" drop-on-latency=true '
                f"latency={self.latency_ms}"
            )
        else:
            source_part = f'uridecodebin uri="{self.uri}"'

        pipeline_desc = (
            source_part
            + " ! queue max-size-buffers=1 leaky=downstream "
            + f" ! {converter} "
            + " ! video/x-raw,format=BGRx "
            + " ! videoconvert "
            + " ! video/x-raw,format=BGR "
            + " ! appsink name=framesink emit-signals=true sync=false max-buffers=1 drop=true"
        )
        log.info("input pipeline: %s", pipeline_desc)
        self.pipeline = Gst.parse_launch(pipeline_desc)
        self.appsink = self.pipeline.get_by_name("framesink")
        if self.appsink is None:
            raise RuntimeError("appsink not found")
        self.appsink.connect("new-sample", self._on_sample)

    def _on_sample(self, sink):
        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.ERROR

        buf = sample.get_buffer()
        caps = sample.get_caps()
        structure = caps.get_structure(0)
        width = int(structure.get_value("width"))
        height = int(structure.get_value("height"))

        success, map_info = buf.map(Gst.MapFlags.READ)
        if not success:
            return Gst.FlowReturn.ERROR

        try:
            arr = np.frombuffer(map_info.data, dtype=np.uint8)
            expected = height * width * 3
            if arr.size < expected:
                log.warning("input buffer too small: got=%d expected=%d", arr.size, expected)
                return Gst.FlowReturn.OK
            frame = arr[:expected].reshape((height, width, 3)).copy()
            with self._lock:
                self._frame = frame
                self._seq += 1
                self._lock.notify_all()
        finally:
            buf.unmap(map_info)

        return Gst.FlowReturn.OK


def utc_now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="milliseconds") + "Z"


def adaptive_merge_boxes(
    boxes: List[Tuple[int, int, int, int]],
    distance_threshold: float,
) -> List[Tuple[int, int, int, int]]:
    merged: List[Tuple[int, int, int, int]] = []

    for x, y, w, h in boxes:
        cx = x + w // 2
        cy = y + h // 2
        found = False

        for i, (mx, my, mw, mh) in enumerate(merged):
            mcx = mx + mw // 2
            mcy = my + mh // 2
            if math.hypot(cx - mcx, cy - mcy) < distance_threshold:
                nx1 = min(x, mx)
                ny1 = min(y, my)
                nx2 = max(x + w, mx + mw)
                ny2 = max(y + h, my + mh)
                merged[i] = (nx1, ny1, nx2 - nx1, ny2 - ny1)
                found = True
                break

        if not found:
            merged.append((x, y, w, h))

    return merged


class CentroidTracker:
    def __init__(self, max_distance: float, max_missing_frames: int) -> None:
        self.max_distance = max_distance
        self.max_missing_frames = max_missing_frames
        self.tracks: Dict[str, Dict[str, float]] = {}

    def assign(self, cx: float, cy: float, frame_id: int) -> str:
        best_id = ""
        best_dist = self.max_distance

        for track_id, track in self.tracks.items():
            dist = math.hypot(cx - track["x"], cy - track["y"])
            if dist < best_dist:
                best_id = track_id
                best_dist = dist

        if not best_id:
            best_id = str(uuid.uuid4())

        self.tracks[best_id] = {"x": cx, "y": cy, "last_frame": float(frame_id)}
        self._expire(frame_id)
        return best_id

    def _expire(self, frame_id: int) -> None:
        stale = [
            track_id for track_id, track in self.tracks.items()
            if frame_id - int(track["last_frame"]) > self.max_missing_frames
        ]
        for track_id in stale:
            del self.tracks[track_id]


class JsonlWriter:
    def __init__(self, path: str) -> None:
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._fh = open(path, "a", buffering=1)

    def write(self, event: dict) -> None:
        self._fh.write(json.dumps(event, separators=(",", ":")) + "\n")

    def close(self) -> None:
        self._fh.close()


def detect_motion_regions(
    frame,
    prev_gray,
    *,
    min_area: float,
    density_threshold: float,
    merge_distance: float,
    motion_width: int,
    blur_kernel: int,
):
    original_h, original_w = frame.shape[:2]
    scale = 1.0
    work = frame
    if motion_width > 0 and original_w > motion_width:
        scale = motion_width / float(original_w)
        motion_height = max(1, int(round(original_h * scale)))
        work = cv2.resize(frame, (motion_width, motion_height), interpolation=cv2.INTER_AREA)

    kernel = max(3, int(blur_kernel))
    if kernel % 2 == 0:
        kernel += 1

    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (kernel, kernel), 0)

    if prev_gray is None:
        return gray, []

    diff = cv2.absdiff(prev_gray, gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=2)
    found = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = found[0] if len(found) == 2 else found[1]

    candidate_boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < min_area * scale * scale:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi = thresh[y:y + h, x:x + w]
        density = cv2.countNonZero(roi) / float(max(w * h, 1))
        if density > density_threshold:
            candidate_boxes.append((x, y, w, h))

    merged = adaptive_merge_boxes(candidate_boxes, merge_distance * scale)
    if scale == 1.0:
        return gray, merged

    scaled_boxes = []
    inv_scale = 1.0 / scale
    for x, y, w, h in merged:
        x1 = int(round(x * inv_scale))
        y1 = int(round(y * inv_scale))
        x2 = int(round((x + w) * inv_scale))
        y2 = int(round((y + h) * inv_scale))
        x1 = max(0, min(x1, original_w - 1))
        y1 = max(0, min(y1, original_h - 1))
        x2 = max(x1 + 1, min(x2, original_w))
        y2 = max(y1 + 1, min(y2, original_h))
        scaled_boxes.append((x1, y1, x2 - x1, y2 - y1))

    return gray, scaled_boxes


def build_vehicle_log_event(
    *,
    detected_at: str,
    track_id: str,
    camera_id: str,
    box: Tuple[int, int, int, int],
) -> dict:
    x, y, w, h = box
    return {
        "detected_at": detected_at,
        "track_id": track_id,
        "camera_id": camera_id,
        "last_seen": detected_at,
        "vehicle_type": None,
        "color": None,
        "brand": None,
        "plate": None,
        "position_x": float(x + w / 2.0),
        "position_y": float(y + h / 2.0),
        "bbox_width": float(w),
        "bbox_height": float(h),
        "event_type": "motion",
        "type_confidence": None,
        "color_confidence": None,
        "brand_confidence": None,
    }


def camera_loop(
    *,
    rtsp_url: str,
    camera_id: str,
    latency_ms: int,
    event_log_path: str,
    process_every_n: int,
    min_area: float,
    density_threshold: float,
    merge_distance: float,
    motion_width: int,
    blur_kernel: int,
    display: bool,
    display_width: int,
) -> None:
    reader: Optional[GstFrameReader] = None
    fail_count = 0
    backoff_s = 0.5
    frames_read = 0
    frames_processed = 0
    events_written = 0
    fps_window_start = time.monotonic()
    fps_window_frames = 0
    fps_window_processed = 0
    fps_window_events = 0
    read_ms_total = 0.0
    read_ms_max = 0.0
    motion_ms_total = 0.0
    motion_ms_max = 0.0
    write_ms_total = 0.0
    write_ms_max = 0.0
    prev_gray = None
    tracker = CentroidTracker(max_distance=merge_distance, max_missing_frames=30)
    writer = JsonlWriter(event_log_path)
    window_name = "motion-reader"

    try:
        while True:
            if reader is None:
                reader = GstFrameReader(rtsp_url, latency_ms)
                if not reader.start():
                    log.warning("input open failed; retrying in %.1fs", backoff_s)
                    reader.close()
                    reader = None
                    time.sleep(backoff_s)
                    backoff_s = min(backoff_s * 2, 5.0)
                    continue
                log.info("input reader ready camera_id=%s", camera_id)
                fail_count = 0
                backoff_s = 0.5

            read_start = time.monotonic()
            ok, frame = reader.read(timeout_s=1.0)
            read_ms = (time.monotonic() - read_start) * 1000.0
            if not ok or frame is None:
                fail_count += 1
                if fail_count % 10 == 0:
                    log.warning("input frame read failed count=%d", fail_count)
                if fail_count >= 30:
                    log.warning("too many input failures; reconnecting")
                    reader.close()
                    reader = None
                    fail_count = 0
                else:
                    time.sleep(0.03)
                continue

            fail_count = 0
            frames_read += 1
            fps_window_frames += 1
            read_ms_total += read_ms
            if read_ms > read_ms_max:
                read_ms_max = read_ms

            if frames_read == 1 or frames_read % 100 == 0:
                log.info("input frames read=%d shape=%s", frames_read, frame.shape)

            if frames_read % max(process_every_n, 1) == 0:
                frames_processed += 1
                fps_window_processed += 1
                motion_start = time.monotonic()
                prev_gray, boxes = detect_motion_regions(
                    frame,
                    prev_gray,
                    min_area=min_area,
                    density_threshold=density_threshold,
                    merge_distance=merge_distance,
                    motion_width=motion_width,
                    blur_kernel=blur_kernel,
                )
                motion_ms = (time.monotonic() - motion_start) * 1000.0
                motion_ms_total += motion_ms
                if motion_ms > motion_ms_max:
                    motion_ms_max = motion_ms
                detected_at = utc_now_iso()
                for box in boxes:
                    x, y, w, h = box
                    cx = x + w / 2.0
                    cy = y + h / 2.0
                    track_id = tracker.assign(cx, cy, frames_processed)
                    write_start = time.monotonic()
                    writer.write(build_vehicle_log_event(
                        detected_at=detected_at,
                        track_id=track_id,
                        camera_id=camera_id,
                        box=box,
                    ))
                    write_ms = (time.monotonic() - write_start) * 1000.0
                    write_ms_total += write_ms
                    if write_ms > write_ms_max:
                        write_ms_max = write_ms
                    events_written += 1
                    fps_window_events += 1

                if display:
                    preview = frame.copy()
                    for x, y, w, h in boxes:
                        cv2.rectangle(preview, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    if display_width > 0 and preview.shape[1] > display_width:
                        scale = display_width / float(preview.shape[1])
                        display_height = max(1, int(round(preview.shape[0] * scale)))
                        preview = cv2.resize(
                            preview,
                            (display_width, display_height),
                            interpolation=cv2.INTER_AREA,
                        )
                    cv2.imshow(window_name, preview)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        log.info("display quit requested")
                        break

            now = time.monotonic()
            if now - fps_window_start >= 1.0:
                window_s = now - fps_window_start
                fps = fps_window_frames / window_s
                read_avg = read_ms_total / max(fps_window_frames, 1)
                motion_avg = motion_ms_total / max(fps_window_processed, 1)
                write_avg = write_ms_total / max(fps_window_events, 1)
                log.info(
                    "fps=%.2f read_avg_ms=%.1f read_max_ms=%.1f "
                    "motion_avg_ms=%.1f motion_max_ms=%.1f "
                    "write_avg_ms=%.3f write_max_ms=%.3f "
                    "frames_read=%d processed=%d events=%d events_window=%d",
                    fps,
                    read_avg, read_ms_max,
                    motion_avg, motion_ms_max,
                    write_avg, write_ms_max,
                    frames_read, frames_processed, events_written, fps_window_events,
                )
                fps_window_start = now
                fps_window_frames = 0
                fps_window_processed = 0
                fps_window_events = 0
                read_ms_total = 0.0
                read_ms_max = 0.0
                motion_ms_total = 0.0
                motion_ms_max = 0.0
                write_ms_total = 0.0
                write_ms_max = 0.0
    finally:
        if reader is not None:
            reader.close()
        if display:
            cv2.destroyAllWindows()
        writer.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="rtsp://10.0.11.153:8554/cctv02")
    parser.add_argument("--camera-id", default="cctv02")
    parser.add_argument("--latency-ms", type=int, default=200)
    parser.add_argument("--event-log", default="vehicle_log.jsonl")
    parser.add_argument("--process-every-n", type=int, default=2)
    parser.add_argument("--min-area", type=float, default=1000.0)
    parser.add_argument("--density-threshold", type=float, default=0.35)
    parser.add_argument("--merge-distance", type=float, default=350.0)
    parser.add_argument("--motion-width", type=int, default=480)
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--display-width", type=int, default=960)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    try:
        camera_loop(
            rtsp_url=args.input,
            camera_id=args.camera_id,
            latency_ms=args.latency_ms,
            event_log_path=args.event_log,
            process_every_n=args.process_every_n,
            min_area=args.min_area,
            density_threshold=args.density_threshold,
            merge_distance=args.merge_distance,
            motion_width=args.motion_width,
            blur_kernel=args.blur_kernel,
            display=args.display,
            display_width=args.display_width,
        )
    except KeyboardInterrupt:
        log.info("stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
