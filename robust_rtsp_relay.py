#!/usr/bin/env python3
"""Robust RTSP input reader for Jetson Nano.

Design goals:
- Read RTSP with OpenCV FFmpeg.
- Reconnect when the stream stalls.
- Log frame counts so rule-based processing can be added next.
"""

import argparse
import logging
import time
from typing import Optional

import cv2


log = logging.getLogger("robust_rtsp_relay")


def open_capture(url: str) -> cv2.VideoCapture:
    log.info("opening input RTSP with OpenCV FFmpeg")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def camera_loop(
    *,
    rtsp_url: str,
) -> None:
    cap: Optional[cv2.VideoCapture] = None
    fail_count = 0
    backoff_s = 0.5
    frames_read = 0

    while True:
        if cap is None or not cap.isOpened():
            cap = open_capture(rtsp_url)
            if cap is None or not cap.isOpened():
                log.warning("input open failed; retrying in %.1fs", backoff_s)
                if cap is not None:
                    cap.release()
                    cap = None
                time.sleep(backoff_s)
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
                time.sleep(0.03)
            continue

        fail_count = 0
        frames_read += 1
        if frames_read == 1 or frames_read % 100 == 0:
            log.info("input frames read=%d shape=%s", frames_read, frame.shape)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="rtsp://10.0.11.153:8554/cctv02")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    try:
        camera_loop(rtsp_url=args.input)
    except KeyboardInterrupt:
        log.info("stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
