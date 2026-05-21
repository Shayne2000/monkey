#!/usr/bin/env python3
"""Robust RTSP input reader with 2-frame motion logging for Jetson Nano.

Design goals:
- Read RTSP with OpenCV FFmpeg.
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
import time
import uuid
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2


log = logging.getLogger("robust_rtsp_relay")


def open_capture(url: str) -> cv2.VideoCapture:
    log.info("opening input RTSP with OpenCV FFmpeg")
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


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
):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

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
        if area < min_area:
            continue
        x, y, w, h = cv2.boundingRect(cnt)
        roi = thresh[y:y + h, x:x + w]
        density = cv2.countNonZero(roi) / float(max(w * h, 1))
        if density > density_threshold:
            candidate_boxes.append((x, y, w, h))

    return gray, adaptive_merge_boxes(candidate_boxes, merge_distance)


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
    event_log_path: str,
    process_every_n: int,
    min_area: float,
    density_threshold: float,
    merge_distance: float,
) -> None:
    cap: Optional[cv2.VideoCapture] = None
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

    try:
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
                log.info("input RTSP connected camera_id=%s", camera_id)
                fail_count = 0
                backoff_s = 0.5

            read_start = time.monotonic()
            ok, frame = cap.read()
            read_ms = (time.monotonic() - read_start) * 1000.0
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
        if cap is not None:
            cap.release()
        writer.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="rtsp://10.0.11.153:8554/cctv02")
    parser.add_argument("--camera-id", default="cctv02")
    parser.add_argument("--event-log", default="vehicle_log.jsonl")
    parser.add_argument("--process-every-n", type=int, default=2)
    parser.add_argument("--min-area", type=float, default=1000.0)
    parser.add_argument("--density-threshold", type=float, default=0.35)
    parser.add_argument("--merge-distance", type=float, default=350.0)
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
            event_log_path=args.event_log,
            process_every_n=args.process_every_n,
            min_area=args.min_area,
            density_threshold=args.density_threshold,
            merge_distance=args.merge_distance,
        )
    except KeyboardInterrupt:
        log.info("stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
