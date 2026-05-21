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
from gi.repository import GLib, Gst


log = logging.getLogger("robust_rtsp_relay")
Gst.init(None)
DEFAULT_CROP_EXPAND = 1.8
VEHICLE_CLASS_IDS = {2, 3, 5, 7}


def gst_element_exists(factory_name: str) -> bool:
    return Gst.ElementFactory.find(factory_name) is not None


class GstFrameReader:
    """Latest-frame reader based on the DeepStream sample's GStreamer source path."""

    def __init__(self, uri: str, latency_ms: int, input_width: int, input_height: int) -> None:
        self.uri = uri
        self.latency_ms = latency_ms
        self.input_width = input_width
        self.input_height = input_height
        self.pipeline = None
        self.appsink = None
        self.loop = None
        self.loop_thread = None
        self._lock = threading.Condition()
        self._frame = None
        self._seq = 0
        self._last_frame_at = 0.0

    def start(self) -> bool:
        attempts = []
        for decoder in self._h264_decoder_candidates():
            for converter in self._converter_candidates():
                attempts.append((
                    f"explicit H264 + {decoder} + {converter}",
                    lambda d=decoder, c=converter: self._build_h264_pipeline(d, c),
                ))

        if gst_element_exists("nvstreammux"):
            for source in self._source_candidates():
                for converter in self._deepstream_converter_candidates():
                    attempts.append((
                        f"DeepStream-style {source} + nvstreammux + {converter}",
                        lambda s=source, c=converter: self._build_deepstream_pipeline(s, c),
                    ))

        for source in self._source_candidates():
            for converter in self._converter_candidates():
                attempts.append((
                    f"dynamic {source} + {converter}",
                    lambda s=source, c=converter: self._build(s, c),
                ))

        for label, builder in attempts:
            try:
                builder()
                ret = self.pipeline.set_state(Gst.State.PLAYING)
                if ret == Gst.StateChangeReturn.FAILURE:
                    raise RuntimeError("set_state(PLAYING) failed")
                self._start_loop()
                if not self.wait_for_first_frame(timeout_s=5.0):
                    raise RuntimeError("pipeline reached PLAYING but produced no frames")
                log.info("input RTSP connected via %s", label)
                return True
            except Exception as exc:
                log.warning("GStreamer input open failed via %s: %r", label, exc)
                self.close()
        return False

    def wait_for_first_frame(self, timeout_s: float) -> bool:
        deadline = time.monotonic() + timeout_s
        with self._lock:
            while self._seq == 0:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._lock.wait(remaining)
            return True

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
        if self.loop is not None:
            self.loop.quit()
        if self.loop_thread is not None:
            self.loop_thread.join(timeout=1.0)
        self.pipeline = None
        self.appsink = None
        self.loop = None
        self.loop_thread = None

    def _start_loop(self) -> None:
        self.loop = GLib.MainLoop()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)
        self.loop_thread = threading.Thread(target=self.loop.run, daemon=True)
        self.loop_thread.start()

    def _on_bus_message(self, bus, message):
        msg_type = message.type
        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            log.warning("GStreamer bus error: %s debug=%s", err, debug)
        elif msg_type == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            log.warning("GStreamer bus warning: %s debug=%s", err, debug)
        elif msg_type == Gst.MessageType.EOS:
            log.warning("GStreamer bus EOS")
        return True

    def _source_candidates(self) -> List[str]:
        candidates = []
        if gst_element_exists("nvurisrcbin"):
            candidates.append("nvurisrcbin")
        candidates.append("uridecodebin")
        return candidates

    def _h264_decoder_candidates(self) -> List[str]:
        candidates = []
        for decoder in ("omxh264dec", "nvv4l2decoder", "avdec_h264"):
            if gst_element_exists(decoder):
                candidates.append(decoder)
        return candidates

    def _deepstream_converter_candidates(self) -> List[str]:
        candidates = []
        if gst_element_exists("nvvideoconvert"):
            candidates.append("nvvideoconvert")
        if gst_element_exists("nvvidconv"):
            candidates.append("nvvidconv")
        return candidates

    def _converter_candidates(self) -> List[str]:
        candidates = []
        if gst_element_exists("nvvidconv"):
            candidates.append("nvvidconv")
        if gst_element_exists("nvvideoconvert"):
            candidates.append("nvvideoconvert")
        candidates.append("videoconvert")
        return candidates

    def _request_mux_sink_pad(self, mux, index: int):
        sinkpad = mux.get_request_pad(f"sink_{index}")
        if sinkpad:
            return sinkpad
        return mux.get_request_pad("sink_%u")

    def _build_deepstream_pipeline(self, source: str, converter: str) -> None:
        self.pipeline = Gst.Pipeline.new("motion-deepstream-input-pipeline")
        if self.pipeline is None:
            raise RuntimeError("failed to create pipeline")

        streammux = Gst.ElementFactory.make("nvstreammux", "stream-muxer")
        conv1 = Gst.ElementFactory.make(converter, "post-mux-converter")
        caps_bgrx = Gst.ElementFactory.make("capsfilter", "caps-bgrx")
        conv2 = Gst.ElementFactory.make("videoconvert", "bgr-converter")
        caps_bgr = Gst.ElementFactory.make("capsfilter", "caps-bgr")
        sink = Gst.ElementFactory.make("appsink", "framesink")
        elements = [streammux, conv1, caps_bgrx, conv2, caps_bgr, sink]
        if any(el is None for el in elements):
            raise RuntimeError(f"failed to create DeepStream chain with converter={converter}")

        source_bin = self._create_source_bin(source)
        if source_bin is None:
            raise RuntimeError("failed to create source bin")

        streammux.set_property("width", 640)
        streammux.set_property("height", 360)
        streammux.set_property("batch-size", 1)
        streammux.set_property("batched-push-timeout", 40000)
        streammux.set_property("live-source", 1)
        caps_bgrx.set_property("caps", Gst.Caps.from_string(self._bgrx_caps_string()))
        caps_bgr.set_property("caps", Gst.Caps.from_string("video/x-raw,format=BGR"))
        sink.set_property("emit-signals", True)
        sink.set_property("sync", False)
        sink.set_property("max-buffers", 1)
        sink.set_property("drop", True)
        sink.connect("new-sample", self._on_sample)

        self.pipeline.add(source_bin)
        for el in elements:
            self.pipeline.add(el)

        mux_sink = self._request_mux_sink_pad(streammux, 0)
        source_src = source_bin.get_static_pad("src")
        if not mux_sink or not source_src:
            raise RuntimeError("failed to get source/mux pads")
        ret = source_src.link(mux_sink)
        if ret != Gst.PadLinkReturn.OK:
            raise RuntimeError(f"failed to link source-bin -> nvstreammux: {ret}")

        chain = [streammux, conv1, caps_bgrx, conv2, caps_bgr, sink]
        for a, b in zip(chain[:-1], chain[1:]):
            if not a.link(b):
                raise RuntimeError(f"failed to link {a.get_name()} -> {b.get_name()}")

        self.appsink = sink
        log.info("DeepStream-style input pipeline built with %s + nvstreammux + %s", source, converter)

    def _create_source_bin(self, source: str):
        source_bin = Gst.Bin.new("source-bin-0")
        if source_bin is None:
            return None

        src = Gst.ElementFactory.make(source, "input-source")
        if src is None:
            return None
        src.set_property("uri", self.uri)
        if source == "nvurisrcbin":
            try:
                src.set_property("drop-on-latency", True)
                src.set_property("latency", self.latency_ms)
            except Exception:
                pass
        try:
            src.connect("child-added", self._source_child_added)
        except TypeError:
            pass
        src.connect("pad-added", self._source_bin_pad_added, source_bin)

        source_bin.add(src)
        ghost_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
        if ghost_pad is None:
            return None
        source_bin.add_pad(ghost_pad)
        return source_bin

    def _source_bin_pad_added(self, decodebin, pad, source_bin):
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps or caps.get_size() == 0:
            log.warning("source-bin pad-added without caps")
            return

        structure_name = caps.get_structure(0).get_name()
        log.info("source-bin pad-added caps=%s", caps.to_string())
        if not structure_name.startswith("video"):
            return

        features = caps.get_features(0)
        if not features or not features.contains("memory:NVMM"):
            log.warning("source-bin video pad is not NVMM; caps=%s", caps.to_string())
            return

        ghost_pad = source_bin.get_static_pad("src")
        if ghost_pad.set_target(pad):
            log.info("linked source-bin NVIDIA/NVMM video pad")
        else:
            log.warning("failed to target source-bin ghost pad")

    def _build_h264_pipeline(self, decoder: str, converter: str) -> None:
        pipeline_desc = (
            f'rtspsrc location="{self.uri}" protocols=tcp latency={self.latency_ms} '
            "drop-on-latency=true "
            "! rtph264depay "
            "! h264parse config-interval=-1 "
            "! queue max-size-buffers=1 leaky=downstream "
            f"! {decoder} "
            f"! {converter} "
            f"! {self._bgrx_caps_string()} "
            "! videoconvert "
            "! video/x-raw,format=BGR "
            "! appsink name=framesink emit-signals=true sync=false max-buffers=1 drop=true"
        )
        log.info("input explicit H264 pipeline: %s", pipeline_desc)
        self.pipeline = Gst.parse_launch(pipeline_desc)
        self.appsink = self.pipeline.get_by_name("framesink")
        if self.appsink is None:
            raise RuntimeError("appsink not found")
        self.appsink.connect("new-sample", self._on_sample)

    def _build(self, source: str, converter: str) -> None:
        self.pipeline = Gst.Pipeline.new("motion-input-pipeline")
        if self.pipeline is None:
            raise RuntimeError("failed to create pipeline")

        src = Gst.ElementFactory.make(source, "input-source")
        queue = Gst.ElementFactory.make("queue", "input-queue")
        conv1 = Gst.ElementFactory.make(converter, "input-converter")
        caps_bgrx = Gst.ElementFactory.make("capsfilter", "caps-bgrx")
        conv2 = Gst.ElementFactory.make("videoconvert", "bgr-converter")
        caps_bgr = Gst.ElementFactory.make("capsfilter", "caps-bgr")
        sink = Gst.ElementFactory.make("appsink", "framesink")

        elements = [src, queue, conv1, caps_bgrx, conv2, caps_bgr, sink]
        if any(el is None for el in elements):
            raise RuntimeError(f"failed to create one or more elements with converter={converter}")

        src.set_property("uri", self.uri)
        if source == "nvurisrcbin":
            try:
                src.set_property("drop-on-latency", True)
                src.set_property("latency", self.latency_ms)
            except Exception:
                pass
        try:
            src.connect("child-added", self._source_child_added)
        except TypeError:
            pass
        src.connect("pad-added", self._source_pad_added, queue)

        queue.set_property("max-size-buffers", 1)
        queue.set_property("leaky", 2)
        caps_bgrx.set_property("caps", Gst.Caps.from_string(self._bgrx_caps_string()))
        caps_bgr.set_property("caps", Gst.Caps.from_string("video/x-raw,format=BGR"))
        sink.set_property("emit-signals", True)
        sink.set_property("sync", False)
        sink.set_property("max-buffers", 1)
        sink.set_property("drop", True)
        sink.connect("new-sample", self._on_sample)

        for el in elements:
            self.pipeline.add(el)

        chain = [queue, conv1, caps_bgrx, conv2, caps_bgr, sink]
        for a, b in zip(chain[:-1], chain[1:]):
            if not a.link(b):
                raise RuntimeError(f"failed to link {a.get_name()} -> {b.get_name()}")

        self.appsink = sink
        log.info("input pipeline built with %s + %s", source, converter)

    def _source_child_added(self, child_proxy, obj, name, user_data=None):
        try:
            obj.connect("child-added", self._source_child_added)
        except Exception:
            pass
        if "source" in name.lower() or "rtspsrc" in name.lower():
            try:
                obj.set_property("latency", self.latency_ms)
                obj.set_property("drop-on-latency", True)
                obj.set_property("protocols", "tcp")
            except Exception:
                pass

    def _bgrx_caps_string(self) -> str:
        caps = "video/x-raw,format=BGRx"
        if self.input_width > 0 and self.input_height > 0:
            caps += f",width={self.input_width},height={self.input_height}"
        return caps

    def _source_pad_added(self, decodebin, pad, queue):
        caps = pad.get_current_caps() or pad.query_caps(None)
        if not caps or caps.get_size() == 0:
            log.warning("input pad-added without caps")
            return

        structure_name = caps.get_structure(0).get_name()
        log.info("input pad-added caps=%s", caps.to_string())
        if not structure_name.startswith("video"):
            return

        sinkpad = queue.get_static_pad("sink")
        if sinkpad.is_linked():
            return

        ret = pad.link(sinkpad)
        if ret == Gst.PadLinkReturn.OK:
            log.info("linked input video pad caps=%s", caps.to_string())
        else:
            log.warning("failed to link input video pad: %s caps=%s", ret, caps.to_string())

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
                self._last_frame_at = time.monotonic()
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
        self._lock = threading.Lock()

    def write(self, event: dict) -> None:
        with self._lock:
            self._fh.write(json.dumps(event, separators=(",", ":")) + "\n")

    def close(self) -> None:
        with self._lock:
            self._fh.close()


def load_label_map(path: str, section: str) -> Dict[int, str]:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    labels = data.get(section, data)
    return {int(k): str(v) for k, v in labels.items()}


def softmax(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    values = values - np.max(values)
    exp = np.exp(values)
    return exp / max(float(np.sum(exp)), 1e-12)


def normalize_model_output(output) -> np.ndarray:
    if isinstance(output, (list, tuple)):
        output = output[0]
    arr = np.asarray(output)
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        return np.empty((0, 0), dtype=np.float32)
    if arr.shape[0] < arr.shape[1] and arr.shape[0] <= 128:
        arr = arr.T
    return arr.astype(np.float32)


def make_input_blob(image, size: int) -> np.ndarray:
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    return np.transpose(blob, (2, 0, 1))[np.newaxis, ...]


def nms_indices(boxes: List[List[int]], confidences: List[float], threshold: float) -> List[int]:
    if not boxes:
        return []

    boxes_np = np.asarray(boxes, dtype=np.float32)
    scores = np.asarray(confidences, dtype=np.float32)
    x1 = boxes_np[:, 0]
    y1 = boxes_np[:, 1]
    x2 = x1 + boxes_np[:, 2]
    y2 = y1 + boxes_np[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = scores.argsort()[::-1]
    keep = []

    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / np.maximum(union, 1e-12)
        order = order[1:][iou <= threshold]

    return keep


class OnnxModelRunner:
    def __init__(self, model_path: str, backend: str) -> None:
        self.backend = backend
        self.model_path = model_path
        self.input_name = ""
        if backend == "opencv":
            self.net = cv2.dnn.readNetFromONNX(model_path)
        elif backend == "onnxruntime":
            import onnxruntime as ort

            self.net = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
            self.input_name = self.net.get_inputs()[0].name
        else:
            raise ValueError(f"unsupported model backend: {backend}")

    def forward(self, image, size: int):
        blob = make_input_blob(image, size)
        if self.backend == "opencv":
            self.net.setInput(blob)
            return self.net.forward()
        return self.net.run(None, {self.input_name: blob})[0]


def choose_model_backend(requested: str) -> str:
    if requested in ("auto", "opencv"):
        if hasattr(cv2, "dnn") and hasattr(cv2.dnn, "readNetFromONNX"):
            return "opencv"
        if requested == "opencv":
            raise RuntimeError("OpenCV was requested, but this build has no cv2.dnn.readNetFromONNX")

    if requested in ("auto", "onnxruntime"):
        try:
            import onnxruntime  # noqa: F401
            return "onnxruntime"
        except ImportError:
            if requested == "onnxruntime":
                raise RuntimeError("onnxruntime was requested, but it is not installed")

    raise RuntimeError(
        "no ONNX backend available; install onnxruntime, install OpenCV with DNN/ONNX support, "
        "or run with --no-models"
    )


class VehicleModelPipeline:
    """Runs YOLO detection and brand classification only on motion crops."""

    def __init__(
        self,
        *,
        yolo_model: str,
        classifier_model: str,
        labels_path: str,
        yolo_size: int,
        classifier_size: int,
        confidence: float,
        nms_threshold: float,
        backend: str,
    ) -> None:
        self.yolo_labels = load_label_map(labels_path, "yolo")
        self.classifier_labels = load_label_map(labels_path, "classifier")
        self.yolo_label_ids = sorted(self.yolo_labels)
        self.yolo_size = yolo_size
        self.classifier_size = classifier_size
        self.confidence = confidence
        self.nms_threshold = nms_threshold
        self.backend = choose_model_backend(backend)
        self.yolo = OnnxModelRunner(yolo_model, self.backend)
        self.classifier = OnnxModelRunner(classifier_model, self.backend)
        self._lock = threading.Lock()

    @classmethod
    def maybe_create(cls, args):
        if not args.enable_models:
            return None
        missing = [
            path for path in (args.yolo_model, args.classifier_model, args.labels)
            if not os.path.exists(path)
        ]
        if missing:
            raise FileNotFoundError("model pipeline files missing: " + ", ".join(missing))
        return cls(
            yolo_model=args.yolo_model,
            classifier_model=args.classifier_model,
            labels_path=args.labels,
            yolo_size=args.yolo_size,
            classifier_size=args.classifier_size,
            confidence=args.model_confidence,
            nms_threshold=args.nms_threshold,
            backend=args.model_backend,
        )

    def detect_and_classify(self, frame, motion_box: Tuple[int, int, int, int]) -> List[dict]:
        x, y, w, h = motion_box
        crop = frame[y:y + h, x:x + w]
        if crop.size == 0:
            return []

        detections = self._detect_vehicles(crop)
        results = []
        for det in detections:
            dx, dy, dw, dh = det["box"]
            fx = max(0, min(frame.shape[1] - 1, x + dx))
            fy = max(0, min(frame.shape[0] - 1, y + dy))
            fw = max(1, min(frame.shape[1] - fx, dw))
            fh = max(1, min(frame.shape[0] - fy, dh))
            vehicle_crop = frame[fy:fy + fh, fx:fx + fw]
            brand, brand_confidence = self._classify_brand(vehicle_crop)
            det.update({
                "box": (fx, fy, fw, fh),
                "brand": brand,
                "brand_confidence": brand_confidence,
            })
            results.append(det)
        return results

    def _detect_vehicles(self, crop) -> List[dict]:
        img_size = self.yolo_size
        height, width = crop.shape[:2]
        with self._lock:
            output = self.yolo.forward(crop, img_size)

        rows = normalize_model_output(output)
        boxes = []
        confidences = []
        class_ids = []
        for row in rows:
            if row.shape[0] < 5:
                continue
            scores = row[4:]
            if scores.size == 0:
                continue
            class_id = int(np.argmax(scores))
            label_id = self._resolve_yolo_label_id(class_id, scores.size)
            confidence = float(scores[class_id])
            if label_id is None or label_id not in VEHICLE_CLASS_IDS or confidence < self.confidence:
                continue
            cx, cy, bw, bh = [float(v) for v in row[:4]]
            left = int(round((cx - bw / 2.0) * width / img_size))
            top = int(round((cy - bh / 2.0) * height / img_size))
            box_w = int(round(bw * width / img_size))
            box_h = int(round(bh * height / img_size))
            left = max(0, min(left, width - 1))
            top = max(0, min(top, height - 1))
            box_w = max(1, min(box_w, width - left))
            box_h = max(1, min(box_h, height - top))
            boxes.append([left, top, box_w, box_h])
            confidences.append(confidence)
            class_ids.append(label_id)

        indices = nms_indices(boxes, confidences, self.nms_threshold)
        if len(indices) == 0:
            return []
        return [
            {
                "box": tuple(int(v) for v in boxes[i]),
                "vehicle_type": self.yolo_labels.get(class_ids[i], str(class_ids[i])),
                "type_confidence": float(confidences[i]),
            }
            for i in indices
        ]

    def _resolve_yolo_label_id(self, class_id: int, score_count: int) -> Optional[int]:
        if class_id in self.yolo_labels:
            return class_id
        if score_count == len(self.yolo_label_ids) and 0 <= class_id < len(self.yolo_label_ids):
            return self.yolo_label_ids[class_id]
        return None

    def _classify_brand(self, crop) -> Tuple[Optional[str], Optional[float]]:
        if crop.size == 0:
            return None, None
        img_size = self.classifier_size
        with self._lock:
            output = np.asarray(self.classifier.forward(crop, img_size)).reshape(-1)
        if output.size == 0:
            return None, None
        probs = output if 0.99 <= float(np.sum(output)) <= 1.01 else softmax(output)
        class_id = int(np.argmax(probs))
        return self.classifier_labels.get(class_id, str(class_id)), float(probs[class_id])


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
        return gray, [
            expand_box_to_square(box, original_w, original_h, DEFAULT_CROP_EXPAND)
            for box in merged
        ]

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

    return gray, [
        expand_box_to_square(box, original_w, original_h, DEFAULT_CROP_EXPAND)
        for box in scaled_boxes
    ]


def expand_box_to_square(
    box: Tuple[int, int, int, int],
    image_width: int,
    image_height: int,
    expand: float,
) -> Tuple[int, int, int, int]:
    x, y, w, h = box
    cx = x + w / 2.0
    cy = y + h / 2.0
    side = max(w, h) * max(expand, 1.0)
    side = max(side, 16.0)

    x1 = int(round(cx - side / 2.0))
    y1 = int(round(cy - side / 2.0))
    x2 = int(round(cx + side / 2.0))
    y2 = int(round(cy + side / 2.0))

    if x1 < 0:
        x2 -= x1
        x1 = 0
    if y1 < 0:
        y2 -= y1
        y1 = 0
    if x2 > image_width:
        shift = x2 - image_width
        x1 = max(0, x1 - shift)
        x2 = image_width
    if y2 > image_height:
        shift = y2 - image_height
        y1 = max(0, y1 - shift)
        y2 = image_height

    return (x1, y1, max(1, x2 - x1), max(1, y2 - y1))


def build_vehicle_log_event(
    *,
    detected_at: str,
    track_id: str,
    camera_id: str,
    box: Tuple[int, int, int, int],
    vehicle_type: Optional[str] = None,
    brand: Optional[str] = None,
    event_type: str = "motion",
    type_confidence: Optional[float] = None,
    brand_confidence: Optional[float] = None,
) -> dict:
    x, y, w, h = box
    return {
        "detected_at": detected_at,
        "track_id": track_id,
        "camera_id": camera_id,
        "last_seen": detected_at,
        "vehicle_type": vehicle_type,
        "color": "black",
        "brand": brand,
        "plate": None,
        "position_x": float(x + w / 2.0),
        "position_y": float(y + h / 2.0),
        "bbox_width": float(w),
        "bbox_height": float(h),
        "event_type": event_type,
        "type_confidence": type_confidence,
        "color_confidence": 0.0,
        "brand_confidence": brand_confidence,
    }


def camera_loop(
    *,
    rtsp_url: str,
    camera_id: str,
    latency_ms: int,
    input_width: int,
    input_height: int,
    event_log_path: str,
    process_every_n: int,
    min_area: float,
    density_threshold: float,
    merge_distance: float,
    motion_width: int,
    blur_kernel: int,
    model_pipeline: Optional[VehicleModelPipeline] = None,
    writer: Optional[JsonlWriter] = None,
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
    model_ms_total = 0.0
    model_ms_max = 0.0
    write_ms_total = 0.0
    write_ms_max = 0.0
    prev_gray = None
    tracker = CentroidTracker(max_distance=merge_distance, max_missing_frames=30)
    owns_writer = writer is None
    if writer is None:
        writer = JsonlWriter(event_log_path)

    try:
        while True:
            if reader is None:
                reader = GstFrameReader(rtsp_url, latency_ms, input_width, input_height)
                if not reader.start():
                    log.warning("input open failed; retrying in %.1fs", backoff_s)
                    reader.close()
                    reader = None
                    time.sleep(backoff_s)
                    backoff_s = min(backoff_s * 2, 5.0)
                    continue
                log.info("[%s] input reader ready", camera_id)
                fail_count = 0
                backoff_s = 0.5

            read_start = time.monotonic()
            ok, frame = reader.read(timeout_s=1.0)
            read_ms = (time.monotonic() - read_start) * 1000.0
            if not ok or frame is None:
                fail_count += 1
                if fail_count % 10 == 0:
                    log.warning("[%s] input frame read failed count=%d", camera_id, fail_count)
                if fail_count >= 30:
                    log.warning("[%s] too many input failures; reconnecting", camera_id)
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
                log.info("[%s] input frames read=%d shape=%s", camera_id, frames_read, frame.shape)

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
                    detections = []
                    if model_pipeline is not None:
                        model_start = time.monotonic()
                        detections = model_pipeline.detect_and_classify(frame, box)
                        model_ms = (time.monotonic() - model_start) * 1000.0
                        model_ms_total += model_ms
                        if model_ms > model_ms_max:
                            model_ms_max = model_ms

                    event_items = detections or [{
                        "box": box,
                        "vehicle_type": None,
                        "brand": None,
                        "event_type": "motion",
                        "type_confidence": None,
                        "brand_confidence": None,
                    }]

                    for item in event_items:
                        event_box = item["box"]
                        x, y, w, h = event_box
                        cx = x + w / 2.0
                        cy = y + h / 2.0
                        track_id = tracker.assign(cx, cy, frames_processed)
                        write_start = time.monotonic()
                        writer.write(build_vehicle_log_event(
                            detected_at=detected_at,
                            track_id=track_id,
                            camera_id=camera_id,
                            box=event_box,
                            vehicle_type=item.get("vehicle_type"),
                            brand=item.get("brand"),
                            event_type=item.get("event_type", "vehicle_detected"),
                            type_confidence=item.get("type_confidence"),
                            brand_confidence=item.get("brand_confidence"),
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
                model_avg = model_ms_total / max(fps_window_events, 1)
                log.info(
                    "[%s] fps=%.2f read_avg_ms=%.1f read_max_ms=%.1f "
                    "motion_avg_ms=%.1f motion_max_ms=%.1f "
                    "model_avg_ms=%.1f model_max_ms=%.1f "
                    "write_avg_ms=%.3f write_max_ms=%.3f "
                    "frames_read=%d processed=%d events=%d events_window=%d",
                    camera_id, fps,
                    read_avg, read_ms_max,
                    motion_avg, motion_ms_max,
                    model_avg, model_ms_max,
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
                model_ms_total = 0.0
                model_ms_max = 0.0
                write_ms_total = 0.0
                write_ms_max = 0.0
    finally:
        if reader is not None:
            reader.close()
        if owns_writer:
            writer.close()


def default_camera_urls(prefix: str, start: int, count: int) -> List[Tuple[str, str]]:
    return [
        (f"cctv{i:02d}", f"{prefix.rstrip('/')}/cctv{i:02d}")
        for i in range(start, start + count)
    ]


def run_multi_camera(args) -> None:
    cameras = default_camera_urls(args.rtsp_prefix, args.camera_start, args.camera_count)
    writer = JsonlWriter(args.event_log)
    model_pipeline = VehicleModelPipeline.maybe_create(args)
    if model_pipeline is not None:
        log.info("model pipeline enabled backend=%s yolo=%s classifier=%s labels=%s",
                 model_pipeline.backend, args.yolo_model, args.classifier_model, args.labels)
    threads = []

    log.info("starting %d cameras", len(cameras))
    for camera_id, rtsp_url in cameras:
        thread = threading.Thread(
            target=camera_loop,
            kwargs={
                "rtsp_url": rtsp_url,
                "camera_id": camera_id,
                "latency_ms": args.latency_ms,
                "input_width": args.input_width,
                "input_height": args.input_height,
                "event_log_path": args.event_log,
                "process_every_n": args.process_every_n,
                "min_area": args.min_area,
                "density_threshold": args.density_threshold,
                "merge_distance": args.merge_distance,
                "motion_width": args.motion_width,
                "blur_kernel": args.blur_kernel,
                "model_pipeline": model_pipeline,
                "writer": writer,
            },
            name=f"camera-{camera_id}",
            daemon=True,
        )
        thread.start()
        threads.append(thread)
        time.sleep(args.camera_start_gap_sec)

    try:
        while True:
            time.sleep(1.0)
    finally:
        writer.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-camera", action="store_true")
    parser.add_argument("--input", default="rtsp://10.0.11.153:8554/cctv02")
    parser.add_argument("--camera-id", default="cctv02")
    parser.add_argument("--rtsp-prefix", default="rtsp://10.0.11.153:8554")
    parser.add_argument("--camera-start", type=int, default=1)
    parser.add_argument("--camera-count", type=int, default=5)
    parser.add_argument("--camera-start-gap-sec", type=float, default=1.0)
    parser.add_argument("--latency-ms", type=int, default=200)
    parser.add_argument("--input-width", type=int, default=640)
    parser.add_argument("--input-height", type=int, default=360)
    parser.add_argument("--event-log", default="vehicle_log.jsonl")
    parser.add_argument("--process-every-n", type=int, default=2)
    parser.add_argument("--min-area", type=float, default=500.0)
    parser.add_argument("--density-threshold", type=float, default=0.20)
    parser.add_argument("--merge-distance", type=float, default=350.0)
    parser.add_argument("--motion-width", type=int, default=480)
    parser.add_argument("--blur-kernel", type=int, default=5)
    parser.add_argument("--no-models", dest="enable_models", action="store_false")
    parser.set_defaults(enable_models=True)
    parser.add_argument("--yolo-model", default="models/yolo.onnx")
    parser.add_argument("--classifier-model", default="models/classifier.onnx")
    parser.add_argument("--labels", default="models/labels.json")
    parser.add_argument("--yolo-size", type=int, default=640)
    parser.add_argument("--classifier-size", type=int, default=224)
    parser.add_argument("--model-confidence", type=float, default=0.25)
    parser.add_argument("--nms-threshold", type=float, default=0.45)
    parser.add_argument("--model-backend", choices=("auto", "opencv", "onnxruntime"), default="auto")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    try:
        if args.single_camera:
            model_pipeline = VehicleModelPipeline.maybe_create(args)
            if model_pipeline is not None:
                log.info("model pipeline enabled backend=%s yolo=%s classifier=%s labels=%s",
                         model_pipeline.backend, args.yolo_model, args.classifier_model, args.labels)
            camera_loop(
                rtsp_url=args.input,
                camera_id=args.camera_id,
                latency_ms=args.latency_ms,
                input_width=args.input_width,
                input_height=args.input_height,
                event_log_path=args.event_log,
                process_every_n=args.process_every_n,
                min_area=args.min_area,
                density_threshold=args.density_threshold,
                merge_distance=args.merge_distance,
                motion_width=args.motion_width,
                blur_kernel=args.blur_kernel,
                model_pipeline=model_pipeline,
            )
        else:
            run_multi_camera(args)
    except KeyboardInterrupt:
        log.info("stopping")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
