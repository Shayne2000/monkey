#!/usr/bin/env python3
"""DeepStream primary YOLO detector that writes vehicle_log-shaped JSONL.

Run this with Jetson system Python, not conda Python:
PYTHONPATH=/opt/nvidia/deepstream/deepstream-5.1/lib /usr/bin/python3 deepstream_vehicle_log.py --single-camera
"""

import argparse
import json
import logging
import os
import threading
import time
import uuid
from datetime import datetime
from typing import Dict, List, Tuple

import gi

gi.require_version("Gst", "1.0")
from gi.repository import GLib, Gst

import pyds


log = logging.getLogger("deepstream_vehicle_log")
Gst.init(None)

VEHICLE_CLASS_IDS = {2, 3, 5, 7}
VEHICLE_LABELS = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}


def utc_now_iso() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def default_camera_urls(prefix: str, start: int, count: int) -> List[Tuple[str, str]]:
    return [
        ("cctv%02d" % i, "%s/cctv%02d" % (prefix.rstrip("/"), i))
        for i in range(start, start + count)
    ]


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


def build_vehicle_log_event(camera_id: str, box, vehicle_type, confidence) -> dict:
    x, y, w, h = box
    detected_at = utc_now_iso()
    return {
        "detected_at": detected_at,
        "track_id": str(uuid.uuid4()),
        "camera_id": camera_id,
        "last_seen": detected_at,
        "vehicle_type": vehicle_type,
        "color": "black",
        "brand": None,
        "plate": None,
        "position_x": float(x + w / 2.0),
        "position_y": float(y + h / 2.0),
        "bbox_width": float(w),
        "bbox_height": float(h),
        "event_type": "vehicle_detected",
        "type_confidence": float(confidence),
        "color_confidence": 0.0,
        "brand_confidence": None,
    }


def make_element(factory_name: str, name: str):
    element = Gst.ElementFactory.make(factory_name, name)
    if element is None:
        raise RuntimeError("Could not create element %s (%s)" % (name, factory_name))
    return element


def request_mux_sink_pad(streammux, index: int):
    pad_name = "sink_%u" % index
    try:
        pad = streammux.get_request_pad(pad_name)
    except AttributeError:
        pad = streammux.request_pad_simple(pad_name)
    if pad is None:
        raise RuntimeError("Unable to request streammux pad %s" % pad_name)
    return pad


class DeepStreamVehicleLogger:
    def __init__(self, args) -> None:
        self.args = args
        self.cameras = self._build_cameras(args)
        self.pipeline = None
        self.loop = None
        self.writer = JsonlWriter(args.event_log)
        self.frames = 0
        self.events = 0
        self.window_start = time.monotonic()
        self.window_frames = 0
        self.window_events = 0

    def _build_cameras(self, args) -> List[Tuple[str, str]]:
        if args.single_camera:
            return [(args.camera_id, args.input)]
        return default_camera_urls(args.rtsp_prefix, args.camera_start, args.camera_count)

    def close(self) -> None:
        if self.pipeline is not None:
            self.pipeline.set_state(Gst.State.NULL)
        self.writer.close()

    def run(self) -> None:
        self.pipeline = self._build_pipeline()
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect("message", self._on_bus_message)

        self.pipeline.set_state(Gst.State.PLAYING)
        self.loop = GLib.MainLoop()
        try:
            self.loop.run()
        finally:
            self.close()

    def _build_pipeline(self):
        pipeline = Gst.Pipeline.new("deepstream-vehicle-log")
        if pipeline is None:
            raise RuntimeError("Could not create pipeline")

        streammux = make_element("nvstreammux", "stream-muxer")
        streammux.set_property("width", self.args.mux_width)
        streammux.set_property("height", self.args.mux_height)
        streammux.set_property("batch-size", len(self.cameras))
        streammux.set_property("batched-push-timeout", self.args.mux_timeout_usec)
        streammux.set_property("live-source", 1)
        pipeline.add(streammux)

        for index, (camera_id, uri) in enumerate(self.cameras):
            source_bin = self._create_source_bin(index, uri)
            pipeline.add(source_bin)
            sinkpad = request_mux_sink_pad(streammux, index)
            srcpad = source_bin.get_static_pad("src")
            if srcpad is None:
                raise RuntimeError("Unable to get source-bin src pad")
            if srcpad.link(sinkpad) != Gst.PadLinkReturn.OK:
                raise RuntimeError("Failed to link source %s to streammux" % camera_id)

        infer_config = self._write_infer_config(len(self.cameras))
        pgie = make_element("nvinfer", "primary-inference")
        pgie.set_property("config-file-path", infer_config)
        pipeline.add(pgie)
        if not streammux.link(pgie):
            raise RuntimeError("Failed to link streammux -> nvinfer")

        srcpad = pgie.get_static_pad("src")
        if srcpad is None:
            raise RuntimeError("Unable to get pgie src pad")
        srcpad.add_probe(Gst.PadProbeType.BUFFER, self._pgie_src_pad_buffer_probe, None)

        sink = make_element("fakesink", "sink")
        sink.set_property("sync", False)
        sink.set_property("enable-last-sample", False)
        pipeline.add(sink)
        if not pgie.link(sink):
            raise RuntimeError("Failed to link nvinfer -> fakesink")

        log.info("DeepStream pipeline built cameras=%d config=%s", len(self.cameras), infer_config)
        return pipeline

    def _create_source_bin(self, index: int, uri: str):
        bin_name = "source-bin-%02d" % index
        source_bin = Gst.Bin.new(bin_name)
        if source_bin is None:
            raise RuntimeError("Unable to create source bin")

        source = make_element("uridecodebin", "uri-decode-bin-%02d" % index)
        source.set_property("uri", uri)
        source.connect("pad-added", self._decodebin_pad_added, source_bin)
        source.connect("child-added", self._decodebin_child_added, None)
        source_bin.add(source)

        ghost_pad = Gst.GhostPad.new_no_target("src", Gst.PadDirection.SRC)
        if not source_bin.add_pad(ghost_pad):
            raise RuntimeError("Failed to add ghost pad to source bin")

        log.info("source %d uri=%s", index, uri)
        return source_bin

    def _decodebin_child_added(self, child_proxy, obj, name, user_data):
        if "source" in name:
            try:
                obj.set_property("latency", self.args.latency_ms)
                obj.set_property("drop-on-latency", True)
            except Exception:
                pass

    def _decodebin_pad_added(self, decodebin, pad, source_bin):
        caps = pad.get_current_caps()
        if caps is None:
            caps = pad.query_caps(None)
        caps_text = caps.to_string() if caps else ""
        if "video" not in caps_text:
            return
        if "memory:NVMM" not in caps_text:
            log.warning("decodebin produced non-NVMM video pad: %s", caps_text)
            return

        ghost_pad = source_bin.get_static_pad("src")
        if ghost_pad.set_target(pad):
            log.info("linked source pad caps=%s", caps_text)
        else:
            log.warning("failed to link source pad caps=%s", caps_text)

    def _write_infer_config(self, batch_size: int) -> str:
        root = os.path.dirname(os.path.abspath(__file__))
        generated_dir = os.path.join(root, ".generated")
        os.makedirs(generated_dir, exist_ok=True)

        onnx_path = os.path.abspath(self.args.yolo_model)
        labels_path = os.path.abspath(self.args.labels)
        parser_path = os.path.abspath(self.args.custom_parser)
        engine_path = os.path.abspath(self.args.engine_file or os.path.join(
            "models", "yolo_b%d_gpu0_fp16.engine" % batch_size
        ))
        config_path = os.path.join(generated_dir, "config_infer_primary_yolo_b%d.txt" % batch_size)

        if not os.path.exists(parser_path):
            raise FileNotFoundError(
                "custom parser not found: %s; build it with: make -C deepstream" % parser_path
            )
        if not os.path.exists(onnx_path):
            raise FileNotFoundError("YOLO model not found: %s" % onnx_path)

        filter_out = ";".join(str(i) for i in range(80) if i not in VEHICLE_CLASS_IDS)
        lines = [
            "[property]",
            "gpu-id=0",
            "net-scale-factor=0.0039215697906911373",
            "onnx-file=%s" % onnx_path,
            "model-engine-file=%s" % engine_path,
            "labelfile-path=%s" % labels_path,
            "infer-dims=3;%d;%d" % (self.args.infer_size, self.args.infer_size),
            "network-mode=2",
            "num-detected-classes=80",
            "filter-out-class-ids=%s" % filter_out,
            "interval=%d" % self.args.infer_interval,
            "gie-unique-id=1",
            "process-mode=1",
            "network-type=0",
            "maintain-aspect-ratio=1",
            "cluster-mode=2",
            "batch-size=%d" % batch_size,
            "workspace-size=1024",
            "parse-bbox-func-name=NvDsInferParseYolo",
            "custom-lib-path=%s" % parser_path,
            "output-blob-names=output0",
            "",
            "[class-attrs-all]",
            "topk=300",
            "pre-cluster-threshold=%.4f" % self.args.confidence,
            "nms-iou-threshold=%.4f" % self.args.nms_iou_threshold,
            "",
        ]
        with open(config_path, "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        return config_path

    def _pgie_src_pad_buffer_probe(self, pad, info, user_data):
        gst_buffer = info.get_buffer()
        if not gst_buffer:
            return Gst.PadProbeReturn.OK

        batch_meta = pyds.gst_buffer_get_nvds_batch_meta(hash(gst_buffer))
        if not batch_meta:
            return Gst.PadProbeReturn.OK

        l_frame = batch_meta.frame_meta_list
        while l_frame is not None:
            try:
                frame_meta = pyds.NvDsFrameMeta.cast(l_frame.data)
            except StopIteration:
                break

            source_id = int(frame_meta.source_id)
            camera_id = self.cameras[source_id][0] if source_id < len(self.cameras) else str(source_id)
            self.frames += 1
            self.window_frames += 1

            l_obj = frame_meta.obj_meta_list
            while l_obj is not None:
                try:
                    obj_meta = pyds.NvDsObjectMeta.cast(l_obj.data)
                except StopIteration:
                    break

                class_id = int(obj_meta.class_id)
                if class_id in VEHICLE_CLASS_IDS:
                    rect = obj_meta.rect_params
                    event = build_vehicle_log_event(
                        camera_id,
                        (float(rect.left), float(rect.top), float(rect.width), float(rect.height)),
                        VEHICLE_LABELS.get(class_id, str(class_id)),
                        float(obj_meta.confidence),
                    )
                    self.writer.write(event)
                    self.events += 1
                    self.window_events += 1

                try:
                    l_obj = l_obj.next
                except StopIteration:
                    break

            try:
                l_frame = l_frame.next
            except StopIteration:
                break

        self._log_window()
        return Gst.PadProbeReturn.OK

    def _log_window(self):
        now = time.monotonic()
        if now - self.window_start < 1.0:
            return
        elapsed = now - self.window_start
        log.info(
            "fps=%.2f frames=%d events=%d events_window=%d",
            self.window_frames / max(elapsed, 1e-6),
            self.frames,
            self.events,
            self.window_events,
        )
        self.window_start = now
        self.window_frames = 0
        self.window_events = 0

    def _on_bus_message(self, bus, message):
        msg_type = message.type
        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            log.error("GStreamer error: %s debug=%s", err, debug)
            if self.loop is not None:
                self.loop.quit()
        elif msg_type == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            log.warning("GStreamer warning: %s debug=%s", err, debug)
        elif msg_type == Gst.MessageType.EOS:
            log.warning("GStreamer EOS")
            if self.loop is not None:
                self.loop.quit()
        return True


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--single-camera", action="store_true")
    parser.add_argument("--input", default="rtsp://10.0.11.153:8554/cctv02")
    parser.add_argument("--camera-id", default="cctv02")
    parser.add_argument("--rtsp-prefix", default="rtsp://10.0.11.153:8554")
    parser.add_argument("--camera-start", type=int, default=1)
    parser.add_argument("--camera-count", type=int, default=5)
    parser.add_argument("--latency-ms", type=int, default=200)
    parser.add_argument("--mux-width", type=int, default=640)
    parser.add_argument("--mux-height", type=int, default=360)
    parser.add_argument("--mux-timeout-usec", type=int, default=40000)
    parser.add_argument("--infer-size", type=int, default=640)
    parser.add_argument("--infer-interval", type=int, default=1)
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--nms-iou-threshold", type=float, default=0.45)
    parser.add_argument("--yolo-model", default="models/yolo.onnx")
    parser.add_argument("--labels", default="models/coco_labels.txt")
    parser.add_argument("--custom-parser", default="deepstream/libnvdsinfer_custom_impl_Yolo.so")
    parser.add_argument("--engine-file", default="")
    parser.add_argument("--event-log", default="vehicle_log.jsonl")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )
    app = DeepStreamVehicleLogger(args)
    try:
        app.run()
    except KeyboardInterrupt:
        log.info("stopping")
        app.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
