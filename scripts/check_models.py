#!/usr/bin/env python3
"""Quick ONNX smoke test for the Jetson."""

import argparse
import json
import os
import time

import cv2
import numpy as np


def choose_backend(requested):
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
        "or run robust_rtsp_relay.py --no-models"
    )


def load_labels(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def make_blob(image, size):
    resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    blob = rgb.astype(np.float32) / 255.0
    return np.transpose(blob, (2, 0, 1))[np.newaxis, ...]


def run_forward(model_path, size, backend):
    image = np.zeros((size, size, 3), dtype=np.uint8)
    blob = make_blob(image, size)
    start = time.perf_counter()
    if backend == "opencv":
        net = cv2.dnn.readNetFromONNX(model_path)
        net.setInput(blob)
        output = net.forward()
    else:
        import onnxruntime as ort

        net = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
        output = net.run(None, {net.get_inputs()[0].name: blob})[0]
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return np.asarray(output).shape, elapsed_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-model", default="models/yolo.onnx")
    parser.add_argument("--classifier-model", default="models/classifier.onnx")
    parser.add_argument("--labels", default="models/labels.json")
    parser.add_argument("--yolo-size", type=int, default=640)
    parser.add_argument("--classifier-size", type=int, default=224)
    parser.add_argument("--backend", choices=("auto", "opencv", "onnxruntime"), default="auto")
    args = parser.parse_args()

    print("cv2:", cv2.__version__)
    for path in (args.yolo_model, args.classifier_model, args.labels):
        print(("OK  " if os.path.exists(path) else "MISS"), path)

    labels = load_labels(args.labels)
    print("yolo labels:", labels.get("yolo", {}))
    print("classifier labels:", labels.get("classifier", {}))
    backend = choose_backend(args.backend)
    print("backend:", backend)

    yolo_shape, yolo_ms = run_forward(args.yolo_model, args.yolo_size, backend)
    print(f"YOLO forward shape={yolo_shape} elapsed_ms={yolo_ms:.1f}")

    cls_shape, cls_ms = run_forward(args.classifier_model, args.classifier_size, backend)
    print(f"Classifier forward shape={cls_shape} elapsed_ms={cls_ms:.1f}")


if __name__ == "__main__":
    main()
