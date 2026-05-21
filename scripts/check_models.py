#!/usr/bin/env python3
"""Quick ONNX/OpenCV DNN smoke test for the Jetson."""

import argparse
import json
import os
import time

import cv2
import numpy as np


def load_labels(path):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def run_forward(model_path, size):
    net = cv2.dnn.readNetFromONNX(model_path)
    image = np.zeros((size, size, 3), dtype=np.uint8)
    blob = cv2.dnn.blobFromImage(
        image,
        scalefactor=1.0 / 255.0,
        size=(size, size),
        swapRB=True,
        crop=False,
    )
    start = time.perf_counter()
    net.setInput(blob)
    output = net.forward()
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    return np.asarray(output).shape, elapsed_ms


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--yolo-model", default="models/yolo.onnx")
    parser.add_argument("--classifier-model", default="models/classifier.onnx")
    parser.add_argument("--labels", default="models/labels.json")
    parser.add_argument("--yolo-size", type=int, default=640)
    parser.add_argument("--classifier-size", type=int, default=224)
    args = parser.parse_args()

    print("cv2:", cv2.__version__)
    for path in (args.yolo_model, args.classifier_model, args.labels):
        print(("OK  " if os.path.exists(path) else "MISS"), path)

    labels = load_labels(args.labels)
    print("yolo labels:", labels.get("yolo", {}))
    print("classifier labels:", labels.get("classifier", {}))

    yolo_shape, yolo_ms = run_forward(args.yolo_model, args.yolo_size)
    print(f"YOLO forward shape={yolo_shape} elapsed_ms={yolo_ms:.1f}")

    cls_shape, cls_ms = run_forward(args.classifier_model, args.classifier_size)
    print(f"Classifier forward shape={cls_shape} elapsed_ms={cls_ms:.1f}")


if __name__ == "__main__":
    main()
