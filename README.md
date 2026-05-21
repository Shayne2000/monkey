# Monkey RTSP Motion Reader

Jetson RTSP reader with a lightweight 2-frame motion gate. When motion is found,
the code can run YOLO detection and a brand classifier before writing
`vehicle_log`-shaped JSONL events.

## Install base dependencies on Jetson

```bash
bash scripts/install_jetson_deps.sh
```

Then check the environment:

```bash
bash scripts/check_jetson_env.sh
```

## Run

Copy model files onto the Jetson first:

```bash
mkdir -p models
cp /path/to/yolo.onnx models/yolo.onnx
cp /path/to/classifier.onnx models/classifier.onnx
```

`models/labels.json` is already tracked in this branch. Large model files are
ignored by git.

Check model loading:

```bash
python3 scripts/check_models.py
```

If Jetson OpenCV is old and prints `module 'cv2' has no attribute 'dnn'`,
run motion logging first with:

```bash
python3 robust_rtsp_relay.py --no-models
```

The model path needs OpenCV DNN/ONNX support or a later DeepStream `nvinfer`
integration.

Run all 5 default cameras:

```bash
python3 robust_rtsp_relay.py
```

Run only one camera:

```bash
python3 robust_rtsp_relay.py --single-camera --input rtsp://10.0.11.153:8554/cctv02 --camera-id cctv02
```

Disable model inference and log motion only:

```bash
python3 robust_rtsp_relay.py --no-models
```

## Important

DeepStream plugins such as `nvstreammux`, `nvurisrcbin`, and `nvinfer` are NVIDIA system plugins. They do not come from pip or a copied `.venv`.

Use `scripts/check_jetson_env.sh` to see what is missing.

Current model inference uses OpenCV DNN (`cv2.dnn.readNetFromONNX`) after the
motion gate. DeepStream `nvinfer` remains a later integration path if the Jetson
runtime has the required plugins.
