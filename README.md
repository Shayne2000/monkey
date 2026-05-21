# Monkey RTSP Motion Reader

This branch contains a Jetson RTSP motion reader that writes `vehicle_log`-shaped JSONL events.

## Install base dependencies on Jetson

```bash
bash scripts/install_jetson_deps.sh
```

Then check the environment:

```bash
bash scripts/check_jetson_env.sh
```

## Run

```bash
python3 robust_rtsp_relay.py
```

With preview on a Jetson display:

```bash
python3 robust_rtsp_relay.py --display
```

## Important

DeepStream plugins such as `nvstreammux`, `nvurisrcbin`, and `nvinfer` are NVIDIA system plugins. They do not come from pip or a copied `.venv`.

Use `scripts/check_jetson_env.sh` to see what is missing.
