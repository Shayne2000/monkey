#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Installing base Python + GStreamer packages"
sudo apt-get update
sudo apt-get install -y \
  python3-pip \
  python3-gi \
  python3-gst-1.0 \
  gir1.2-gstreamer-1.0 \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav \
  python3-opencv

echo "[INFO] Installing Python API packages"
python3 -m pip install --user --upgrade pip
python3 -m pip install --user fastapi uvicorn

echo
echo "[INFO] Base install done. Checking environment..."
bash "$(dirname "$0")/check_jetson_env.sh"

cat <<'EOF'

[NEXT]
If these are still missing:
  - nvstreammux
  - nvurisrcbin
  - nvinfer
then DeepStream is not installed or its plugin path is not visible.

If these are missing:
  - nvv4l2decoder
  - nvvidconv
then Jetson multimedia/GStreamer plugins are not installed or not visible.

DeepStream is a system NVIDIA install, not a pip package and not solved by copying .venv.
Install the DeepStream version that matches your JetPack, then re-run:

  bash scripts/check_jetson_env.sh

EOF
