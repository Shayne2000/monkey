#!/usr/bin/env bash
set -u

echo "== Python =="
python3 --version || true
python3 - <<'PY' || true
try:
    import cv2
    print("cv2:", cv2.__version__)
    info = cv2.getBuildInformation()
    for line in info.splitlines():
        if "GStreamer" in line or "FFMPEG" in line:
            print(line)
except Exception as exc:
    print("cv2 import failed:", repr(exc))
try:
    import gi
    print("gi: ok")
except Exception as exc:
    print("gi import failed:", repr(exc))
try:
    import pyds
    print("pyds: ok")
except Exception as exc:
    print("pyds import failed:", repr(exc))
PY

echo
echo "== GStreamer core =="
for el in rtspsrc rtph264depay h264parse avdec_h264 videoconvert appsink; do
  if gst-inspect-1.0 "$el" >/dev/null 2>&1; then
    echo "OK   $el"
  else
    echo "MISS $el"
  fi
done

echo
echo "== NVIDIA / Jetson plugins =="
for el in nvv4l2decoder omxh264dec nvvidconv nvvideoconvert; do
  if gst-inspect-1.0 "$el" >/dev/null 2>&1; then
    echo "OK   $el"
  else
    echo "MISS $el"
  fi
done

echo
echo "== DeepStream plugins =="
for el in nvstreammux nvurisrcbin nvinfer nvstreamdemux nvmultistreamtiler nvdsosd; do
  if gst-inspect-1.0 "$el" >/dev/null 2>&1; then
    echo "OK   $el"
  else
    echo "MISS $el"
  fi
done

echo
echo "== DeepStream folders =="
ls -ld /opt/nvidia/deepstream* 2>/dev/null || echo "No /opt/nvidia/deepstream* folder found"
