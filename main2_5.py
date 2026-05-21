import cv2
import time
import math

# =============================
# CONFIG
# =============================
rtsp_in = "rtsp://10.0.11.153:8554/cctv08"
rtsp_out = "rtsp://192.168.1.10:8554/jetson"  # 👈 PC IP

resized_w, resized_h = 480, 270
process_interval = 0.05

# =============================
# INPUT (NVDEC)
# =============================
def build_input(url):
    return (
        f"rtspsrc location={url} latency=0 drop-on-latency=true ! "
        f"rtph264depay ! h264parse ! "
        f"nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw,format=BGRx ! "
        f"videoconvert ! appsink drop=1 sync=false"
    )

cap = cv2.VideoCapture(build_input(rtsp_in), cv2.CAP_GSTREAMER)

# =============================
# OUTPUT (RTSP STREAM OUT)
# =============================
def build_output(url):
    return (
        f"appsrc ! videoconvert ! nvvidconv ! "
        f"nvv4l2h264enc bitrate=2000000 ! "
        f"h264parse ! rtspclientsink location={url}"
    )

out = cv2.VideoWriter(
    build_output(rtsp_out),
    cv2.CAP_GSTREAMER,
    0,
    20,
    (resized_w, resized_h)
)

# =============================
# MERGE
# =============================
def adaptive_merge(boxes, dist_th=180):
    merged = []

    for x, y, w, h in boxes:
        cx, cy = x + w // 2, y + h // 2
        found = False

        for i, (mx, my, mw, mh) in enumerate(merged):
            mcx, mcy = mx + mw // 2, my + mh // 2

            if math.hypot(cx - mcx, cy - mcy) < dist_th:
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

# =============================
# LOOP
# =============================
prev = None
last_time = 0

try:
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        # throttle
        now = time.perf_counter()
        if now - last_time < process_interval:
            continue
        last_time = now

        # =========================
        # PROCESS
        # =========================
        frame = cv2.resize(frame, (resized_w, resized_h))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev is None:
            prev = gray
            continue

        diff = cv2.absdiff(prev, gray)
        _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        th = cv2.dilate(th, None, 1)

        prev = gray

        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours[:10]:
            if cv2.contourArea(c) < 1000:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h))

        merged = adaptive_merge(boxes)

        for x, y, w, h in merged:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # =========================
        # SEND TO LOCAL HOST (RTSP)
        # =========================
        out.write(frame)

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    out.release()
    print("Stopped")