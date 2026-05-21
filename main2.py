import cv2
import time
import math

# =============================
# CONFIG
# =============================
rtsp_in = "rtsp://10.0.11.153:8554/cctv08"
rtsp_out = "rtsp://127.0.0.1:8554/jetson"

w, h = 480, 270
process_interval = 0.05

# =============================
# NVDEC INPUT PIPELINE
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
# RTSP OUTPUT PIPELINE (NVENC)
# =============================
out_pipeline = (
    "appsrc ! videoconvert ! video/x-raw,format=BGRx ! "
    "nvvidconv ! nvv4l2h264enc bitrate=2000000 ! "
    "h264parse ! rtph264pay config-interval=1 pt=96 ! "
    "udpsink host=127.0.0.1 port=8554"
)

out = cv2.VideoWriter(
    out_pipeline,
    cv2.CAP_GSTREAMER,
    20,
    (w, h),
    True
)

# =============================
# MOTION PIPELINE
# =============================
prev = None
last = 0

def adaptive_merge(boxes, dist=180):
    merged = []

    for x, y, bw, bh in boxes:
        cx, cy = x + bw // 2, y + bh // 2
        found = False

        for i, (mx, my, mw, mh) in enumerate(merged):
            mcx, mcy = mx + mw // 2, my + mh // 2

            if math.hypot(cx - mcx, cy - mcy) < dist:
                nx1 = min(x, mx)
                ny1 = min(y, my)
                nx2 = max(x + bw, mx + mw)
                ny2 = max(y + bh, my + mh)

                merged[i] = (nx1, ny1, nx2 - nx1, ny2 - ny1)
                found = True
                break

        if not found:
            merged.append((x, y, bw, bh))

    return merged

# =============================
# LOOP
# =============================
try:
    while True:

        ret, frame = cap.read()
        if not ret:
            break

        now = time.perf_counter()
        if now - last < process_interval:
            continue
        last = now

        # ---------------- resize
        frame = cv2.resize(frame, (w, h))

        # ---------------- grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev is None:
            prev = gray
            continue

        # ---------------- motion detect
        diff = cv2.absdiff(prev, gray)
        _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        th = cv2.dilate(th, None, 1)

        prev = gray

        # ---------------- contours
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        boxes = []
        for c in contours[:10]:
            if cv2.contourArea(c) < 1000:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            boxes.append((x, y, bw, bh))

        merged = adaptive_merge(boxes)

        # ---------------- DRAW BOX
        for x, y, bw, bh in merged:
            cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

        # ---------------- SEND TO RTSP
        out.write(frame)

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    out.release()
    print("stopped")