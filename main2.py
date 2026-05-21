import cv2
import time
import math

# =============================
# CONFIG
# =============================
rtsp_url = "rtsp://10.0.11.153:8554/cctv08"

resized_w, resized_h = 480, 270
process_interval = 0.05  # ~20 FPS cap

# =============================
# NVDEC GSTREAMER PIPELINE
# =============================
def build_pipeline(url):
    return (
        f"rtspsrc location={url} latency=0 drop-on-latency=true ! "
        f"rtph264depay ! h264parse ! "
        f"nvv4l2decoder ! "               # 🔥 NVDEC (hardware decode)
        f"nvvidconv ! video/x-raw,format=BGRx ! "
        f"videoconvert ! "
        f"appsink drop=1 sync=false"
    )

cap = cv2.VideoCapture(build_pipeline(rtsp_url), cv2.CAP_GSTREAMER)

# =============================
# STATS
# =============================
stats = {
    "resize": [],
    "gray": [],
    "blur": [],
    "diff": [],
    "thresh": [],
    "contour": [],
    "merge": [],
    "total": []
}

# =============================
# ADAPTIVE MERGE (LIGHT)
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

        # -------------------------
        # throttle
        # -------------------------
        now = time.perf_counter()
        if now - last_time < process_interval:
            continue
        last_time = now

        t0 = time.perf_counter()

        # =========================
        # PROCESS PIPELINE (CPU only after decode)
        # =========================

        t = time.perf_counter()
        frame = cv2.resize(frame, (resized_w, resized_h))
        stats["resize"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        stats["gray"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        stats["blur"].append((time.perf_counter() - t) * 1000)

        if prev is None:
            prev = gray
            continue

        t = time.perf_counter()
        diff = cv2.absdiff(prev, gray)
        stats["diff"].append((time.perf_counter() - t) * 1000)

        t = time.perf_counter()
        _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
        th = cv2.dilate(th, None, 1)
        stats["thresh"].append((time.perf_counter() - t) * 1000)

        prev = gray

        # =========================
        # CONTOURS
        # =========================
        t = time.perf_counter()
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        stats["contour"].append((time.perf_counter() - t) * 1000)

        boxes = []
        for c in contours[:10]:
            if cv2.contourArea(c) < 1000:
                continue
            x, y, w, h = cv2.boundingRect(c)
            boxes.append((x, y, w, h))

        # =========================
        # MERGE
        # =========================
        t = time.perf_counter()
        merged = adaptive_merge(boxes)
        stats["merge"].append((time.perf_counter() - t) * 1000)

        # =========================
        # DRAW
        # =========================
        for x, y, w, h in merged:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # =========================
        # FPS REPORT
        # =========================
        stats["total"].append((time.perf_counter() - t0) * 1000)

        if len(stats["total"]) >= 30:

            print("\n===== NVDEC JETSON PROFILING =====")

            for k in stats:
                avg = sum(stats[k]) / len(stats[k])
                print(f"{k:10s}: {avg:.2f} ms")

            cpu_ms = sum(stats["total"]) / len(stats["total"])
            cpu_fps = 1000 / cpu_ms

            print("----------------------------------")
            print(f"Pipeline FPS: {cpu_fps:.2f}")
            print("==================================\n")

            for k in stats:
                stats[k].clear()

except KeyboardInterrupt:
    pass

finally:
    cap.release()
    print("Stopped")