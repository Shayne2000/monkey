import cv2
import time
import math
import threading
import gi

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

# =============================
# CONFIG
# =============================
rtsp_in = "rtsp://10.0.11.153:8554/cctv08"
rtsp_port = "8554"
rtsp_mount = "/jetson"

W, H = 480, 270
FPS = 20
process_interval = 1.0 / FPS

# =============================
# GLOBAL FRAME (SAFE)
# =============================
latest_frame = None
frame_lock = threading.Lock()

# =============================
# NVDEC INPUT
# =============================
def build_input(url):
    pipeline = (
        "rtspsrc location=" + url + " latency=0 drop-on-latency=true ! "
        "rtph264depay ! h264parse ! "
        "nvv4l2decoder ! "
        "nvvidconv ! video/x-raw,format=BGRx ! "
        "videoconvert ! "
        "appsink drop=1 sync=false"
    )
    return pipeline

# =============================
# MERGE BOX
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
# CAMERA THREAD
# =============================
def camera_loop():
    global latest_frame

    cap = cv2.VideoCapture(build_input(rtsp_in), cv2.CAP_GSTREAMER)

    prev = None
    last = 0
    merged = []

    print("[INFO] Camera started")

    while True:
        ret, frame = cap.read()
        if not ret:
            time.sleep(1)
            cap = cv2.VideoCapture(build_input(rtsp_in), cv2.CAP_GSTREAMER)
            continue

        now = time.perf_counter()
        if now - last < process_interval:
            continue
        last = now

        frame = cv2.resize(frame, (W, H))

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        if prev is not None:
            diff = cv2.absdiff(prev, gray)
            _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
            th = cv2.dilate(th, None, 1)

            contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            boxes = []
            for c in contours[:10]:
                if cv2.contourArea(c) < 1000:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                boxes.append((x, y, w, h))

            merged = adaptive_merge(boxes)

        prev = gray

        for x, y, w, h in merged:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        with frame_lock:
            latest_frame = frame.copy()

# =============================
# RTSP SERVER
# =============================
class Factory(GstRtspServer.RTSPMediaFactory):

    def __init__(self):
        super().__init__()
        self.set_shared(True)

        self.launch = (
            f"appsrc name=source is-live=true format=GST_FORMAT_TIME "
            f"caps=video/x-raw,format=BGR,width={W},height={H},framerate={FPS}/1 ! "
            f"videoconvert ! nvvidconv ! "
            f"nvv4l2h264enc bitrate=2000000 insert-sps-pps=true ! "
            f"h264parse ! rtph264pay name=pay0 pt=96"
        )

        self.frame_duration = int(1e9 / FPS)
        self.frame_id = 0

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch)

    def do_configure(self, media):
        appsrc = media.get_element().get_child_by_name("source")
        appsrc.connect("need-data", self.on_need_data)

    def on_need_data(self, src, length):
        global latest_frame

        with frame_lock:
            if latest_frame is None:
                return
            frame = latest_frame.copy()

        data = frame.tobytes()

        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)

        buf.duration = self.frame_duration
        buf.pts = buf.dts = self.frame_id * self.frame_duration

        self.frame_id += 1

        src.emit("push-buffer", buf)

# =============================
# MAIN
# =============================
if __name__ == "__main__":

    Gst.init(None)

    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

    server = GstRtspServer.RTSPServer()
    server.set_service(rtsp_port)

    factory = Factory()
    server.get_mount_points().add_factory(rtsp_mount, factory)
    server.attach(None)

    print(f"RTSP: rtsp://<JETSON_IP>:{rtsp_port}{rtsp_mount}")

    loop = GLib.MainLoop()
    loop.run()