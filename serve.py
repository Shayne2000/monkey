import cv2
import time
import math
import threading
import gi
import numpy as np

gi.require_version('Gst', '1.0')
gi.require_version('GstRtspServer', '1.0')
from gi.repository import Gst, GstRtspServer, GLib

# =============================
# CONFIG
# =============================
rtsp_in = "rtsp://10.0.11.153:8554/cctv02"
rtsp_port = "8554"
rtsp_mount = "/jetson"

W, H = 480, 270
FPS = 15
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
    return (
        "rtspsrc location=" + url + " protocols=tcp latency=500 drop-on-latency=true ! "
        "rtph264depay ! h264parse ! avdec_h264 ! "
        "videoconvert ! video/x-raw,format=BGR ! "
        "appsink drop=1 sync=false max-buffers=1"
    )

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

    def open_cap():
        cap = cv2.VideoCapture(build_input(rtsp_in), cv2.CAP_GSTREAMER)
        print("cap A")
        time.sleep(1)
        print("cap b")
        return cap

    cap = open_cap()
    print("cap c")

    prev = None
    last = 0
    merged = []

    print("[INFO] Camera started")

    while True:

        ret, frame = cap.read()

        if not ret or frame is None:
            print("[WARN] reconnect RTSP")
            cap.release()
            time.sleep(1)
            cap = open_cap()
            print(not ret or frame is None)
            continue

        now = time.perf_counter()
        if now - last < process_interval:
            print("13.5")
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
            for c in contours:
                if cv2.contourArea(c) < 1000:
                    continue
                x, y, w, h = cv2.boundingRect(c)
                boxes.append((x, y, w, h))

            merged = adaptive_merge(boxes)

        prev = gray

        for x, y, w, h in merged:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        with frame_lock:
            latest_frame = frame

# =============================
# RTSP SERVER
# =============================
class Factory(GstRtspServer.RTSPMediaFactory):

    def __init__(self):
        super().__init__()
        self.set_shared(True)

        self.launch = (
            "appsrc name=source is-live=true block=true format=GST_FORMAT_TIME "
            f"caps=video/x-raw,format=BGR,width={W},height={H},framerate={FPS}/1 ! "
            "videoconvert ! "
            "video/x-raw,format=NV12 ! "
            "nvvidconv ! video/x-raw(memory:NVMM),format=NV12 ! "
            "nvv4l2h264enc bitrate=2000000 insert-sps-pps=true maxperf-enable=1 ! "
            "h264parse ! rtph264pay name=pay0 pt=96 config-interval=1"
        )

        print('test2')

        self.frame_id = 0

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch)

    def do_configure(self, media):
        appsrc = media.get_element().get_child_by_name("source")
        appsrc.connect("need-data", self.on_need_data)

    def on_need_data(self, src, length):
        global latest_frame

        print("push frame2")

        with frame_lock:
            if latest_frame is None:
                print("latest_frame is None")
                return
            frame = latest_frame.copy()

        print("check point1")

        frame = cv2.resize(frame, (W, H))
        frame = np.ascontiguousarray(frame)

        buf = Gst.Buffer.new_allocate(None, frame.nbytes, None)
        buf.fill(0, frame.tobytes())

        buf.pts = buf.dts = int(self.frame_id * Gst.SECOND / FPS)
        buf.duration = int(Gst.SECOND / FPS)
        buf.offset = self.frame_id

        self.frame_id += 1

        src.emit("push-buffer", buf)

        print("check point2")

# =============================
# MAIN
# =============================
if __name__ == "__main__":

    Gst.init(None)

    t = threading.Thread(target=camera_loop, daemon=True)
    t.start()

    server = GstRtspServer.RTSPServer()
    server.set_service(rtsp_port)
    print("push frame3")

    factory = Factory()
    server.get_mount_points().add_factory(rtsp_mount, factory)
    server.attach(None)

    print('test')

    print(f"RTSP: rtsp://<JETSON_IP>:{rtsp_port}{rtsp_mount}")

    loop = GLib.MainLoop()
    loop.run()