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

resized_w, resized_h = 480, 270
process_interval = 0.05

# ตัวแปร Global สำหรับเก็บภาพล่าสุดเตรียมส่งออกสตรีม
latest_frame = None

# =============================
# INPUT (NVDEC)
# =============================
def build_input(url):
    return (
        f"rtspsrc location={url} latency=0 drop-on-latency=true ! "
        f"rtph264depay ! h264parse ! nvv4l2decoder ! "
        f"nvvidconv ! video/x-raw,width={resized_w},height={resized_h},format=BGRx ! "
        f"videoconvert ! video/x-raw,format=BGR ! "
        f"appsink drop=1 sync=false"
    )

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
# CAMERA & PROCESS THREAD
# =============================
def process_camera():
    global latest_frame
    cap = cv2.VideoCapture(build_input(rtsp_in), cv2.CAP_GSTREAMER)
    
    prev = None
    last_time = 0
    merged = []

    print("[INFO] Camera started...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Stream disconnected, retrying...")
            time.sleep(2)
            cap = cv2.VideoCapture(build_input(rtsp_in), cv2.CAP_GSTREAMER)
            continue

        now = time.perf_counter()
        if now - last_time >= process_interval:
            last_time = now
            
            # แปลงภาพเพื่อหา Motion
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray = cv2.GaussianBlur(gray, (5, 5), 0)

            if prev is not None:
                diff = cv2.absdiff(prev, gray)
                _, th = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
                th = cv2.dilate(th, None, iterations=1)

                contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                boxes = []
                for c in contours[:10]:
                    if cv2.contourArea(c) < 1000:
                        continue
                    x, y, w, h = cv2.boundingRect(c)
                    boxes.append((x, y, w, h))

                merged = adaptive_merge(boxes)
            
            prev = gray

        # วาดกล่องลงเฟรม
        for x, y, w, h in merged:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # อัปเดตเฟรมล่าสุดให้ RTSP Server หยิบไปใช้
        latest_frame = frame.copy()

# =============================
# RTSP SERVER PIPELINE
# =============================
class GstServerFactory(GstRtspServer.RTSPMediaFactory):
    def __init__(self):
        super(GstServerFactory, self).__init__()
        # ท่อส่งภาพออกไปเป็น H264
        self.launch_string = (
            "appsrc name=source is-live=true block=true format=GST_FORMAT_TIME "
            f"caps=video/x-raw,format=BGR,width={resized_w},height={resized_h},framerate=20/1 ! "
            "videoconvert ! video/x-raw,format=BGRx ! nvvidconv ! "
            "nvv4l2h264enc bitrate=2000000 insert-sps-pps=true ! "
            "h264parse ! rtph264pay name=pay0 pt=96"
        )
        self.number_frames = 0
        self.duration = 1 / 20 * Gst.SECOND

    def do_create_element(self, url):
        return Gst.parse_launch(self.launch_string)

    def do_configure(self, rtsp_media):
        self.number_frames = 0
        appsrc = rtsp_media.get_element().get_child_by_name('source')
        appsrc.connect('need-data', self.on_need_data)

    def on_need_data(self, src, length):
        global latest_frame
        if latest_frame is not None:
            data = latest_frame.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            buf.duration = self.duration
            timestamp = self.number_frames * self.duration
            buf.pts = buf.dts = int(timestamp)
            buf.offset = timestamp
            self.number_frames += 1
            src.emit('push-buffer', buf)

# =============================
# MAIN RUNNER
# =============================
if __name__ == '__main__':
    Gst.init(None)

    # รันตัวประมวลผลกล้องแยกไปอีก Thread
    cam_thread = threading.Thread(target=process_camera, daemon=True)
    cam_thread.start()

    # ตั้งค่าเซิร์ฟเวอร์
    server = GstRtspServer.RTSPServer()
    server.set_service(rtsp_port)
    factory = GstServerFactory()
    factory.set_shared(True)
    server.get_mount_points().add_factory(rtsp_mount, factory)
    server.attach(None)

    print(f"\n[SUCCESS] RTSP Server is running!")
    print(f"Watch the stream at: rtsp://<JETSON_IP>:{rtsp_port}{rtsp_mount}")
    
    # รัน MainLoop ให้ Server เปิดรอรับคอนเนกชัน
    loop = GLib.MainLoop()
    try:
        loop.run()
    except KeyboardInterrupt:
        print("Stopping...")
        loop.quit()