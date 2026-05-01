
import cv2
from mjpeg_streamer import MjpegServer, Stream

CAMERA_INDEX = 0  # HD Pro Webcam C920 is at /dev/video0
PORT = 7124

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
  print(f"% usb_cam_stream: camera  /dev/video{CAMERA_INDEX} failed to open")
  exit(1)

# cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(f"% usb_cam_stream: USB camera opened ({w}x{h})")

stream = Stream("usb_stream.mjpg", size=(w, h), quality=80, fps=30)
#stream = Stream("stream", size=(w, h), quality=80, fps=30)
server = MjpegServer("0.0.0.0", PORT)
server.add_stream(stream)
server.start()
print("% USE THE STREAMING ADRESS BELOW!!!!!!!")
print(f"% USB WEBCAM streaming on http://0.0.0.0:{PORT}/usb_stream.mjpg")
print(f"% TO KILL STREAM: sudo pkill -f usb_stream_server.py")
print(f"% START STREAM: python3 ~/Autonomous_Robot_Car/svn/robobot/stream_server/usb_stream_server.py")

try:
  while True:
    ret, frame = cap.read()
    if ret:
      stream.set_frame(frame)
finally:
  server.stop()
  cap.release()
  print("% usb_cam_stream: stopped")
