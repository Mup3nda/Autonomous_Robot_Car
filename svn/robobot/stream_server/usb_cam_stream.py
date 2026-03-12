
import cv2
from mjpeg_streamer import MjpegServer, Stream

CAMERA_INDEX = 0  # HD Pro Webcam C920 is at /dev/video0
PORT = 7124

cap = cv2.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
  print(f"% usb_cam_stream: camera  /dev/video{CAMERA_INDEX} failed to open")
  exit(1)

h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
print(f"% usb_cam_stream: USB camera opened ({w}x{h})")

stream = Stream("usb_camera", size=(w, h), quality=80, fps=30)
server = MjpegServer("0.0.0.0", PORT)
server.add_stream(stream)
server.start()
print("USE THE STREAMING ADRESS BELOW!!!!!!!")
print(f"% USB WEBCAM streaming on http://0.0.0.0:{PORT}/usb_camera")

try:
  while True:
    ret, frame = cap.read()
    if ret:
      stream.set_frame(frame)
finally:
  server.stop()
  cap.release()
  print("% usb_cam_stream: stopped")
