#/***************************************************************************
#*   Copyright (C) 2025 by DTU
#*   jcan@dtu.dk
#*
#*
#* The MIT License (MIT)  https://mit-license.org/
#*
#* Permission is hereby granted, free of charge, to any person obtaining a copy of this software
#* and associated documentation files (the "Software"), to deal in the Software without restriction,
#* including without limitation the rights to use, copy, modify, merge, publish, distribute,
#* sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
#* is furnished to do so, subject to the following conditions:
#*
#* The above copyright notice and this permission notice shall be included in all copies
#* or substantial portions of the Software.
#*
#* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
#* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
#* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#* THE SOFTWARE. */
#
#  USB camera MJPEG stream server.
#  Opens VideoCapture(0) (HD Pro Webcam C920 at /dev/video0) and serves
#  frames as MJPEG over HTTP on port 7124 at /usb_camera
#
#  Run on the Raspberry Pi:
#    python3 usb_cam_stream.py
#
#  Connect a client (e.g. scam_usb.py) via:
#    http://<host>:7124/usb_camera

import cv2 as cv
from mjpeg_streamer import MjpegServer, Stream

CAMERA_INDEX = 0  # HD Pro Webcam C920 is at /dev/video0
PORT = 7124

cap = cv.VideoCapture(CAMERA_INDEX)
if not cap.isOpened():
  print(f"% usb_cam_stream: camera index {CAMERA_INDEX} failed to open")
  exit(1)

h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
print(f"% usb_cam_stream: USB camera opened ({h}x{w})")

stream = Stream("usb_camera", size=(w, h), quality=80, fps=30)
server = MjpegServer("0.0.0.0", PORT)
server.add_stream(stream)
server.start()
print(f"% usb_cam_stream: streaming on http://0.0.0.0:{PORT}/usb_camera")

try:
  while True:
    ret, frame = cap.read()
    if ret:
      stream.set_frame(frame)
finally:
  server.stop()
  cap.release()
  print("% usb_cam_stream: stopped")
