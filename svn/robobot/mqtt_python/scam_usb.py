import cv2 as cv
from threading import Thread
import time as t
from datetime import *
from uservice import service

class SUsbCam:

  cap = {}        # capture device
  th = {}         # thread
  savedFrame = {}
  frameTime = datetime.now()
  getFrame = True
  cnt = 0
  useCam = True
  imageFailCnt = 0
  stop = False
  camhost = '0.0.0.0'
  port = 7124

  def setup(self):
    if self.useCam:
      #self.cap = cv.VideoCapture(f'http://{service.host}:7124/usb_camera')
      self.cap = cv.VideoCapture(f'http://{self.camhost}:{self.port}/stream.mjpeg')
      if self.cap.isOpened():
        print(f"% SUsbCam:: Connected to {service.host}")
        self.th = Thread(target=cam_usb.run)
        self.th.start()
      else:
        print(f"% SUsbCam:: Failed to connect to {service.host}")
        self.terminate()
    else:
      print("% SUsbCam:: Camera disabled (in scam_usb.py)")
    print("# cam_usb setup finished")

  def getImage(self):
    fail = False
    if not self.useCam:
      if self.imageFailCnt == 0:
        print("% SUsbCam:: not using cam")
      fail = True
    if not fail and not self.cap.isOpened():
      if self.imageFailCnt == 0:
        print("% SUsbCam:: could not open")
      fail = True
    if not fail:
      self.getFrame = True
      cnt = 0  # timeout
      while self.getFrame and cnt < 100 and not service.stop:
        t.sleep(0.01)
        cnt += 1
      fail = self.getFrame
    if fail:
      self.imageFailCnt += 1
      return False, self.savedFrame, self.frameTime
    else:
      self.imageFailCnt = 0
      return True, self.savedFrame, self.frameTime

  def run(self):
    print("% SUsbCam:: camera thread running")
    first = True
    ret = False
    while self.cap.isOpened() and not self.stop:
      if self.getFrame or first:
        try:
          ret, self.savedFrame = self.cap.read()
        except:
          ret = False
        self.frameTime = datetime.now()
        if ret:
          self.getFrame = False
          self.cnt += 1
          if first:
            first = False
            h, w, ch = self.savedFrame.shape
            print(f"% SUsbCam:: Camera available: size ({h}x{w}, {ch} channels)")
      else:
        # just discard unused images
        self.cap.read()
      if not ret:
        print("% SUsbCam:: Failed to receive frame (stream end?). Exiting ...")
        self.terminate()
    print("% SUsbCam:: Camera thread stopped")

  def terminate(self):
    self.stop = True
    try:
      self.th.join()
    except:
      print("% SUsbCam:: join cam failed")
      pass
    if isinstance(self.cap, cv.VideoCapture):
      self.cap.release()
    cv.destroyAllWindows()
    print("% SUsbCam:: Camera terminated")

# create instance of this class
cam_usb = SUsbCam()
