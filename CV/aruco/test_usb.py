
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), '../svn/robobot/mqtt_python'))

from scam_usb import cam_usb
import cv2

# # Start the camera connection
cam_usb.setup()

# Get a frame
ok, frame, timestamp = cam_usb.getImage()
if ok:
    cv2.imshow("cam", frame)

# When done
cam_usb.terminate()

##_---------------------------