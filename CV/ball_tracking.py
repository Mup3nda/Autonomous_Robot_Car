from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time

ap = argparse.ArgumentParser()
ap.add_argument("-v","--video", 
                help="add path tp video (optional)")
ap.add_argument("-b", "--buffer",
                help="add buffer size")
args = ap.parse_args()



