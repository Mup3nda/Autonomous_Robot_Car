#/***************************************************************************
#*   Copyright (C) 2024 by DTU
#*   jcan@dtu.dk
#*
#*
#* The MIT License (MIT)  https://mit-license.org/
#*
#* Permission is hereby granted, free of charge, to any person obtaining a copy of this software
#* and associated documentation files (the “Software”), to deal in the Software without restriction,
#* including without limitation the rights to use, copy, modify, merge, publish, distribute,
#* sublicense, and/or sell copies of the Software, and to permit persons to whom the Software
#* is furnished to do so, subject to the following conditions:
#*
#* The above copyright notice and this permission notice shall be included in all copies
#* or substantial portions of the Software.
#*
#* THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
#* INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
#* PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE
#* FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
#* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
#* THE SOFTWARE. */


from datetime import *
import time as t
from threading import Thread
import cv2 as cv
from ulog import flog

class SEdge:
    # raw AD values
    edge = [0, 0, 0 , 0, 0, 0, 0, 0]
    edgeUpdCnt = 0
    edgeTime = datetime.now()
    edgeInterval = 0
    # normalizing white values
    edge_n_w = [0, 0, 0 , 0, 0, 0, 0, 0]
    edge_n_wUpdCnt = 0
    edge_n_wTime = datetime.now()
    # normalized after white calibration
    edge_n = [0, 0, 0 , 0, 0, 0, 0, 0]
    edge_nUpdCnt = 0
    edge_nTime = datetime.now()
    edge_nInterval = 0
    edgeIntervalSetup = 0.1
    # line detection levels
    lineValidThreshold = 750 # 1000 is calibrated white
    crossingThreshold = 700 # average above this is assumed to be crossing line
    # level for relevant white values
    low = lineValidThreshold - 100;
    # line detection values
    posLeft = 0.0
    posRight = 0.0
    followLeft = True
    refPosition = 0.0 # distance from detected edge
    lineValid = False
    lineValidCnt = 0 # a value up to 20 for most confident line detect
    lineLastSeenTime = datetime.now() # timestamp of last valid line detection
    crossingLine = False
    crossingLineCnt = 0  # a value up to 20 for most confident crossing line
    average = 0
    high = 0 # highest reflectivity
    low = 0  # the darkest value found in latest sample
    #
    topicLip = ""
    sendCalibRequest = False
    #
    # follow line controller
    lineCtrl = False # private
    # Motor velocity limits
    wheelbase = 0.23  # Distance between wheels (m)
    maxWheelVel = 1.3  # Maximum wheel velocity (m/s)
    # PID profiles for different velocities
    pidProfiles = {
        'slow': {
            'Kp': 1.5,    # More aggressive proportional gain for slow speeds
            'Ki': 0.4,    # Increased integral gain
            'Kd': 0.15,   # Increased derivative gain
            'derivativeAlpha': 0.5,    # More low-pass filtering for noise reduction
            'maxIntegral': 1.5
        },
        'medium': {
            'Kp': 1.0,    # Moderate proportional gain
            'Ki': 0.45,    # Moderate integral gain
            'Kd': 0.11,   # Moderate derivative gain
            'derivativeAlpha': 0.5,   # Moderate filtering
            'maxIntegral': 1.1
        },
        'fast': {
            'Kp': 0.8,   # Original proportional gain (works well at 0.95 m/s)
            'Ki': 0.2,    # Original integral gain
            'Kd': 0.25,   # Original derivative gain
            'derivativeAlpha': 0.6,    # Original low-pass filter
            'maxIntegral': 1.2
        }
    }
    # Currently active profile parameters
    lineKp = 0.75  # Proportional gain (rad/s per sensor value)
    lineKi = 0.2  # Integral gain (rad/s per (sensor value * sec))
    lineKd = 0.08 # Derivative gain (rad/s per (sensor value / sec)))
    derivativeAlpha = 0.6  # Low-pass filter for derivative (0-1, lower = more filtering)
    maxIntegral = 1.2 # anti-windup limit for integral term
    currentProfile = 'fast'  # Track which profile is active
    # PID state variables
    lineE0 = 0.0      # previous error
    lineIntegral = 0.0 # accumulated integral error
    lineDerivFiltered = 0.0  # filtered derivative term
    lineY = 0.0       # control output (rad/s)
    # management
    # topicRc = ""
    topicCmdT0 = ""
    lostLineCnt = 0
    u = 0 # turn rate control signal


    ##########################################################

    def setup(self):
      from uservice import service
      sendBlack = False
      loops = 0
      # turn line sensor on (command 'lip 1')
      print("% Edge (sedge.py):: turns on line sensor")
      self.topicCmdT0 = "robobot/cmd/T0"
      service.send(self.topicCmdT0, "lip 1")
      # request fast update (every 3 ms)
      service.send(self.topicCmdT0,"sub livn 10")
      # request data
      while not service.stop:
        t.sleep(0.02)
        # white calibrate requested
        if service.args.white:
          if not sendBlack:
            # make sure black level is black
            topic = self.topicCmdT0
            param = "litb 0 0 0 0 0 0 0 0"
            sendBlack = service.send(topic, param)
          elif self.edgeUpdCnt < 3:
            # request raw AD reflectivity
            service.send(self.topicCmdT0,"livi")
            pass
          elif not self.sendCalibRequest:
            # send calibration request, averaged over 100 samples
            service.send(self.topicCmdT0,"liwi")
            t.sleep(0.02)
            # calibrate using current white level averaged over 100 samples
            service.send(self.topicCmdT0,"licw 100")
            # allow communication to settle
            print("# Edge (sedge.py):: sending calibration request")
            # wait for calibration to finish (each sample takes 1-2 ms)
            t.sleep(0.25)
            # save the calibration as new default
            service.send(self.topicCmdT0,"eew")
            self.sendCalibRequest = True
            # ask for new white values
            service.send(self.topicCmdT0,"liwi")
            t.sleep(0.02)
          else:
            t.sleep(0.25)
            service.args.white = False
            print(f"% Edge (sedge.py):: calibration should be fine, terminates.")
            # terminate mission
            service.stop = True
        elif self.edge_n_wUpdCnt == 0:
          # get calibrated white value
          service.send(self.topicCmdT0,"liwi")
          pass
        elif self.edge_nUpdCnt == 0:
          # wait for line sensor data
          pass
        else:
          print(f"% Edge (sedge.py):: got data stream; after {loops} loops")
          break
        loops += 1
        if loops > 30:
          print(f"% Edge (sedge.py):: got no data after {loops} (continues edge_n_wUpdCnt={self.edge_n_wUpdCnt}, edgeUpdCnt={self.edgeUpdCnt}, edge_nUpdCnt={self.edge_nUpdCnt})")
          break
      pass

    ##########################################################

    def print(self):
      from uservice import service
      print("% Edge (sedge.py):: " + str(self.edgeTime - service.startTime) +
            f" ({self.edge[0]}, " +
            f"{self.edge[1]}, " +
            f"{self.edge[2]}, " +
            f"{self.edge[3]}, " +
            f"{self.edge[4]}, " +
            f"{self.edge[5]}, " +
            f"{self.edge[6]}, " +
            f"{self.edge[7]})" +
            f" {self.edgeInterval:.2f} ms " +
            str(self.edgeUpdCnt))
    def printn(self):
      from uservice import service
      print("% Edge (sedge.py):: normalized " + str(self.edge_nTime - service.startTime) +
            f" ({self.edge_n[0]}, " +
            f"{self.edge_n[1]}, " +
            f"{self.edge_n[2]}, " +
            f"{self.edge_n[3]}, " +
            f"{self.edge_n[4]}, " +
            f"{self.edge_n[5]}, " +
            f"{self.edge_n[6]}, " +
            f"{self.edge_n[7]})" +
            f" {self.edge_nInterval:.2f} ms " +
            str(self.edge_nUpdCnt))
    def printnw(self):
      from uservice import service
      print("% Edge (sedge.py):: white level " + str(self.edge_n_wTime) +
            f" ({self.edge_n_w[0]}, " +
            f"{self.edge_n_w[1]}, " +
            f"{self.edge_n_w[2]}, " +
            f"{self.edge_n_w[3]}, " +
            f"{self.edge_n_w[4]}, " +
            f"{self.edge_n_w[5]}, " +
            f"{self.edge_n_w[6]}, " +
            f"{self.edge_n_w[7]}) " +
            str(self.edge_n_wUpdCnt))

    ##########################################################

    def decode(self, topic, msg):
        # decode MQTT message
        used = True
        if topic == "T0/liv": # raw AD value
          from uservice import service
          gg = msg.split(" ")
          if (len(gg) >= 4):
            t0 = self.edgeTime;
            self.edgeTime = datetime.fromtimestamp(float(gg[0]))
            self.edge[0] = int(gg[1])
            self.edge[1] = int(gg[2])
            self.edge[2] = int(gg[3])
            self.edge[3] = int(gg[4])
            self.edge[4] = int(gg[5])
            self.edge[5] = int(gg[6])
            self.edge[6] = int(gg[7])
            self.edge[7] = int(gg[8])
            t1 = self.edgeTime;
            if self.edgeUpdCnt == 2:
              self.edgeInterval = (t1 -t0).total_seconds()*1000
            elif self.edgeUpdCnt > 2:
              self.edgeInterval = (self.edgeInterval * 99 + (t1 -t0).total_seconds()*1000) / 100
            self.edgeUpdCnt += 1
            # self.print()
        elif topic == "T0/livn": # normalized after calibration range (0..1000)
          from uservice import service
          gg = msg.split(" ")
          if (len(gg) >= 4):
            t0 = self.edge_nTime;
            self.edge_nTime = datetime.fromtimestamp(float(gg[0]))
            self.edge_n[0] = int(gg[1])
            self.edge_n[1] = int(gg[2])
            self.edge_n[2] = int(gg[3])
            self.edge_n[3] = int(gg[4])
            self.edge_n[4] = int(gg[5])
            self.edge_n[5] = int(gg[6])
            self.edge_n[6] = int(gg[7])
            self.edge_n[7] = int(gg[8])
            t1 = self.edge_nTime;
            if self.edge_nUpdCnt == 2:
              self.edge_nInterval = (t1 -t0).total_seconds()*1000
            elif self.edge_nUpdCnt > 2:
              self.edge_nInterval = (self.edge_nInterval * 99 + (t1 -t0).total_seconds()*1000) / 100
            self.edge_nUpdCnt += 1
            # got new normalized values
            # debug save as a remark with timestamp
            # flog.writeDataString(f" {msg}");
            #
            # calculate line values based on new values
            self.LineDetect()
            #
            # use to control, if active
            if self.lineCtrl:
              self.followLine()
            # log relevant line sensor data
            if self.edge_nUpdCnt % 10 == 0:
              flog.write()
            #self.printn()
        elif topic == "T0/liw": # get white level
          from uservice import service
          gg = msg.split(" ")
          if (len(gg) >= 4):
            self.edge_n_wTime = datetime.fromtimestamp(float(gg[0]))
            self.edge_n_w[0] = int(gg[1])
            self.edge_n_w[1] = int(gg[2])
            self.edge_n_w[2] = int(gg[3])
            self.edge_n_w[3] = int(gg[4])
            self.edge_n_w[4] = int(gg[5])
            self.edge_n_w[5] = int(gg[6])
            self.edge_n_w[6] = int(gg[7])
            self.edge_n_w[7] = int(gg[8])
            self.edge_n_wUpdCnt += 1
            # self.printnw()
        else:
          used = False
        return used

    ##########################################################

    def LineDetect(self):
      sum = 0
      posSum = 0
      high = int(1)
      # find levels (and average)
      # using normalised readings (0 (no reflection) to 1000 (calibrated white)))
      for i in range(8):
        sum += self.edge_n[i] # for average
        if self.edge_n[i] > high:
          high = self.edge_n[i] # most bright value (floor level)
      self.high = high # most white level
      # print(f"% Edge (sedge.py):: {low}, {high} - what")
      # average white level
      self.average = sum / 8.0;
      # detect if we have a crossing line
      self.crossingLine = self.average >= self.crossingThreshold
      # is line valid (high above threshold)
      self.lineValid = self.high >= self.lineValidThreshold
      if self.lineValid:
        self.lineLastSeenTime = datetime.now()
      # find line position
      # from left side - stop at first value above half of the brightest
      if self.lineValid:
        posLeft = -3.5 # max left
        if self.edge_n[0] < self.lineValidThreshold:
          posLeft = -3 # between sensor 1 and 2 or more right
          for i in range(1,8):
            if self.edge_n[i] < self.lineValidThreshold:
              posLeft += 1;
            else:
              break;
        posRight = 3.5 # max right
        if self.edge_n[7] < self.lineValidThreshold:
          posRight = 3 # may be between sensor 8 and 7 or more left
          for i in range(1,8):
            if self.edge_n[7-i] < self.lineValidThreshold:
              posRight -= 1;
            else:
              break;
        self.posLeft = posLeft
        self.posRight = posRight
      else:
        # just keep old value
        pass
      #
      if self.lineValid and self.lineValidCnt < 20:
        self.lineValidCnt += 1
        # Update last seen timestamp when line is valid with confidence >= 2
        #if self.lineValidCnt >= 2:
        #  self.lineLastSeenTime = datetime.now()
        
        # Update last seen timestamp whenever the line is detected.
        # This avoids stale timestamps after brief reacquisitions.
        self.lineLastSeenTime = datetime.now()
      elif not self.lineValid:
        if self.lineValidCnt > 0:
          self.lineValidCnt -= 1
        else:
          self.lineValidCnt = 0
      if self.crossingLine and self.crossingLineCnt < 20:
        self.crossingLineCnt += 1
      elif not self.crossingLine:
        self.crossingLineCnt -= 1
        if self.crossingLineCnt < 0:
          self.crossingLineCnt = 0
      pass
      # print(f"% Edge (sedge.py):: ({self.edge_n[0]} {self.edge_n[1]} {self.edge_n[2]} {self.edge_n[3]} {self.edge_n[4]} {self.edge_n[5]} {self.edge_n[6]}), high={self.high}, left={self.posLeft:.2f}, right={self.posRight:.2f}.")

    ##########################################################

    def selectAndApplyProfile(self, velocity):
      """Select appropriate PID profile based on velocity and apply parameters"""
      profileName = 'fast'  # default
      
      if velocity <= 0.35:
        profileName = 'slow'
      elif velocity < 0.55:
        profileName = 'medium'
      else:
        profileName = 'fast'
      
      # Only update if profile changed (reduces console spam)
      if profileName != self.currentProfile:
        self.currentProfile = profileName
        profile = self.pidProfiles[profileName]
        self.lineKp = profile['Kp']
        self.lineKi = profile['Ki']
        self.lineKd = profile['Kd']
        self.derivativeAlpha = profile['derivativeAlpha']
        self.maxIntegral = profile['maxIntegral']
        print(f"% Edge::selectAndApplyProfile: Switched to '{profileName}' profile (velocity={velocity:.3f})")
        print(f"  Kp={self.lineKp}, Ki={self.lineKi}, Kd={self.lineKd}, alpha={self.derivativeAlpha}")

    ##########################################################

    def lineControl(self, velocity, followLeft = True, refPosition = 0):
      self.velocity = velocity
      self.followLeft = followLeft
      self.refPosition = refPosition
      # Select and apply the appropriate PID profile
      self.selectAndApplyProfile(velocity)
      # velocity 0 (or negative) is turning off line control
      wasActive = self.lineCtrl
      self.lineCtrl = velocity > 0.001
      # Reset PID when starting new control session
      if self.lineCtrl and not wasActive:
        self.resetPID()
      pass

    ##########################################################

    def followLine(self):
      from uservice import service
      # Calculate current error
      if self.followLeft:
        e = self.refPosition - self.posLeft
      else:
        e = self.refPosition - self.posRight
      # when line (posLeft or posRight) is to (much) to the right edge position is positive.
      # The robot is thus too much to the left.
      # To correct we need a negative turn rate (CV),
      # so sign of e is OK
      #
      # Get sample time in seconds
      dt = self.edge_nInterval / 1000.0  # convert ms to seconds
      if dt < 0.001:  # safety check
        dt = 0.05  # assume 50ms if invalid
      #
      # PID Controller
      # Proportional term
      P = self.lineKp * e
      #
      # Integral term (with anti-windup)
      self.lineIntegral += e * dt
      # Anti-windup: clamp integral term
      if self.lineIntegral > self.maxIntegral:
        self.lineIntegral = self.maxIntegral
      elif self.lineIntegral < -self.maxIntegral:
        self.lineIntegral = -self.maxIntegral
      I = self.lineKi * self.lineIntegral
      #
      # Derivative term with low-pass filtering to reduce noise
      de_raw = (e - self.lineE0) / dt
      self.lineDerivFiltered = self.derivativeAlpha * de_raw + (1 - self.derivativeAlpha) * self.lineDerivFiltered
      D = self.lineKd * self.lineDerivFiltered
      #
      # Control output
      self.lineY = P + I + D
      #
      # Output saturation
      if self.lineY > 4:
        self.lineY = 4
      elif self.lineY < -4:
        self.lineY = -4
      #
      # Rate limiting: prevent wheel velocity saturation
      # velDif = wheelbase * turnrate, so:
      # v_right = linVel + velDif/2 <= maxWheelVel
      # Therefore: turnrate <= 2 * (maxWheelVel - linVel) / wheelbase
      if self.velocity > 0.001:
        max_turnrate = 2.0 * (self.maxWheelVel - self.velocity) / self.wheelbase
        if max_turnrate < 0:
          max_turnrate = 0  # Can't turn faster than physical limits allow
        if abs(self.lineY) > max_turnrate:
          # Clamp turn rate to physically achievable limit
          self.lineY = max_turnrate if self.lineY > 0 else -max_turnrate
      #
      # Save error for next iteration
      self.lineE0 = e
      #
      # make response
      par = f"rc {self.velocity:.3f} {self.lineY:.3f} {t.time()}"
      service.send("robobot/cmd/ti", par) # send new turn command, maintaining velocity
      # debug print
      if True: # self.edge_nUpdCnt % 20 == 0:
        print(f"% Edge::followLine PID: e={e:.3f}, P={P:.3f}, I={I:.3f}, D={D:.3f}, y={self.lineY:.3f} -> {par}")

    ##########################################################

    def resetPID(self):
      """Reset PID controller state (call when starting new line following)"""
      self.lineE0 = 0.0
      self.lineIntegral = 0.0
      self.lineDerivFiltered = 0.0
      self.lineY = 0.0
      print("% LineCtrl:: PID controller reset")


    ##########################################################

    def terminate(self):
      from uservice import service
      self.need_data = False
      print("% Edge (sedge.py):: turn off line sensor")
      service.send(self.topicCmdT0, "lip 0")
      # try:
      #   self.th.join()
      #   # stop subscription service from Teensy
      #   service.send(service.topicCmd + "T0/sub","livn 0")
      # except:
      #   print("% Edge thread not running")
      print("% Edge (sedge.py):: terminated")
      pass

    ##########################################################

    def paint(self, img):
      h, w, ch = img.shape
      pl = int(h - h/4) # base position bottom (most positive y)
      st = int(w/10) # distance between sensors
      gh = int(h/2) # graph height
      x = st # base position left
      y = pl
      dtuGreen = (0x35, 0x88, 0) # BGR
      dtuBlue = (0xea, 0x3e, 0x2f)
      dtuRed = (0x00, 0x00, 0x99)
      dtuPurple = (0x8e, 0x23, 0x77)
      # paint baseline
      cv.line(img, (x,y), (int(x + 7*st), int(y)), dtuGreen, thickness=1, lineType=8)
      # paint calibrated white line (top)
      cv.line(img, (x,int(y-gh)), (int(x + 7*st), int(y-gh)), dtuGreen, thickness=1, lineType=8)
      # paint threshold line for line valid
      cv.line(img, (x,int(y-gh*self.lineValidThreshold/1000.0)), (int(x + 7*st), int(y-gh*self.lineValidThreshold/1000.0)), dtuBlue, thickness=1, lineType=4)
      # draw current sensor readings
      for i in range(8):
        y = int(pl - self.edge_n[i]/1000 * gh)
        cv.drawMarker(img, (x,y), dtuRed, markerType=cv.MARKER_STAR, thickness=2, line_type=8, markerSize = 10)
        x += st
      # paint line position
      print(f" Edge::paint: posLeft {self.posLeft}, right {self.posRight}")
      pixP = int((self.posLeft + 4.5)*st)
      cv.line(img, (pixP, int(pl)), (pixP, int(pl-gh)), dtuRed, thickness=3, lineType=4)
      pixP = int((self.posRight + 4.5)*st)
      cv.line(img, (pixP, int(pl)), (pixP, int(pl-gh)), dtuGreen, thickness=3, lineType=4)
      # paint low line position
      pixL = pl - int(gh * 0.0)
      cv.line(img, (st, pixL), (st*8, pixL), dtuRed, thickness=1, lineType=4)
      # some axis marking
      cv.putText(img, "Left", (st,pl - 2), cv.FONT_HERSHEY_PLAIN, 1, dtuPurple, thickness=2)
      cv.putText(img, "Right", (int(st+6*st),pl - 2), cv.FONT_HERSHEY_PLAIN, 1, dtuPurple, thickness=2)
      cv.putText(img, "White (1000)", (int(st),pl - gh - 2), cv.FONT_HERSHEY_PLAIN, 1, dtuPurple, thickness=2)
      if self.crossingLine:
        cv.putText(img, "Crossing", (int(st),int(pl - 20)), cv.FONT_HERSHEY_PLAIN, 1, dtuRed, thickness=2)


# create the data object
edge = SEdge()

