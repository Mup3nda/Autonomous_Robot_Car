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
import os
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
            'Kp': 0.85,    # More aggressive proportional gain for slow speeds
            'Ki': 0.3,    # Increased integral gain
            'Kd': 0.18,   # Increased derivative gain
            'derivativeAlpha': 0.22,    # More low-pass filtering for noise reduction
            'maxIntegral': 1.0
        },
        'medium': {
          'Kp': 0.72,    # Reduced proportional gain to lower oscillation at ~0.45 m/s
          'Ki': 0.3,    # Reduced integral gain to avoid windup-driven wobble
          'Kd': 0.18,    # Increased derivative damping
          'derivativeAlpha': 0.20,   # More low-pass filtering on derivative term
          'maxIntegral': 0.75
        },
        'fast': {
            'Kp': 0.7,   # Original proportional gain (works well at 0.95 m/s)
            'Ki': 0.2,    # Original integral gain
            'Kd': 0.25,   # Original derivative gain
            'derivativeAlpha': 0.7,    # Original low-pass filter
            'maxIntegral': 0.6
        }
    }
    # Currently active profile parameters
    lineKp = 0.75  # Proportional gain (rad/s per sensor value)
    lineKi = 0.2  # Integral gain (rad/s per (sensor value * sec))
    lineKd = 0.08 # Derivative gain (rad/s per (sensor value / sec)))
    derivativeAlpha = 0.6  # Low-pass filter for derivative (0-1, lower = more filtering)
    maxErrorRate = 40.0  # Clamp on error derivative to limit D-kick from sensor jumps
    maxIntegral = 1.2 # anti-windup limit for integral term
    currentProfile = 'fast'  # Track which profile is active
    # PID state variables
    lineE0 = 0.0      # previous error
    lineIntegral = 0.0 # accumulated integral error
    lineDerivFiltered = 0.0  # filtered derivative term
    lineY = 0.0       # control output (rad/s)
    centroidPosition = 0.0
    centroidWeightSum = 0.0
    centroidWeightedPosSum = 0.0
    centroidValid = False
    centroidWeights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    debugLineFollow = False
    debugPrintEvery = 10
    # management
    # topicRc = ""
    topicCmdT0 = ""
    lostLineCnt = 0
    u = 0 # turn rate control signal
    # velocity ramping for smoother line-follow acceleration/deceleration
    velocity = 0.0
    targetVelocity = 0.0
    maxAccelUp = 0.15    # m/s^2, limit when increasing speed
    maxAccelDown = 0.6   # m/s^2, limit when decreasing speed
    # PID logging
    pidLogDir = ""
    pidLogFiles = {}
    pidLogFlushDecimation = 20
    pidLogCount = 0


    ##########################################################

    def setup(self):
      from uservice import service
      sendBlack = False
      loops = 0
      self.initPIDLogging()
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

    def initPIDLogging(self):
      """Create one CSV log file per PID profile in the pid folder."""
      if len(self.pidLogFiles) > 0:
        return
      self.pidLogDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "pid")
      os.makedirs(self.pidLogDir, exist_ok=True)
      ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
      header = (
        "timestamp,edge_n_timestamp,edge_timestamp,edge_n_upd_cnt,edge_upd_cnt,profile,"
        "follow_left,ref_position,target_velocity,velocity,dt,error,line_position,pos_left,pos_right,"
        "centroid_position,centroid_weight_sum,centroid_weighted_pos_sum,"
        "centroid_w0,centroid_w1,centroid_w2,centroid_w3,centroid_w4,centroid_w5,centroid_w6,centroid_w7,"
        "P,I,D,de_raw,Y,lineKp,lineKi,lineKd,derivative_alpha,max_integral,"
        "line_integral,line_deriv_filtered,ctrl_sat,rate_limited,max_turnrate,wheelbase,max_wheel_vel,"
        "cmd_timestamp,cmd_velocity,cmd_turnrate,line_valid,line_valid_cnt,crossing,crossing_cnt,high,average,line_valid_threshold,crossing_threshold,"
        "edge_n0,edge_n1,edge_n2,edge_n3,edge_n4,edge_n5,edge_n6,edge_n7,"
        "edge_raw0,edge_raw1,edge_raw2,edge_raw3,edge_raw4,edge_raw5,edge_raw6,edge_raw7\n"
      )
      for profile in ("slow", "medium", "fast"):
        fn = os.path.join(self.pidLogDir, f"pid_{profile}.csv")
        f = open(fn, "w", encoding="ascii")
        f.write(f"# PID telemetry log for profile '{profile}'\n")
        f.write(f"# generated_at,{ts}\n")
        f.write(header)
        self.pidLogFiles[profile] = f
      print(f"% Edge:: PID logs initialized in {self.pidLogDir}")

    def setDebugLogging(self, enabled=True, printEvery=10):
      """Enable or disable verbose line-follow diagnostics."""
      self.debugLineFollow = bool(enabled)
      self.debugPrintEvery = max(1, int(printEvery))
      print(f"% Edge:: debug line follow {'enabled' if self.debugLineFollow else 'disabled'}; printEvery={self.debugPrintEvery}")

    def logPIDSample(self, dt, e, linePosition, P, I, D, de_raw, ctrlSaturated, rateLimited, max_turnrate, cmdTimestamp):
      """Write one PID control sample to the active profile log file."""
      if self.currentProfile not in self.pidLogFiles:
        return
      f = self.pidLogFiles[self.currentProfile]
      row = (
        f"{datetime.now().timestamp():.6f},{self.edge_nTime.timestamp():.6f},{self.edgeTime.timestamp():.6f},"
        f"{self.edge_nUpdCnt},{self.edgeUpdCnt},{self.currentProfile},"
        f"{1 if self.followLeft else 0},{self.refPosition:.5f},{self.targetVelocity:.4f},{self.velocity:.4f},"
        f"{dt:.5f},{e:.5f},{linePosition:.5f},{self.posLeft:.5f},{self.posRight:.5f},"
        f"{self.centroidPosition:.5f},{self.centroidWeightSum:.5f},{self.centroidWeightedPosSum:.5f},"
        f"{self.centroidWeights[0]:.5f},{self.centroidWeights[1]:.5f},{self.centroidWeights[2]:.5f},{self.centroidWeights[3]:.5f},"
        f"{self.centroidWeights[4]:.5f},{self.centroidWeights[5]:.5f},{self.centroidWeights[6]:.5f},{self.centroidWeights[7]:.5f},"
        f"{P:.5f},{I:.5f},{D:.5f},{de_raw:.5f},{self.lineY:.5f},"
        f"{self.lineKp:.5f},{self.lineKi:.5f},{self.lineKd:.5f},{self.derivativeAlpha:.5f},{self.maxIntegral:.5f},"
        f"{self.lineIntegral:.5f},{self.lineDerivFiltered:.5f},"
        f"{1 if ctrlSaturated else 0},{1 if rateLimited else 0},{max_turnrate:.5f},{self.wheelbase:.5f},{self.maxWheelVel:.5f},"
        f"{cmdTimestamp:.6f},{self.velocity:.5f},{self.lineY:.5f},"
        f"{1 if self.lineValid else 0},{self.lineValidCnt},{1 if self.crossingLine else 0},{self.crossingLineCnt},"
        f"{self.high},{self.average:.2f},{self.lineValidThreshold},{self.crossingThreshold},"
        f"{self.edge_n[0]},{self.edge_n[1]},{self.edge_n[2]},{self.edge_n[3]},"
        f"{self.edge_n[4]},{self.edge_n[5]},{self.edge_n[6]},{self.edge_n[7]},"
        f"{self.edge[0]},{self.edge[1]},{self.edge[2]},{self.edge[3]},"
        f"{self.edge[4]},{self.edge[5]},{self.edge[6]},{self.edge[7]}\n"
      )
      f.write(row)
      self.pidLogCount += 1
      if self.pidLogCount % self.pidLogFlushDecimation == 0:
        f.flush()

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
      # Find line position using centroid on darkness weights.
      # Sensor coordinates are fixed from left to right.
      if self.lineValid:
        sensorPos = (-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5)
        weightSum = 0.0
        weightedPosSum = 0.0
        for i in range(8):
          # Darker-than-threshold readings carry weight in the centroid.
          w = float(self.lineValidThreshold - self.edge_n[i])
          if w < 0.0:
            w = 0.0
          self.centroidWeights[i] = w
          weightSum += w
          weightedPosSum += w * sensorPos[i]
        self.centroidWeightSum = weightSum
        self.centroidWeightedPosSum = weightedPosSum
        self.centroidValid = weightSum > 1e-6
        if self.centroidValid:
          centroid = weightedPosSum / weightSum
          self.centroidPosition = centroid
          self.posLeft = centroid
          self.posRight = centroid
      else:
        self.centroidValid = False
        self.centroidWeightSum = 0.0
        self.centroidWeightedPosSum = 0.0
        self.centroidWeights = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
      # If line is not valid (or no darkness weight), keep previous position.
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
      self.targetVelocity = max(0.0, velocity)
      self.followLeft = followLeft
      self.refPosition = refPosition
      # velocity 0 (or negative) is turning off line control
      wasActive = self.lineCtrl
      self.lineCtrl = self.targetVelocity > 0.001
      # Reset PID when starting new control session
      if self.lineCtrl and not wasActive:
        self.resetPID()
        self.velocity = 0.0
      elif not self.lineCtrl:
        self.velocity = 0.0
      pass

    ##########################################################

    def updateVelocityRamp(self, dt):
      """Move current velocity toward target velocity with acceleration limits."""
      if dt < 0.001:
        dt = 0.001
      # keep target in valid range
      if self.targetVelocity > self.maxWheelVel:
        self.targetVelocity = self.maxWheelVel
      elif self.targetVelocity < 0:
        self.targetVelocity = 0
      delta = self.targetVelocity - self.velocity
      if delta > 0:
        step = min(delta, self.maxAccelUp * dt)
      else:
        step = max(delta, -self.maxAccelDown * dt)
      self.velocity += step
      # final clamp for safety
      if self.velocity > self.maxWheelVel:
        self.velocity = self.maxWheelVel
      elif self.velocity < 0:
        self.velocity = 0

    ##########################################################

    def applyTurnLimits(self, turnrate):
      """Apply controller output and kinematic turn-rate limits."""
      ctrlSaturated = False
      if turnrate > 4:
        turnrate = 4
        ctrlSaturated = True
      elif turnrate < -4:
        turnrate = -4
        ctrlSaturated = True
      max_turnrate = 0.0
      rateLimited = False
      if self.velocity > 0.001:
        max_turnrate = 2.0 * (self.maxWheelVel - self.velocity) / self.wheelbase
        if max_turnrate < 0:
          max_turnrate = 0
        if abs(turnrate) > max_turnrate:
          turnrate = max_turnrate if turnrate > 0 else -max_turnrate
          rateLimited = True
      return turnrate, ctrlSaturated, rateLimited, max_turnrate

    ##########################################################

    def followLine(self):
      from uservice import service
      # Calculate current error
      if self.followLeft:
        linePosition = self.posLeft
      else:
        linePosition = self.posRight
      e = self.refPosition - linePosition
      # when line (posLeft or posRight) is to (much) to the right edge position is positive.
      # The robot is thus too much to the left.
      # To correct we need a negative turn rate (CV),
      # so sign of e is OK
      #
      # Get sample time in seconds
      dt = self.edge_nInterval / 1000.0  # convert ms to seconds
      if dt < 0.001:  # safety check
        dt = 0.05  # assume 50ms if invalid
      # Apply acceleration-limited ramp toward requested velocity
      self.updateVelocityRamp(dt)
      # Tune PID profile based on currently achieved velocity
      self.selectAndApplyProfile(self.velocity)
      #
      # PID Controller
      # Proportional term
      P = self.lineKp * e
      #
      # Derivative term with low-pass filtering to reduce noise
      de_raw = (e - self.lineE0) / dt
      if de_raw > self.maxErrorRate:
        de_raw = self.maxErrorRate
      elif de_raw < -self.maxErrorRate:
        de_raw = -self.maxErrorRate
      self.lineDerivFiltered = self.derivativeAlpha * de_raw + (1 - self.derivativeAlpha) * self.lineDerivFiltered
      D = self.lineKd * self.lineDerivFiltered
      #
      # Conditional integral update (anti-windup + invalid-line protection)
      oldIntegral = self.lineIntegral
      candidateIntegral = oldIntegral
      if self.lineValid:
        candidateIntegral += e * dt
        if candidateIntegral > self.maxIntegral:
          candidateIntegral = self.maxIntegral
        elif candidateIntegral < -self.maxIntegral:
          candidateIntegral = -self.maxIntegral

      candidateI = self.lineKi * candidateIntegral
      yCandidate = P + candidateI + D
      _, candCtrlSat, candRateLimited, _ = self.applyTurnLimits(yCandidate)

      if self.lineValid and not candCtrlSat and not candRateLimited:
        self.lineIntegral = candidateIntegral
        I = candidateI
      else:
        self.lineIntegral = oldIntegral
        I = self.lineKi * self.lineIntegral

      # Final control output with physical limits.
      self.lineY = P + I + D
      self.lineY, ctrlSaturated, rateLimited, max_turnrate = self.applyTurnLimits(self.lineY)
      if self.debugLineFollow and (self.pidLogCount % self.debugPrintEvery == 0 or not self.lineValid or ctrlSaturated or rateLimited):
        print(
          "% Edge::centroid debug: "
          f"valid={int(self.lineValid)} cnt={self.lineValidCnt} "
          f"centroid={self.centroidPosition:.3f} wsum={self.centroidWeightSum:.1f} "
          f"posL={self.posLeft:.3f} posR={self.posRight:.3f} "
          f"e={e:.3f} de_raw={de_raw:.3f} d_f={self.lineDerivFiltered:.3f} "
          f"P={P:.3f} I={I:.3f} D={D:.3f} y={self.lineY:.3f} "
          f"sat={int(ctrlSaturated)} rate={int(rateLimited)} profile={self.currentProfile}"
        )
      #
      # Save error for next iteration
      self.lineE0 = e
      #
      # make response
      cmdTimestamp = t.time()
      par = f"rc {self.velocity:.3f} {self.lineY:.3f} {cmdTimestamp}"
      service.send("robobot/cmd/ti", par) # send new turn command, maintaining velocity
      # Save PID tuning telemetry per active profile
      self.logPIDSample(dt, e, linePosition, P, I, D, de_raw, ctrlSaturated, rateLimited, max_turnrate, cmdTimestamp)
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
      for f in self.pidLogFiles.values():
        try:
          f.flush()
          f.close()
        except:
          pass
      self.pidLogFiles = {}
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

