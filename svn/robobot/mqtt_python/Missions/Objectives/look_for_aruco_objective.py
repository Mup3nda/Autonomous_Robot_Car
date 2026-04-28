import math
from enum import IntEnum
import time
import math

from mission_context import MissionContext
from objective import Objective
#from Autonomous_Robot_Car.svn.robobot.mqtt_python.aruco_detector import ArucoDetector
from aruco_detector2 import ArucoDetector
from sodom import odom


class LookForArucoState(IntEnum):
	SEARCHING = 0
	FOUND = 1
	DONE = 99


class LookForArucoObjective(Objective):
	"""Rotate in place until the ball detector sees a target."""
	SCAN_MODE_SPIN = "spin"
	SCAN_MODE_SWEEP_90 = "sweep_90"

	def __init__(
		self,
		turn_rate=0.18,
		min_confidence=1,
		print_interval=20,
		marker_id=0,
		fallback_marker_id=None,
		search_timeout_s=None,
		max_sweep_deg = 180
  	):
		super().__init__()
		self.turn_rate = float(turn_rate)
		self.min_confidence = int(min_confidence)
		self.print_interval = int(print_interval)
  
		self.max_sweep_deg = float(max_sweep_deg)
		self.max_sweep_rad = math.radians(self.max_sweep_deg)
		self.sweep_sign = 1
		self.origin_heading = None
  
  
		self.marker_id = marker_id
		self.fallback_marker_id = fallback_marker_id
		self.search_timeout_s = search_timeout_s
		self.search_start_time = None
		self.current_target_id = self.marker_id
		self.fallback_used = False
		self.scan_mode = str(scan_mode).lower()
		if self.scan_mode not in (self.SCAN_MODE_SPIN, self.SCAN_MODE_SWEEP_90):
			self.scan_mode = self.SCAN_MODE_SPIN
		self.heading_tolerance_rad = math.radians(abs(float(heading_tolerance_deg)))
		self.sweep_turn_rate = abs(self.turn_rate)
		self.start_heading = 0.0
		self.sweep_headings = []
		self.sweep_index = 0

	@staticmethod
	def _wrap_to_pi(angle_rad):
		while angle_rad > math.pi:
			angle_rad -= 2.0 * math.pi
		while angle_rad < -math.pi:
			angle_rad += 2.0 * math.pi
		return angle_rad

	def start(self, ctx: MissionContext):
		self._done = False
		self.tick_count = 0
		self.state = LookForArucoState.SEARCHING
		ctx.memory["aruco_found_id"] = None
		ctx.memory["last_visible_target"] = None
		ctx.memory["fallback_flag"] = 0
		self.search_start_time = time.time()
		self.current_target_id = self.marker_id
	
		# Reset tripAh so we start from a known baseline
		ctx.pose.tripAreset()
		self.origin_heading = 0.0
  
        # record heading to limit sweep from this origin
		#self.origin_heading = getattr(ctx.pose, "tripAh", None)
  
		ctx.memory["aruco_found_id"] = None
		ctx.memory["last_visible_target"] = None
  
		self.detector = ArucoDetector(cam=ctx.cam, gpio=ctx.gpio, service=ctx.service, target_id=self.current_target_id)
		self.detector.start()

		print(
			f"% Objective: Look For {self.current_target_id} (turn_rate={self.turn_rate:.2f}, "
			f"min_conf={self.min_confidence}, scan_mode={self.scan_mode}, "
			f"fallback={self.fallback_marker_id}, timeout={self.search_timeout_s})"
		)

	def tick(self, ctx: MissionContext):
		if self._done:
			return

		self.tick_count += 1

		if self.detector and self.detector.is_target_visible(self.min_confidence):
			ctx.actions.drive.stop()
			self.state = LookForArucoState.FOUND
			self._done = True
			target = self.detector.get_target()
			if target is not None:
				ctx.memory["last_visible_target"] = target
			ctx.memory["aruco_found_id"] = self.current_target_id
			print(f"% Look For {self.current_target_id} target detected, stopping rotation")
			return

		if (
			self.fallback_marker_id is not None
			and self.search_timeout_s is not None
			and self.current_target_id == self.marker_id
		):
			elapsed = time.time() - self.search_start_time
			if elapsed >= self.search_timeout_s:
				print(
					f"% Look For {self.marker_id}: timeout after {elapsed:.1f}s, "
					f"switching to fallback {self.fallback_marker_id}"
				)
				self.current_target_id = self.fallback_marker_id
				self.fallback_used = True
				ctx.memory["fallback_flag"] = 1
				self.search_start_time = time.time()
				if self.detector and hasattr(self.detector, "set_target_id"):
					self.detector.set_target_id(self.current_target_id)
     
		# if self.origin_heading is None:
		# 	turn_cmd = self.turn_rate
		# else:
		if self.turn_rate != 0:
			current_heading = getattr(ctx.pose, "tripAh", 0.0)
			print(f"Current heading: {current_heading}")
			def _wrap_to_pi(angle):
				while angle >= math.pi:
					angle -= 2*math.pi
				while angle <= -math.pi:
					angle += 2*math.pi
				return angle
			
			# relative heading change
			relative_heading = _wrap_to_pi(current_heading - self.origin_heading)
			half_sweep_rad = self.max_sweep_rad / 2

			if relative_heading >= half_sweep_rad:
				self.sweep_sign = -1
			elif relative_heading <= - half_sweep_rad:
				self.sweep_sign = 1

		turn_cmd = self.turn_rate * self.sweep_sign
   
		ctx.actions.drive.rc(0.0, turn_cmd)

		if self.tick_count % self.print_interval == 0:
			print(f"% Look For {self.current_target_id}: searching...")

	def stop(self, ctx: MissionContext):
		ctx.actions.drive.stop()
		if self.detector and hasattr(self.detector, "stop"):
			self.detector.stop()
		self.state = LookForArucoState.DONE
		print(f"% Look For {self.current_target_id} objective stopped")
