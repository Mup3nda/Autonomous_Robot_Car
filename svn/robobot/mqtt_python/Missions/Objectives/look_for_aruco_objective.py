"""Look For  Ball Objective - rotate until  ball is detected."""

from enum import IntEnum
import time

from mission_context import MissionContext
from objective import Objective
from aruco_detector2 import ArucoDetector


class LookForArucoState(IntEnum):
	SEARCHING_PRIMARY = 0
	SEARCHING_FALLBACK = 1
	FOUND = 2
	DONE = 99
 
class LookForArucoObjective(Objective):
	"""Rotate in place until the ball detector sees a target."""

	def __init__(
		self, turn_rate=0.18, 
		min_confidence=1, 
		print_interval=20, 
		marker_id=0,
		fallback_marker_id=None,
		search_timeout_s=None
  	):
		super().__init__()
		self.turn_rate = float(turn_rate)
		self.min_confidence = int(min_confidence)
		self.print_interval = int(print_interval)
  
		self.marker_id = marker_id
		self.fallback_marker_id = fallback_marker_id
		self.search_timeout_s = search_timeout_s
  
		self.tick_count = 0
		self.detector = None
		self.state = LookForArucoState.SEARCHING_PRIMARY
		self.search_start_time = None
		self.current_target_id = self.marker_id
		self.fallback_used = False  # Track if fallback was used

	def start(self, ctx: MissionContext):
		self._done = False
		self.tick_count = 0
		self.state = LookForArucoState.SEARCHING_PRIMARY
		self.search_start_time = time.time()
		self.current_target_id = self.marker_id
  
		ctx.memory["aruco_found_id"] = None
		ctx.memory["last_visible_target"] = None
  
		self.detector = ArucoDetector(cam=ctx.cam, gpio=ctx.gpio, service=ctx.service, target_id=self.current_target_id)
		self.detector.start()

		print(
			f"% Objective: Look For {self.current_target_id} (turn_rate={self.turn_rate:.2f}, "
			f"{f' then {self.fallback_marker_id}' if self.fallback_marker_id is not None else ''}"
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
			print(f"% Look For {self.marker_id} target detected, stopping rotation")
			return

		if (
			self.state == LookForArucoState.SEARCHING_PRIMARY
			and self.fallback_marker_id is not None
			and self.search_timeout_s is not None
		):
			elapsed = time.time() - self.search_start_time
			if elapsed >= self.search_timeout_s:
				print(
                    f"% Look For {self.marker_id}: timeout after {elapsed:.1f}s, "
                    f"switching to fallback {self.fallback_marker_id}"
                )

				self.current_target_id = self.fallback_marker_id
				self.fallback_used = True  # Mark fallback as used
				ctx.memory["fallback_flag"] = 1  # Set flag in mission context
				self.state = LookForArucoState.SEARCHING_FALLBACK
				self.search_start_time = time.time()
    
				if self.detector and hasattr(self.detector, "set_target_id"):
					self.detector.set_target_id(self.current_target_id)

		ctx.actions.drive.rc(0.0, self.turn_rate)

		if self.tick_count % self.print_interval == 0:
			print(f"% Look For {self.marker_id}: searching...")

	def stop(self, ctx: MissionContext):
		ctx.actions.drive.stop()
		if self.detector and hasattr(self.detector, "stop"):
			self.detector.stop()
		self.state = LookForArucoState.DONE
		print(f"% Look For Aruco objective stopped")
