"""Look For  Ball Objective - rotate until  ball is detected."""

import math
from enum import IntEnum

from mission_context import MissionContext
from objective import Objective
from sball_saray import SBall
from sodom import odom


class LookForBlueBallState(IntEnum):
	SEARCHING = 0
	FOUND = 1
	DONE = 99


class LookForBallObjective(Objective):
	"""Rotate in place until the ball detector sees a target."""
	SCAN_MODE_SPIN = "spin"
	SCAN_MODE_SWEEP_90 = "sweep_90"

	def __init__(
		self,
		turn_rate=0.18,
		min_confidence=1,
		print_interval=20,
		color="blue",
		scan_mode=SCAN_MODE_SPIN,
		heading_tolerance_deg=4.0,
	):
		super().__init__()
		self.turn_rate = float(turn_rate)
		self.min_confidence = int(min_confidence)
		self.print_interval = int(print_interval)
		self.tick_count = 0
		self.detector = None
		self.state = LookForBlueBallState.SEARCHING
		self.color = str(color).lower()
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
		self.state = LookForBlueBallState.SEARCHING
		_, _, self.start_heading = odom.get_world_pose()
		self.sweep_headings = [
			self._wrap_to_pi(self.start_heading + math.radians(90.0)),
			self._wrap_to_pi(self.start_heading),
			self._wrap_to_pi(self.start_heading - math.radians(90.0)),
			self._wrap_to_pi(self.start_heading),
		]
		self.sweep_index = 0

		self.detector = SBall(cam=ctx.cam, gpio=ctx.gpio, service=ctx.service)
		self.detector.set_detection_color(self.color)
		self.detector.start()

		print(
			f"% Objective: Look For {self.color.capitalize()} Ball (turn_rate={self.turn_rate:.2f}, "
			f"min_conf={self.min_confidence}, scan_mode={self.scan_mode})"
		)

	def tick(self, ctx: MissionContext):
		if self._done:
			return

		self.tick_count += 1

		if self.detector and self.detector.is_target_visible(self.min_confidence):
			ctx.actions.drive.stop()
			self.state = LookForBlueBallState.FOUND
			self._done = True
			target = self.detector.get_target()
			if target is not None:
				ctx.memory["last_visible_target"] = target
			print(f"% Look For {self.color.capitalize()} Ball: target detected, stopping rotation")
			return

		if self.scan_mode == self.SCAN_MODE_SWEEP_90:
			_, _, current_heading = odom.get_world_pose()
			target_heading = self.sweep_headings[self.sweep_index]
			err = self._wrap_to_pi(target_heading - current_heading)

			if abs(err) <= self.heading_tolerance_rad:
				self.sweep_index = (self.sweep_index + 1) % len(self.sweep_headings)
				target_heading = self.sweep_headings[self.sweep_index]
				err = self._wrap_to_pi(target_heading - current_heading)

			if abs(err) > self.heading_tolerance_rad:
				turn_cmd = self.sweep_turn_rate if err > 0.0 else -self.sweep_turn_rate
				ctx.actions.drive.rc(0.0, turn_cmd)
			else:
				ctx.actions.drive.stop()
		else:
			ctx.actions.drive.rc(0.0, self.turn_rate)

		if self.tick_count % self.print_interval == 0:
			if self.scan_mode == self.SCAN_MODE_SWEEP_90:
				_, _, current_heading = odom.get_world_pose()
				target_heading = self.sweep_headings[self.sweep_index]
				err = self._wrap_to_pi(target_heading - current_heading)
				print(
					f"% Look For {self.color.capitalize()} Ball: searching sweep "
					f"step={self.sweep_index}, err_deg={math.degrees(err):.1f}"
				)
			else:
				print(f"% Look For {self.color.capitalize()} Ball: searching spin...")

	def stop(self, ctx: MissionContext):
		ctx.actions.drive.stop()
		if self.detector and hasattr(self.detector, "stop"):
			self.detector.stop()
		self.state = LookForBlueBallState.DONE
		print(f"% Look For {self.color.capitalize()} Ball objective stopped")
