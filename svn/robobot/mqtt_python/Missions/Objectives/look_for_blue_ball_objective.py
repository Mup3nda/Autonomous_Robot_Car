"""Look For Blue Ball Objective - rotate until blue ball is detected."""

from enum import IntEnum

from mission_context import MissionContext
from objective import Objective
from sball_saray import SBall


class LookForBlueBallState(IntEnum):
	SEARCHING = 0
	FOUND = 1
	DONE = 99


class LookForBlueBallObjective(Objective):
	"""Rotate in place until the blue ball detector sees a target."""

	def __init__(self, turn_rate=0.18, min_confidence=1, print_interval=20):
		super().__init__()
		self.turn_rate = float(turn_rate)
		self.min_confidence = int(min_confidence)
		self.print_interval = int(print_interval)
		self.tick_count = 0
		self.detector = None
		self.state = LookForBlueBallState.SEARCHING

	def start(self, ctx: MissionContext):
		self._done = False
		self.tick_count = 0
		self.state = LookForBlueBallState.SEARCHING

		self.detector = SBall(cam=ctx.cam, gpio=ctx.gpio, service=ctx.service)
		self.detector.set_detection_color("blue")
		self.detector.start()

		print(
			f"% Objective: Look For Blue Ball (turn_rate={self.turn_rate:.2f}, "
			f"min_conf={self.min_confidence})"
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
			print("% Look For Blue Ball: target detected, stopping rotation")
			return

		ctx.actions.drive.rc(0.0, self.turn_rate)

		if self.tick_count % self.print_interval == 0:
			print("% Look For Blue Ball: searching...")

	def stop(self, ctx: MissionContext):
		ctx.actions.drive.stop()
		if self.detector and hasattr(self.detector, "stop"):
			self.detector.stop()
		self.state = LookForBlueBallState.DONE
		print("% Look For Blue Ball objective stopped")
