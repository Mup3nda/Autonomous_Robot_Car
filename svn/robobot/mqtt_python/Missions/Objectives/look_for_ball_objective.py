"""Look For  Ball Objective - rotate until  ball is detected."""

from enum import IntEnum

from mission_context import MissionContext
from objective import Objective
from sball_saray import SBall


class LookForBlueBallState(IntEnum):
	SEARCHING = 0
	FOUND = 1
	DONE = 99


class LookForBallObjective(Objective):
	"""Rotate in place until the ball detector sees a target."""

	def __init__(self, turn_rate=0.18, min_confidence=1, print_interval=20,color="blue"):
		super().__init__()
		self.turn_rate = float(turn_rate)
		self.min_confidence = int(min_confidence)
		self.print_interval = int(print_interval)
		self.tick_count = 0
		self.detector = None
		self.state = LookForBlueBallState.SEARCHING
		self.color = str(color).lower()

	def start(self, ctx: MissionContext):
		self._done = False
		self.tick_count = 0
		self.state = LookForBlueBallState.SEARCHING

		self.detector = SBall(cam=ctx.cam, gpio=ctx.gpio, service=ctx.service)
		self.detector.set_detection_color(self.color)
		self.detector.start()

		print(
			f"% Objective: Look For {self.color.capitalize()} Ball (turn_rate={self.turn_rate:.2f}, "
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
			print(f"% Look For {self.color.capitalize()} Ball: target detected, stopping rotation")
			return

		ctx.actions.drive.rc(0.0, self.turn_rate)

		if self.tick_count % self.print_interval == 0:
			print(f"% Look For {self.color.capitalize()} Ball: searching...")

	def stop(self, ctx: MissionContext):
		ctx.actions.drive.stop()
		if self.detector and hasattr(self.detector, "stop"):
			self.detector.stop()
		self.state = LookForBlueBallState.DONE
		print(f"% Look For {self.color.capitalize()} Ball objective stopped")
