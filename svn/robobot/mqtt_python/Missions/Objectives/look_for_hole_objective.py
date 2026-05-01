"""Look For Hole Objective - rotate until hole is detected."""

from enum import IntEnum

from mission_context import MissionContext
from objective import Objective
from shole import SHole


class LookForHoleState(IntEnum):
	SEARCHING = 0
	FOUND = 1
	DONE = 99


class LookForHoleObjective(Objective):
	"""Rotate in place until the hole detector sees a target."""

	def __init__(self, turn_rate=0.18, min_confidence=1, print_interval=20):
		super().__init__()
		self.turn_rate = float(turn_rate)
		self.min_confidence = int(min_confidence)
		self.print_interval = int(print_interval)
		self.tick_count = 0
		self.detector = None
		self.state = LookForHoleState.SEARCHING

	def start(self, ctx: MissionContext):
		self._done = False
		self.tick_count = 0
		self.state = LookForHoleState.SEARCHING

		self.detector = SHole(cam=ctx.cam, gpio=ctx.gpio, service=ctx.service)
		self.detector.start()

		print(
			f"% Objective: Look For Hole (turn_rate={self.turn_rate:.2f}, "
			f"min_conf={self.min_confidence})"
		)

	def tick(self, ctx: MissionContext):
		if self._done:
			return

		self.tick_count += 1

		if self.detector and self.detector.is_target_visible(self.min_confidence):
			ctx.actions.drive.stop()
			self.state = LookForHoleState.FOUND
			self._done = True

			target = self.detector.get_target()
			if target is not None:
				ctx.memory["last_visible_target"] = target

			print("% Look For Hole: target detected, stopping rotation")
			return

		ctx.actions.drive.rc(0.0, self.turn_rate)

		if self.tick_count % self.print_interval == 0:
			print("% Look For Hole: searching...")

	def stop(self, ctx: MissionContext):
		ctx.actions.drive.stop()
		if self.detector and hasattr(self.detector, "stop"):
			self.detector.stop()
		self.state = LookForHoleState.DONE
		print("% Look For Hole objective stopped")