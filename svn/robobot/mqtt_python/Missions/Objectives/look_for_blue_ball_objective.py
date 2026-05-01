"""Look For Blue Ball Objective - rotate until blue ball is detected."""

from Objectives.look_for_ball_objective import LookForBallObjective


class LookForBlueBallObjective(LookForBallObjective):
	"""Rotate in place until the blue ball detector sees a target."""

	def __init__(
		self,
		turn_rate=0.18,
		min_confidence=1,
		print_interval=20,
		scan_mode=LookForBallObjective.SCAN_MODE_SPIN,
		heading_tolerance_deg=4.0,
	):
		super().__init__(
			turn_rate=turn_rate,
			min_confidence=min_confidence,
			print_interval=print_interval,
			color="blue",
			scan_mode=scan_mode,
			heading_tolerance_deg=heading_tolerance_deg,
		)
