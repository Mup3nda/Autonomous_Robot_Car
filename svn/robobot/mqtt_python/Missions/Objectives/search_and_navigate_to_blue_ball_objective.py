"""Composite objective: search for blue ball, then navigate to it."""

from objective import CompositeObjective
from Objectives.look_for_blue_ball_objective import LookForBlueBallObjective
from Objectives.navigate_to_blue_ball_objective import NavigateToBlueBallObjective

from Objectives.look_for_ball_objective import LookForBallObjective
from Objectives.navigate_to_ball_objective import NavigateToBallObjective

class SearchAndNavigateToBlueBall(CompositeObjective):
    """Run blue-ball search first, then start navigation to the detected ball."""

    name = "Search_And_Navigate_To_Blue_Ball"

    def __init__(
        self,
        turn_rate=0.5,
        min_confidence=1,
        search_print_interval=20,
        desired_distance=0.38,
        navigate_print_interval=20,
    ):
        objectives = [
            LookForBallObjective(
                turn_rate=turn_rate,
                min_confidence=min_confidence,
                print_interval=search_print_interval,
                color="blue",
                scan_mode=LookForBallObjective.SCAN_MODE_SWEEP_90,
            ),
            NavigateToBallObjective(
                desired_distance=desired_distance,
                print_interval=navigate_print_interval,
                nav_mode="sequential",
                color="blue"
            ),
        ]
        super().__init__(objectives)
