"""Composite objective: search for blue ball, then navigate to it."""

from objective import CompositeObjective
from Objectives.look_for_blue_ball_objective import LookForBlueBallObjective
from Objectives.navigate_to_blue_ball_objective import NavigateToBlueBallObjective


class SearchAndNavigateToBlueBall(CompositeObjective):
    """Run blue-ball search first, then start navigation to the detected ball."""

    name = "Search_And_Navigate_To_Blue_Ball"

    def __init__(
        self,
        turn_rate=-0.5,
        min_confidence=1,
        search_print_interval=20,
        desired_distance=0.41,
        navigate_print_interval=20,
    ):
        objectives = [
            LookForBlueBallObjective(
                turn_rate=turn_rate,
                min_confidence=min_confidence,
                print_interval=search_print_interval,
            ),
            NavigateToBlueBallObjective(
                desired_distance=desired_distance,
                print_interval=navigate_print_interval,
                nav_mode="sequential"
            ),
        ]
        super().__init__(objectives)
