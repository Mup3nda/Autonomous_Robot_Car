"""Composite objective: search for blue ball, then navigate to it."""

from objective import CompositeObjective
from Objectives.look_for_ball_objective import LookForBallObjective
from Objectives.navigate_to_ball_objective import NavigateToBallObjective


class SearchAndNavigateToGolfBall(CompositeObjective):
    """Run blue-ball search first, then start navigation to the detected ball."""

    name = "Search_And_Navigate_To_Golf_Ball"

    def __init__(
        self,
        turn_rate=0.5,
        min_confidence=1,
        search_print_interval=20,
        desired_distance=0.50,
        navigate_print_interval=20,
        COMPENSATE_PARAMETER = 30,
    ):
        objectives = [
            LookForBallObjective(
                turn_rate=turn_rate,
                min_confidence=min_confidence,
                print_interval=search_print_interval,
                color="red_orange",
            ),
            NavigateToBallObjective(
                desired_distance=desired_distance,
                print_interval=navigate_print_interval,
                nav_mode="sequential",
                color="red_orange",
                COMPENSATE_PARAMETER=COMPENSATE_PARAMETER
            ),
        ]
        super().__init__(objectives)
