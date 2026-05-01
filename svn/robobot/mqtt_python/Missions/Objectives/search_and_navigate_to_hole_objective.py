"""Composite objective: search for blue ball, then navigate to it."""

from objective import CompositeObjective
from Objectives.look_for_hole_objective import LookForHoleObjective 
from Objectives.navigate_to_hole_objective import NavigateToHoleObjective


class SearchAndNavigateToHole(CompositeObjective):
    """Run blue-ball search first, then start navigation to the detected ball."""

    name = "Search_And_Navigate_To_Hole"

    def __init__(
        self,
        turn_rate=-0.5,
        min_confidence=1,
        search_print_interval=20,
        desired_distance=0.35,
        navigate_print_interval=20,
        COMPENSATE_PARAMETER = 35
    ):
        objectives = [
            LookForHoleObjective(
                turn_rate=turn_rate,
                min_confidence=min_confidence,
                print_interval=search_print_interval,
            ),
            NavigateToHoleObjective(
                desired_distance=desired_distance,
                print_interval=navigate_print_interval,
                nav_mode="sequential",
                COMPENSATE_PARAMETER = COMPENSATE_PARAMETER
            ),
        ]
        super().__init__(objectives)
