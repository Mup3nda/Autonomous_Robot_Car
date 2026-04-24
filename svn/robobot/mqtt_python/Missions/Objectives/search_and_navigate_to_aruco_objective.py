"""Composite objective: search for blue ball, then navigate to it."""

from objective import CompositeObjective
from Objectives.look_for_aruco_objective import LookForArucoObjective
from Objectives.navigate_to_aruco_objective import NavigateToArucoObjective


class SearchAndNavigateToAruco(CompositeObjective):
    """Navigating to platform"""

    name = "Search_And_Navigate_To_Aruco"

    def __init__(
        self,
        marker_id=0,
        fallback_marker_id=None,
        search_time_out=5.0,
        turn_rate=0.5,
        min_confidence=1,
        search_print_interval=20,
        desired_distance=0.4,
        navigate_print_interval=20,
    ):
        objectives = [
            LookForArucoObjective(
                marker_id = marker_id,
                fallback_marker_id=None,
                search_time_out=5.0,
                turn_rate=turn_rate,
                min_confidence=min_confidence,
                print_interval=search_print_interval,
            ),
            NavigateToArucoObjective(
                marker_id = marker_id,
                desired_distance=desired_distance,
                print_interval=navigate_print_interval,
                nav_mode="aruco"
            ),
        ]
        super().__init__(objectives)
