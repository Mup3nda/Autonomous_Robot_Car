"""Composite objective: search for an ArUco marker, then navigate to it."""

from objective import CompositeObjective
from Objectives.look_for_aruco_objective import LookForArucoObjective
from Objectives.navigate_to_aruco_objective import NavigateToArucoObjective


class SearchAndNavigateToAruco(CompositeObjective):
    """Search for an ArUco marker, then navigate to it."""

    name = "Search_And_Navigate_To_Aruco"

    def __init__(
        self,
        marker_id=0,
        fallback_marker_id=None,
        search_timeout_s=5.0,
        turn_rate=0.5,
        min_confidence=1,
        scan_mode="spin",
        heading_tolerance_deg=4.0,
        desired_distance=0.4,
        print_interval=20,
        nav_mode="aruco",
    ):
        self.fallback_flag = 0  # 0 = using primary marker, 1 = using fallback marker
        objectives = [
            LookForArucoObjective(
                marker_id=marker_id,
                fallback_marker_id=fallback_marker_id,
                search_timeout_s=search_timeout_s,
                turn_rate=turn_rate,
                min_confidence=min_confidence,
                print_interval=print_interval,
                scan_mode=scan_mode,
                heading_tolerance_deg=heading_tolerance_deg,
            ),
            NavigateToArucoObjective(
                marker_id=marker_id,
                fallback_marker_id=fallback_marker_id,
                search_timeout_s=search_timeout_s,
                desired_distance=desired_distance,
                print_interval=print_interval,
                nav_mode=nav_mode,
            ),
        ]
        super().__init__(objectives)
    
    def start(self, ctx):
        """Initialize fallback_flag in mission context and start child objectives."""
        ctx.memory["fallback_flag"] = 0  # Initialize flag
        super().start(ctx)
    
    def tick(self, ctx):
        """Override tick to check for fallback flag from child objectives."""
        super().tick(ctx)
        # Update fallback_flag based on mission context
        if ctx.memory.get("fallback_flag") == 1:
            self.fallback_flag = 1
    
    def get_fallback_flag(self):
        """Return the fallback flag status (0 = primary marker, 1 = fallback marker)."""
        return self.fallback_flag
