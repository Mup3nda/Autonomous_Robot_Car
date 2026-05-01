"""Navigate To Hole Objective - Move towards detected hole target."""

from enum import IntEnum

from mission_context import MissionContext
from objective import Objective
from shole import SHole


class NavigateToHoleState(IntEnum):
    MOVING = 0
    COMPLETE = 1
    DONE = 99


class NavigateToHoleObjective(Objective):
    """Move the robot towards a detected hole target.

    Uses the same navigation pipeline as balls, but with SHole detector.
    """

    def __init__(self, desired_distance=0.40, print_interval=20, nav_mode="sequential", COMPENSATE_PARAMETER = 35):
        super().__init__()
        self.desired_distance = desired_distance
        self.print_interval = print_interval
        self.nav_mode = str(nav_mode).lower()
        self.tick_count = 0
        self.COMPENSATE_PARAMETER = COMPENSATE_PARAMETER

    def start(self, ctx: MissionContext):
        """Initialize navigation to hole."""
        self.state = NavigateToHoleState.MOVING
        self.tick_count = 0

        # Create hole detector
        detector = SHole(cam=ctx.cam, gpio=ctx.gpio, service=ctx.service)

        # Setup navigation with hole detector
        ctx.actions.navigation.setup_detector(detector)
        ctx.actions.navigation.setup(
            desired_distance=self.desired_distance,
            ctx=ctx,
            nav_mode=self.nav_mode,
            COMPENSATE_PARAMETER=self.COMPENSATE_PARAMETER
        )
        ctx.actions.navigation.start()

        print(f"% Objective: Navigate To Hole (target_distance={self.desired_distance}m, nav_mode={self.nav_mode})")

    def tick(self, ctx: MissionContext):
        """Execute one iteration of navigation."""
        self.tick_count += 1

        if ctx.actions.navigation.is_complete():
            self.state = NavigateToHoleState.COMPLETE
            self._done = True
            print("% Navigate To Hole objective complete!")

        elif self.tick_count % self.print_interval == 0:
            target_info = ctx.actions.navigation.get_target_info()
            if target_info:
                print(f"% Navigating to hole: dist={target_info.get('distance', 0):.2f}m, "
                      f"conf={target_info.get('confidence', 0)}")

    def stop(self, ctx: MissionContext):
        """Clean up when objective is stopped or interrupted."""
        ctx.actions.navigation.stop()
        print("% Navigate To Hole objective stopped")