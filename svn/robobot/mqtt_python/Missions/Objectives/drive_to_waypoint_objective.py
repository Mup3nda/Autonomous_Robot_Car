"""Drive to one world-space waypoint using SWorldPoint."""

from enum import IntEnum

from mission_context import MissionContext
from objective import Objective

# Add parent directory to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sworld_point import SWorldPoint


class DriveToWaypointState(IntEnum):
    NAVIGATING = 0
    COMPLETE = 1
    DONE = 99


class DriveToWaypointObjective(Objective):
    """Navigate the robot to one world-space waypoint.

    Parameters:
    -----------
    waypoint: Tuple[float, float]
        Target waypoint as (x, y) in meters in the local world frame.

    reset_origin: bool
        If True, reset odometry origin when starting this objective.

    print_interval: Int
        Print status every N ticks (default 20 = ~1 second at 50ms tick rate)
    """
    
    def __init__(self, waypoint=(0.0, 0.0), reset_origin=False, print_interval=20, nav_mode="sequential"):
        super().__init__()
        self.waypoint = (float(waypoint[0]), float(waypoint[1]))
        self.reset_origin = bool(reset_origin)
        self.print_interval = int(print_interval)
        self.nav_mode = str(nav_mode)
        self.tick_count = 0

    def set_waypoint(self, waypoint):
        """Set waypoint before starting objective."""
        self.waypoint = (float(waypoint[0]), float(waypoint[1]))

    def start(self, ctx: MissionContext):
        """Initialize single-waypoint navigation."""
        self.state = DriveToWaypointState.NAVIGATING
        self.tick_count = 0
        self._done = False

        # Create single world-point detector
        detector = SWorldPoint(self.waypoint[0], self.waypoint[1])

        # Setup navigation action with this detector
        ctx.actions.navigation.setup_detector(detector)
        if self.reset_origin:
            ctx.actions.navigation.reset_origin()
        ctx.actions.navigation.setup(desired_distance=0.0, ctx=ctx, nav_mode=self.nav_mode)
        ctx.actions.navigation.start()

        print(
            f"% Objective: Drive To Waypoint ({self.waypoint[0]:.2f}, {self.waypoint[1]:.2f}), "
            f"reset_origin={self.reset_origin}, nav_mode={self.nav_mode}"
        )

    def tick(self, ctx: MissionContext):
        """Execute one iteration of waypoint navigation."""
        self.tick_count += 1
        
        # Check if navigation is complete
        if ctx.actions.navigation.is_complete():
            self.state = DriveToWaypointState.COMPLETE
            self._done = True
            print(f"% Drive To Waypoint objective complete!")
        elif self.tick_count % self.print_interval == 0:
            # Print status periodically
            target_info = ctx.actions.navigation.get_target_info()
            if target_info:
                dist = target_info.get("distance", 0)
                bearing = target_info.get("bearing", 0)
                print(
                    f"% Waypoint ({self.waypoint[0]:.2f}, {self.waypoint[1]:.2f}): "
                    f"dist={dist:.2f}m, bearing={bearing:.3f}rad, "
                    f"conf={target_info.get('confidence', 0)}"
                )

    def stop(self, ctx: MissionContext):
        """Clean up when objective is stopped or interrupted."""
        ctx.actions.navigation.stop()
        print(f"% Drive To Waypoint objective stopped")
