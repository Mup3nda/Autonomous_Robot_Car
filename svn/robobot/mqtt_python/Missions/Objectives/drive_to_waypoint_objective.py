"""Drive to one waypoint using SWorldPoint."""

from enum import IntEnum
import math

from mission_context import MissionContext
from objective import Objective
from sodom import odom

# Add parent directory to path
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from sworld_point import SWorldPoint


class DriveToWaypointState(IntEnum):
    NAVIGATING = 0
    ALIGNING_HEADING = 1
    COMPLETE = 2
    DONE = 99


class DriveToWaypointObjective(Objective):
    """Navigate the robot to one waypoint.

    Parameters:
    -----------
    waypoint: Tuple[float, float]
        Target waypoint as (x, y).

    is_local: bool
        If False, waypoint and heading inputs use global frame.
        If True, waypoint and heading inputs use local robot frame at objective start.

    reset_origin: bool
        If True, reset odometry origin when starting this objective.

    print_interval: Int
        Print status every N ticks (default 20 = ~1 second at 50ms tick rate)

    relative_heading_deg: Optional[float]
        If provided, require final robot heading to be this signed angle in the
        selected frame:
        - global frame (is_local=False): absolute heading in world frame.
        - local frame (is_local=True): heading relative to start heading.
    """

    def __init__(
        self,
        waypoint=(0.0, 0.0),
        reset_origin=False,
        print_interval=20,
        nav_mode="smooth",
        relative_heading_deg=None,
        heading_tolerance_deg=3.0,
        heading_kp=1.4,
        heading_max_turn_cmd=0.65,
        heading_min_turn_cmd=0.12,
        is_local=False,
    ):
        super().__init__()
        self.waypoint = (float(waypoint[0]), float(waypoint[1]))
        self.reset_origin = bool(reset_origin)
        self.print_interval = int(print_interval)
        self.nav_mode = str(nav_mode)
        self.is_local = bool(is_local)
        self.frame = "local" if self.is_local else "global"
        self.relative_heading_rad = None if relative_heading_deg is None else math.radians(float(relative_heading_deg))
        self.heading_tolerance_rad = math.radians(abs(float(heading_tolerance_deg)))
        self.heading_kp = float(heading_kp)
        self.heading_max_turn_cmd = abs(float(heading_max_turn_cmd))
        self.heading_min_turn_cmd = abs(float(heading_min_turn_cmd))
        self.tick_count = 0
        self.start_heading_rad = 0.0
        self.target_heading_rad = None

    @staticmethod
    def _wrap_to_pi(angle_rad):
        while angle_rad > math.pi:
            angle_rad -= 2.0 * math.pi
        while angle_rad < -math.pi:
            angle_rad += 2.0 * math.pi
        return angle_rad

    @staticmethod
    def _clamp(value, lo=-1.0, hi=1.0):
        return max(lo, min(hi, value))

    def set_waypoint(self, waypoint):
        """Set waypoint before starting objective."""
        self.waypoint = (float(waypoint[0]), float(waypoint[1]))

    def start(self, ctx: MissionContext):
        """Initialize single-waypoint navigation."""
        self.state = DriveToWaypointState.NAVIGATING
        self.tick_count = 0
        self._done = False
        _, _, self.start_heading_rad = odom.get_world_pose()

        if self.relative_heading_rad is None:
            self.target_heading_rad = None
            heading_text = "None"
        else:
            if self.is_local:
                # Local frame heading command is relative to objective start heading.
                self.target_heading_rad = self._wrap_to_pi(self.start_heading_rad + self.relative_heading_rad)
            else:
                # Global frame heading command is absolute in world frame.
                self.target_heading_rad = self._wrap_to_pi(self.relative_heading_rad)
            heading_text = f"{math.degrees(self.relative_heading_rad):.1f}"

        # Create single point detector with selected frame.
        detector = SWorldPoint(self.waypoint[0], self.waypoint[1], frame=self.frame)

        # Setup navigation action with this detector
        ctx.actions.navigation.setup_detector(detector)
        if self.reset_origin:
            ctx.actions.navigation.reset_origin()
        ctx.actions.navigation.setup(desired_distance=0.0, ctx=ctx, nav_mode=self.nav_mode)
        ctx.actions.navigation.start()

        print(
            f"% Objective: Drive To Waypoint ({self.waypoint[0]:.2f}, {self.waypoint[1]:.2f}), "
            f"frame={self.frame}, reset_origin={self.reset_origin}, nav_mode={self.nav_mode}, "
            f"relative_heading_deg={heading_text}"
        )

    def tick(self, ctx: MissionContext):
        """Execute one iteration of waypoint navigation."""
        self.tick_count += 1

        # Check if navigation is complete
        if self.state == DriveToWaypointState.NAVIGATING:
            if ctx.actions.navigation.is_complete():
                if self.target_heading_rad is None:
                    self.state = DriveToWaypointState.COMPLETE
                    self._done = True
                    print(f"% Drive To Waypoint objective complete!")
                else:
                    # Stop nav first to avoid command contention with heading alignment.
                    ctx.actions.navigation.stop()
                    self.state = DriveToWaypointState.ALIGNING_HEADING
                    print(
                        "% Drive To Waypoint reached position; aligning heading to "
                        f"{math.degrees(self.target_heading_rad):.1f} deg"
                    )
            elif self.tick_count % self.print_interval == 0:
                # Print status periodically
                target_info = ctx.actions.navigation.get_target_info()
                if target_info:
                    dist = target_info.get("distance", 0)
                    bearing = target_info.get("bearing", 0)
                    print(
                        f"% Waypoint ({self.waypoint[0]:.2f}, {self.waypoint[1]:.2f}) "
                        f"[{self.frame}]: dist={dist:.2f}m, bearing={bearing:.3f}rad, "
                        f"conf={target_info.get('confidence', 0)}"
                    )
        elif self.state == DriveToWaypointState.ALIGNING_HEADING:
            _, _, current_heading = odom.get_world_pose()
            err = self._wrap_to_pi(self.target_heading_rad - current_heading)

            if abs(err) <= self.heading_tolerance_rad:
                ctx.actions.drive.stop()
                self.state = DriveToWaypointState.COMPLETE
                self._done = True
                print(
                    "% Drive To Waypoint objective complete with heading: "
                    f"err={math.degrees(err):.2f} deg"
                )
            else:
                w_cmd = self._clamp(
                    self.heading_kp * err,
                    -self.heading_max_turn_cmd,
                    self.heading_max_turn_cmd,
                )
                if abs(w_cmd) < self.heading_min_turn_cmd:
                    w_cmd = math.copysign(self.heading_min_turn_cmd, w_cmd)
                ctx.actions.drive.rc(0.0, w_cmd)

                if self.tick_count % self.print_interval == 0:
                    print(
                        "% Waypoint heading align: "
                        f"target={math.degrees(self.target_heading_rad):.1f}deg, "
                        f"current={math.degrees(current_heading):.1f}deg, "
                        f"err={math.degrees(err):.2f}deg"
                    )

    def stop(self, ctx: MissionContext):
        """Clean up when objective is stopped or interrupted."""
        ctx.actions.navigation.stop()
        ctx.actions.drive.stop()
        print(f"% Drive To Waypoint objective stopped")
