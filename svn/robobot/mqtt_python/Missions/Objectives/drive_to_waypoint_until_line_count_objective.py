"""Drive toward a waypoint and stop after detecting a target line count.

Behavior:
- Keep waypoint navigation active.
- Count distinct line detections using edge sensor confidence thresholds.
- Ignore early detections by configuring stop_line_count (default: stop on 2nd line).
"""

import math
import os
import sys
import time as t
from enum import IntEnum

from mission_context import MissionContext
from objective import Objective

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from sworld_point import SWorldPoint


class DriveToWaypointUntilLineCountState(IntEnum):
    NAVIGATING = 0
    COMPLETE = 2
    DONE = 99


class DriveToWaypointUntilLineCountObjective(Objective):
    """Navigate to waypoint; stop navigation when the Nth line is detected."""

    name = "drive_to_waypoint_until_line_count"

    def __init__(
        self,
        waypoint=(0.0, 0.0),
        is_local=False,
        print_interval=20,
        nav_mode="smooth",
        line_detect_confidence=4,
        line_clear_confidence=1,
        stop_line_count=2,
        max_duration_s=0.0,
    ):
        super().__init__()
        self.waypoint = (float(waypoint[0]), float(waypoint[1]))
        self.is_local = bool(is_local)
        self.frame = "local" if self.is_local else "global"
        self.print_interval = int(print_interval)
        self.nav_mode = str(nav_mode)
        self.line_detect_confidence = int(line_detect_confidence)
        self.line_clear_confidence = int(line_clear_confidence)
        self.stop_line_count = max(1, int(stop_line_count))
        self.max_duration_s = float(max_duration_s)

        self.tick_count = 0
        self.line_count = 0
        self.in_line = False
        self.start_time = 0.0

    def start(self, ctx: MissionContext):
        self.state = DriveToWaypointUntilLineCountState.NAVIGATING
        self.tick_count = 0
        self.line_count = 0
        self.in_line = False
        self.start_time = t.time()
        self._done = False

        detector = SWorldPoint(self.waypoint[0], self.waypoint[1], frame=self.frame)
        ctx.actions.navigation.setup_detector(detector)
        ctx.actions.navigation.setup(desired_distance=0.0, ctx=ctx, nav_mode=self.nav_mode)
        ctx.actions.navigation.start()

        print(
            "% Objective: Drive To Waypoint Until Line Count "
            f"({self.waypoint[0]:.2f}, {self.waypoint[1]:.2f}), frame={self.frame}, "
            f"stop_line_count={self.stop_line_count}, nav_mode={self.nav_mode}"
        )

    def tick(self, ctx: MissionContext):
        self.tick_count += 1

        if self.state != DriveToWaypointUntilLineCountState.NAVIGATING:
            return

        line_detected = ctx.actions.edge.is_line_valid(confidence=self.line_detect_confidence)
        line_still_present = ctx.actions.edge.is_line_valid(confidence=self.line_clear_confidence)

        if line_detected and not self.in_line:
            self.in_line = True
            self.line_count += 1
            print(f"% Line event #{self.line_count} detected while navigating")

            if self.line_count >= self.stop_line_count:
                ctx.actions.navigation.stop()
                ctx.actions.drive.stop()
                self.state = DriveToWaypointUntilLineCountState.COMPLETE
                self._done = True
                print(
                    "% Drive To Waypoint Until Line Count complete: "
                    f"reached line event #{self.line_count}"
                )
                return

        if self.in_line and not line_still_present:
            self.in_line = False

        if self.max_duration_s > 0.0 and (t.time() - self.start_time) >= self.max_duration_s:
            ctx.actions.navigation.stop()
            ctx.actions.drive.stop()
            self.state = DriveToWaypointUntilLineCountState.COMPLETE
            self._done = True
            print(
                "% Drive To Waypoint Until Line Count complete: "
                f"max_duration_s {self.max_duration_s:.1f}s reached"
            )
            return

        if ctx.actions.navigation.is_complete():
            self.state = DriveToWaypointUntilLineCountState.COMPLETE
            self._done = True
            print(
                "% Drive To Waypoint Until Line Count complete: "
                "waypoint reached before stop_line_count"
            )
            return

        if self.tick_count % self.print_interval == 0:
            target_info = ctx.actions.navigation.get_target_info()
            if target_info:
                dist = target_info.get("distance", 0.0)
                bearing = target_info.get("bearing", 0.0)
                print(
                    f"% Waypoint ({self.waypoint[0]:.2f}, {self.waypoint[1]:.2f}) [{self.frame}]: "
                    f"dist={dist:.2f}m, bearing={bearing:.3f}rad, line_count={self.line_count}"
                )

    def stop(self, ctx: MissionContext):
        ctx.actions.navigation.stop()
        ctx.actions.drive.stop()
        print("% Drive To Waypoint Until Line Count objective stopped")
