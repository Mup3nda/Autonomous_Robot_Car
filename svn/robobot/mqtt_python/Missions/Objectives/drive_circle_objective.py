"""Drive in a circle for a configurable number of revolutions.

This objective uses rc(v, w) drive commands:
- v: forward command
- w: turning command

The stop condition is based on odometry heading traveled (tripBh).
"""

import math
from enum import IntEnum

from objective import Objective
from mission_context import MissionContext


class DriveCircleState(IntEnum):
    START = 0
    DRIVING = 1
    STOPPED = 2


class DriveCircleObjective(Objective):
    name = "drive_circle"

    def __init__(
        self,
        radius_m=0.8,
        revolutions=1.0,
        forward_cmd=0.3,
        turn_cmd=None,
        turn_rate_scale=1.0,
        clockwise=False,
        timeout_s=30.0,
        settle_velocity_threshold=0.01,
        print_interval=20,
    ):
        super().__init__()
        self.radius_m = max(0.05, float(radius_m))
        self.revolutions = max(0.01, float(revolutions))
        self.forward_cmd = float(forward_cmd)
        self.turn_cmd = None if turn_cmd is None else float(turn_cmd)
        self.turn_rate_scale = float(turn_rate_scale)
        self.clockwise = bool(clockwise)
        self.timeout_s = float(timeout_s)
        self.settle_velocity_threshold = float(settle_velocity_threshold)
        self.print_interval = max(1, int(print_interval))

        self.state = DriveCircleState.START
        self.tick_count = 0
        self.target_angle_rad = 2.0 * math.pi * self.revolutions
        self._effective_turn_cmd = 0.0

    def _clamp(self, value, lo=-1.0, hi=1.0):
        return max(lo, min(hi, value))

    def _compute_turn_cmd(self):
        if self.turn_cmd is not None:
            cmd = self.turn_cmd
        else:
            # Approximate turn command from desired radius.
            # turn_rate_scale lets us calibrate this on real hardware.
            cmd = (self.forward_cmd / self.radius_m) * self.turn_rate_scale

        cmd = self._clamp(cmd)
        if self.clockwise:
            cmd = -abs(cmd)
        else:
            cmd = abs(cmd)
        return cmd

    def start(self, ctx: MissionContext):
        self._done = False
        self.state = DriveCircleState.START
        self.tick_count = 0
        self.target_angle_rad = 2.0 * math.pi * self.revolutions
        self._effective_turn_cmd = self._compute_turn_cmd()

        ctx.pose.tripBreset()
        ctx.actions.drive.leds(0, 100, 0)
        print(
            "% Objective: Drive Circle "
            f"(radius={self.radius_m:.2f}m, rev={self.revolutions:.2f}, "
            f"v={self.forward_cmd:.3f}, w={self._effective_turn_cmd:.3f}, "
            f"clockwise={self.clockwise})"
        )

    def tick(self, ctx: MissionContext):
        if self._done:
            return

        self.tick_count += 1

        if self.state == DriveCircleState.START:
            ctx.actions.drive.rc(self._clamp(self.forward_cmd), self._effective_turn_cmd)
            self.state = DriveCircleState.DRIVING
            return

        if self.state == DriveCircleState.DRIVING:
            turned = abs(ctx.pose.tripBh)
            elapsed = ctx.pose.tripBtimePassed()

            if turned >= self.target_angle_rad or elapsed >= self.timeout_s:
                ctx.actions.drive.stop()
                self.state = DriveCircleState.STOPPED
            elif self.tick_count % self.print_interval == 0:
                print(
                    f"% Drive Circle: turned={turned:.3f}/{self.target_angle_rad:.3f} rad, "
                    f"time={elapsed:.2f}s"
                )
            return

        if self.state == DriveCircleState.STOPPED:
            if abs(ctx.pose.velocity()) <= self.settle_velocity_threshold:
                print(
                    "% Drive Circle complete: "
                    f"turned={abs(ctx.pose.tripBh):.3f} rad, "
                    f"distance={ctx.pose.tripB:.3f}m, "
                    f"time={ctx.pose.tripBtimePassed():.2f}s"
                )
                self._done = True

    def stop(self, ctx: MissionContext):
        ctx.actions.drive.stop()
        ctx.actions.drive.leds(0, 0, 0)
        print("% Drive Circle objective stopped")
