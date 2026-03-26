"""Rotate the robot by a configurable signed angle.

Positive angle means turn left, negative angle means turn right.
A small linear command can be added while turning if needed.
"""

import math
import time as t
from enum import IntEnum

from objective import Objective
from mission_context import MissionContext


class DriveTurnAngleState(IntEnum):
    START = 0
    ROTATING = 1
    STOPPED = 2


class DriveTurnAngleObjective(Objective):
    name = "drive_turn_angle"
    PROGRESS_KEY = "drive_turn_angle"

    def __init__(
        self,
        angle_deg=30.0,
        linear_cmd=0.0,
        turn_cmd=0.8,
        timeout_s=8.0,
        settle_velocity_threshold=0.01,
        settle_turnrate_threshold=0.01,
        print_interval=20,
    ):
        super().__init__()
        self.angle_deg = float(angle_deg)
        self.linear_cmd = float(linear_cmd)
        self.turn_cmd = abs(float(turn_cmd))
        self.timeout_s = float(timeout_s)
        self.settle_velocity_threshold = float(settle_velocity_threshold)
        self.settle_turnrate_threshold = float(settle_turnrate_threshold)
        self.print_interval = max(1, int(print_interval))

        self.state = DriveTurnAngleState.START
        self.tick_count = 0
        self.target_angle_rad = math.radians(abs(self.angle_deg))
        self._effective_turn_cmd = 0.0

    def start(self, ctx: MissionContext):
        self._done = False
        self.state = DriveTurnAngleState.START
        self.tick_count = 0
        self.target_angle_rad = math.radians(abs(self.angle_deg))

        if self.angle_deg >= 0.0:
            self._effective_turn_cmd = self.turn_cmd
            direction = "left"
        else:
            self._effective_turn_cmd = -self.turn_cmd
            direction = "right"

        ctx.start_local_progress(self.PROGRESS_KEY)
        ctx.actions.drive.leds(100, 100, 0)
        print(
            "% Objective: Drive Turn Angle "
            f"(angle={self.angle_deg:.1f}deg, direction={direction}, "
            f"v={self.linear_cmd:.3f}, w={self._effective_turn_cmd:.3f})"
        )

    def tick(self, ctx: MissionContext):
        if self._done:
            return

        self.tick_count += 1

        if self.state == DriveTurnAngleState.START:
            ctx.actions.drive.rc(self.linear_cmd, self._effective_turn_cmd)
            self.state = DriveTurnAngleState.ROTATING
            return

        if self.state == DriveTurnAngleState.ROTATING:
            # Keep issuing the command while rotating for robust control.
            ctx.actions.drive.rc(self.linear_cmd, self._effective_turn_cmd)

            marker = ctx.memory["_local_progress"][self.PROGRESS_KEY]
            turned = abs(ctx.pose.tripAh - marker["tripAh"])
            elapsed = t.time() - marker["time_s"]

            if turned >= self.target_angle_rad or elapsed >= self.timeout_s:
                ctx.actions.drive.stop()
                self.state = DriveTurnAngleState.STOPPED
            elif self.tick_count % self.print_interval == 0:
                print(
                    "% Drive Turn Angle: "
                    f"turned={math.degrees(turned):.1f}/{abs(self.angle_deg):.1f} deg, "
                    f"time={elapsed:.2f}s"
                )
            return

        if self.state == DriveTurnAngleState.STOPPED:
            if (
                abs(ctx.pose.velocity()) <= self.settle_velocity_threshold
                and abs(ctx.pose.turnrate()) <= self.settle_turnrate_threshold
            ):
                marker = ctx.memory["_local_progress"][self.PROGRESS_KEY]
                turned_total = abs(ctx.pose.tripAh - marker["tripAh"])
                elapsed_total = t.time() - marker["time_s"]
                print(
                    "% Drive Turn Angle complete: "
                    f"turned={turned_total:.3f} rad "
                    f"({math.degrees(turned_total):.1f} deg), "
                    f"time={elapsed_total:.2f}s"
                )
                self._done = True

    def stop(self, ctx: MissionContext):
        ctx.actions.drive.stop()
        ctx.actions.drive.leds(0, 0, 0)
        print("% Drive Turn Angle objective stopped")
