"""Align robot heading to the tangent direction of a target circle.

This objective rotates in place using odometry feedback. It is intended for
circle-entry correction when the robot reaches the entry point with variable
heading (for example after ramp approach).

Geometry assumption:
- Circle center lies on a known horizontal line y = center_line_y_m in the
  local odom frame used by the mission.
- Heading 0 rad is along +x of this frame.
"""

import math
import time as t
from enum import IntEnum

from mission_context import MissionContext
from objective import Objective
from sodom import odom


class AlignToCircleTangentState(IntEnum):
    START = 0
    ALIGNING = 1
    STOPPED = 2


class AlignToCircleTangentObjective(Objective):
    name = "align_to_circle_tangent"

    def __init__(
        self,
        radius_m=0.35,
        clockwise=True,
        center_line_y_m=0.0,
        linear_cmd=0.0,
        max_turn_cmd=0.35,
        min_turn_cmd=0.12,
        kp_heading=1.4,
        heading_tolerance_deg=2.0,
        timeout_s=8.0,
        settle_velocity_threshold=0.01,
        settle_turnrate_threshold=0.01,
        print_interval=20,
    ):
        super().__init__()
        self.radius_m = max(0.05, float(radius_m))
        self.clockwise = bool(clockwise)
        self.center_line_y_m = float(center_line_y_m)
        self.linear_cmd = float(linear_cmd)
        self.max_turn_cmd = abs(float(max_turn_cmd))
        self.min_turn_cmd = abs(float(min_turn_cmd))
        self.kp_heading = float(kp_heading)
        self.heading_tolerance_rad = math.radians(abs(float(heading_tolerance_deg)))
        self.timeout_s = float(timeout_s)
        self.settle_velocity_threshold = float(settle_velocity_threshold)
        self.settle_turnrate_threshold = float(settle_turnrate_threshold)
        self.print_interval = max(1, int(print_interval))

        self.state = AlignToCircleTangentState.START
        self.tick_count = 0
        self.start_time_s = 0.0
        self.last_error_rad = 0.0
        self.last_desired_heading_rad = 0.0
        self.timed_out = False

    @staticmethod
    def _clamp(value, lo=-1.0, hi=1.0):
        return max(lo, min(hi, value))

    @staticmethod
    def _wrap_to_pi(angle_rad):
        while angle_rad > math.pi:
            angle_rad -= 2.0 * math.pi
        while angle_rad < -math.pi:
            angle_rad += 2.0 * math.pi
        return angle_rad

    def _compute_desired_heading(self, y_rel_m, current_heading_rad):
        # Tangent heading candidates from y-offset and radius.
        # For clockwise:  cos(h) = +y/r
        # For counterclockwise: cos(h) = -y/r
        cos_term = (y_rel_m / self.radius_m) if self.clockwise else (-y_rel_m / self.radius_m)
        cos_term = self._clamp(cos_term, -1.0, 1.0)
        base = math.acos(cos_term)
        candidates = (base, -base)

        # Pick the branch requiring the smallest in-place correction.
        desired = min(
            candidates,
            key=lambda angle: abs(self._wrap_to_pi(angle - current_heading_rad)),
        )
        return desired

    def start(self, ctx: MissionContext):
        self._done = False
        self.state = AlignToCircleTangentState.START
        self.tick_count = 0
        self.start_time_s = t.time()
        self.last_error_rad = 0.0
        self.last_desired_heading_rad = 0.0
        self.timed_out = False

        x, y, h = odom.get_world_pose()
        y_rel = y - self.center_line_y_m
        desired_h = self._compute_desired_heading(y_rel, h)
        err = self._wrap_to_pi(desired_h - h)

        ctx.actions.drive.leds(100, 60, 0)
        print(
            "% Objective: Align To Circle Tangent "
            f"(x={x:.3f}, y={y:.3f}, h={math.degrees(h):.1f}deg, "
            f"y_rel={y_rel:.3f}m, desired_h={math.degrees(desired_h):.1f}deg, "
            f"err={math.degrees(err):.1f}deg, clockwise={self.clockwise})"
        )

    def tick(self, ctx: MissionContext):
        if self._done:
            return

        self.tick_count += 1

        if self.state == AlignToCircleTangentState.START:
            self.state = AlignToCircleTangentState.ALIGNING
            return

        if self.state == AlignToCircleTangentState.ALIGNING:
            elapsed = t.time() - self.start_time_s
            x, y, h = odom.get_world_pose()
            y_rel = y - self.center_line_y_m

            desired_h = self._compute_desired_heading(y_rel, h)
            err = self._wrap_to_pi(desired_h - h)

            self.last_error_rad = err
            self.last_desired_heading_rad = desired_h

            if abs(err) <= self.heading_tolerance_rad:
                ctx.actions.drive.stop()
                self.state = AlignToCircleTangentState.STOPPED
            elif elapsed >= self.timeout_s:
                self.timed_out = True
                ctx.actions.drive.stop()
                self.state = AlignToCircleTangentState.STOPPED
            else:
                w_cmd = self._clamp(
                    self.kp_heading * err,
                    -self.max_turn_cmd,
                    self.max_turn_cmd,
                )
                if abs(w_cmd) < self.min_turn_cmd:
                    w_cmd = math.copysign(self.min_turn_cmd, w_cmd)
                ctx.actions.drive.rc(self.linear_cmd, w_cmd)

            if self.tick_count % self.print_interval == 0:
                print(
                    "% Align Circle Tangent: "
                    f"y_rel={y_rel:.3f}m, "
                    f"h={math.degrees(h):.1f}deg, "
                    f"target={math.degrees(desired_h):.1f}deg, "
                    f"err={math.degrees(err):.1f}deg, "
                    f"time={elapsed:.2f}s"
                )
            return

        if self.state == AlignToCircleTangentState.STOPPED:
            if (
                abs(ctx.pose.velocity()) <= self.settle_velocity_threshold
                and abs(ctx.pose.turnrate()) <= self.settle_turnrate_threshold
            ):
                x, y, h = odom.get_world_pose()
                result = "timeout" if self.timed_out else "aligned"
                print(
                    "% Align To Circle Tangent complete: "
                    f"result={result}, "
                    f"x={x:.3f}, y={y:.3f}, "
                    f"h={math.degrees(h):.1f}deg, "
                    f"target={math.degrees(self.last_desired_heading_rad):.1f}deg, "
                    f"err={math.degrees(self.last_error_rad):.2f}deg"
                )
                self._done = True

    def stop(self, ctx: MissionContext):
        ctx.actions.drive.stop()
        ctx.actions.drive.leds(0, 0, 0)
        print("% Align To Circle Tangent objective stopped")
