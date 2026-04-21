"""Drive to line, then lock heading to recent straight segment.

This objective extends DriveToLineObjective and adds a final in-place heading
alignment step after line following stops. The heading reference is estimated
from a configurable recent time window while line following is active.
"""

import math
import time as t
from collections import deque

from sodom import odom

from Objectives.drive_to_line_objective import DriveToLineObjective, DriveToLineState


class DriveToLineLockStraightHeadingObjective(DriveToLineObjective):
    name = "drive_to_line_lock_straight_heading"

    ALIGNING_HEADING_STATE = 200

    def __init__(
        self,
        heading_track_window_s=0.45,
        heading_track_confidence=4,
        heading_track_max_turnrate=0.45,
        heading_track_min_samples=6,
        heading_tolerance_deg=2.5,
        heading_kp=1.4,
        heading_max_turn_cmd=0.55,
        heading_min_turn_cmd=0.10,
        heading_align_timeout_s=3.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.heading_track_window_s = max(0.05, float(heading_track_window_s))
        self.heading_track_confidence = max(0, int(heading_track_confidence))
        self.heading_track_max_turnrate = abs(float(heading_track_max_turnrate))
        self.heading_track_min_samples = max(1, int(heading_track_min_samples))

        self.heading_tolerance_rad = math.radians(abs(float(heading_tolerance_deg)))
        self.heading_kp = float(heading_kp)
        self.heading_max_turn_cmd = abs(float(heading_max_turn_cmd))
        self.heading_min_turn_cmd = abs(float(heading_min_turn_cmd))
        self.heading_align_timeout_s = max(0.2, float(heading_align_timeout_s))

        self._heading_samples = deque()
        self._target_heading_rad = None
        self._alignment_started = False
        self._align_start_time = 0.0

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

    def _prune_heading_samples(self, now_s):
        cutoff = now_s - self.heading_track_window_s
        while self._heading_samples and self._heading_samples[0][0] < cutoff:
            self._heading_samples.popleft()

    def _track_heading_sample(self, ctx):
        if not ctx.actions.edge.is_line_valid(confidence=self.heading_track_confidence):
            return
        if abs(ctx.pose.turnrate()) > self.heading_track_max_turnrate:
            return

        now_s = t.time()
        _, _, heading = odom.get_world_pose()
        self._heading_samples.append((now_s, float(heading)))
        self._prune_heading_samples(now_s)

    def _compute_target_heading(self):
        if len(self._heading_samples) < self.heading_track_min_samples:
            return None

        sin_sum = 0.0
        cos_sum = 0.0
        for _, h in self._heading_samples:
            sin_sum += math.sin(h)
            cos_sum += math.cos(h)
        return math.atan2(sin_sum, cos_sum)

    def start(self, ctx):
        super().start(ctx)
        self._heading_samples.clear()
        self._target_heading_rad = None
        self._alignment_started = False
        self._align_start_time = 0.0

    def tick(self, ctx):
        prev_state = self.state
        super().tick(ctx)

        # Track straight-segment heading only while actively line-following.
        if self.state == DriveToLineState.LINE_FOLLOWING:
            self._track_heading_sample(ctx)
            return

        # Detect transition out of line-following and switch to heading lock.
        if (
            not self._alignment_started
            and prev_state == DriveToLineState.LINE_FOLLOWING
            and self.state in (DriveToLineState.STOPPED, DriveToLineState.DONE)
        ):
            self._target_heading_rad = self._compute_target_heading()
            if self._target_heading_rad is None:
                print(
                    "% DriveToLineLockStraightHeading: insufficient heading samples; "
                    "skipping heading lock"
                )
                return

            self._alignment_started = True
            self._align_start_time = t.time()
            self._done = False
            self.state = self.ALIGNING_HEADING_STATE
            print(
                "% DriveToLineLockStraightHeading: locking heading to "
                f"{math.degrees(self._target_heading_rad):.1f} deg "
                f"using last {self.heading_track_window_s:.2f}s"
            )
            return

        if self.state != self.ALIGNING_HEADING_STATE:
            return

        _, _, current_heading = odom.get_world_pose()
        err = self._wrap_to_pi(self._target_heading_rad - current_heading)

        if abs(err) <= self.heading_tolerance_rad:
            ctx.actions.drive.stop(instant=self.instant_stop)
            self._done = True
            print(
                "% DriveToLineLockStraightHeading complete: "
                f"heading error={math.degrees(err):.2f} deg"
            )
            return

        elapsed = t.time() - self._align_start_time
        if elapsed >= self.heading_align_timeout_s:
            ctx.actions.drive.stop(instant=self.instant_stop)
            self._done = True
            print(
                "% DriveToLineLockStraightHeading: heading lock timeout; "
                f"final error={math.degrees(err):.2f} deg"
            )
            return

        w_cmd = self._clamp(
            self.heading_kp * err,
            -self.heading_max_turn_cmd,
            self.heading_max_turn_cmd,
        )
        if abs(w_cmd) < self.heading_min_turn_cmd:
            w_cmd = math.copysign(self.heading_min_turn_cmd, w_cmd)
        ctx.actions.drive.rc(0.0, w_cmd)

    def stop(self, ctx):
        ctx.actions.drive.stop(instant=self.instant_stop)
        super().stop(ctx)
