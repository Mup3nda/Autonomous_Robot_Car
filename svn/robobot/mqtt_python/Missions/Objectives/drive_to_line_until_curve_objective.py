#!/usr/bin/env python3

import time
from enum import IntEnum
from objective import Objective


class DriveToLineUntilCurveState(IntEnum):
    START = 0
    SEARCHING = 1
    CENTERING = 2
    LINE_FOLLOWING = 3
    DONE = 99


class DriveToLineUntilCurveObjective(Objective):
    name = "drive_to_line_until_curve"

    def __init__(
        self,
        follow_left=False,
        search_speed=0.25,
        centering_speed=0.20,
        follow_speed=0.60,
        curve_threshold=1.0,
        min_follow_time_s=1.0,
        curve_persist_time_s=0.3,
        line_found_confidence=4,
        centered_confidence=8,
        lost_line_timeout_s=1.0,
        max_duration_s=30.0,
        curve_detection_delay_s=1.0,
    ):
        super().__init__()
        self.follow_left = bool(follow_left)
        self.search_speed = float(search_speed)
        self.centering_speed = float(centering_speed)
        self.follow_speed = float(follow_speed)
        self.curve_threshold = float(curve_threshold)
        self.min_follow_time_s = float(min_follow_time_s)
        self.curve_persist_time_s = float(curve_persist_time_s)
        self.line_found_confidence = int(line_found_confidence)
        self.centered_confidence = int(centered_confidence)
        self.lost_line_timeout_s = float(lost_line_timeout_s)
        self.max_duration_s = float(max_duration_s)
        self.curve_detection_delay_s = float(curve_detection_delay_s)

    def start(self, ctx):
        self.state = DriveToLineUntilCurveState.START
        self.start_time = time.time()
        self.follow_start_time = None
        self.curve_start_time = None
        self.curve_detection_start_time = None
        self._done = False
        ctx.actions.drive.leds(0, 100, 0)
        print("% DriveToLineUntilCurveObjective: searching for line")

    def _line_lost(self, ctx):
        last_seen_time = ctx.actions.edge.last_seen_time_passed()
        return not ctx.actions.edge.is_line_valid(confidence=self.centered_confidence) and last_seen_time > self.lost_line_timeout_s

    def _get_line_turn_rate(self, ctx):
        return float(getattr(ctx.actions.edge.edge, "lineY", 0.0))

    def tick(self, ctx):
        now = time.time()

        if self.state == DriveToLineUntilCurveState.START:
            ctx.actions.drive.rc(self.search_speed, 0.0)
            self.state = DriveToLineUntilCurveState.SEARCHING

        elif self.state == DriveToLineUntilCurveState.SEARCHING:
            if ctx.actions.edge.is_line_valid(confidence=self.line_found_confidence):
                ctx.actions.edge.start_following(velocity=self.centering_speed, follow_left=self.follow_left)
                self.follow_start_time = now
                self.state = DriveToLineUntilCurveState.CENTERING

        elif self.state == DriveToLineUntilCurveState.CENTERING:
            centered = ctx.actions.edge.is_line_valid(confidence=self.centered_confidence)
            if centered or (now - self.follow_start_time) >= 3.0:
                ctx.actions.edge.start_following(velocity=self.follow_speed, follow_left=self.follow_left)
                self.curve_detection_start_time = now + self.curve_detection_delay_s
                self.state = DriveToLineUntilCurveState.LINE_FOLLOWING

        elif self.state == DriveToLineUntilCurveState.LINE_FOLLOWING:
            if self._line_lost(ctx):
                print("% DriveToLineUntilCurveObjective: lost line, stopping")
                ctx.actions.edge.stop_following()
                ctx.actions.drive.stop()
                self._done = True
                return

            if now - self.start_time > self.max_duration_s:
                print("% DriveToLineUntilCurveObjective: max duration reached, stopping")
                ctx.actions.edge.stop_following()
                ctx.actions.drive.stop()
                self._done = True
                return

            line_turn = abs(self._get_line_turn_rate(ctx))
            if self.curve_detection_start_time is not None and now >= self.curve_detection_start_time:
                # Check if edge profile switched to 'slow' (immediate curve detection)
                if hasattr(ctx.actions.edge, 'edge') and hasattr(ctx.actions.edge.edge, 'currentProfile'):
                    if ctx.actions.edge.edge.currentProfile == 'slow':
                        print(f"% DriveToLineUntilCurveObjective: edge profile switched to 'slow', stopping immediately")
                        ctx.actions.edge.stop_following()
                        ctx.actions.drive.stop()
                        self._done = True
                        return
                
                if line_turn >= self.curve_threshold:
                    if self.curve_start_time is None:
                        self.curve_start_time = now
                    elif (now - self.curve_start_time) >= self.curve_persist_time_s:
                        print(f"% DriveToLineUntilCurveObjective: sustained curve detected (turn rate={line_turn:.2f}), stopping")
                        ctx.actions.edge.stop_following()
                        ctx.actions.drive.stop()
                        self._done = True
                        return
                else:
                    self.curve_start_time = None

        else:
            self._done = True

    def stop(self, ctx):
        ctx.actions.edge.stop_following()
        ctx.actions.drive.stop()
        ctx.actions.drive.leds(0, 0, 0)
        print("% DriveToLineUntilCurveObjective: stopped")
