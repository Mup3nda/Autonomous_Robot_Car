"""Drive forward until the edge sensor detects a line, then automatically follow it.

State machine:
- State 0: Wait for IR proximity detection (skipped by default)
- State 1: Drive forward until line detected, then switch to line following
- State 2: Stop if no line was found or timeout occurred
- State 10: Follow the line (active line control)
- State 99: Line following complete
"""
from enum import IntEnum
import time as t
from objective import Objective
from robot_actions import RobotActions
from mission_context import MissionContext


# Objective tuning constants
SEARCH_SPEED = 0.2
CENTERING_SPEED = 0.2
FOLLOW_SPEED = 0.80
SEARCH_MAX_DISTANCE_M = 1.0
SEARCH_TIMEOUT_S = 15.0
LINE_FOUND_CONFIDENCE = 4
CENTERED_CONFIDENCE = 8
CENTERED_MIN_TIME_S = 2.0
CENTERING_TIMEOUT_S = 4.0
FOLLOW_VALID_CONFIDENCE = 2
LOST_LINE_TIMEOUT_S = 5
STOPPED_VELOCITY_EPS = 0.001
FOLLOW_LEFT = False  # Set to True to follow line on left side instead of right


class DriveToLineState(IntEnum):
    START = 0
    SEARCHING = 1
    STOPPED = 2
    CENTERING = 3
    RAMPING = 4
    LINE_FOLLOWING = 10
    DONE = 99

class DriveToLineObjective(Objective):
    name = "drive_to_line"
    SEARCH_PROGRESS_KEY = "drive_to_line_search"
    ALONG_LINE_PROGRESS_KEY = "drive_to_line_along"

    def __init__(
        self,
        follow_left=FOLLOW_LEFT,
        follow_speed=FOLLOW_SPEED,
        search_speed=SEARCH_SPEED,
        centering_speed=CENTERING_SPEED,
        follow_ramp_time_s=FOLLOW_RAMP_TIME_S,
        lost_line_timeout_s=LOST_LINE_TIMEOUT_S,
    ):
        super().__init__()
        self.follow_left = bool(follow_left)
        self.follow_speed = float(follow_speed)
        self.search_speed = float(search_speed)
        self.centering_speed = float(centering_speed)
        self.follow_ramp_time_s = max(0.0, float(follow_ramp_time_s))
        self.lost_line_timeout_s = float(lost_line_timeout_s)

    def start(self, ctx):
        """Initialize local progress trackers without resetting global odometry."""
        self.state = DriveToLineState.START
        self.dist_to_line = 0.0  # Track distance traveled before finding line
        self.along_line_started = False
        self.centering_start_time = 0.0  # Wall-clock when centering started
        self.centering_deadline = 0.0  # Hard timeout for centering phase
        self.follow_ramp_start_time = 0.0  # Wall-clock when follow speed ramp starts
        ctx.start_local_progress(self.SEARCH_PROGRESS_KEY)
        ctx.actions.drive.leds(0, 100, 0)  # Green LED
        print("% Driving to line ---------------------- right ir start ---")

    def _line_lost(self, ctx):
        return (
            not ctx.actions.edge.is_line_valid(confidence=FOLLOW_VALID_CONFIDENCE)
            and ctx.actions.edge.last_seen_time_passed() > self.lost_line_timeout_s
        )

    def tick(self, ctx):
        """Update objective state and control the robot."""
        if self.state == DriveToLineState.START:
            # State 0: Start driving forward (IR check was removed)
            ctx.actions.drive.rc(self.search_speed, 0.0)  # Search speed, straight
            ctx.actions.drive.lognow(3)  # Log sensor data
            self.state = DriveToLineState.SEARCHING
        elif self.state == DriveToLineState.SEARCHING:
            # State 1: Searching for line while driving forward
            search_marker = ctx.memory["_local_progress"][self.SEARCH_PROGRESS_KEY]
            search_dist = ctx.distance_since_start(self.SEARCH_PROGRESS_KEY)
            search_elapsed = t.time() - search_marker["time_s"]
            if search_dist > SEARCH_MAX_DISTANCE_M or search_elapsed > SEARCH_TIMEOUT_S:
                # Stop if traveled >1m or >15s timeout without finding line
                ctx.actions.drive.stop(instant=False)
                self.state = DriveToLineState.STOPPED
            if ctx.actions.edge.is_line_valid(confidence=LINE_FOUND_CONFIDENCE):
                # Line detected! Switch to centering mode at low speed
                ctx.actions.edge.start_following(velocity=self.centering_speed, follow_left=self.follow_left)
                self.dist_to_line = search_dist  # Record distance to line
                ctx.start_local_progress(self.ALONG_LINE_PROGRESS_KEY)
                self.along_line_started = True
                self.centering_start_time = t.time()
                self.centering_deadline = self.centering_start_time + CENTERING_TIMEOUT_S
                self.state = DriveToLineState.CENTERING
        elif self.state == DriveToLineState.CENTERING:
            # State 3: Center on line at low speed before accelerating.
            # Promote to high speed when line confidence is strong, or after timeout.
            now = t.time()
            centered = ctx.actions.edge.is_line_valid(confidence=CENTERED_CONFIDENCE)
            centered_long_enough = now - self.centering_start_time > CENTERED_MIN_TIME_S
            timed_out = now >= self.centering_deadline
            if (centered and centered_long_enough) or timed_out:
                if self.follow_ramp_time_s > 0.0 and self.follow_speed > self.centering_speed:
                    self.follow_ramp_start_time = now
                    self.state = DriveToLineState.RAMPING
                else:
                    ctx.actions.edge.start_following(velocity=self.follow_speed, follow_left=self.follow_left)
                    self.state = DriveToLineState.LINE_FOLLOWING
        elif self.state == DriveToLineState.RAMPING:
            # State 4: Gradually ramp follow speed to avoid steering jerk and oscillation.
            now = t.time()
            ramp_elapsed = now - self.follow_ramp_start_time
            if self.follow_ramp_time_s <= 0.0:
                alpha = 1.0
            else:
                alpha = max(0.0, min(1.0, ramp_elapsed / self.follow_ramp_time_s))
            ramp_speed = self.centering_speed + (self.follow_speed - self.centering_speed) * alpha

            ctx.actions.edge.start_following(velocity=ramp_speed, follow_left=self.follow_left)

            if self._line_lost(ctx):
                ctx.actions.edge.stop_following()
                ctx.actions.drive.stop()
                self.state = DriveToLineState.STOPPED
            elif alpha >= 1.0:
                self.state = DriveToLineState.LINE_FOLLOWING
        elif self.state == DriveToLineState.STOPPED:
            # State 2: Stopped after timeout - wait for robot to settle
            if abs(ctx.pose.velocity()) < STOPPED_VELOCITY_EPS:
                self.state = DriveToLineState.DONE  # Mark as done
        elif self.state == DriveToLineState.LINE_FOLLOWING:
            # State 10: Following line - check if line is still valid
            if self._line_lost(ctx):
                # Lost the line - stop and try to recover
                ctx.actions.edge.stop_following()
                ctx.actions.drive.stop(instant=False)
                self.state = DriveToLineState.STOPPED  # Go to stopped state
        else:
            # Final state - log results and mark complete
            along_line_dist = 0.0
            along_line_time = 0.0
            if self.along_line_started:
                along_marker = ctx.memory["_local_progress"][self.ALONG_LINE_PROGRESS_KEY]
                along_line_dist = ctx.distance_since_start(self.ALONG_LINE_PROGRESS_KEY)
                along_line_time = t.time() - along_marker["time_s"]
            print(
                f"# drive to line {self.dist_to_line:.3f}m, then along line "
                f"{along_line_dist:.3f}m in {along_line_time:.3f} seconds"
            )
            ctx.actions.drive.stop(instant=False)
            self._done = True  # Mark objective as complete

    def stop(self, ctx):
        """Clean up: turn off LED and stop the robot."""
        ctx.actions.drive.leds(0, 0, 0)  # Turn off LEDs
        ctx.actions.drive.stop(instant=False)  # Stop all movement
        print("% Driving to line ------------------------- end")
