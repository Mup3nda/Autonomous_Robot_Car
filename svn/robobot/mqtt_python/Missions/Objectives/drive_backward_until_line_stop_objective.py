"""Drive backward until a line is detected, then stop.

State machine:
- START: Initialize reverse drive and progress tracking
- DRIVING: Reverse while checking line validity
- STOPPING: Wait until robot settles after stop command
- DONE: Objective complete
"""

from enum import IntEnum
import time as t

from objective import Objective


class DriveBackwardUntilLineStopState(IntEnum):
    START = 0
    DRIVING = 1
    STOPPING = 2
    DONE = 99


class DriveBackwardUntilLineStopObjective(Objective):
    name = "drive_backward_until_line_stop"
    PROGRESS_KEY = "drive_backward_until_line_stop"

    DEFAULT_REVERSE_SPEED = -0.25
    DEFAULT_LINE_FOUND_CONFIDENCE = 4
    DEFAULT_TIMEOUT_S = 8.0
    DEFAULT_MAX_DISTANCE_M = 0.0
    STOPPED_VELOCITY_EPS = 0.001

    def __init__(
        self,
        reverse_speed=DEFAULT_REVERSE_SPEED,
        line_found_confidence=DEFAULT_LINE_FOUND_CONFIDENCE,
        timeout_s=DEFAULT_TIMEOUT_S,
        max_distance_m=DEFAULT_MAX_DISTANCE_M,
        instant_stop=True,
    ):
        super().__init__()
        self.reverse_speed = float(reverse_speed)
        self.line_found_confidence = int(line_found_confidence)
        self.timeout_s = float(timeout_s)
        self.max_distance_m = float(max_distance_m)
        self.instant_stop = bool(instant_stop)

    def start(self, ctx):
        self.state = DriveBackwardUntilLineStopState.START
        self.stop_reason = ""
        ctx.actions.drive.leds(0, 100, 0)
        ctx.start_local_progress(self.PROGRESS_KEY)
        print("% Drive backward until line stop -------------------------")

    def tick(self, ctx):
        if self.state == DriveBackwardUntilLineStopState.START:
            ctx.actions.drive.rc(self.reverse_speed, 0.0)
            ctx.actions.drive.lognow(3)
            self.state = DriveBackwardUntilLineStopState.DRIVING
            return

        if self.state == DriveBackwardUntilLineStopState.DRIVING:
            marker = ctx.memory["_local_progress"][self.PROGRESS_KEY]
            driven = abs(ctx.distance_since_start(self.PROGRESS_KEY))
            elapsed = t.time() - marker["time_s"]

            if ctx.actions.edge.is_line_valid(confidence=self.line_found_confidence):
                self.stop_reason = "line_detected"
                ctx.actions.drive.stop(instant=self.instant_stop)
                self.state = DriveBackwardUntilLineStopState.STOPPING
                return

            if self.max_distance_m > 0.0 and driven >= self.max_distance_m:
                self.stop_reason = "max_distance"
                ctx.actions.drive.stop(instant=self.instant_stop)
                self.state = DriveBackwardUntilLineStopState.STOPPING
                return

            if self.timeout_s > 0.0 and elapsed >= self.timeout_s:
                self.stop_reason = "timeout"
                ctx.actions.drive.stop(instant=self.instant_stop)
                self.state = DriveBackwardUntilLineStopState.STOPPING
                return

            return

        if self.state == DriveBackwardUntilLineStopState.STOPPING:
            if abs(ctx.pose.velocity()) < self.STOPPED_VELOCITY_EPS:
                marker = ctx.memory["_local_progress"][self.PROGRESS_KEY]
                driven = abs(ctx.distance_since_start(self.PROGRESS_KEY))
                elapsed = t.time() - marker["time_s"]
                print(
                    f"# drive backward until line stop: reason={self.stop_reason}, "
                    f"distance={driven:.3f}m, time={elapsed:.3f}s"
                )
                self.state = DriveBackwardUntilLineStopState.DONE
                self._done = True

    def stop(self, ctx):
        ctx.actions.drive.leds(0, 0, 0)
        ctx.actions.drive.stop(instant=self.instant_stop)
        print("% Drive backward until line stop ------------------------- end")
