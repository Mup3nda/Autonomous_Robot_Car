"""Drive forward a fixed distance, then drive backward the same distance.

State machine:
- FORWARD_START: initialize forward motion
- FORWARD_DRIVING: keep driving forward until target distance or timeout
- FORWARD_STOPPED: wait until robot has stopped
- BACKWARD_START: initialize backward motion
- BACKWARD_DRIVING: keep driving backward until target distance or timeout
- BACKWARD_STOPPED: wait until robot has stopped, then finish
"""

from enum import IntEnum
import time as t
from objective import Objective

TARGET_DISTANCE_M = 2.0
OFFSET_STOP_DIST = 0.10
FORWARD_THROTTLE = 0.40 #0.20
BACKWARD_THROTTLE = -0.40
STEERING = 0.0
TIMEOUT_S = 20.0
WAIT_BEFORE_BACKWARD_S = 1.0
STOP_VELOCITY_THRESHOLD = 0.001

class DriveToTimerAndBackState(IntEnum):
    FORWARD_START = 0
    FORWARD_DRIVING = 1
    FORWARD_STOPPED = 2
    WAIT_BEFORE_BACKWARD = 3
    BACKWARD_START = 4
    BACKWARD_DRIVING = 5
    BACKWARD_STOPPED = 6


class DriveToTimerAndBackObjective(Objective):
    name = "drive_to_timer_and_back_objective"
    FORWARD_PROGRESS_KEY = "drive_to_timer_forward"
    BACKWARD_PROGRESS_KEY = "drive_to_timer_backward"

    def __init__(
        self,
        target_distance = TARGET_DISTANCE_M,
        offset_stop_distance = OFFSET_STOP_DIST,
        forward_throttle = FORWARD_THROTTLE,
        backward_throttle = BACKWARD_THROTTLE,
        
        ):
        super().__init__()
        self.target_distance = target_distance
        self.offset_stop_distance = offset_stop_distance,
        self.forward_throttle = forward_throttle,
        self.backward_throttle = backward_throttle,
        
    def start(self, ctx):
        self.state = DriveToTimerAndBackState.FORWARD_START
        ctx.actions.drive.leds(0, 100, 0)  # Green
        print("% Drive out and back -------------------------")

    def tick(self, ctx):
        if self.state == DriveToTimerAndBackState.FORWARD_START:
            ctx.start_local_progress(self.FORWARD_PROGRESS_KEY)
            #ctx.actions.drive.rc(self.forward_throttle, self.STEERING)
            ctx.actions.drive.ramp_to(self.forward_throttle, STEERING)
            self.state = DriveToTimerAndBackState.FORWARD_DRIVING

        elif self.state == DriveToTimerAndBackState.FORWARD_DRIVING:
            marker = ctx.memory["_local_progress"][self.FORWARD_PROGRESS_KEY]
            driven = ctx.distance_since_start(self.FORWARD_PROGRESS_KEY)
            elapsed = t.time() - marker["time_s"]

            if driven >= self.TARGET_DISTANCE_M or elapsed > TIMEOUT_S:
                ctx.actions.drive.stop(instant=False)
                self.state = DriveToTimerAndBackState.FORWARD_STOPPED

        elif self.state == DriveToTimerAndBackState.FORWARD_STOPPED:
            if abs(ctx.pose.velocity()) < STOP_VELOCITY_THRESHOLD:
                marker = ctx.memory["_local_progress"][self.FORWARD_PROGRESS_KEY]
                driven = ctx.distance_since_start(self.FORWARD_PROGRESS_KEY)
                elapsed = t.time() - marker["time_s"]
                print(f"# forward leg drove {driven:.3f}m in {elapsed:.3f}s")
                ctx.reset_state_time()
                self.state = DriveToTimerAndBackState.WAIT_BEFORE_BACKWARD
        
        elif self.state == DriveToTimerAndBackState.WAIT_BEFORE_BACKWARD:
            if ctx.state_time_passed() >= WAIT_BEFORE_BACKWARD_S:
                self.state = DriveToTimerAndBackState.BACKWARD_START
                
        elif self.state == DriveToTimerAndBackState.BACKWARD_START:
            ctx.start_local_progress(self.BACKWARD_PROGRESS_KEY)
            #ctx.actions.drive.rc(self.bacward_throttle, STEERING)
            ctx.actions.drive.ramp_to(self.bacward_throttle, STEERING)
            self.state = DriveToTimerAndBackState.BACKWARD_DRIVING

        elif self.state == DriveToTimerAndBackState.BACKWARD_DRIVING:
            marker = ctx.memory["_local_progress"][self.BACKWARD_PROGRESS_KEY]
            driven = abs(ctx.distance_since_start(self.BACKWARD_PROGRESS_KEY))
            elapsed = t.time() - marker["time_s"]

            if driven >= self.TARGET_DISTANCE_M + self.OFFSET_STOP_DIST or elapsed > TIMEOUT_S:
                ctx.actions.drive.stop()
                self.state = DriveToTimerAndBackState.BACKWARD_STOPPED

        elif self.state == DriveToTimerAndBackState.BACKWARD_STOPPED:
            if abs(ctx.pose.velocity()) < STOP_VELOCITY_THRESHOLD:
                marker = ctx.memory["_local_progress"][self.BACKWARD_PROGRESS_KEY]
                driven = abs(ctx.distance_since_start(self.BACKWARD_PROGRESS_KEY))
                elapsed = t.time() - marker["time_s"]
                print(f"# backward leg drove {driven:.3f}m in {elapsed:.3f}s")
                self._done = True

        self._log_progress(ctx)

    def _log_progress(self, ctx):
        if self.state in (
            DriveToTimerAndBackState.FORWARD_START,
            DriveToTimerAndBackState.FORWARD_DRIVING,
            DriveToTimerAndBackState.FORWARD_STOPPED,
            DriveToTimerAndBackState.WAIT_BEFORE_BACKWARD,
        ):
            if self.FORWARD_PROGRESS_KEY in ctx.memory.get("_local_progress", {}):
                marker = ctx.memory["_local_progress"][self.FORWARD_PROGRESS_KEY]
                driven = ctx.distance_since_start(self.FORWARD_PROGRESS_KEY)
                elapsed = t.time() - marker["time_s"]
                print(
                    f"# state {int(self.state)} forward {driven:.3f}m in {elapsed:.3f}s; "
                )

        elif self.state in (
            DriveToTimerAndBackState.BACKWARD_START,
            DriveToTimerAndBackState.BACKWARD_DRIVING,
            DriveToTimerAndBackState.BACKWARD_STOPPED,
        ):
            if self.BACKWARD_PROGRESS_KEY in ctx.memory.get("_local_progress", {}):
                marker = ctx.memory["_local_progress"][self.BACKWARD_PROGRESS_KEY]
                driven = abs(ctx.distance_since_start(self.BACKWARD_PROGRESS_KEY))
                elapsed = t.time() - marker["time_s"]
                print(
                    f"# state {int(self.state)} backward {driven:.3f}m in {elapsed:.3f}s; "
                )

    def stop(self, ctx):
        ctx.actions.drive.leds(0, 0, 0)
        ctx.actions.drive.stop()
        print("% Drive out and back ------------------------- end")
