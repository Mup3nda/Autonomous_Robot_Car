"""Drive the robot forward 1 meter in a straight line.

This objective uses a simple state machine:
- State 0: Start driving forward
- State 1: Continue driving until 1m is traveled or 15s timeout
- State 2: Stop and wait for robot to come to rest
- Done: Log results and mark objective complete
"""
from enum import IntEnum
import time as t
from objective import Objective


class DriveDistanceState(IntEnum):
    START = 0
    DRIVING = 1
    STOPPED = 2

class DriveDistanceObjective(Objective):
    name = "drive_distance_meter"
    PROGRESS_KEY = "drive_meter"
    
    def __init__(self, target_distance_m=1.0, throttle=0.2, timeout_s=15.0, instant_stop=True):
        super().__init__()
        self.target_distance_m = float(target_distance_m)
        self.throttle = float(throttle)
        self.timeout_s = float(timeout_s)
        self.instant_stop = bool(instant_stop)

    def _distance_reached(self, driven_m):
        """Distance completion should work for both forward and reverse driving."""
        return abs(float(driven_m)) >= abs(self.target_distance_m)
    

    def start(self, ctx):
        """Initialize the objective: reset distance tracker and turn on green LED."""
        self.state = DriveDistanceState.START
        ctx.start_local_progress(self.PROGRESS_KEY)
        ctx.actions.drive.leds(0, 100, 0)  # Green LED
        print("% Driving 1m -------------------------")

    def tick(self, ctx):
        """Update the objective state and control the robot."""
        if self.state == DriveDistanceState.START:
            # State 0: Start driving forward at 20% throttle with steering adjustment
            ctx.actions.drive.rc(self.throttle, 0.0)  # rc(throttle, steering): 0.2 forward, 0.0 straight
            self.state = DriveDistanceState.DRIVING
        elif self.state == DriveDistanceState.DRIVING:
            # State 1: Driving - check if 1m reached or timeout
            marker = ctx.memory["_local_progress"][self.PROGRESS_KEY]
            driven = ctx.distance_since_start(self.PROGRESS_KEY)
            elapsed = t.time() - marker["time_s"]
            if self._distance_reached(driven) or (self.timeout_s > 0.0 and elapsed >= self.timeout_s):
                ctx.actions.drive.stop(instant = self.instant_stop)  # Stop driving
                self.state = DriveDistanceState.STOPPED
        elif self.state == DriveDistanceState.STOPPED:
            # State 2: Stopped - wait for velocity to settle to near-zero
            if abs(ctx.pose.velocity()) < 0.001:
                marker = ctx.memory["_local_progress"][self.PROGRESS_KEY]
                driven = ctx.distance_since_start(self.PROGRESS_KEY)
                elapsed = t.time() - marker["time_s"]
                print(
                    f"# drive 1m drove {driven:.3f}m in {elapsed:.3f} seconds"
                )
                self._done = True  # Mark objective as complete
        marker = ctx.memory["_local_progress"][self.PROGRESS_KEY]
        driven = ctx.distance_since_start(self.PROGRESS_KEY)
        elapsed = t.time() - marker["time_s"]
        print(
            f"# drive {int(self.state)}, now {driven:.3f}m in {elapsed:.3f} seconds; "
            f"left {ctx.actions.edge.get_left_position()}, right {ctx.actions.edge.get_right_position()}"
        )

    def stop(self, ctx):
        """Clean up: turn off LED and stop the robot."""
        ctx.actions.drive.leds(0, 0, 0)  # Turn off LEDs
        ctx.actions.drive.stop(instant=self.instant_stop)  # Stop all movement
        print("% Driving 1m ------------------------- end")