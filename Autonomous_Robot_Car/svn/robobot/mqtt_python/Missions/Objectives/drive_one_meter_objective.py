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


class DriveOneMeterState(IntEnum):
    START = 0
    DRIVING = 1
    STOPPED = 2

class DriveOneMeterObjective(Objective):
    name = "drive_one_meter"
    PROGRESS_KEY = "drive_one_meter"

    def start(self, ctx):
        """Initialize the objective: reset distance tracker and turn on green LED."""
        self.state = DriveOneMeterState.START
        ctx.start_local_progress(self.PROGRESS_KEY)
        ctx.actions.drive.leds(0, 100, 0)  # Green LED
        print("% Driving 1m -------------------------")

    def tick(self, ctx):
        """Update the objective state and control the robot."""
        if self.state == DriveOneMeterState.START:
            # State 0: Start driving forward at 20% throttle with steering adjustment
            ctx.actions.drive.rc(0.2, 0.0)  # rc(throttle, steering): 0.2 forward, 0.0 straight
            ctx.actions.drive.servo(1, -800, 300)  # Adjust servo position
            self.state = DriveOneMeterState.DRIVING
        elif self.state == DriveOneMeterState.DRIVING:
            # State 1: Driving - check if 1m reached or timeout
            marker = ctx.memory["_local_progress"][self.PROGRESS_KEY]
            driven = ctx.distance_since_start(self.PROGRESS_KEY)
            elapsed = t.time() - marker["time_s"]
            if driven > 1.0 or elapsed > 15:
                ctx.actions.drive.stop()  # Stop driving
                ctx.actions.drive.servo(1, 0, 0)  # Center servo
                self.state = DriveOneMeterState.STOPPED
        elif self.state == DriveOneMeterState.STOPPED:
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
        ctx.actions.drive.stop()  # Stop all movement
        print("% Driving 1m ------------------------- end")