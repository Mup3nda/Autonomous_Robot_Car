"""Drive the robot forward 1 meter in a straight line.

This objective uses a simple state machine:
- State 0: Start driving forward
- State 1: Continue driving until 1m is traveled or 15s timeout
- State 2: Stop and wait for robot to come to rest
- Done: Log results and mark objective complete
"""
from objective import Objective

class DriveOneMeterObjective(Objective):
    name = "drive_one_meter"

    def start(self, ctx):
        """Initialize the objective: reset distance tracker and turn on green LED."""
        self.state = 0
        ctx.pose.tripBreset()  # Reset distance counter
        ctx.actions.drive.leds(0, 100, 0)  # Green LED
        print("% Driving 1m -------------------------")

    def tick(self, ctx):
        """Update the objective state and control the robot."""
        if self.state == 0:
            # State 0: Start driving forward at 20% throttle with steering adjustment
            ctx.actions.drive.rc(0.2, 0.0)  # rc(throttle, steering): 0.2 forward, 0.0 straight
            ctx.actions.drive.servo(1, -800, 300)  # Adjust servo position
            self.state = 1
        elif self.state == 1:
            # State 1: Driving - check if 1m reached or timeout
            if ctx.pose.tripB > 1.0 or ctx.pose.tripBtimePassed() > 15:
                ctx.actions.drive.stop()  # Stop driving
                ctx.actions.drive.servo(1, 0, 0)  # Center servo
                self.state = 2
        elif self.state == 2:
            # State 2: Stopped - wait for velocity to settle to near-zero
            if abs(ctx.pose.velocity()) < 0.001:
                print(
                    f"# drive 1m drove {ctx.pose.tripB:.3f}m in {ctx.pose.tripBtimePassed():.3f} seconds"
                )
                self._done = True  # Mark objective as complete
        print(
            f"# drive {self.state}, now {ctx.pose.tripB:.3f}m in {ctx.pose.tripBtimePassed():.3f} seconds; "
            f"left {ctx.actions.edge.get_left_position()}, right {ctx.actions.edge.get_right_position()}"
        )

    def stop(self, ctx):
        """Clean up: turn off LED and stop the robot."""
        ctx.actions.drive.leds(0, 0, 0)  # Turn off LEDs
        ctx.actions.drive.stop()  # Stop all movement
        print("% Driving 1m ------------------------- end")