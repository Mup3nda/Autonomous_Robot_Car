"""Rotate the robot 180 degrees (π radians) in place.

State machine:
- State 0: Start rotation with combined forward throttle and rotation
- State 1: Continue rotating until π radians reached or 15s timeout
- State 2: Stop and wait for rotation to settle
- Done: Log results and mark objective complete
"""
from objective import Objective
from robot_actions import RobotActions
from mission_context import MissionContext
import time as t

class DriveTurnPiObjective(Objective):
    name = "drive_turn_pi"

    def start(self, ctx):
        """Initialize: reset angle tracker and turn on green LED."""
        self.state = 0
        ctx.pose.tripBreset()  # Reset pose counter (includes angle)
        ctx.actions.drive.leds(0, 100, 0)  # Green LED
        print("% Driving a Pi turn -------------------------")

    def tick(self, ctx):
        """Update objective state and control the robot."""
        if self.state == 0:
            # State 0: Start rotation - forward throttle + steering for rotation
            # rc(v, w): v=0.2 forward, w=0.5 rotation
            ctx.actions.drive.rc(0.2, 0.5)
            self.state = 1
        elif self.state == 1:
            # State 1: Rotating - check if π radians reached or timeout
            # tripBh is the angle in radians
            if ctx.pose.tripBh > 3.14 or ctx.pose.tripBtimePassed() > 15:
                ctx.actions.drive.stop()  # Stop rotation
                self.state = 2
        elif self.state == 2:
            # State 2: Stopped - wait for rotation to settle (zero velocity and turnrate)
            if abs(ctx.pose.velocity()) < 0.001 and abs(ctx.pose.turnrate()) < 0.001:
                print(
                    f"# drive turned {ctx.pose.tripBh:.3f} rad in {ctx.pose.tripBtimePassed():.3f} seconds"
                )
                self._done = True  # Mark objective as complete
        print(
            f"# turn {self.state}, now {ctx.pose.tripBh:.3f} rad in {ctx.pose.tripBtimePassed():.3f} seconds; "
            f"left {ctx.actions.edge.get_left_position()}, right {ctx.actions.edge.get_right_position()}"
        )

    def stop(self, ctx):
        """Clean up: turn off LED and stop the robot."""
        ctx.actions.drive.leds(0, 0, 0)  # Turn off LEDs
        ctx.actions.drive.stop()  # Stop all movement
        print("% Driving a Pi turn ------------------------- end")