"""Drive forward until the edge sensor detects a line, then automatically follow it.

State machine:
- State 0: Wait for IR proximity detection (skipped by default)
- State 1: Drive forward until line detected, then switch to line following
- State 2: Stop if no line was found or timeout occurred
- State 10: Follow the line (active line control)
- State 99: Line following complete
"""
from objective import Objective
from robot_actions import RobotActions
from mission_context import MissionContext

class DriveToLineObjective(Objective):
    name = "drive_to_line"

    def start(self, ctx):
        """Initialize: reset distance tracker, set green LED."""
        self.state = 0
        self.dist_to_line = 0.0  # Track distance traveled before finding line
        ctx.pose.tripBreset()  # Reset distance counter
        ctx.actions.drive.leds(0, 100, 0)  # Green LED
        print("% Driving to line ---------------------- right ir start ---")

    def tick(self, ctx):
        """Update objective state and control the robot."""
        if self.state == 0:
            # State 0: Start driving forward (IR check was removed)
            ctx.actions.drive.rc(0.2, 0.0)  # 20% throttle, straight
            ctx.actions.drive.lognow(3)  # Log sensor data
            ctx.actions.drive.servo(1, -800, 300)  # Adjust servo
            self.state = 1
        elif self.state == 1:
            # State 1: Searching for line while driving forward
            if ctx.pose.tripB > 1.0 or ctx.pose.tripBtimePassed() > 15:
                # Stop if traveled >1m or >15s timeout without finding line
                ctx.actions.drive.stop()
                self.state = 2
            if ctx.actions.edge.is_line_valid(confidence=4):
                # Line detected! Switch to line following mode
                ctx.actions.edge.start_following(velocity=0.2, follow_left=False)
                ctx.actions.drive.servo(1, 0, 0)  # Center servo
                self.dist_to_line = ctx.pose.tripB  # Record distance to line
                ctx.pose.tripBreset()  # Reset counter for line following distance
                self.state = 10  # Enter line following state
        elif self.state == 2:
            # State 2: Stopped after timeout - wait for robot to settle
            if abs(ctx.pose.velocity()) < 0.001:
                self.state = 99  # Mark as done
        elif self.state == 10:
            # State 10: Following line - check if line is still valid
            if not ctx.actions.edge.is_line_valid(confidence=2) and ctx.actions.edge.last_seen_time_passed() > 5.0:
                # Lost the line - stop and try to recover
                ctx.actions.edge.stop_following()
                ctx.actions.drive.stop()
                ctx.pose.tripBreset()
                self.state = 2  # Go to stopped state
        else:
            # Final state - log results and mark complete
            print(
                f"# drive to line {self.dist_to_line:.3f}m, then along line "
                f"{ctx.pose.tripB:.3f}m in {ctx.pose.tripBtimePassed():.3f} seconds"
            )
            ctx.actions.drive.stop()
            ctx.actions.drive.servo(1, 500, 200)  # Position servo
            self._done = True  # Mark objective as complete

    def stop(self, ctx):
        """Clean up: turn off LED and stop the robot."""
        ctx.actions.drive.leds(0, 0, 0)  # Turn off LEDs
        ctx.actions.drive.stop()  # Stop all movement
        print("% Driving to line ------------------------- end")
