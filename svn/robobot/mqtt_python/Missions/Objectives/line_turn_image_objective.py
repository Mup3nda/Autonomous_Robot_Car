"""Complex mission combining multiple actions: drive forward, turn 90°, and capture images.

State machine:
- State 0: Initialize mission
- State 12: Drive forward ~0.5m
- State 14: Rotate 90° (π/2 radians)
- State 20: Capture 10 images while toggling LEDs
- Done: Mission complete
"""
from objective import Objective
from robot_actions import RobotActions
from mission_context import MissionContext
from datetime import datetime
import numpy as np

class LineTurnImageObjective(Objective):
    name = "line_turn_image"

    def start(self, ctx):
        """Initialize mission: disable initial line control, set yellow LED."""
        self.state = 0
        self.images = 0  # Counter for images captured
        self.ledon = True  # LED blink state toggle
        self.state_time = datetime.now()  # Track time for state transitions
        ctx.actions.edge.stop_following()  # Disable line control initially
        ctx.actions.drive.leds(30, 30, 0)  # Yellow LED (R+G)
        print("% Starting line/turn/image objective")

    def _state_time_passed(self):
        """Helper: calculate elapsed time in current state."""
        return (datetime.now() - self.state_time).total_seconds()

    def _set_state(self, state):
        """Helper: set new state and reset the state timer."""
        self.state = state
        self.state_time = datetime.now()

    def tick(self, ctx):
        """Update objective state and control the robot."""
        if self.state == 0:
            # State 0: Initialize mission and transition to driving
            start = True
            if start:
                ctx.actions.drive.leds(0, 0, 30)  # Blue LED
                ctx.actions.drive.rc(0.25, 0.0)  # 25% throttle forward, straight
                ctx.actions.drive.servo(1, 100, 300)  # Servo position
                ctx.pose.tripBreset()  # Reset distance counter
                self._set_state(12)  # Move to driving state
        elif self.state == 12:
            # State 12: Drive forward ~0.5m
            if ctx.pose.tripB > 0.5 or ctx.pose.tripBtimePassed() > 10:
                # Driven enough or timeout - transition to turning
                ctx.actions.edge.stop_following()  # Disable line control
                ctx.pose.tripBreset()  # Reset for angle tracking
                ctx.actions.drive.rc(0.1, 0.5)  # Slow forward + rotation
                ctx.actions.drive.servo(1, -800, 1000)  # Servo for rotation
                self._set_state(14)  # Move to turning state
        elif self.state == 14:
            # State 14: Rotate 90 degrees (π/2 radians)
            if ctx.pose.tripBh > np.pi / 2 or ctx.pose.tripBtimePassed() > 10:
                # Rotated enough or timeout - transition to image capture
                ctx.actions.drive.stop()  # Stop all movement
                ctx.actions.drive.servo(1, 0, 1000)  # Center servo
                self._set_state(20)  # Move to image capture state
        elif self.state == 20:
            # State 20: Capture images with blinking LED
            ctx.actions.vision.image_analysis(self.images == 2)  # Save image 3 (index 2)
            self.images += 1
            if self.ledon:
                # LED on - green
                ctx.actions.drive.leds(0, 64, 0)
                ctx.actions.drive.set_gpio(20, 1)  # GPIO pin on
            else:
                # LED off - cyan (B+G)
                ctx.actions.drive.leds(0, 30, 30)
                ctx.actions.drive.set_gpio(20, 0)  # GPIO pin off
            self.ledon = not self.ledon  # Toggle LED state
            # Complete when 10 images captured or timeout or no camera
            if self.images >= 10 or (not ctx.cam.useCam) or self._state_time_passed() > 20:
                self._done = True  # Mark objective as complete

    def stop(self, ctx):
        """Clean up: turn off LED, GPIO, and stop the robot."""
        ctx.actions.drive.leds(0, 0, 0)  # Turn off LEDs
        ctx.actions.drive.set_gpio(20, 0)  # Turn off GPIO pin
        ctx.actions.edge.stop_following()  # Disable line control
        ctx.actions.drive.stop()  # Stop all movement
        ctx.actions.drive.servo(1, 0, 0)  # Center servo
        print("% Line/turn/image objective end")