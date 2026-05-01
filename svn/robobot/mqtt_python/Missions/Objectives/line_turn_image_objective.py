"""Complex mission combining multiple actions: drive forward, turn 90°, and capture images.

State machine:
- State 0: Initialize mission
- State 12: Drive forward ~0.5m
- State 14: Rotate 90° (π/2 radians)
- State 20: Capture 10 images while toggling LEDs
- Done: Mission complete
 - Done: Mission complete
"""
from enum import IntEnum
from objective import Objective
from robot_actions import RobotActions
from mission_context import MissionContext
from datetime import datetime
import time as t
import numpy as np


class LineTurnImageState(IntEnum):
    INIT = 0
    DRIVE_FORWARD = 12
    TURN_90_DEG = 14
    CAPTURE_IMAGES = 20

class LineTurnImageObjective(Objective):
    name = "line_turn_image"
    DRIVE_PROGRESS_KEY = "line_turn_image_drive"
    TURN_PROGRESS_KEY = "line_turn_image_turn"

    def start(self, ctx):
        """Initialize mission: disable initial line control, set yellow LED."""
        self.state = LineTurnImageState.INIT
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
        if self.state == LineTurnImageState.INIT:
            # State 0: Initialize mission and transition to driving
            start = True
            if start:
                ctx.actions.drive.leds(0, 0, 30)  # Blue LED
                ctx.actions.drive.rc(0.25, 0.0)  # 25% throttle forward, straight
                ctx.actions.drive.servo(1, 100, 300)  # Servo position
                ctx.start_local_progress(self.DRIVE_PROGRESS_KEY)
                self._set_state(LineTurnImageState.DRIVE_FORWARD)  # Move to driving state
        elif self.state == LineTurnImageState.DRIVE_FORWARD:
            # State 12: Drive forward ~0.5m
            drive_marker = ctx.memory["_local_progress"][self.DRIVE_PROGRESS_KEY]
            driven = ctx.distance_since_start(self.DRIVE_PROGRESS_KEY)
            drive_elapsed = t.time() - drive_marker["time_s"]
            if driven > 0.5 or drive_elapsed > 10:
                # Driven enough or timeout - transition to turning
                ctx.actions.edge.stop_following()  # Disable line control
                ctx.start_local_progress(self.TURN_PROGRESS_KEY)
                ctx.actions.drive.rc(0.1, 0.5)  # Slow forward + rotation
                ctx.actions.drive.servo(1, -800, 1000)  # Servo for rotation
                self._set_state(LineTurnImageState.TURN_90_DEG)  # Move to turning state
        elif self.state == LineTurnImageState.TURN_90_DEG:
            # State 14: Rotate 90 degrees (π/2 radians)
            turn_marker = ctx.memory["_local_progress"][self.TURN_PROGRESS_KEY]
            turned = abs(ctx.pose.tripAh - turn_marker["tripAh"])
            turn_elapsed = t.time() - turn_marker["time_s"]
            if turned > np.pi / 2 or turn_elapsed > 10:
                # Rotated enough or timeout - transition to image capture
                ctx.actions.drive.stop()  # Stop all movement
                ctx.actions.drive.servo(1, 0, 1000)  # Center servo
                self._set_state(LineTurnImageState.CAPTURE_IMAGES)  # Move to image capture state
        elif self.state == LineTurnImageState.CAPTURE_IMAGES:
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