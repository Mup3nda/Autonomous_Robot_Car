"""Drive along a line, detect a slope using the IMU, and stop when the surface becomes flat.

State machine:
- State 0: Start driving/following the line
- State 1: Follow line on flat ground, wait for slope (pitch > SLOPE_THRESHOLD)
- State 2: Climbing slope (pitch remains > FLAT_THRESHOLD)
- State 3: Reached flat surface (pitch < FLAT_THRESHOLD) -> Stop
"""
from enum import IntEnum
import time as t
import math
from objective import Objective
from simu import SImu

# Tuning constants
FOLLOW_SPEED = 0.6
SLOPE_THRESHOLD_RAD = 0.15  # Approx 8.5 degrees to trigger slope detection
FLAT_THRESHOLD_RAD = 0.05   # Approx 2.8 degrees to consider surface flat again
LOST_LINE_TIMEOUT_S = 3.0
FOLLOW_LEFT = False

class DriveUpSlopeState(IntEnum):
    START = 0
    WAITING_FOR_SLOPE = 1
    CLIMBING = 2
    DONE = 99

class DriveUpSlopeObjective(Objective):
    name = "drive_up_slope"

    def __init__(
        self,
        follow_speed=FOLLOW_SPEED,
        follow_left=FOLLOW_LEFT,
        slope_threshold=SLOPE_THRESHOLD_RAD,
        flat_threshold=FLAT_THRESHOLD_RAD
    ):
        super().__init__()
        self.follow_speed = float(follow_speed)
        self.follow_left = bool(follow_left)
        self.slope_threshold = float(slope_threshold)
        self.flat_threshold = float(flat_threshold)
        self.imu = SImu()  # IMU data object

    def start(self, ctx):
        """Start the objective and begin tracking IMU."""
        self.state = DriveUpSlopeState.START
        self.start_time = t.time()
        ctx.actions.drive.leds(100, 100, 0)  # Yellow LED for slope objective
        print("% [Slope] Started objective - Starting line following")
        
        # Immediately start following the line
        ctx.actions.edge.start_following(velocity=self.follow_speed, follow_left=self.follow_left)
        self.state = DriveUpSlopeState.WAITING_FOR_SLOPE

    def tick(self, ctx):
        """Update state machine based on IMU pitch."""
        
        # Calculate pitch from IMU accelerometer data
        # atan2(acc_x, acc_z) gives the pitch angle in radians
        # Note: If your accelerometer axes are different, you may need to swap these.
        pitch = math.atan2(self.imu.acc[0], self.imu.acc[2])

        if self.state == DriveUpSlopeState.WAITING_FOR_SLOPE:
            # Check if we have hit the slope
            if pitch > self.slope_threshold:
                print(f"% [Slope] Slope detected! Pitch: {pitch:.3f} rad. Switching to CLIMBING state.")
                self.state = DriveUpSlopeState.CLIMBING
                ctx.actions.drive.leds(100, 0, 0)  # Red LED while climbing

        elif self.state == DriveUpSlopeState.CLIMBING:
            # Check if we have leveled out
            if pitch < self.flat_threshold:
                print(f"% [Slope] Surface flattened. Pitch: {pitch:.3f} rad. Stopping.")
                ctx.actions.edge.stop_following()
                ctx.actions.drive.stop(instant=True)
                self.state = DriveUpSlopeState.DONE
                self._done = True

        # Safety check: if line is lost for too long while looking or climbing, stop.
        last_seen = ctx.actions.edge.last_seen_time_passed()
        if not ctx.actions.edge.is_line_valid(confidence=2) and last_seen > LOST_LINE_TIMEOUT_S:
            print(f"% [Slope] Line lost for {last_seen:.2f}s! Aborting.")
            ctx.actions.edge.stop_following()
            ctx.actions.drive.stop(instant=True)
            self.state = DriveUpSlopeState.DONE
            self._done = True

    def stop(self, ctx):
        """Clean up."""
        ctx.actions.drive.leds(0, 0, 0)
        ctx.actions.drive.stop(instant=True)
        ctx.actions.edge.stop_following()
        print("% [Slope] Objective finished and stopped.")
