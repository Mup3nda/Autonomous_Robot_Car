from enum import IntEnum
import time as t
import math
from simu import SImu
from objective import Objective
from robot_actions import RobotActions
from mission_context import MissionContext

# Objective tuning constants
SEARCH_SPEED = 0.2
CENTERING_SPEED = 0.2
FOLLOW_SPEED = 0.80
SEARCH_MAX_DISTANCE_M = 1.0
SEARCH_TIMEOUT_S = 15.0
LINE_FOUND_CONFIDENCE = 4
CENTERED_CONFIDENCE = 8
CENTERED_MIN_TIME_S = 2.0
CENTERING_TIMEOUT_S = 4.0
FOLLOW_VALID_CONFIDENCE = 2
LOST_LINE_TIMEOUT_S = 5
RAMP_PITCH_THRESHOLD = 0.2  
STOPPED_VELOCITY_EPS = 0.001
FOLLOW_LEFT = False 

class DriveToLineStateIMU(IntEnum):
    START = 0
    SEARCHING = 1
    STOPPED = 2
    CENTERING = 3
    RAMPING = 4
    LINE_FOLLOWING = 10
    DONE = 99

class DriveUntilEndRamp(Objective):
    name = "drive_until_end_ramp"
    SEARCH_PROGRESS_KEY = "drive_to_line_search"
    ALONG_LINE_PROGRESS_KEY = "drive_to_line_along"

    def __init__(
        self,
        follow_left=FOLLOW_LEFT,
        follow_speed=FOLLOW_SPEED,
        search_speed=SEARCH_SPEED,
        centering_speed=CENTERING_SPEED,
        lost_line_timeout_s=LOST_LINE_TIMEOUT_S,
        instant_stop=True,
        max_duration=0.0,
        
    ):
        super().__init__()
        self.follow_left = bool(follow_left)
        self.follow_speed = float(follow_speed)
        self.search_speed = float(search_speed)
        self.centering_speed = float(centering_speed)
        self.lost_line_timeout_s = float(lost_line_timeout_s)
        self.instant_stop = bool(instant_stop)
        self.max_duration = float(max_duration)
        self.ramp_pitch_threshold = RAMP_PITCH_THRESHOLD
        self.imu = SImu()

    def start(self, ctx):
        """Initialize local progress trackers."""
        self.state = DriveToLineStateIMU.START
        self.dist_to_line = 0.0  
        self.along_line_started = False
        self.centering_start_time = 0.0  
        self.centering_deadline = 0.0  
        self.start_time = t.time()
        self.on_ramp = False
        
        # Eliminar la creación de SImu(). Usaremos ctx.pose
        self.baseline_pitch = None
        
        # Buffer for 10 samples of pitch_diff for mean-based detection
        self.pitch_diff_buffer = []

        ctx.start_local_progress(self.SEARCH_PROGRESS_KEY)
        ctx.actions.drive.leds(0, 100, 0)  
        print("% Driving to line ---------------------- right ir start ---")

    def _line_lost(self, ctx):
        last_seen_time = ctx.actions.edge.last_seen_time_passed()
        line_lost = not ctx.actions.edge.is_line_valid(confidence=FOLLOW_VALID_CONFIDENCE) \
                    and last_seen_time > self.lost_line_timeout_s
        return line_lost

    def tick(self, ctx):
        """Update objective state and control the robot."""
        
        # 1. Read the raw pitch from the pose context
        try:
            raw_pitch = math.atan2(self.imu.acc[0], self.imu.acc[2])
        except (IndexError, TypeError):
            raw_pitch = 0.0 # Fallback por si la IMU tarda unos milisegundos en arrancar
        
        # 2. Set the baseline "zero" pitch on the very first tick while on flat ground
        if self.baseline_pitch is None and self.state == DriveToLineStateIMU.START:
            self.baseline_pitch = raw_pitch
            
        # 3. Calculate the absolute difference between current pitch and baseline
        pitch_diff = 0.0
        if self.baseline_pitch is not None:
            pitch_diff = abs(raw_pitch - self.baseline_pitch)
        
        # 4. Maintain a 10-sample buffer and calculate mean
        self.pitch_diff_buffer.append(pitch_diff)
        if len(self.pitch_diff_buffer) > 30:
            self.pitch_diff_buffer.pop(0)  # Keep only the last 10 samples
        
        # Calculate mean of pitch_diff from the buffer
        pitch_diff_mean = sum(self.pitch_diff_buffer) / len(self.pitch_diff_buffer) if self.pitch_diff_buffer else 0.0

        print("Pitch_diff: ", pitch_diff, " | Mean(10): ", pitch_diff_mean)
        if self.state == DriveToLineStateIMU.START:
            ctx.actions.drive.rc(self.search_speed, 0.0)  
            self.state = DriveToLineStateIMU.SEARCHING
            
        elif self.state == DriveToLineStateIMU.SEARCHING:
            search_marker = ctx.memory["_local_progress"][self.SEARCH_PROGRESS_KEY]
            search_dist = ctx.distance_since_start(self.SEARCH_PROGRESS_KEY)
            search_elapsed = t.time() - search_marker["time_s"]
            
            if search_dist > SEARCH_MAX_DISTANCE_M or search_elapsed > SEARCH_TIMEOUT_S:
                ctx.actions.drive.stop(instant=self.instant_stop)
                self.state = DriveToLineStateIMU.STOPPED
                
            if ctx.actions.edge.is_line_valid(confidence=LINE_FOUND_CONFIDENCE):
                ctx.actions.edge.start_following(velocity=self.centering_speed, follow_left=self.follow_left)
                self.dist_to_line = search_dist  
                ctx.start_local_progress(self.ALONG_LINE_PROGRESS_KEY)
                self.along_line_started = True
                self.centering_start_time = t.time()
                self.centering_deadline = self.centering_start_time + CENTERING_TIMEOUT_S
                self.state = DriveToLineStateIMU.CENTERING
                
        elif self.state == DriveToLineStateIMU.CENTERING:
            now = t.time()
            centered = ctx.actions.edge.is_line_valid(confidence=CENTERED_CONFIDENCE)
            centered_long_enough = now - self.centering_start_time > CENTERED_MIN_TIME_S
            timed_out = now >= self.centering_deadline
            
            if (centered and centered_long_enough) or timed_out:
                ctx.actions.edge.start_following(velocity=self.follow_speed, follow_left=self.follow_left)
                self.state = DriveToLineStateIMU.LINE_FOLLOWING
                
        elif self.state == DriveToLineStateIMU.STOPPED:
            if abs(ctx.pose.velocity()) < STOPPED_VELOCITY_EPS:
                self.state = DriveToLineStateIMU.DONE  
                
        elif self.state == DriveToLineStateIMU.LINE_FOLLOWING:
            if self.max_duration > 0 and (t.time() - self.start_time) > self.max_duration:
                ctx.actions.edge.stop_following()
                ctx.actions.drive.stop(instant=self.instant_stop)
                self.state = DriveToLineStateIMU.STOPPED
                return

            # --- SMART RAMP DETECTION LOGIC (using mean of 10 samples) ---
            
            # UP THRESHOLD: If pitch_diff mean changes by more than 0.12 rad (~7 degrees), we are going up
            if pitch_diff_mean > 0.35:
                if not self.on_ramp:
                    self.on_ramp = True
                    print(f"% [RAMP] Going UP! Mean pitch_diff (10 samples): {pitch_diff_mean:.3f} rad"* 10)

            # TOP THRESHOLD: If we were on the ramp, and pitch_diff mean drops below 0.05 rad (~3 degrees), we reached the top
            if self.on_ramp and pitch_diff_mean < 0.05:
                print(f"% [RAMP] Reached the top! Mean pitch_diff (10 samples): {pitch_diff_mean:.3f} rad"* 10)
                ctx.actions.edge.stop_following()
                ctx.actions.drive.stop(instant=self.instant_stop)
                self.state = DriveToLineStateIMU.STOPPED
                return

            if self._line_lost(ctx):
                # ALWAYS turn off PID before taking other actions to prevent motor conflicts
                ctx.actions.edge.stop_following() 
                
                if self.on_ramp:
                    # Survival mode: drive straight blindly to overcome the bump
                    ctx.actions.drive.rc(self.follow_speed, 0.0)
                    print("% Line lost on ramp! Driving straight blind.")
                else:
                    ctx.actions.drive.stop(instant=self.instant_stop)
                    self.state = DriveToLineStateIMU.STOPPED
        else:
            ctx.actions.drive.stop(instant=self.instant_stop)
            self._done = True

    def stop(self, ctx):
        ctx.actions.drive.leds(0, 0, 0)  
        ctx.actions.drive.stop(instant=self.instant_stop)