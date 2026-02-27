"""Drive to Ball Objective - Search for, approach, and center on a ball."""

from mission_context import MissionContext
from objective import Objective


class DriveToBallObjective(Objective):
    """Search for a ball, approach it, and stop at target distance.
    
    This objective uses the camera and ball tracking to:
    1. Search for a ball (rotating in place if needed)
    2. Approach the ball using automatic ball following control
    3. Stop when within target distance and centered
    
    State Machine:
    --------------
    State 0 (SEARCHING): Wait for ball detection
        - Checks if ball is visible with sufficient confidence
        - If ball detected -> State 1 (APPROACHING)
        - Optional: Could add rotation to search if ball not found
    
    State 1 (APPROACHING): Move toward ball using automatic following
        - Enables ball following control (auto-steering + speed control)
        - Monitors distance and centering
        - If at target distance and centered -> State 2 (STOPPING)
    
    State 2 (STOPPING): Stop motors and finish
        - Disables ball following control
        - Stops all motors
        - -> DONE
    
    Parameters:
    -----------
    velocity: Float (0.0 to 1.0)
        Nominal forward speed for approaching ball (default 0.2 = 20% throttle)
        Actual speed adjusts based on distance
    
    target_distance: Float (meters)
        Desired final distance from ball (default 0.5m = 50cm)
    
    distance_tolerance: Float (meters)
        How close to target distance is "close enough" (default 0.1m = 10cm)
    
    centering_tolerance: Int (pixels)
        How close to image center is "centered" (default 50 pixels)
    
    min_confidence: Int (0-20)
        Minimum confidence required for valid detection (default 3)
    """
    
    def __init__(
        self,
        velocity=0.2,
        target_distance=0.5,
        distance_tolerance=0.1,
        centering_tolerance=50,
        min_confidence=3
    ):
        super().__init__()
        self.velocity = velocity
        self.target_distance = target_distance
        self.distance_tolerance = distance_tolerance
        self.centering_tolerance = centering_tolerance
        self.min_confidence = min_confidence
    
    def start(self, ctx: MissionContext):
        """Initialize ball tracking objective."""
        self.state = 0
        print(f"% Objective: Drive to Ball (v={self.velocity}, target={self.target_distance}m)")
    
    def tick(self, ctx: MissionContext):
        """Execute one iteration of the ball tracking state machine."""
        
        # State 0: SEARCHING - Wait for ball detection
        if self.state == 0:
            if ctx.actions.ball.is_ball_visible(confidence=self.min_confidence):
                print(f"% Ball detected! Approaching...")
                ctx.actions.ball.start_following(
                    velocity=self.velocity,
                    target_distance=self.target_distance
                )
                self.state = 1
            else:
                # Optional: Could add rotation search behavior here
                # For now, just wait for ball to appear
                if self.ticks % 20 == 0:  # Print every second (20 * 50ms)
                    print(f"% Searching for ball... (confidence={ctx.actions.ball.get_ball_confidence()})")
        
        # State 1: APPROACHING - Move toward ball
        elif self.state == 1:
            # Check if we've reached target position
            if ctx.actions.ball.is_ball_visible(confidence=self.min_confidence):
                is_centered = ctx.actions.ball.is_centered(tolerance=self.centering_tolerance)
                at_distance = ctx.actions.ball.is_at_target_distance(tolerance=self.distance_tolerance)
                
                # Debug output every second
                if self.ticks % 20 == 0:
                    status = ctx.actions.ball.get_status()
                    print(f"% Approaching: dist={status['distance']:.2f}m centered={is_centered} " +
                          f"conf={status['confidence']}")
                
                # Check if we've reached the goal
                if is_centered and at_distance:
                    print(f"% Target reached! Stopping...")
                    self.state = 2
            else:
                # Lost sight of ball
                print(f"% Ball lost! Stopping and re-searching...")
                ctx.actions.ball.stop_following()
                ctx.actions.drive.stop()
                self.state = 0
        
        # State 2: STOPPING - Clean up and finish
        elif self.state == 2:
            ctx.actions.ball.stop_following()
            ctx.actions.drive.stop()
            self.state = self.DONE
            print(f"% Drive to Ball objective complete!")
    
    def stop(self, ctx: MissionContext):
        """Clean up when objective is stopped or interrupted."""
        ctx.actions.ball.stop_following()
        ctx.actions.drive.stop()
        print(f"% Drive to Ball objective stopped")
