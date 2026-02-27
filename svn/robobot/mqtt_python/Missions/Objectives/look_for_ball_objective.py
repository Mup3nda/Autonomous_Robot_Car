"""Look For Ball Objective - Rotate in place until ball is detected."""

from mission_context import MissionContext
from objective import Objective


class LookForBallObjective(Objective):
    """Rotate the robot on the spot until a ball is detected in the camera frame.
    
    This objective performs a rotating search pattern:
    1. Continuously rotate the robot on the spot
    2. Monitor the camera feed for ball detection
    3. Stop and complete once ball is visible
    
    State Machine:
    --------------
    State 0 (SEARCHING): Rotate and scan for ball
        - Sends continuous rotation command (yaw)
        - Checks each frame for ball detection
        - If ball detected with sufficient confidence -> DONE
        - If timeout reached -> DONE (failure)
    
    Parameters:
    -----------
    angular_velocity: Float (0.0 to 1.0)
        Rotation speed, where 0.5 = 50% max angular velocity (default 0.3)
        Lower values = slower rotation, better for deliberate search
        Higher values = faster rotation, better for quick sweeps
    
    min_confidence: Int (0-20)
        Minimum confidence required for valid ball detection (default 2)
        Higher values require more confident detection
    
    timeout_seconds: Float
        Maximum time to search before giving up (default 30 seconds)
        Prevents infinite rotation if ball is not in environment
    
    print_interval: Int
        Print status every N ticks (default 20 = ~1 second at 50ms tick rate)
        Set to 0 to disable status printing
    """
    
    def __init__(
        self,
        angular_velocity=0.3,
        min_confidence=2,
        timeout_seconds=30,
        print_interval=20
    ):
        super().__init__()
        self.angular_velocity = angular_velocity
        self.min_confidence = min_confidence
        self.timeout_seconds = timeout_seconds
        self.print_interval = print_interval
        self.max_ticks = int(timeout_seconds * 1000 / 50)  # 50ms per tick
    
    def start(self, ctx: MissionContext):
        """Initialize ball search objective."""
        self.state = 0
        print(f"% Objective: Look For Ball (omega={self.angular_velocity}, timeout={self.timeout_seconds}s)")
        print(f"% Robot will rotate in place to search for ball...")
    
    def tick(self, ctx: MissionContext):
        """Execute one iteration of the ball search state machine."""
        
        # State 0: SEARCHING - Rotate and look for ball
        if self.state == 0:
            # Apply continuous rotation (0 forward, angular_velocity for turn)
            ctx.actions.drive.rc(0, self.angular_velocity)
            
            # Check if ball is visible
            if ctx.actions.ball.is_ball_visible(confidence=self.min_confidence):
                print(f"% Ball found at tick {self.ticks}!")
                status = ctx.actions.ball.get_status()
                print(f"% Ball position: x={status['x']:.0f}, y={status['y']:.0f}, " +
                      f"r={status['radius']:.0f}px, conf={status['confidence']}")
                ctx.actions.drive.stop()
                self.state = self.DONE
                print(f"% Look for Ball objective complete!")
            
            # Periodic status output
            elif self.print_interval > 0 and self.ticks % self.print_interval == 0:
                confidence = ctx.actions.ball.get_ball_confidence()
                time_elapsed = self.ticks * 50 / 1000  # Convert ticks to seconds
                print(f"% Searching ({time_elapsed:.1f}s): ball_confidence={confidence}")
            
            # Check timeout
            if self.ticks >= self.max_ticks:
                print(f"% Timeout: Ball not found after {self.timeout_seconds}s")
                ctx.actions.drive.stop()
                self.state = self.DONE
                print(f"% Look for Ball objective failed (timeout)")
    
    def stop(self, ctx: MissionContext):
        """Clean up when objective is stopped or interrupted."""
        ctx.actions.drive.stop()
        print(f"% Look for Ball objective stopped")
