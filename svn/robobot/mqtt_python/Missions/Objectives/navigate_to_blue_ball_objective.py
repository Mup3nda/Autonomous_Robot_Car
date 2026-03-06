"""Navigate To Ball Objective - Move towards detected ball target."""

from enum import IntEnum

from mission_context import MissionContext
from objective import Objective
from sball_saray import SBall
from Nav import Nav


class NavigateToBallState(IntEnum):
    MOVING = 0
    REACHED = 1
    DONE = 99


class NavigateToBallObjective(Objective):
    """Move the robot towards a detected ball target.
    
    This objective demonstrates how to use the Nav class with a TargetDetector:
    1. Setup a detector to find ball targets
    2. Setup and start navigation towards target
    3. Check if target distance is reached
    4. Stop and complete
    
    Parameters:
    -----------
    desired_distance: Float
        Target distance to maintain from ball (default 0.5 = 50cm)
    
    print_interval: Int
        Print status every N ticks (default 20 = ~1 second at 50ms tick rate)
    """
    
    def __init__(self, desired_distance=0.41, print_interval=20):
        super().__init__()
        self.desired_distance = desired_distance
        self.print_interval = print_interval
        # Navigation components
        self.detector = None
        self.nav = None
    
    def start(self, ctx: MissionContext):
        """Initialize navigation to ball."""
        self.state = NavigateToBallState.MOVING
        # Setup detector to find ball targets (use the camera from context)
        self.detector = SBall(cam=ctx.cam, gpio=ctx.gpio, service=ctx.service)
        self.detector.set_detection_color("blue")
        
        # Start the detector (this begins the threaded camera processing)
        self.detector.start()
        
        # Setup navigation controller
        self.nav = Nav()
        self.nav.setup(self.detector, self.desired_distance, ctx)
        
        # Start moving towards target
        self.nav.start()
        
        print(f"% Objective: Navigate To Ball (target_distance={self.desired_distance}m)")
        print(f"% Started moving towards ball target...")
    
    def tick(self, ctx: MissionContext):
        """Execute one iteration of the navigation state machine."""
        
        # State 0: MOVING - Navigate towards ball
        if self.state == NavigateToBallState.MOVING:
            
            # Check if navigation has reached the target
            if self.nav.hasReachedTarget:
                #print(f"% Reached target at tick {self.tick}!")
                self.nav.stop()
                self.state = NavigateToBallState.REACHED
                self._done = True
                print(f"% Navigate To Ball objective complete!")
            else:
                # Show navigation status
                target_info = self.detector.get_target()
                if target_info:
                    distance = self.detector.get_target_distance()
                    print(f"% Navigating: ball at ({target_info['x']}, {target_info['y']}), "
                          f"dist={distance:.2f}m, conf={target_info['confidence']}")
                
    
    def stop(self, ctx: MissionContext):
        """Clean up when objective is stopped or interrupted."""
        if self.nav:
            self.nav.stop()
        if self.detector:
            self.detector.stop()
        print(f"% Navigate To Ball objective stopped")
