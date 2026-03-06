"""Navigate To Ball Objective - Move towards detected ball target."""

from enum import IntEnum
import sys
import os

# Add parent directories to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from mission_context import MissionContext
from objective import Objective
from target_detector import TargetDetector
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
    
    def __init__(self, desired_distance=0.5, print_interval=20):
        super().__init__()
        self.desired_distance = desired_distance
        self.print_interval = print_interval
        
        # Navigation components
        self.detector = None
        self.nav = None
    
    def start(self, ctx: MissionContext):
        """Initialize navigation to ball."""
        self.state = NavigateToBallState.MOVING
        
        # Setup detector to find ball targets
        self.detector = BallDetector("Blue")
        
        # Setup navigation controller
        self.nav = Nav()
        self.nav.setup(self.detector, self.desired_distance)
        
        # Start moving towards target
        self.nav.start()
        
        print(f"% Objective: Navigate To Ball (target_distance={self.desired_distance}m)")
        print(f"% Started moving towards ball target...")
    
    def tick(self, ctx: MissionContext):
        """Execute one iteration of the navigation state machine."""
        
        # State 0: MOVING - Navigate towards ball
        if self.state == NavigateToBallState.MOVING:
            
            # Get current target from navigation
            target = self.nav.target
            
            if target is None:
                if self.print_interval > 0 and self.ticks % self.print_interval == 0:
                    print(f"% Searching for ball target...")
            else:
                
                if self.nav.hasReachedTarget():
                    print(f"% Reached target at tick {self.ticks}!")
                    self.nav.stop()
                    self.state = NavigateToBallState.REACHED
                    self._done = True
                    print(f"% Navigate To Ball objective complete!")
    
    def stop(self, ctx: MissionContext):
        """Clean up when objective is stopped or interrupted."""
        if self.nav:
            self.nav.stop()
        print(f"% Navigate To Ball objective stopped")
