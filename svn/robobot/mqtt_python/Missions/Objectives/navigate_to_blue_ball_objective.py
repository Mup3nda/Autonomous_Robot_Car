"""Navigate To Ball Objective - Move towards detected ball target."""

from enum import IntEnum

from mission_context import MissionContext
from objective import Objective
from sball_saray import SBall
from Nav import Nav


class NavigateToBallState(IntEnum):
    MOVING = 0
    COMPLETE = 1
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
        self.tick_count = 0

    def start(self, ctx: MissionContext):
        """Initialize navigation to blue ball using NavigationAction."""
        self.state = NavigateToBallState.MOVING
        self.tick_count = 0
        
        # Create detector for blue balls
        detector = SBall(cam=ctx.cam, gpio=ctx.gpio, service=ctx.service)
        detector.set_detection_color("blue")
        
        # Setup navigation action with this detector
        ctx.actions.navigation.setup_detector(detector)
        ctx.actions.navigation.setup(desired_distance=self.desired_distance, ctx=ctx)
        ctx.actions.navigation.start()
        
        print(f"% Objective: Navigate To Ball (target_distance={self.desired_distance}m)")

    def tick(self, ctx: MissionContext):
        """Execute one iteration of navigation."""
        self.tick_count += 1
        
        # Check if navigation objective is complete
        if ctx.actions.navigation.is_complete():
            self.state = NavigateToBallState.COMPLETE
            self._done = True
            print(f"% Navigate To Ball objective complete!")
        elif self.tick_count % self.print_interval == 0:
            # Print status periodically
            target_info = ctx.actions.navigation.get_target_info()
            if target_info:
                print(f"% Navigating: dist={target_info.get('distance', 0):.2f}m, "
                      f"conf={target_info.get('confidence', 0)}")

    def stop(self, ctx: MissionContext):
        """Clean up when objective is stopped or interrupted."""
        ctx.actions.navigation.stop()
        print(f"% Navigate To Ball objective stopped")
