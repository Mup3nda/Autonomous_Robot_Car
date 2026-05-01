"""Navigate To Blue Ball Objective - Move towards detected ball target."""

from Objectives.navigate_to_ball_objective import NavigateToBallObjective


class NavigateToBlueBallObjective(NavigateToBallObjective):
    """Move the robot towards a detected blue ball target.
    
    This objective demonstrates how to use the Nav class with a TargetDetector:
    1. Setup a detector to find ball targets
    2. Setup and start navigation towards target
    3. Check if target distance is reached
    4. Stop and complete
    
    Parameters:
    -----------
    desired_distance: Float
        Target distance to maintain from ball (default 0.41 = 41cm)
    
    print_interval: Int
        Print status every N ticks (default 20 = ~1 second at 50ms tick rate)

    nav_mode: str
        "sequential" (rotate-then-drive) or "smooth" (simultaneous drive+turn)
    """
    
    def __init__(self, desired_distance=0.41, print_interval=20, nav_mode="sequential"): #NavMode "sequential" or "smooth". Sequential is Nav.py Smooth is NavSmooth.py
        super().__init__(desired_distance=desired_distance, print_interval=print_interval, nav_mode=nav_mode, color="blue")
  

    
