"""Navigate To ArUco Objective - Move towards a detected ArUco marker."""

from enum import IntEnum

from mission_context import MissionContext
from objective import Objective
#from Autonomous_Robot_Car.svn.robobot.mqtt_python.aruco_detector import ArucoDetector
from aruco_detector import ArucoDetector


class NavigateToArucoState(IntEnum):
    MOVING = 0
    COMPLETE = 1
    DONE = 99


class NavigateToArucoObjective(Objective):
    """Move the robot towards a detected ArUco marker.
    
    This objective demonstrates how to use the Nav class with a TargetDetector:
    1. Setup an ArUco detector for the target marker
    2. Setup and start navigation towards the target
    3. Check if target distance is reached
    4. Stop and complete
    
    Parameters:
    -----------
    marker_id: Int
        ArUco marker ID to navigate towards (default 53)

    desired_distance: Float
        Target distance to maintain from marker (default 0.41 = 41cm)
    
    print_interval: Int
        Print status every N ticks (default 20 = ~1 second at 50ms tick rate)

    nav_mode: str
        "sequential" (rotate-then-drive) or "smooth" (simultaneous drive+turn)
    """
    
    def __init__(self, marker_id=53, desired_distance=0.41, print_interval=20, nav_mode="aruco", COMPENSATE_PARAMETER= 20): #NavMode "aruco" for aruco controller. Sequential is Nav.py Smooth is NavSmooth.py
        super().__init__()
        self.desired_distance = desired_distance
        self.print_interval = print_interval
        self.nav_mode = str(nav_mode).lower()
        self.tick_count = 0
        self.marker_id = marker_id
        self.COMPENSATE_PARAMETER = COMPENSATE_PARAMETER

    def start(self, ctx: MissionContext):
        """Initialize navigation to ArUco marker using NavigationAction."""
        self.state = NavigateToArucoState.MOVING
        self.tick_count = 0
        
        # Create detector for the target ArUco marker
        detector = ArucoDetector(cam=ctx.cam, gpio=ctx.gpio, service=ctx.service, target_id=self.marker_id)
        
        # Setup navigation action with this detector
        ctx.actions.navigation.setup_detector(detector)
        ctx.actions.navigation.setup(desired_distance=self.desired_distance, 
                                     ctx=ctx, 
                                     nav_mode=self.nav_mode, COMPENSATE_PARAMETER = self.COMPENSATE_PARAMETER)
        ctx.actions.navigation.start()
        
        print(f"% Objective: Navigate To ArUco Marker {self.marker_id} (target_distance={self.desired_distance}m, nav_mode={self.nav_mode})")
    
    def tick(self, ctx: MissionContext):
        """Execute one iteration of navigation."""
        self.tick_count += 1
        
        # Check if navigation objective is complete
        if ctx.actions.navigation.is_complete():
            self.state = NavigateToArucoState.COMPLETE
            self._done = True
            print(f"% Navigate To ArUco objective complete!")
        elif self.tick_count % self.print_interval == 0:
            # Print status periodically
            target_info = ctx.actions.navigation.get_target_info()
            if target_info:
                print(
                    f"% Navigating to marker {target_info.get('id')}: "
                    f"dist={target_info.get('distance', 0):.2f}m"
                )

    def stop(self, ctx: MissionContext):
        """Clean up when objective is stopped or interrupted."""
        ctx.actions.navigation.stop()
        print(f"% Navigate To ArUco objective stopped")
