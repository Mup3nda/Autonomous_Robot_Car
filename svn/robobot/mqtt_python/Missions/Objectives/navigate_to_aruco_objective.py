"""Navigate To ArUco Objective - Move towards a detected ArUco marker."""

from enum import IntEnum
import time

from mission_context import MissionContext
from objective import Objective
#from Autonomous_Robot_Car.svn.robobot.mqtt_python.aruco_detector import ArucoDetector
from aruco_detector2 import ArucoDetector


class NavigateToArucoState(IntEnum):
    MOVING_PRIMARY = 0
    MOVING_FALLBACK = 1
    COMPLETE = 2
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
    
    def __init__(self, marker_id=53, desired_distance=0.41, print_interval=20, nav_mode="aruco", fallback_marker_id=None, search_timeout_s=None): #NavMode "aruco" for aruco controller. Sequential is Nav.py Smooth is NavSmooth.py
        super().__init__()
        self.desired_distance = desired_distance
        self.print_interval = print_interval
        self.nav_mode = str(nav_mode).lower()
        self.tick_count = 0
        self.marker_id = marker_id
        self.fallback_marker_id = fallback_marker_id
        self.search_timeout_s = search_timeout_s
        
        self.detector = None
        self.state = NavigateToArucoState.MOVING_PRIMARY
        self.search_start_time = None
        self.current_target_id = self.marker_id
        self.fallback_used = False  # Track if fallback was used
        self.has_target = False  # Track if we currently have target

    def start(self, ctx: MissionContext):
        """Initialize navigation to ArUco marker using NavigationAction."""
        self.state = NavigateToArucoState.MOVING_PRIMARY
        self.tick_count = 0
        self.search_start_time = None  # Don't start counting until we lose the target
        self.current_target_id = self.marker_id
        self.has_target = False  # Initially assume no target
        
        # Create detector for the target ArUco marker
        self.detector = ArucoDetector(cam=ctx.cam, gpio=ctx.gpio, service=ctx.service, target_id=self.current_target_id)
        
        # Setup navigation action with this detector
        ctx.actions.navigation.setup_detector(self.detector)
        ctx.actions.navigation.setup(desired_distance=self.desired_distance, 
                                     ctx=ctx, 
                                     nav_mode=self.nav_mode)
        ctx.actions.navigation.start()
        
        print(f"% Objective: Navigate To ArUco Marker {self.current_target_id} (target_distance={self.desired_distance}m, nav_mode={self.nav_mode}, "
              f"{f'then {self.fallback_marker_id}' if self.fallback_marker_id is not None else 'no fallback'})")
    
    def tick(self, ctx: MissionContext):
        """Execute one iteration of navigation."""
        if self._done:
            return
            
        self.tick_count += 1
        
        # Check if we currently have the target visible
        target_info = ctx.actions.navigation.get_target_info()
        target_visible = target_info is not None
        
        # Handle timeout only when we DON'T have the target in MOVING_PRIMARY state
        if (
            self.state == NavigateToArucoState.MOVING_PRIMARY
            and self.fallback_marker_id is not None
            and self.search_timeout_s is not None
        ):
            # If we just lost the target, start the timeout clock
            if self.has_target and not target_visible:
                self.search_start_time = time.time()
                self.has_target = False
            # If we regain the target, stop the clock
            elif not self.has_target and target_visible:
                self.search_start_time = None
                self.has_target = True
            # If we still don't have the target, check if timeout elapsed
            elif not target_visible and self.search_start_time is not None:
                elapsed = time.time() - self.search_start_time
                if elapsed >= self.search_timeout_s:
                    print(
                        f"% Navigate To ArUco {self.marker_id}: timeout after {elapsed:.1f}s without target, "
                        f"switching to fallback {self.fallback_marker_id}"
                    )
                    
                    self.current_target_id = self.fallback_marker_id
                    self.fallback_used = True  # Mark fallback as used
                    ctx.memory["fallback_flag"] = 1  # Set flag in mission context
                    self.state = NavigateToArucoState.MOVING_FALLBACK
                    self.search_start_time = None
                    self.has_target = False
                    
                    if self.detector and hasattr(self.detector, "set_target_id"):
                        self.detector.set_target_id(self.current_target_id)
        
        # Check if navigation objective is complete
        if ctx.actions.navigation.is_complete():
            self.state = NavigateToArucoState.COMPLETE
            self._done = True
            print(f"% Navigate To ArUco objective complete!")
        elif self.tick_count % self.print_interval == 0:
            # Print status periodically
            if target_info:
                print(
                    f"% Navigating to marker {target_info.get('id')}: "
                    f"dist={target_info.get('distance', 0):.2f}m"
                )

    def stop(self, ctx: MissionContext):
        """Clean up when objective is stopped or interrupted."""
        ctx.actions.navigation.stop()
        if self.detector and hasattr(self.detector, "stop"):
            self.detector.stop()
        self.state = NavigateToArucoState.DONE
        print(f"% Navigate To ArUco objective stopped")
