"""Navigation actions: high-level interface for detector-based navigation controller."""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from Nav import Nav #import nav2  for the new version
from NavSmooth import NavSmooth


class NavigationAction:
    """High-level navigation interface that works with any TargetDetector.
    
    Wraps the Nav controller to provide a clean objective-level interface
    for detector-based navigation (balls, waypoints, aruco codes, etc.).
    
    The detector is injected at setup time, allowing NavAction to work
    with any TargetDetector implementation (SBall, SWorldPoints, etc.).
    
    Args:
        None at init time; detector is injected via setup_detector()
    """
    
    def __init__(self):
        self.detector = None
        self.nav = None
        self.desired_distance = 0.0
        self.started = False
    
    def setup_detector(self, detector):
        """Set the target detector (SBall, SWorldPoints, etc.).
        
        Args:
            detector: Instance of a TargetDetector subclass
        """
        self.detector = detector
    
    def setup_route(self, waypoints):
        """Setup a route for world-point navigation.
        
        Only applicable if using a detector that supports routes (e.g., SWorldPoints).
        
        Args:
            waypoints: List of (x, y) or (x, y, yaw) tuples
        """
        if not self.detector:
            raise ValueError("Detector not set. Call setup_detector() first.")
        self.detector.set_route(waypoints)

    def reset_origin(self):
        """Reset detector origin when supported by the detector implementation."""
        if not self.detector:
            raise ValueError("Detector not set. Call setup_detector() first.")
        self.detector.reset_origin()
    
    def setup(self, desired_distance=0.41, ctx=None, nav_mode="sequential"):
        """Initialize the navigation controller.
        
        Args:
            desired_distance: Target distance to maintain from target (meters)
            ctx: Mission context with actions, pose, service, etc.
            nav_mode: "sequential" (rotate-then-drive) or "smooth" (simultaneous drive+turn)
        """
        if not self.detector:
            raise ValueError("Detector not set. Call setup_detector() first.")
        
        self.desired_distance = float(desired_distance)
        if str(nav_mode).lower() == "smooth":
            self.nav = NavSmooth()
        else:
            self.nav = Nav()
        self.nav.setup(self.detector, self.desired_distance, ctx)
    
    def start(self):
        """Start navigation towards the target."""
        if not self.nav:
            raise ValueError("Must call setup() before start()")
        
        self.detector.start()
        
        self.nav.start()
        self.started = True
    
    def stop(self):
        """Stop navigation and clean up resources."""
        if self.nav:
            self.nav.stop()
        
        if self.detector and hasattr(self.detector, 'stop'):
            self.detector.stop()
        
        self.started = False
    
    def is_complete(self):
        """Check if target has been reached.
        
        Returns:
            bool: True if navigation objective is complete
        """
        if not self.nav:
            return False
        return self.nav.hasReachedTarget
    
    def get_target_info(self):
        """Get current target information from detector.
        
        Returns:
            dict: Target info or None if no target detected
        """
        if not self.detector:
            return None
        return self.detector.get_target()
    
    def get_status(self):
        """Get detailed navigation status.
        
        Returns:
            dict: Status information including detector and nav state
        """
        status = {
            "nav_started": self.started,
            "nav_complete": self.is_complete() if self.nav else False,
            "desired_distance": self.desired_distance,
        }
        
        if self.detector and hasattr(self.detector, 'get_status'):
            status["detector_status"] = self.detector.get_status()
        
        return status
