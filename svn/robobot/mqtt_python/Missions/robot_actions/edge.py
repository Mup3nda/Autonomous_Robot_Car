"""Edge/line sensor actions: high-level interface for line detection and following."""


class EdgeActions:
    """High-level interface for line following control.
    
    Wraps the sedge.SEdge module to provide clean, objective-level interface
    for line detection and automatic line following control.
    
    Args:
        edge: Instance of sedge.SEdge (the low-level edge sensor interface)
    """
    
    def __init__(self, edge):
        self.edge = edge  # Reference to sedge.SEdge instance
    
    def start_following(self, velocity=0.2, follow_left=True, ref_position=0.0, stop_on_intersection=False, stop_on_left_turn=False):
        """Begin automatic line following.
        
        Enables closed-loop control that automatically steers the robot
        to follow a detected line. The control runs asynchronously in
        response to sensor updates from the Teensy.
        
        Args:
            velocity: Forward speed (0.0 to 1.0, where 0.2 = 20% throttle)
            follow_left: If True, follow left edge; if False, follow right edge
            ref_position: Target distance from line edge (default 0.0 = on line)
            stop_on_intersection: If True, stop line following when intersection detected
            stop_on_left_turn: If True, stop line following when a 90-degree left turn is detected
        """
        self.edge.lineControl(velocity, follow_left, ref_position, stop_on_intersection, stop_on_left_turn)
    
    def stop_following(self):
        """Stop automatic line following.
        
        Disables the closed-loop line following control. Does not stop
        the robot motors - use ctx.actions.drive.stop() for that.
        """
        self.edge.lineControl(0, True)
    
    def is_line_valid(self, confidence=2):
        """Check if line is currently detected with sufficient confidence.
        
        Args:
            confidence: Minimum lineValidCnt threshold (0-20, default 2)
                       Higher values require more confident detection
        
        Returns:
            bool: True if line is detected with required confidence
        """
        return self.edge.lineValidCnt > confidence
    
    def get_line_position(self):
        """Get current line position relative to robot center.
        
        Returns:
            float: Line position (-3.5 to +3.5, where 0 is centered)
                  Negative = line is to the left
                  Positive = line is to the right
        """
        return self.edge.posLeft if self.edge.followLeft else self.edge.posRight
    
    def is_crossing(self, confidence=2):
        """Check if robot is on a crossing line (perpendicular line).
        
        Detects when the robot is crossing over a line, useful for
        navigation waypoints or intersection detection.
        
        Args:
            confidence: Minimum crossingLineCnt threshold (0-20, default 2)
        
        Returns:
            bool: True if crossing line is detected
        """
        return self.edge.crossingLineCnt > confidence

    def is_line_control_active(self):
        """Check if line control is currently active (being processed in sedge).

        Returns:
            bool: True if lineControl is actively running PID
        """
        return self.edge.lineCtrl

    def get_line_control_stop_reason(self):
        """Return why line control last stopped, if available."""
        return getattr(self.edge, "stop_reason", None)

    def is_intersection(self, confidence=2):
        """Check if an intersection is detected with sufficient confidence.

        Args:
            confidence: Minimum intersectionCnt threshold (0-20, default 2)
        Returns:
            bool: True if intersection is detected
        """
        return self.edge.intersectionCnt > confidence
    
    def get_line_confidence(self):
        """Get line detection confidence level.
        
        Returns:
            int: Confidence counter (0-20), higher is more confident
        """
        return self.edge.lineValidCnt
    
    def get_sensor_values(self):
        """Get normalized sensor values for debugging.
        
        Returns:
            list: 8 normalized sensor values (0.0 to 1.0)
        """
        return self.edge.edge_n
    
    def get_left_position(self):
        """Get left edge position for debugging/logging.
        
        Returns:
            float: Left edge position
        """
        return self.edge.posLeft
    
    def get_right_position(self):
        """Get right edge position for debugging/logging.
        
        Returns:
            float: Right edge position
        """
        return self.edge.posRight

    def last_seen_time_passed(self):
        """Get time in seconds since line was last seen with sufficient confidence.
        
        Returns:
            float: Time in seconds since line was last detected
        """
        from datetime import datetime
        return (datetime.now() - self.edge.lineLastSeenTime).total_seconds() 
