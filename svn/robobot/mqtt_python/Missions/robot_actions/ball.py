"""Ball tracking actions: high-level interface for ball detection and following."""


class BallActions:
    """High-level interface for ball tracking and following control.
    
    Wraps the sball.SBall module to provide clean, objective-level interface
    for ball detection and automatic ball following control.
    
    Args:
        ball: Instance of sball.SBall (the low-level ball tracking interface)
    """
    
    def __init__(self, ball):
        self.ball = ball  # Reference to sball.SBall instance
    
    def start_following(self, velocity=0.2, target_distance=0.5):
        """Begin automatic ball tracking and approach.
        
        Enables closed-loop control that automatically steers the robot
        to center the ball in view and approach to target distance.
        The control runs asynchronously in the ball tracking thread.
        
        Args:
            velocity: Nominal forward speed (0.0 to 1.0, where 0.2 = 20% throttle)
                     Actual speed adjusts based on distance to ball
            target_distance: Target distance to maintain from ball (meters, default 0.5m)
        """
        self.ball.ballControl(velocity, target_distance)
    
    def stop_following(self):
        """Stop automatic ball following.
        
        Disables the closed-loop ball following control. Does not stop
        the robot motors - use ctx.actions.drive.stop() for that.
        """
        self.ball.ballControl(0, 0)
    
    def is_ball_visible(self, confidence=2):
        """Check if ball is currently detected with sufficient confidence.
        
        Args:
            confidence: Minimum ball_confidence threshold (0-20, default 2)
                       Higher values require more confident detection
        
        Returns:
            bool: True if ball is detected with required confidence
        """
        return self.ball.ball_valid and self.ball.ball_confidence >= confidence
    
    def get_ball_position(self):
        """Get ball position in image coordinates.
        
        Returns:
            tuple: (x, y, radius) where:
                   x, y = center position in pixels
                   radius = ball radius in pixels
        """
        return (self.ball.ball_x, self.ball.ball_y, self.ball.ball_radius)
    
    def is_centered(self, tolerance=50):
        """Check if ball is centered in the image.
        
        Args:
            tolerance: Maximum deviation from center in pixels (default 50)
        
        Returns:
            bool: True if ball is within tolerance of image center
        """
        if not self.ball.ball_valid:
            return False
        
        center_x = self.ball.image_width / 2
        deviation = abs(self.ball.ball_x - center_x)
        return deviation < tolerance
    
    def get_estimated_distance(self):
        """Get estimated distance to ball based on apparent size.
        
        Uses camera calibration and known ball size to estimate distance.
        
        Returns:
            float: Estimated distance to ball in meters
        """
        return self.ball.estimated_distance()
    
    def is_at_target_distance(self, tolerance=0.1):
        """Check if robot is at target distance from ball.
        
        Args:
            tolerance: Distance tolerance in meters (default 0.1m = 10cm)
        
        Returns:
            bool: True if within tolerance of target distance
        """
        if not self.ball.ball_valid:
            return False
        
        current_dist = self.ball.estimated_distance()
        target_dist = self.ball.target_distance
        return abs(current_dist - target_dist) < tolerance
    
    def get_ball_confidence(self):
        """Get ball detection confidence level.
        
        Returns:
            int: Confidence counter (0-20), higher is more confident
        """
        return self.ball.ball_confidence
    
    def set_color_range(self, color_name='red'):
        """Set the color to track.
        
        Args:
            color_name: Color to track ('red', 'blue', 'green', 'yellow')
        """
        # Predefined HSV ranges for common colors
        color_ranges = {
            'red': {
                'lower1': (0, 245, 150),
                'upper1': (10, 255, 255),
                'lower2': (170, 245, 150),
                'upper2': (180, 255, 255)
            },
            'blue': {
                'lower1': (100, 150, 100),
                'upper1': (130, 255, 255),
                'lower2': (100, 150, 100),  # Blue doesn't wrap
                'upper2': (130, 255, 255)
            },
            'green': {
                'lower1': (40, 100, 100),
                'upper1': (80, 255, 255),
                'lower2': (40, 100, 100),  # Green doesn't wrap
                'upper2': (80, 255, 255)
            },
            'yellow': {
                'lower1': (20, 100, 100),
                'upper1': (40, 255, 255),
                'lower2': (20, 100, 100),  # Yellow doesn't wrap
                'upper2': (40, 255, 255)
            }
        }
        
        if color_name in color_ranges:
            ranges = color_ranges[color_name]
            self.ball.color_lower1 = ranges['lower1']
            self.ball.color_upper1 = ranges['upper1']
            self.ball.color_lower2 = ranges['lower2']
            self.ball.color_upper2 = ranges['upper2']
            print(f"% Ball: Color set to {color_name}")
        else:
            print(f"% Ball: Unknown color '{color_name}', keeping current settings")
    
    def get_status(self):
        """Get full ball tracking status for debugging.
        
        Returns:
            dict: Status dictionary with detection info
        """
        return {
            'visible': self.ball.ball_valid,
            'x': self.ball.ball_x,
            'y': self.ball.ball_y,
            'radius': self.ball.ball_radius,
            'confidence': self.ball.ball_confidence,
            'distance': self.ball.estimated_distance() if self.ball.ball_valid else None,
            'centered': self.is_centered(),
            'following': self.ball.ballCtrl
        }
