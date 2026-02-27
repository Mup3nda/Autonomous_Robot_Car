"""Robot actions module for sending commands to the robot hardware.

Provides high-level interfaces for drive commands (motor control, LEDs, servos),
edge/line detection and following, and vision/image analysis operations.
"""
from .drive import DriveActions
from .edge import EdgeActions
from .vision import VisionActions


class RobotActions:
    """Aggregates all robot control actions: drive, edge, and vision.
    
    Args:
        service: MQTT service for sending commands to Teensy
        gpio: GPIO interface for direct pin control
        cam: Camera interface for image capture
        edge: Edge sensor interface for line detection
    """
    def __init__(self, service, gpio, cam, edge):
        self.drive = DriveActions(service, gpio)  # Motor, LED, servo control
        self.edge = EdgeActions(edge)  # Line detection and following
        self.vision = VisionActions(cam, edge, gpio, service)  # Image capture and analysis
