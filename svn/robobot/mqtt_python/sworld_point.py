"""Single world-space point detector for odometry-based navigation."""

import math

from sodom import odom
from target_detector import TargetDetector


class SWorldPoint(TargetDetector):
    """Reports distance and bearing to a single world-space (x, y) waypoint.

    Returns a target dict with 'distance' (metres) and 'bearing' (radians,
    positive = target is to the left of the robot).  Nav uses these directly
    with its bearing-based PID path.

    Args:
        wx: Target x coordinate in world frame (metres).
        wy: Target y coordinate in world frame (metres).
        odometry: Optional odometry object; defaults to the global odom singleton.
    """

    def __init__(self, wx, wy, odometry=None):
        super().__init__()
        self.wx = float(wx)
        self.wy = float(wy)
        self.odometry = odometry or odom
        self.running = False

    def reset_origin(self):
        self.odometry.reset_origin()

    def start(self):
        self.running = True
        print(f"% SWorldPoint:: tracking ({self.wx:.2f}, {self.wy:.2f})")

    def stop(self):
        self.running = False
        print(f"% SWorldPoint:: stopped")

    def get_target(self):
        """Return distance and bearing to the waypoint, or None if not running."""
        if not self.running:
            return None

        rx, ry, rh = self.odometry.get_world_pose()

        dx = self.wx - rx
        dy = self.wy - ry
        distance = math.hypot(dx, dy)

        cr, sr = math.cos(rh), math.sin(rh)
        rel_x =  cr * dx + sr * dy
        rel_y = -sr * dx + cr * dy
        bearing = math.atan2(rel_y, rel_x)  # positive = target is to the left

        conf01 = self.odometry.confidence()
        confidence = int(max(0, min(20, round(20.0 * conf01))))

        return {
            "valid": True,
            "distance": distance,
            "bearing": bearing,
            "confidence": confidence,
        }
