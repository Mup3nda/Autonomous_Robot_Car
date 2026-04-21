"""Single world-space point detector for odometry-based navigation."""

import math

from sodom import odom
from target_detector import TargetDetector


class SWorldPoint(TargetDetector):
    """Reports distance and bearing to a single waypoint.

    Returns a target dict with 'distance' (metres) and 'bearing' (radians,
    positive = target is to the left of the robot). Nav uses these directly
    with its bearing-based PID path.

    Args:
        wx: Target x coordinate.
        wy: Target y coordinate.
        frame: Waypoint frame, either "global" or "local".
            - "global": (wx, wy) is interpreted in world frame.
            - "local": (wx, wy) is interpreted in robot frame at start().
        odometry: Optional odometry object; defaults to the global odom singleton.
    """

    def __init__(self, wx, wy, frame="global", odometry=None):
        super().__init__()
        self.wx = float(wx)
        self.wy = float(wy)
        self.frame = str(frame).lower()
        if self.frame not in ("global", "local"):
            raise ValueError(f"SWorldPoint frame must be 'global' or 'local', got {frame!r}")
        self.odometry = odometry or odom
        self.running = False
        self._target_wx = self.wx
        self._target_wy = self.wy

    def reset_origin(self):
        self.odometry.reset_origin()

    def _resolve_target_world(self):
        if self.frame == "global":
            self._target_wx = self.wx
            self._target_wy = self.wy
            return

        rx, ry, rh = self.odometry.get_world_pose()
        cr, sr = math.cos(rh), math.sin(rh)

        # Local waypoint is interpreted in robot frame at start() (x forward, y left).
        self._target_wx = rx + cr * self.wx - sr * self.wy
        self._target_wy = ry + sr * self.wx + cr * self.wy

    def start(self):
        self._resolve_target_world()
        self.running = True
        print(
            "% SWorldPoint:: tracking "
            f"({self._target_wx:.2f}, {self._target_wy:.2f}) [frame={self.frame}]"
        )

    def stop(self):
        self.running = False
        print(f"% SWorldPoint:: stopped")

    def get_target(self):
        """Return distance and bearing to the waypoint, or None if not running."""
        if not self.running:
            return None

        rx, ry, rh = self.odometry.get_world_pose()

        dx = self._target_wx - rx
        dy = self._target_wy - ry
        distance = math.hypot(dx, dy)

        cr, sr = math.cos(rh), math.sin(rh)
        rel_x = cr * dx + sr * dy
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
