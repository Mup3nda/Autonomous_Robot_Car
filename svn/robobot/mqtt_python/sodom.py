"""Lightweight odometry helper for mission-level navigation.

This module wraps SPose and exposes a world frame where (0,0,0) is the
robot pose at reset time. It is intentionally simple and fast for Raspberry Pi.
"""

from datetime import datetime
import math

from spose import pose


class SOdom:
    """Mission odometry view based on SPose pose stream.

    The Teensy already estimates robot pose and publishes T0/pose.
    This class only applies an origin transform so missions can work in a
    local world frame that starts at (0, 0, 0) at reset.
    """

    def __init__(self, pose_source=None, pose_timeout=0.5):
        self.pose_source = pose_source or pose
        self.pose_timeout = pose_timeout
        self.origin_x = 0.0
        self.origin_y = 0.0
        self.origin_h = 0.0
        self.reset_time = datetime.now()
        self.reset_origin()

    @staticmethod
    def _wrap_to_pi(angle):
        while angle > math.pi:
            angle -= 2.0 * math.pi
        while angle < -math.pi:
            angle += 2.0 * math.pi
        return angle

    def reset_origin(self):
        """Set current robot pose as world origin (0,0,0)."""
        self.origin_x = float(self.pose_source.pose[0])
        self.origin_y = float(self.pose_source.pose[1])
        self.origin_h = float(self.pose_source.pose[2])
        self.reset_time = datetime.now()

    def _get_encoder_world_pose(self):
        """Return world pose from encoder-only odometry transform."""
        x_raw = float(self.pose_source.pose[0])
        y_raw = float(self.pose_source.pose[1])
        h_raw = float(self.pose_source.pose[2])

        # Translate to origin, then rotate so heading at reset is 0 rad.
        dx = x_raw - self.origin_x
        dy = y_raw - self.origin_y
        c0 = math.cos(self.origin_h)
        s0 = math.sin(self.origin_h)
        x = c0 * dx + s0 * dy
        y = -s0 * dx + c0 * dy
        h = self._wrap_to_pi(h_raw - self.origin_h)
        return x, y, h

    def get_world_pose(self):
        """Return odometry in local world frame as (x, y, yaw)."""
        return self._get_encoder_world_pose()

    def pose_age(self):
        """Seconds since last SPose update."""
        return (datetime.now() - self.pose_source.poseTime).total_seconds()

    def confidence(self):
        """Simple confidence based on freshness of pose stream."""
        age = self.pose_age()
        if age <= self.pose_timeout:
            return 1.0
        if age >= 3.0:
            return 0.0
        return max(0.0, 1.0 - (age - self.pose_timeout) / (3.0 - self.pose_timeout))


# Shared module instance, same pattern as spose.py/simu.py.
odom = SOdom()
