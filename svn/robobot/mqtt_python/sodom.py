"""Lightweight odometry helper for mission-level navigation.

This module wraps SPose and exposes a world frame where (0,0,0) is the
robot pose at reset time. It is intentionally simple and fast for Raspberry Pi.
"""

from datetime import datetime
import math

from spose import pose
from simu import imu


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
        # Optional IMU slip suppression for odometry translation updates.
        self.imu_source = imu
        self.use_imu_slip_guard = True
        self.imu_timeout = 0.35
        self.stationary_gyro_threshold = 0.08      # rad/s (assumed from imu.gyro[2])
        self.stationary_accel_dev_threshold = 0.08 # normalized accel magnitude delta
        self.slip_translation_threshold = 0.006    # m/update before suppression kicks in
        self.slip_translation_scale = 0.15         # keep only 15% of suspicious translation
        self._acc_norm_lp = None
        self._fused_x = 0.0
        self._fused_y = 0.0
        self._fused_h = 0.0
        self._last_enc_x = None
        self._last_enc_y = None
        self._last_enc_h = None
        self.slip_suppression_active = False
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

        self._fused_x = 0.0
        self._fused_y = 0.0
        self._fused_h = 0.0
        self._last_enc_x = None
        self._last_enc_y = None
        self._last_enc_h = None
        self.slip_suppression_active = False

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

    def _imu_data_fresh(self):
        now = datetime.now()
        gyro_age = (now - self.imu_source.gyroTime).total_seconds()
        acc_age = (now - self.imu_source.accTime).total_seconds()
        return gyro_age <= self.imu_timeout and acc_age <= self.imu_timeout

    def _imu_stationary(self):
        """True when IMU indicates no meaningful body motion."""
        if not self._imu_data_fresh():
            return False

        ax = float(self.imu_source.acc[0])
        ay = float(self.imu_source.acc[1])
        az = float(self.imu_source.acc[2])
        gyro_z = abs(float(self.imu_source.gyro[2]))

        acc_norm = math.sqrt(ax * ax + ay * ay + az * az)
        if self._acc_norm_lp is None:
            self._acc_norm_lp = acc_norm
        else:
            self._acc_norm_lp = 0.99 * self._acc_norm_lp + 0.01 * acc_norm

        acc_dev = abs(acc_norm - self._acc_norm_lp)
        return gyro_z < self.stationary_gyro_threshold and acc_dev < self.stationary_accel_dev_threshold

    def get_world_pose(self):
        """Return odometry in local world frame as (x, y, yaw)."""
        enc_x, enc_y, enc_h = self._get_encoder_world_pose()

        if not self.use_imu_slip_guard:
            return enc_x, enc_y, enc_h

        if self._last_enc_x is None:
            self._fused_x = enc_x
            self._fused_y = enc_y
            self._fused_h = enc_h
            self._last_enc_x = enc_x
            self._last_enc_y = enc_y
            self._last_enc_h = enc_h
            self.slip_suppression_active = False
            return self._fused_x, self._fused_y, self._fused_h

        dx = enc_x - self._last_enc_x
        dy = enc_y - self._last_enc_y
        dh = self._wrap_to_pi(enc_h - self._last_enc_h)
        ds = math.hypot(dx, dy)

        stationary_imu = self._imu_stationary()
        if stationary_imu and ds > self.slip_translation_threshold:
            scale = self.slip_translation_scale
            self.slip_suppression_active = True
        else:
            scale = 1.0
            self.slip_suppression_active = False

        self._fused_x += dx * scale
        self._fused_y += dy * scale
        self._fused_h = self._wrap_to_pi(self._fused_h + dh)

        self._last_enc_x = enc_x
        self._last_enc_y = enc_y
        self._last_enc_h = enc_h

        return self._fused_x, self._fused_y, self._fused_h

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
