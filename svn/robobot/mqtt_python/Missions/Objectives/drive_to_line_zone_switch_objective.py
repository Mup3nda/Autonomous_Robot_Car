"""DriveToLine objective with world-coordinate switch zones.

Each switch zone can update follow side and speed while line-following:
- x, y, radius_m: circular trigger area in world frame
- trigger_dist_m: switch after along-line distance reaches this value
- trigger_time_s: switch after along-line time reaches this value
- follow_left: optional bool, if provided updates follow side
- follow_speed: optional float, if provided updates follow speed
- lost_line_timeout_s: optional float, if provided updates line-loss timeout
- name: optional label for logging

This objective inherits DriveToLineObjective and reuses its state machine.
"""

import time as t
from typing import Dict, List, Optional

from sodom import odom

from Objectives.drive_to_line_objective import DriveToLineObjective, DriveToLineState


class DriveToLineZoneSwitchObjective(DriveToLineObjective):
    name = "drive_to_line_zone_switch"

    def __init__(
        self,
        switch_zones: Optional[List[Dict]] = None,
        ordered_switches: bool = True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.switch_zones = list(switch_zones or [])
        self.ordered_switches = bool(ordered_switches)
        self._triggered: List[bool] = []

    def start(self, ctx):
        super().start(ctx)
        self._triggered = [False] * len(self.switch_zones)

    @staticmethod
    def _inside_zone(x_world: float, y_world: float, zone: Dict) -> bool:
        zx = float(zone.get("x", 0.0))
        zy = float(zone.get("y", 0.0))
        radius = max(0.0, float(zone.get("radius_m", 0.0)))
        dx = x_world - zx
        dy = y_world - zy
        return (dx * dx + dy * dy) <= (radius * radius)

    def _apply_switch(self, ctx, zone: Dict, zone_index: int):
        prev_side = self.follow_left
        prev_speed = self.follow_speed
        prev_timeout = self.lost_line_timeout_s

        if "follow_left" in zone:
            self.follow_left = bool(zone["follow_left"])
        if "follow_speed" in zone:
            self.follow_speed = float(zone["follow_speed"])
        if "lost_line_timeout_s" in zone:
            self.lost_line_timeout_s = max(0.0, float(zone["lost_line_timeout_s"]))

        # Re-issue line-follow command with updated parameters.
        ctx.actions.edge.start_following(
            velocity=self.follow_speed,
            follow_left=self.follow_left,
        )

        zone_name = str(zone.get("name", f"zone_{zone_index}"))
        print(
            f"% Zone switch [{zone_name}] side {prev_side}->{self.follow_left}, "
            f"speed {prev_speed:.3f}->{self.follow_speed:.3f}, "
            f"lost_line_timeout {prev_timeout:.3f}->{self.lost_line_timeout_s:.3f}"
        )
        self._triggered[zone_index] = True

    def _is_triggered(
        self,
        ctx,
        zone: Dict,
        x_world: float,
        y_world: float,
        along_dist_m: Optional[float],
        along_time_s: Optional[float],
    ) -> bool:
        if "trigger_dist_m" in zone:
            if along_dist_m is None:
                return False
            return along_dist_m >= float(zone["trigger_dist_m"])

        if "trigger_time_s" in zone:
            if along_time_s is None:
                return False
            return along_time_s >= float(zone["trigger_time_s"])

        if all(k in zone for k in ("x", "y", "radius_m")):
            return self._inside_zone(x_world, y_world, zone)

        return False

    def tick(self, ctx):
        super().tick(ctx)

        if self.state != DriveToLineState.LINE_FOLLOWING:
            return
        if not self.switch_zones:
            return

        x_world, y_world, _ = odom.get_world_pose()
        along_dist_m: Optional[float] = None
        along_time_s: Optional[float] = None

        if self.along_line_started:
            try:
                along_dist_m = float(ctx.distance_since_start(self.ALONG_LINE_PROGRESS_KEY))
                marker = ctx.memory["_local_progress"][self.ALONG_LINE_PROGRESS_KEY]
                along_time_s = t.time() - float(marker["time_s"])
            except Exception:
                along_dist_m = None
                along_time_s = None

        if self.ordered_switches:
            for i, done in enumerate(self._triggered):
                if done:
                    continue
                zone = self.switch_zones[i]
                if self._is_triggered(ctx, zone, x_world, y_world, along_dist_m, along_time_s):
                    self._apply_switch(ctx, zone, i)
                break
            return

        for i, done in enumerate(self._triggered):
            if done:
                continue
            zone = self.switch_zones[i]
            if self._is_triggered(ctx, zone, x_world, y_world, along_dist_m, along_time_s):
                self._apply_switch(ctx, zone, i)
                break
