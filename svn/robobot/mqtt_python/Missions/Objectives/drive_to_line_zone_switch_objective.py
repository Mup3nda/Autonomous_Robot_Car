"""DriveToLine objective with world-coordinate switch zones.

Each switch zone can update follow side and speed while line-following:
- x, y, radius_m: circular trigger area in world frame
- follow_left: optional bool, if provided updates follow side
- follow_speed: optional float, if provided updates follow speed
- name: optional label for logging

This objective inherits DriveToLineObjective and reuses its state machine.
"""

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

        if "follow_left" in zone:
            self.follow_left = bool(zone["follow_left"])
        if "follow_speed" in zone:
            self.follow_speed = float(zone["follow_speed"])

        # Re-issue line-follow command with updated parameters.
        ctx.actions.edge.start_following(
            velocity=self.follow_speed,
            follow_left=self.follow_left,
        )

        zone_name = str(zone.get("name", f"zone_{zone_index}"))
        print(
            f"% Zone switch [{zone_name}] side {prev_side}->{self.follow_left}, "
            f"speed {prev_speed:.3f}->{self.follow_speed:.3f}"
        )
        self._triggered[zone_index] = True

    def tick(self, ctx):
        super().tick(ctx)

        if self.state != DriveToLineState.LINE_FOLLOWING:
            return
        if not self.switch_zones:
            return

        x_world, y_world, _ = odom.get_world_pose()

        if self.ordered_switches:
            for i, done in enumerate(self._triggered):
                if done:
                    continue
                zone = self.switch_zones[i]
                if self._inside_zone(x_world, y_world, zone):
                    self._apply_switch(ctx, zone, i)
                break
            return

        for i, done in enumerate(self._triggered):
            if done:
                continue
            zone = self.switch_zones[i]
            if self._inside_zone(x_world, y_world, zone):
                self._apply_switch(ctx, zone, i)
                break
