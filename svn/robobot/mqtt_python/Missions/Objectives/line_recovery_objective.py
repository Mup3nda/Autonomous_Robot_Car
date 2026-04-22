
from objective import Objective
from Objectives.drive_to_line_objective import DriveToLineObjective
from Objectives.drive_turn_angle_objective import DriveTurnAngleObjective
from Objectives.drive_to_timer_and_back_objective import DriveToTimerAndBackObjective

LINE_RECOVERY_FOLLOW_LEFT = False
LINE_RECOVERY_FOLLOW_SPEED = 0.75
LINE_RECOVERY_SEARCH_SPEED = 0.35
LINE_RECOVERY_LOST_LINE_TIMEOUT_S = 0.35
LINE_RECOVERY_SEARCH_TIMEOUT_S = 4.0
LINE_RECOVERY_TURN_ANGLE_DEG = 90.0
LINE_RECOVERY_TURN_TIMEOUT_S = 6.0
LINE_MAX_DISTANCE = 25

class LineRecovery(Objective):
    name = "line_recovery"

    def __init__(
        self,
        follow_left=LINE_RECOVERY_FOLLOW_LEFT,
        follow_speed=LINE_RECOVERY_FOLLOW_SPEED,
        search_speed=LINE_RECOVERY_SEARCH_SPEED,
        lost_line_timeout_s=LINE_RECOVERY_LOST_LINE_TIMEOUT_S,
        search_timeout_s=LINE_RECOVERY_SEARCH_TIMEOUT_S,
        turn_angle_deg=LINE_RECOVERY_TURN_ANGLE_DEG,
        turn_timeout_s=LINE_RECOVERY_TURN_TIMEOUT_S,
        max_line_distance_m=LINE_MAX_DISTANCE
    ):
        super().__init__()
        self.objectives = [
            DriveTurnAngleObjective(
                angle_deg=turn_angle_deg,
                linear_cmd=0.0,
                timeout_s=turn_timeout_s,
            ),
            DriveToLineObjective(
                follow_left=follow_left,
                follow_speed=follow_speed,
                search_speed=search_speed,
                lost_line_timeout_s=lost_line_timeout_s,
                search_timeout_s=search_timeout_s,
                max_line_distance_m=max_line_distance_m,
            ),
            DriveToTimerAndBackObjective(drive_back=True),
            
        ]
        self.index = 0
        self.active = False

    def start(self, ctx):
        self.index = 0
        self.active = bool(ctx.memory.get("line_failed", False))
        if not self.active:
            self._done = True
            return
        self.objectives[0].start(ctx)

    def tick(self, ctx):
        if not self.active:
            self._done = True
            return

        current = self.objectives[self.index]
        if not current.is_done(ctx):
            current.tick(ctx)
            return

        current.stop(ctx)
        self.index += 1
        if self.index >= len(self.objectives):
            ctx.memory["line_failed"] = False
            self._done = True
            return
        self.objectives[self.index].start(ctx)

    def stop(self, ctx):
        if self.active and 0 <= self.index < len(self.objectives):
            self.objectives[self.index].stop(ctx)

