from objective import Objective
from robot_actions import RobotActions
from mission_context import MissionContext
import time as t

class DriveTurnPiObjective(Objective):
    name = "drive_turn_pi"

    def start(self, ctx):
        self.state = 0
        ctx.pose.tripBreset()
        ctx.actions.drive.leds(0, 100, 0)
        print("% Driving a Pi turn -------------------------")

    def tick(self, ctx):
        if self.state == 0:
            ctx.actions.drive.rc(0.2, 0.5)
            self.state = 1
        elif self.state == 1:
            if ctx.pose.tripBh > 3.14 or ctx.pose.tripBtimePassed() > 15:
                ctx.actions.drive.stop()
                self.state = 2
        elif self.state == 2:
            if abs(ctx.pose.velocity()) < 0.001 and abs(ctx.pose.turnrate()) < 0.001:
                print(
                    f"# drive turned {ctx.pose.tripBh:.3f} rad in {ctx.pose.tripBtimePassed():.3f} seconds"
                )
                self._done = True
        print(
            f"# turn {self.state}, now {ctx.pose.tripBh:.3f} rad in {ctx.pose.tripBtimePassed():.3f} seconds; "
            f"left {ctx.edge.posLeft}, right {ctx.edge.posRight}"
        )

    def stop(self, ctx):
        ctx.actions.drive.leds(0, 0, 0)
        ctx.actions.drive.stop()
        print("% Driving a Pi turn ------------------------- end")