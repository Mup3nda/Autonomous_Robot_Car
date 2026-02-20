from objective import Objective
from robot_actions import RobotActions
from mission_context import MissionContext
import time as t

class DriveToLineObjective(Objective):
    name = "drive_to_line"

    def start(self, ctx):
        self.state = 0
        self.dist_to_line = 0.0
        ctx.pose.tripBreset()
        ctx.actions.drive.leds(0, 100, 0)
        print("% Driving to line ---------------------- right ir start ---")

    def tick(self, ctx):
        if self.state == 0:
            if ctx.ir.ir[0] < 0.2:
                ctx.actions.drive.rc(0.2, 0.0)
                ctx.actions.drive.lognow(3)
                ctx.actions.drive.servo(1, -800, 300)
                self.state = 1
        elif self.state == 1:
            if ctx.pose.tripB > 1.0 or ctx.pose.tripBtimePassed() > 15:
                ctx.actions.drive.stop()
                self.state = 2
            if ctx.edge.lineValidCnt > 4:
                ctx.edge.lineControl(0.2, True)
                ctx.actions.drive.servo(1, 0, 0)
                self.dist_to_line = ctx.pose.tripB
                ctx.pose.tripBreset()
                self.state = 10
        elif self.state == 2:
            if abs(ctx.pose.velocity()) < 0.001:
                self.state = 99
        elif self.state == 10:
            if ctx.edge.lineValidCnt < 2:
                ctx.edge.lineControl(0, True)
                ctx.actions.drive.stop()
                ctx.pose.tripBreset()
                self.state = 2
        else:
            print(
                f"# drive to line {self.dist_to_line:.3f}m, then along line "
                f"{ctx.pose.tripB:.3f}m in {ctx.pose.tripBtimePassed():.3f} seconds"
            )
            ctx.actions.drive.stop()
            ctx.actions.drive.servo(1, 500, 200)
            self._done = True
        t.sleep(0.01)

    def stop(self, ctx):
        ctx.actions.drive.leds(0, 0, 0)
        ctx.actions.drive.stop()
        print("% Driving to line ------------------------- end")
