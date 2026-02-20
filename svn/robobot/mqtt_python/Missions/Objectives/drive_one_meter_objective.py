from objective import Objective

class DriveOneMeterObjective(Objective):
    name = "drive_one_meter"

    def start(self, ctx):
        self.state = 0
        ctx.pose.tripBreset()
        ctx.actions.drive.leds(0, 100, 0)
        print("% Driving 1m -------------------------")

    def tick(self, ctx):
        if self.state == 0:
            ctx.actions.drive.rc(0.2, 0.0)
            ctx.actions.drive.servo(1, -800, 300)
            self.state = 1
        elif self.state == 1:
            if ctx.pose.tripB > 1.0 or ctx.pose.tripBtimePassed() > 15:
                ctx.actions.drive.stop()
                ctx.actions.drive.servo(1, 0, 0)
                self.state = 2
        elif self.state == 2:
            if abs(ctx.pose.velocity()) < 0.001:
                print(
                    f"# drive 1m drove {ctx.pose.tripB:.3f}m in {ctx.pose.tripBtimePassed():.3f} seconds"
                )
                self._done = True
        print(
            f"# drive {self.state}, now {ctx.pose.tripB:.3f}m in {ctx.pose.tripBtimePassed():.3f} seconds; "
            f"left {ctx.edge.posLeft}, right {ctx.edge.posRight}"
        )

    def stop(self, ctx):
        ctx.actions.drive.leds(0, 0, 0)
        ctx.actions.drive.stop()
        print("% Driving 1m ------------------------- end")