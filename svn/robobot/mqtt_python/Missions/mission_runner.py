import time as t


class MissionRunner:
    def __init__(self, objectives, ctx):
        self.objectives = objectives
        self.ctx = ctx

    def run(self):
        from ulog import flog

        for obj in self.objectives:
            if self.ctx.service.stop:
                break
            self.ctx.reset_state_time()
            flog.writeRemark(f"% Objective start {obj.name}")
            obj.start(self.ctx)
            while not self.ctx.service.stop and not obj.is_done(self.ctx):
                obj.tick(self.ctx)
                t.sleep(0.05)
            obj.stop(self.ctx)
            flog.writeRemark(f"% Objective end {obj.name}")
