import time as t


class MissionRunner:
    def __init__(self, objectives, ctx, refresh_time=0.01, tick_hook=None):
        self.objectives = objectives
        self.ctx = ctx
        self.refresh_time = refresh_time  # Time between ticks in seconds
        self.tick_hook = tick_hook

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
                if self.tick_hook is not None:
                    self.tick_hook(self.ctx)
                t.sleep(self.refresh_time)
            obj.stop(self.ctx)
            flog.writeRemark(f"% Objective end {obj.name}")
