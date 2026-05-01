import time as t


class MissionRunner:
    def __init__(
        self,
        objectives,
        ctx,
        refresh_time=0.001,
        tick_hook=None,
        hold_arm_position=True,
        arm_hold_interval_s=0.25,
    ):
        self.objectives = objectives
        self.ctx = ctx
        self.refresh_time = refresh_time  # Time between ticks in seconds
        self.tick_hook = tick_hook
        self.hold_arm_position = bool(hold_arm_position)
        self.arm_hold_interval_s = float(arm_hold_interval_s)

    def _hold_arm_position_tick(self):
        """Re-send arm target periodically to keep servo position stable."""
        if not self.hold_arm_position:
            return
        if "_arm_state" not in self.ctx.memory:
            return
        if not hasattr(self.ctx.actions, "arm"):
            return

        now = t.time()
        last_sent = float(self.ctx.memory.get("_arm_hold_last_sent_s", 0.0))
        if now - last_sent < self.arm_hold_interval_s:
            return

        self.ctx.actions.arm.hold_position()
        self.ctx.memory["_arm_hold_last_sent_s"] = now

    def run(self):
        from ulog import flog

        for idx, obj in enumerate(self.objectives):
            if self.ctx.service.stop:
                break
            self.ctx.reset_state_time()
            self.ctx.memory["_mission_current_objective"] = obj.name
            self.ctx.memory["_mission_current_objective_index"] = idx
            flog.writeRemark(f"% Objective start {obj.name}")
            obj.start(self.ctx)
            while not self.ctx.service.stop and not obj.is_done(self.ctx):
                obj.tick(self.ctx)
                self._hold_arm_position_tick()
                if self.tick_hook is not None:
                    self.tick_hook(self.ctx)
                t.sleep(self.refresh_time)
            obj.stop(self.ctx)
            flog.writeRemark(f"% Objective end {obj.name}")

        self.ctx.memory["_mission_current_objective"] = None
        self.ctx.memory["_mission_current_objective_index"] = None
