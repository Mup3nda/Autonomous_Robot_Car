"""Wait for the shared IR reader to produce data, then finish."""

from objective import Objective


class StartIRObjective(Objective):
    name = "start_ir"

    def __init__(self, timeout_s=1.0, memory_key="_ir_latest"):
        super().__init__()
        self.timeout_s = float(timeout_s)
        self.memory_key = memory_key

    def start(self, ctx):
        ready = ctx.actions.ir.wait_until_ready(timeout_s=self.timeout_s)
        data = ctx.actions.ir.remember_latest(self.memory_key)
        data["ready"] = bool(ready)
        ctx.memory[self.memory_key] = data
        if not ready:
            print(f"% StartIRObjective: no IR update within {self.timeout_s:.2f}s")
        self._done = True

    def tick(self, ctx):
        pass

    def stop(self, ctx):
        pass
