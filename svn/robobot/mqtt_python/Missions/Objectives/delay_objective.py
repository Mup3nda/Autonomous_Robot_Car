"""Reusable delay objective for composite missions."""

import time as t

from objective import Objective


class DelayObjective(Objective):
    """Wait for a fixed amount of time using a local progress marker."""

    name = "delay"
    _instance_counter = 0

    def __init__(self, duration_s):
        super().__init__()
        self.duration_s = float(duration_s)
        self._progress_key = f"delay_{DelayObjective._instance_counter}"
        DelayObjective._instance_counter += 1

    def start(self, ctx):
        self._done = self.duration_s <= 0.0
        if not self._done:
            ctx.start_local_progress(self._progress_key)

    def tick(self, ctx):
        if self._done:
            return

        progress_map = ctx.memory.get("_local_progress", {})
        marker = progress_map.get(self._progress_key)
        if marker is None:
            ctx.start_local_progress(self._progress_key)
            marker = ctx.memory["_local_progress"][self._progress_key]

        if t.time() - marker["time_s"] >= self.duration_s:
            self._done = True

    def stop(self, ctx):
        pass