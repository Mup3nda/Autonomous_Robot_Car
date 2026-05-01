class Objective:
    name = "Objective"

    def __init__(self):
        self._done = False

    def start(self, ctx):
        pass

    def tick(self, ctx):
        pass

    def is_done(self, ctx):
        return self._done

    def stop(self, ctx):
        pass

class CompositeObjective(Objective):

    def __init__(self, objectives):
        super().__init__()
        self.objectives = objectives
        self.index = 0

    def start(self, ctx):
        self.index = 0
        if self.objectives:
            self.objectives[0].start(ctx)

    def tick(self, ctx):
        current = self.objectives[self.index]

        if not current.is_done(ctx):
            current.tick(ctx)
        else:
            current.stop(ctx)
            self.index += 1

            if self.index >= len(self.objectives):
                self._done = True
                return

            self.objectives[self.index].start(ctx)

    def stop(self, ctx):
        # Propagate stop to the currently active child objective.
        if 0 <= self.index < len(self.objectives):
            self.objectives[self.index].stop(ctx)