"""Reset mission odometry origin so current pose becomes (0,0,0)."""

from objective import Objective
from sodom import odom


class ResetOriginObjective(Objective):
    """Reset world-frame origin and complete immediately."""

    name = "reset_origin"

    def start(self, ctx):
        odom.reset_origin()
        print("% Objective: Reset odometry origin -> (0.0, 0.0, 0.0)")
        self._done = True

    def tick(self, ctx):
        pass

    def stop(self, ctx):
        pass
