"""Reset mission odometry origin so current pose becomes (0,0,0)."""

from objective import Objective
from sodom import odom


class ResetOriginObjective(Objective):
    """Reset world-frame origin and complete immediately."""

    name = "reset_origin"

    def start(self, ctx):
        raw_before = float(odom.pose_source.pose[2])
        _, _, world_before = odom.get_world_pose()
        odom.reset_origin()
        raw_after = float(odom.pose_source.pose[2])
        _, _, world_after = odom.get_world_pose()
        print(
            "% Objective: Reset odometry origin -> (0.0, 0.0, 0.0); "
            f"raw_h {raw_before:+.4f}->{raw_after:+.4f}, "
            f"world_h {world_before:+.4f}->{world_after:+.4f}"
        )
        self._done = True

    def tick(self, ctx):
        pass

    def stop(self, ctx):
        pass
