"""Drive a custom multi-leg route as single-waypoint objectives."""

from objective import CompositeObjective
from Objectives.drive_to_waypoint_objective import DriveToWaypointObjective


class DriveSquareWorldPointsObjective(CompositeObjective):
    """Drive a custom test route in the local world frame.

    Route from (0,0):
    (0.6, 0.0) -> (0.6, 3.0) -> (0.6, -3.0) -> (0.6, 0.0) -> (0.0, 0.0)
    """

    name = "DriveSquareWorldPointsObjective"

    def __init__(self, side_length=0.6, print_interval=20, nav_mode="smooth"):
        waypoints = [
            (0.4, 0.0),
            (0.4, 2.0),
            (0.6, -2.0),
            (0.4, 0.0),
            (0.0, 0.0),
        ]

        objectives = []
        for i, wp in enumerate(waypoints):
            objectives.append(
                DriveToWaypointObjective(
                    waypoint=wp,
                    reset_origin=(i == 0),
                    print_interval=print_interval,
                    nav_mode=nav_mode,
                )
            )

        super().__init__(objectives)
