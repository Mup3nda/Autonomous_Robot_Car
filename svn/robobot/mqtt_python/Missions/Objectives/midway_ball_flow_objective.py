#!/usr/bin/env python3

import time

from objective import Objective
from Objectives.search_and_navigate_to_red_ball import SearchAndNavigateToRedBall
from Objectives.search_and_navigate_to_blue_ball_objective import SearchAndNavigateToBlueBall
from Objectives.look_for_aruco_objective import LookForArucoObjective
from Objectives.search_and_navigate_to_aruco_objective import SearchAndNavigateToAruco
from Objectives.drive_to_waypoint_objective import DriveToWaypointObjective
from Objectives.grab_target_objective import GrabTargetObjective
from Objectives.drop_target_objective import DropTargetObjective
from Objectives.drive_turn_angle_objective import DriveTurnAngleObjective


class MissionBallFlowObjective(Objective):
    """Mission-specific ball flow with red-first behavior and blue fallback."""

    RED_SEARCH_TIMEOUT_S = 10.0
    BLUE_SEARCH_TIMEOUT_S = 10.0

    def __init__(self, waypoint_nav_mode="smooth"):
        super().__init__()
        self.waypoint_nav_mode = waypoint_nav_mode
        self.phase = "idle"
        self.phase_objective = None
        self.phase_started_at = None
        self.blue_search_failed = False

    def _make_red_pickup_waypoint(self):
        return DriveToWaypointObjective(
            waypoint=(0.6, 1.2),
            is_local=False,
            print_interval=20,
            relative_heading_deg=-100.0,
            nav_mode=self.waypoint_nav_mode,
        )

    def _make_red_search(self):
        return SearchAndNavigateToRedBall(desired_distance=0.35)

    def _make_red_dropoff_waypoint(self):
        return DriveToWaypointObjective(
            waypoint=(0.9, 1.3),
            is_local=False,
            print_interval=20,
            relative_heading_deg=0.0,
            nav_mode=self.waypoint_nav_mode,
        )

    def _make_blue_pickup_waypoint(self):
        return DriveToWaypointObjective(
            waypoint=(0.5, 1.0),
            is_local=False,
            print_interval=20,
            relative_heading_deg=-100.0,
            nav_mode=self.waypoint_nav_mode,
        )

    def _make_blue_search(self):
        return SearchAndNavigateToBlueBall(turn_rate=0.3, desired_distance=0.35)

    def _make_blue_dropoff_waypoint(self):
        return DriveToWaypointObjective(
            waypoint=(1.35, 0.8),
            is_local=False,
            print_interval=20,
            relative_heading_deg=90.0,
            nav_mode=self.waypoint_nav_mode,
        )

    def _make_red_dropoff_aruco(self):
        return SearchAndNavigateToAruco(
            marker_id=12,
            desired_distance=0.35,
            scan_mode=LookForArucoObjective.SCAN_MODE_SWEEP_90,
        )

    def _make_blue_dropoff_aruco(self):
        return SearchAndNavigateToAruco(
            marker_id=15,
            desired_distance=0.35,
            scan_mode=LookForArucoObjective.SCAN_MODE_SWEEP_90,
        )

    def _start_phase(self, ctx, phase, objective=None):
        self.phase = phase
        self.phase_objective = objective
        self.phase_started_at = time.time()
        if self.phase_objective is not None:
            self.phase_objective.start(ctx)

    def start(self, ctx):
        self._done = False
        self._start_phase(ctx, "red_pickup_waypoint", self._make_red_pickup_waypoint())

    def _advance_from_red_search(self, ctx):
        if self.phase_objective is not None:
            self.phase_objective.stop(ctx)
        print("% Red ball not found in time; switching to blue ball flow")
        self._start_phase(ctx, "blue_pickup_waypoint", self._make_blue_pickup_waypoint())

    def _advance_from_blue_search(self, ctx):
        if self.phase_objective is not None:
            self.phase_objective.stop(ctx)
        self.blue_search_failed = True
        print("% Blue ball not found in time; going to blue drop-off location and ending")
        self.phase = "blue_turn"
        self._done = True;
        return

    def _advance_after_phase(self, ctx):
        if self.phase == "red_pickup_waypoint":
            self._start_phase(ctx, "red_search", self._make_red_search())
            return
        if self.phase == "red_search":
            self._start_phase(ctx, "red_grab", GrabTargetObjective(nav_mode=self.waypoint_nav_mode))
            return
        if self.phase == "red_grab":
            self._start_phase(ctx, "red_dropoff_waypoint", self._make_red_dropoff_waypoint())
            return
        if self.phase == "red_dropoff_waypoint":
            self._start_phase(ctx, "red_dropoff_aruco", self._make_red_dropoff_aruco())
            return
        if self.phase == "red_dropoff_aruco":
            self._start_phase(ctx, "red_drop", DropTargetObjective(delay_s=1.0))
            return
        if self.phase == "red_drop":
            self._start_phase(ctx, "red_turn", DriveTurnAngleObjective(angle_deg=170.0, linear_cmd=0.0, timeout_s=6.0))
            return
        if self.phase == "red_turn":
            self._start_phase(ctx, "blue_pickup_waypoint", self._make_blue_pickup_waypoint())
            return
        if self.phase == "blue_pickup_waypoint":
            self._start_phase(ctx, "blue_search", self._make_blue_search())
            return
        if self.phase == "blue_search":
            self._start_phase(ctx, "blue_grab", GrabTargetObjective(nav_mode=self.waypoint_nav_mode))
            return
        if self.phase == "blue_grab":
            self._start_phase(ctx, "blue_dropoff_waypoint", self._make_blue_dropoff_waypoint())
            return
        if self.phase == "blue_dropoff_waypoint":
            if self.blue_search_failed:
                self._done = True
                return
            self._start_phase(ctx, "blue_dropoff_aruco", self._make_blue_dropoff_aruco())
            return
        if self.phase == "blue_dropoff_aruco":
            self._start_phase(ctx, "blue_drop", DropTargetObjective(delay_s=1.0))
            return
        if self.phase == "blue_drop":
            self._start_phase(ctx, "blue_turn", DriveTurnAngleObjective(angle_deg=-90.0, linear_cmd=0.0, timeout_s=6.0))
            return
        if self.phase == "blue_turn":
            self._done = True

    def tick(self, ctx):
        if self._done or self.phase_objective is None:
            return

        if self.phase == "red_search" and self.phase_started_at is not None:
            if time.time() - self.phase_started_at > self.RED_SEARCH_TIMEOUT_S:
                self._advance_from_red_search(ctx)
                return

        if self.phase == "blue_search" and self.phase_started_at is not None:
            if time.time() - self.phase_started_at > self.BLUE_SEARCH_TIMEOUT_S:
                self._advance_from_blue_search(ctx)
                return

        self.phase_objective.tick(ctx)
        if self.phase_objective.is_done(ctx):
            self.phase_objective.stop(ctx)
            self._advance_after_phase(ctx)

    def stop(self, ctx):
        if self.phase_objective is not None:
            self.phase_objective.stop(ctx)
