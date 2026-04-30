#!/usr/bin/env python3

import time

from objective import Objective
from Objectives.look_for_aruco_objective import LookForArucoObjective
from Objectives.search_and_navigate_to_aruco_objective import SearchAndNavigateToAruco
from Objectives.drive_to_waypoint_objective import DriveToWaypointObjective
from Objectives.grab_target_objective import GrabArucoObjective, GrabTargetObjective
from Objectives.drop_target_objective import DropTargetObjective


class MissionArucoCubeFlowObjective(Objective):
    """Mission-specific aruco cube flow with cube 20 first and cube 53 fallback."""

    CUBE20_SEARCH_TIMEOUT_S = 10.0
    CUBE53_SEARCH_TIMEOUT_S = 10.0

    def __init__(self, waypoint_nav_mode="smooth"):
        super().__init__()
        self.waypoint_nav_mode = waypoint_nav_mode
        self.phase = "idle"
        self.phase_objective = None
        self.phase_started_at = None
        self.cube20_search_failed = False
        self.cube53_search_failed = False

    def _make_cube20_pickup_waypoint(self):
        return DriveToWaypointObjective(
            waypoint=(1.94, 0.7),
            is_local=False,
            print_interval=20,
            relative_heading_deg=0.0,
            nav_mode=self.waypoint_nav_mode,
        )

    def _make_cube20_search(self):
        return SearchAndNavigateToAruco(
            marker_id=20,
            desired_distance=0.38,
            scan_mode=LookForArucoObjective.SCAN_MODE_SWEEP_90,
        )

    def _make_dropoff_a_waypoint(self):
        return DriveToWaypointObjective(
            waypoint=(1.94, 1.6),
            is_local=False,
            print_interval=20,
            relative_heading_deg=110.0,
            nav_mode=self.waypoint_nav_mode,
        )

    def _make_dropoff_a_approach_waypoint(self):
        return DriveToWaypointObjective(
            waypoint=(1.5, 1.7),
            is_local=False,
            print_interval=20,
            relative_heading_deg=-90.0,
            nav_mode=self.waypoint_nav_mode,
        )

    def _make_pickup_again_waypoint(self):
        return DriveToWaypointObjective(
            waypoint=(1.94, 0.7),
            is_local=False,
            print_interval=20,
            relative_heading_deg=0.0,
            nav_mode=self.waypoint_nav_mode,
        )

    def _make_cube53_search(self):
        return SearchAndNavigateToAruco(
            marker_id=53,
            desired_distance=0.38,
            scan_mode=LookForArucoObjective.SCAN_MODE_SWEEP_90,
        )

    def _make_dropoff_d_waypoint(self):
        return DriveToWaypointObjective(
            waypoint=(1.8, 0.7),
            is_local=False,
            print_interval=20,
            relative_heading_deg=165.0,
            nav_mode=self.waypoint_nav_mode,
        )

    def _make_dropoff_d_search(self):
        return SearchAndNavigateToAruco(
            marker_id=17,
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
        self.cube20_search_failed = False
        self.cube53_search_failed = False
        self._start_phase(ctx, "cube20_pickup_waypoint", self._make_cube20_pickup_waypoint())

    def _advance_from_cube20_search(self, ctx):
        if self.phase_objective is not None:
            self.phase_objective.stop(ctx)
        self.cube20_search_failed = True
        print("% Cube 20 not found in time; switching to cube 53 flow")
        self._start_phase(ctx, "pickup_again_waypoint", self._make_pickup_again_waypoint())

    def _advance_from_cube53_search(self, ctx):
        if self.phase_objective is not None:
            self.phase_objective.stop(ctx)
        self.cube53_search_failed = True
        print("% Cube 53 not found in time; going to drop-off D location and ending")
        self._start_phase(ctx, "dropoff_d_waypoint", self._make_dropoff_d_waypoint())

    def _advance_after_phase(self, ctx):
        if self.phase == "cube20_pickup_waypoint":
            self._start_phase(ctx, "cube20_search", self._make_cube20_search())
            return
        if self.phase == "cube20_search":
            self._start_phase(ctx, "cube20_grab", GrabArucoObjective(nav_mode=self.waypoint_nav_mode))
            return
        if self.phase == "cube20_grab":
            self._start_phase(ctx, "dropoff_a_waypoint", self._make_dropoff_a_waypoint())
            return
        if self.phase == "dropoff_a_waypoint":
            self._start_phase(ctx, "dropoff_a_approach_waypoint", self._make_dropoff_a_approach_waypoint())
            return
        if self.phase == "dropoff_a_approach_waypoint":
            self._start_phase(ctx, "dropoff_a_search", SearchAndNavigateToAruco(marker_id=11, desired_distance=0.35, scan_mode=LookForArucoObjective.SCAN_MODE_SWEEP_90))
            return
        if self.phase == "dropoff_a_search":
            self._start_phase(ctx, "dropoff_a_drop", DropTargetObjective(delay_s=1.0))
            return
        if self.phase == "dropoff_a_drop":
            self._start_phase(ctx, "pickup_again_waypoint", self._make_pickup_again_waypoint())
            return
        if self.phase == "pickup_again_waypoint":
            self._start_phase(ctx, "cube53_search", self._make_cube53_search())
            return
        if self.phase == "cube53_search":
            self._start_phase(ctx, "cube53_grab", GrabArucoObjective(nav_mode=self.waypoint_nav_mode))
            return
        if self.phase == "cube53_grab":
            self._start_phase(ctx, "dropoff_d_waypoint", self._make_dropoff_d_waypoint())
            return
        if self.phase == "dropoff_d_waypoint":
            if self.cube53_search_failed:
                self._done = True
                return
            self._start_phase(ctx, "dropoff_d_search", self._make_dropoff_d_search())
            return
        if self.phase == "dropoff_d_search":
            self._start_phase(ctx, "dropoff_d_drop", DropTargetObjective(delay_s=1.0))
            return
        if self.phase == "dropoff_d_drop":
            self._done = True

    def tick(self, ctx):
        if self._done or self.phase_objective is None:
            return

        if self.phase == "cube20_search" and self.phase_started_at is not None:
            if time.time() - self.phase_started_at > self.CUBE20_SEARCH_TIMEOUT_S:
                self._advance_from_cube20_search(ctx)
                return

        if self.phase == "cube53_search" and self.phase_started_at is not None:
            if time.time() - self.phase_started_at > self.CUBE53_SEARCH_TIMEOUT_S:
                self._advance_from_cube53_search(ctx)
                return

        self.phase_objective.tick(ctx)
        if self.phase_objective.is_done(ctx):
            self.phase_objective.stop(ctx)
            self._advance_after_phase(ctx)

    def stop(self, ctx):
        if self.phase_objective is not None:
            self.phase_objective.stop(ctx)
