"""Trajectory generators for moving preys and predators in PreyPredEnv."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from numpy.typing import NDArray

from contgrid.core import Landmark
from contgrid.core.typing import Position

from .configs import (
    CircularOrbitTrajectoryConfig,
    LinearPatrolTrajectoryConfig,
    LissajousTrajectoryConfig,
    StationaryTrajectoryConfig,
    TrajectoryConfig,
    WaypointPatrolTrajectoryConfig,
)


class BaseTrajectory(ABC):
    """Abstract base class for deterministic entity trajectories."""

    @abstractmethod
    def get_state(
        self, t: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Compute entity position and velocity at time t.

        Args:
            t: Simulation time in seconds.

        Returns:
            Tuple of (pos, vel) as 1D float64 arrays of shape (2,).
        """

    @abstractmethod
    def get_path_waypoints(self, num_points: int = 100) -> list[Position]:
        """Generate sampled waypoint coordinates along one full period for rendering.

        Args:
            num_points: Number of points to sample.

        Returns:
            List of (x, y) coordinates defining the trajectory path.
        """


class StationaryTrajectory(BaseTrajectory):
    """A stationary entity trajectory fixed at a constant position."""

    def __init__(self, position: Position) -> None:
        self.pos: NDArray[np.float64] = np.array(position, dtype=np.float64)
        self.vel: NDArray[np.float64] = np.zeros(2, dtype=np.float64)

    def get_state(
        self, t: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.pos.copy(), self.vel.copy()

    def get_path_waypoints(self, num_points: int = 100) -> list[Position]:
        return [(float(self.pos[0]), float(self.pos[1]))]


class LinearPatrolTrajectory(BaseTrajectory):
    """A linear back-and-forth oscillating patrol trajectory."""

    def __init__(
        self, config: LinearPatrolTrajectoryConfig, center: Position
    ) -> None:
        self.config = config
        self.center = np.array(center, dtype=np.float64)
        self.amplitude = config.amplitude
        self.speed = config.speed
        self.omega = (self.speed / self.amplitude) if self.amplitude > 0 else 0.0
        self.phase = config.phase

        d = 1.0 / math.sqrt(2.0)
        dirs = {"horizontal": (1.0, 0.0), "vertical": (0.0, 1.0), "diagonal": (d, d)}
        self.dir = np.array(dirs.get(config.axis, (1.0, 0.0)), dtype=np.float64)

    def get_state(
        self, t: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        angle = self.omega * t + self.phase
        disp = self.amplitude * math.sin(angle)
        speed = self.speed * math.cos(angle)
        pos = self.center + disp * self.dir
        vel = speed * self.dir
        return pos, vel

    def get_path_waypoints(self, num_points: int = 100) -> list[Position]:
        p_start = self.center - self.amplitude * self.dir
        p_end = self.center + self.amplitude * self.dir
        return [
            (float(p_start[0]), float(p_start[1])),
            (float(p_end[0]), float(p_end[1])),
        ]


class CircularOrbitTrajectory(BaseTrajectory):
    """A circular orbit trajectory around a fixed or dynamic center point."""

    def __init__(
        self,
        config: CircularOrbitTrajectoryConfig,
        default_center: Position,
        partner_entity: Landmark | None = None,
    ) -> None:
        self.config = config
        self.default_center = np.array(default_center, dtype=np.float64)
        self.partner_entity = partner_entity
        self.radius = config.radius
        self.speed = config.speed
        self.omega = (self.speed / self.radius) if self.radius > 0 else 0.0
        self.sign = -1.0 if config.clockwise else 1.0
        self.start_angle = config.start_angle

    def _get_center(self) -> NDArray[np.float64]:
        if (
            self.config.orbit_center == "partner_entity"
            and self.partner_entity is not None
        ):
            return self.partner_entity.state.pos
        return self.default_center

    def get_state(
        self, t: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        center = self._get_center()
        theta = self.start_angle + self.sign * self.omega * t
        pos = center + self.radius * np.array(
            [math.cos(theta), math.sin(theta)], dtype=np.float64
        )
        vel = self.speed * self.sign * np.array(
            [-math.sin(theta), math.cos(theta)], dtype=np.float64
        )
        return pos, vel

    def get_path_waypoints(self, num_points: int = 100) -> list[Position]:
        center = self._get_center()
        cx, cy = float(center[0]), float(center[1])
        return [
            (
                cx + self.radius * math.cos(2.0 * math.pi * i / num_points),
                cy + self.radius * math.sin(2.0 * math.pi * i / num_points),
            )
            for i in range(num_points + 1)
        ]


class WaypointPatrolTrajectory(BaseTrajectory):
    """A closed-loop or open patrol trajectory through waypoints."""

    def __init__(self, config: WaypointPatrolTrajectoryConfig) -> None:
        self.config = config
        self.waypoints = [
            np.array(wp, dtype=np.float64) for wp in config.waypoints
        ]
        if not self.waypoints:
            self.waypoints = [np.zeros(2, dtype=np.float64)]

        self.speed = config.speed
        self.loop = config.loop

        self.seg_vectors: list[NDArray[np.float64]] = []
        self.seg_lengths: list[float] = []
        n = len(self.waypoints)
        num_segs = n if (self.loop and n > 1) else max(1, n - 1)

        for i in range(num_segs):
            p1 = self.waypoints[i]
            p2 = self.waypoints[(i + 1) % n]
            diff = p2 - p1
            length = float(np.linalg.norm(diff))
            self.seg_vectors.append(diff)
            self.seg_lengths.append(length)

        self.total_length = max(1e-6, sum(self.seg_lengths))

    def get_state(
        self, t: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        if len(self.waypoints) <= 1 or self.total_length <= 1e-6:
            return self.waypoints[0].copy(), np.zeros(2, dtype=np.float64)

        d = (self.speed * t) % self.total_length
        accum = 0.0
        for i, length in enumerate(self.seg_lengths):
            if accum + length >= d or i == len(self.seg_lengths) - 1:
                seg_d = d - accum
                unit = (
                    self.seg_vectors[i] / length
                    if length > 1e-6
                    else np.zeros(2, dtype=np.float64)
                )
                pos = self.waypoints[i] + seg_d * unit
                vel = self.speed * unit
                return pos, vel
            accum += length

        return self.waypoints[0].copy(), np.zeros(2, dtype=np.float64)

    def get_path_waypoints(self, num_points: int = 100) -> list[Position]:
        pts = [(float(wp[0]), float(wp[1])) for wp in self.waypoints]
        if self.loop and len(pts) > 1:
            pts.append(pts[0])
        return pts


class LissajousTrajectory(BaseTrajectory):
    """A harmonic figure-8 / Lissajous trajectory."""

    def __init__(
        self, config: LissajousTrajectoryConfig, center: Position
    ) -> None:
        self.config = config
        self.center = np.array(center, dtype=np.float64)
        max_amp = max(config.amp_x, config.amp_y, 1e-3)
        self.omega_0 = config.speed / max_amp
        self.omega_x = config.freq_x * self.omega_0
        self.omega_y = config.freq_y * self.omega_0
        self.amp_x = config.amp_x
        self.amp_y = config.amp_y
        self.phase_delta = config.phase_delta

    def get_state(
        self, t: float
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        angle_x = self.omega_x * t + self.phase_delta
        angle_y = self.omega_y * t
        pos = self.center + np.array(
            [self.amp_x * math.sin(angle_x), self.amp_y * math.sin(angle_y)],
            dtype=np.float64,
        )
        vel = np.array(
            [
                self.amp_x * self.omega_x * math.cos(angle_x),
                self.amp_y * self.omega_y * math.cos(angle_y),
            ],
            dtype=np.float64,
        )
        return pos, vel

    def get_path_waypoints(self, num_points: int = 100) -> list[Position]:
        period = (2.0 * math.pi / self.omega_0) if self.omega_0 > 0 else 1.0
        return [
            (
                float(self.get_state(period * i / num_points)[0][0]),
                float(self.get_state(period * i / num_points)[0][1]),
            )
            for i in range(num_points + 1)
        ]


def create_trajectory(
    config: TrajectoryConfig,
    center: Position,
    partner_entity: Landmark | None = None,
) -> BaseTrajectory:
    """Factory creating a trajectory instance from configuration.

    Args:
        config: Polymorphic TrajectoryConfig instance.
        center: Reference center position (region center or spawn position).
        partner_entity: Optional partner landmark for partner-centered orbits.

    Returns:
        BaseTrajectory implementation instance.

    Raises:
        TypeError: If an unsupported trajectory config is provided.
    """
    match config:
        case StationaryTrajectoryConfig():
            return StationaryTrajectory(center)
        case LinearPatrolTrajectoryConfig():
            return LinearPatrolTrajectory(config, center)
        case CircularOrbitTrajectoryConfig():
            return CircularOrbitTrajectory(
                config, center, partner_entity=partner_entity
            )
        case WaypointPatrolTrajectoryConfig():
            return WaypointPatrolTrajectory(config)
        case LissajousTrajectoryConfig():
            return LissajousTrajectory(config, center)
        case _:
            raise TypeError(
                f"Unsupported trajectory config: {type(config).__name__}"
            )
