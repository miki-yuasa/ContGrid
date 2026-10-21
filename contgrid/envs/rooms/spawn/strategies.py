"""Obstacle spawning strategies for the Rooms environment."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Discriminator

from contgrid.core import World
from contgrid.core.grid_rust import WallCollisionCheckerAccelerated
from contgrid.core.typing import Position
from contgrid.envs.rooms.topology import (
    LineSegment,
    RoomTopology,
    get_relevant_path_segments,
)

from .reference import (
    _get_existing_obstacles,
    sample_path_gaussian_reference,
)
from .simple import FixedSpawnStrategy, UniformRandomSpawnStrategy

if TYPE_CHECKING:
    from contgrid.envs.rooms.configs import ObjConfig
    from contgrid.envs.rooms.scenario import RoomsScenario


class SpawnMode(StrEnum):
    """Enumeration of available obstacle spawning methods."""

    FIXED = "fixed"
    PATH_GAUSSIAN = "path_gaussian"
    UNIFORM_RANDOM = "uniform_random"


class PathGaussianConfig(BaseModel):
    """Configuration for path-based Gaussian spawning."""

    mode: Literal[SpawnMode.PATH_GAUSSIAN] = SpawnMode.PATH_GAUSSIAN
    gaussian_std: float = 0.6
    min_spacing: float = 1.0
    edge_buffer: float = 0.05
    include_agent_paths: bool = True
    use_rust: bool = True


class UniformRandomConfig(BaseModel):
    """Configuration for uniform random spawning."""

    mode: Literal[SpawnMode.UNIFORM_RANDOM] = SpawnMode.UNIFORM_RANDOM
    min_spacing: float = 0.5


class FixedSpawnConfig(BaseModel):
    """Configuration for fixed position spawning."""

    mode: Literal[SpawnMode.FIXED] = SpawnMode.FIXED


# Union type for all spawn method configs with discriminator.
SpawnMethodConfig = Annotated[
    PathGaussianConfig | UniformRandomConfig | FixedSpawnConfig,
    Discriminator("mode"),
]


class SpawnStrategy(ABC):
    """Abstract base class for object spawning strategies."""

    @abstractmethod
    def spawn_obstacles(
        self,
        num_obstacles: int,
        obstacle_type: str,
        world: World,
        scenario: RoomsScenario,
        np_random: np.random.Generator,
        agent_pos: NDArray[np.float64] | None = None,
        obstacle_configs: Sequence[ObjConfig] | None = None,
    ) -> list[Position]:
        """Generate obstacle positions."""


class PathGaussianSpawnStrategy(SpawnStrategy):
    """Obstacle spawning strategy near shortest paths with Gaussian noise."""

    def __init__(self, config: PathGaussianConfig) -> None:
        self.config = config
        self.topology: RoomTopology | None = None

    def spawn_obstacles(
        self,
        num_obstacles: int,
        obstacle_type: str,
        world: World,
        scenario: RoomsScenario,
        np_random: np.random.Generator,
        agent_pos: NDArray[np.float64] | None = None,
        obstacle_configs: Sequence[ObjConfig] | None = None,
    ) -> list[Position]:
        """Spawn obstacles along relevant paths with Gaussian perturbation."""
        if num_obstacles == 0:
            return []
        if scenario.config is None:
            raise ValueError("Scenario config must not be None.")
        if self.topology is None:
            self.topology = RoomTopology(scenario.config.spawn_config.doorways)

        # Get relevant path segments.
        if agent_pos is None and self.config.include_agent_paths:
            segs = _get_doorway_segments_only(
                scenario.doorways, scenario.goal_pos, self.topology
            )
        else:
            p = agent_pos if agent_pos is not None else scenario.goal_pos
            segs = get_relevant_path_segments(
                p,
                scenario.goal_pos,
                scenario.doorways,
                self.topology,
                world.grid,
            )

        if not segs:
            return FixedSpawnStrategy().spawn_obstacles(
                num_obstacles,
                obstacle_type,
                world,
                scenario,
                np_random,
                agent_pos,
                obstacle_configs,
            )

        rooms = [
            obstacle_configs[i].room
            if obstacle_configs and i < len(obstacle_configs)
            else None
            for i in range(num_obstacles)
        ]
        if not self.config.use_rust:
            if self.topology is None:
                raise ValueError("Topology must be initialized before spawn.")
            return sample_path_gaussian_reference(
                self.config,
                num_obstacles,
                obstacle_type,
                world,
                scenario,
                np_random,
                agent_pos,
                rooms,
                segs,
                self.topology.room_boundaries,
            )

        return self._spawn_with_rust(
            num_obstacles,
            obstacle_type,
            world,
            scenario,
            np_random,
            agent_pos,
            rooms,
            segs,
        )

    def _spawn_with_rust(
        self,
        num_obstacles: int,
        obstacle_type: str,
        world: World,
        scenario: RoomsScenario,
        np_random: np.random.Generator,
        agent_pos: NDArray[np.float64] | None,
        obstacle_rooms: Sequence[str | None],
        segments: Sequence[LineSegment],
    ) -> list[Position]:
        """Spawn obstacles using native Rust acceleration."""
        if scenario.config is None or self.topology is None:
            raise ValueError("Scenario config and topology must not be None.")

        coords = [
            [
                float(s.start[0]),
                float(s.start[1]),
                float(s.end[0]),
                float(s.end[1]),
            ]
            for s in segments
            if s.length() > 1e-6
        ]
        r_bounds = [
            _to_bound_rect(r, self.topology.room_boundaries)
            for r in obstacle_rooms
        ]
        lim = world.grid.wall_limits
        gb = [
            float(lim.min_x),
            float(lim.max_x),
            float(lim.min_y),
            float(lim.max_y),
        ]
        gp = (float(scenario.goal_pos[0]), float(scenario.goal_pos[1]))
        sc = scenario.config.spawn_config
        obs_r = sc.lava_size if obstacle_type == "lava" else sc.hole_size
        a_pos = (
            (float(agent_pos[0]), float(agent_pos[1]))
            if agent_pos is not None
            else None
        )

        checker = world.wall_collision_checker
        if not isinstance(checker, WallCollisionCheckerAccelerated):
            return sample_path_gaussian_reference(
                self.config,
                num_obstacles,
                obstacle_type,
                world,
                scenario,
                np_random,
                agent_pos,
                obstacle_rooms,
                segments,
                self.topology.room_boundaries,
            )

        return checker.sample_path_gaussian(
            segments=coords,
            num_obstacles=num_obstacles,
            obstacle_radius=obs_r,
            gaussian_std=self.config.gaussian_std,
            min_spacing=self.config.min_spacing,
            edge_buffer=self.config.edge_buffer,
            grid_bounds=gb,
            room_bounds=r_bounds,
            existing_obstacles=_get_existing_obstacles(scenario),
            goal_pos=gp,
            goal_clearance=scenario.goal_thr_dist + 0.5,
            agent_pos=a_pos,
            min_agent_dist=obs_r + sc.agent_size,
            seed=int(np_random.integers(0, 2**63 - 1)),
        )


def _to_bound_rect(
    room: str | None, room_boundaries: Mapping[str, Mapping[str, float]]
) -> list[float] | None:
    """Extract rectangle bounds as list of floats for a given room."""
    if room and (b := room_boundaries.get(room)):
        return [
            float(b["min_x"]),
            float(b["max_x"]),
            float(b["min_y"]),
            float(b["max_y"]),
        ]
    return None


def _get_doorway_segments_only(
    doorways: Mapping[str, NDArray[np.float64]],
    goal_pos: NDArray[np.float64],
    topology: RoomTopology,
) -> list[LineSegment]:
    """Get doorway segments and segment to goal."""
    segs = [
        LineSegment(doorways[d1].copy(), doorways[d2].copy())
        for d1, d2 in topology.get_neighbor_pairs()
        if d1 in doorways and d2 in doorways
    ]
    dist_by_name = {
        k: float(np.linalg.norm(goal_pos - v)) for k, v in doorways.items()
    }
    closest = min(dist_by_name, key=dist_by_name.__getitem__)
    segs.append(LineSegment(doorways[closest].copy(), goal_pos.copy()))
    return segs


__all__ = [
    "FixedSpawnConfig",
    "FixedSpawnStrategy",
    "PathGaussianConfig",
    "PathGaussianSpawnStrategy",
    "SpawnMethodConfig",
    "SpawnMode",
    "SpawnStrategy",
    "UniformRandomConfig",
    "UniformRandomSpawnStrategy",
]
