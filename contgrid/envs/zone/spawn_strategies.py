"""Obstacle spawning strategies for the Rooms environment."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Discriminator

from contgrid.core import World
from contgrid.core.typing import Position

if TYPE_CHECKING:
    from .configs import ObjConfig
    from .scenario import ZoneScenario


class SpawnMode(str, Enum):
    """Enumeration of available obstacle spawning methods."""

    FIXED = "fixed"
    UNIFORM_RANDOM = "uniform_random"


class UniformRandomConfig(BaseModel):
    """Configuration for uniform random spawning."""

    mode: Literal[SpawnMode.UNIFORM_RANDOM] = SpawnMode.UNIFORM_RANDOM
    min_spacing: float = 0.5


class FixedSpawnConfig(BaseModel):
    """Configuration for fixed position spawning."""

    mode: Literal[SpawnMode.FIXED] = SpawnMode.FIXED


# Union type for all spawn method configs with discriminator
SpawnMethodConfig = Annotated[
    UniformRandomConfig | FixedSpawnConfig,
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
        scenario: "ZoneScenario",  # type: ignore
        np_random: np.random.Generator,
        agent_pos: NDArray[np.float64] | None = None,
        obstacle_configs: list["ObjConfig"] | None = None,
    ) -> list[Position]:
        """Generate obstacle positions."""
        pass


class FixedSpawnStrategy(SpawnStrategy):
    """Original fixed-position spawning strategy."""

    def spawn_obstacles(
        self,
        num_obstacles: int,
        obstacle_type: str,
        world: World,
        scenario: "ZoneScenario",  # type: ignore
        np_random: np.random.Generator,
        agent_pos: NDArray[np.float64] | None = None,
        obstacle_configs: list["ObjConfig"] | None = None,
    ) -> list[Position]:
        """Spawn obstacles at configured positions or random free cells."""
        positions: list[Position] = []

        obstacle_landmarks_by_type = {
            "yellow": scenario.yellow,
            "red": scenario.red,
            "white": scenario.white,
            "black": scenario.black,
        }
        obstacle_landmarks = obstacle_landmarks_by_type.get(obstacle_type, [])

        obstacle_radius = scenario.config.spawn_config.zone_size
        agent_radius = scenario.config.spawn_config.agent_size
        min_agent_distance = obstacle_radius + agent_radius

        for landmark in obstacle_landmarks[:num_obstacles]:
            max_attempts = 100
            for _attempt in range(max_attempts):
                new_pos = scenario._choose_new_pos(
                    landmark.reset_config.spawn_pos, scenario.free_cells, np_random
                )
                new_pos_array = np.array(new_pos, dtype=np.float64)

                # Check if position is valid (not overlapping with agent)
                is_valid = True
                if not world.wall_collision_checker.is_position_valid(
                    obstacle_radius,
                    world.contact_margin,
                    (float(new_pos_array[0]), float(new_pos_array[1])),
                ):
                    is_valid = False

                if agent_pos is not None:
                    if np.linalg.norm(new_pos_array - agent_pos) < min_agent_distance:
                        is_valid = False

                if is_valid:
                    positions.append((float(new_pos_array[0]), float(new_pos_array[1])))

                    # Remove chosen position from free-cells if it maps exactly to a cell.
                    if (
                        float(new_pos_array[0]).is_integer()
                        and float(new_pos_array[1]).is_integer()
                    ):
                        cell_pos = (int(new_pos_array[0]), int(new_pos_array[1]))
                        if cell_pos in scenario.free_cells:
                            scenario.free_cells.remove(cell_pos)
                    break

        return positions


class UniformRandomSpawnStrategy(SpawnStrategy):
    """Spawn obstacles uniformly at random in free space."""

    def __init__(self, config: UniformRandomConfig):
        self.config = config

    def spawn_obstacles(
        self,
        num_obstacles: int,
        obstacle_type: str,
        world: World,
        scenario: "ZoneScenario",  # type: ignore
        np_random: np.random.Generator,
        agent_pos: NDArray[np.float64] | None = None,
        obstacle_configs: list["ObjConfig"] | None = None,
    ) -> list[Position]:
        """Spawn obstacles uniformly in the walled area."""
        positions: list[Position] = []

        obstacle_radius = scenario.config.spawn_config.zone_size
        agent_radius = scenario.config.spawn_config.agent_size
        min_agent_distance = obstacle_radius + agent_radius

        wall_limits = world.grid.wall_limits
        min_x = wall_limits.min_x + obstacle_radius
        max_x = wall_limits.max_x - obstacle_radius
        min_y = wall_limits.min_y + obstacle_radius
        max_y = wall_limits.max_y - obstacle_radius

        if min_x >= max_x or min_y >= max_y:
            return positions

        max_attempts_per_obstacle = 200

        for _ in range(num_obstacles):
            found_valid_position = False

            for _attempt in range(max_attempts_per_obstacle):
                candidate = np.array(
                    [
                        np_random.uniform(min_x, max_x),
                        np_random.uniform(min_y, max_y),
                    ],
                    dtype=np.float64,
                )

                # Keep candidates inside the non-wall free region.
                if not world.wall_collision_checker.is_position_valid(
                    obstacle_radius,
                    world.contact_margin,
                    (float(candidate[0]), float(candidate[1])),
                ):
                    continue

                if agent_pos is not None and (
                    np.linalg.norm(candidate - agent_pos) < min_agent_distance
                ):
                    continue

                if any(
                    np.linalg.norm(candidate - np.array(pos, dtype=np.float64))
                    < self.config.min_spacing
                    for pos in positions
                ):
                    continue

                positions.append((float(candidate[0]), float(candidate[1])))
                found_valid_position = True
                break

            if not found_valid_position:
                break

        return positions
