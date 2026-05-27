"""Obstacle spawning strategies for the Rooms environment."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Discriminator, Field

from contgrid.core import World
from contgrid.core.typing import Position

if TYPE_CHECKING:
    from ..configs import ObjConfig
    from ..scenario import ZoneScenario


class SpawnMode(str, Enum):
    """Enumeration of available obstacle spawning methods."""

    FIXED = "fixed"
    FIXED_RANDOM_SWAP = "fixed_random_swap"
    GAUSSIAN = "gaussian"
    UNIFORM_RANDOM = "uniform_random"


class UniformRandomConfig(BaseModel):
    """Configuration for uniform random spawning."""

    mode: Literal[SpawnMode.UNIFORM_RANDOM] = SpawnMode.UNIFORM_RANDOM
    min_spacing: float = 1.5


class GaussianSpawnConfig(BaseModel):
    """Configuration for Gaussian spawning centered at the map center."""

    mode: Literal[SpawnMode.GAUSSIAN] = SpawnMode.GAUSSIAN
    gaussian_std: float = 1.0
    min_spacing: float = 1.5


class FixedSpawnConfig(BaseModel):
    """Configuration for fixed position spawning."""

    mode: Literal[SpawnMode.FIXED] = SpawnMode.FIXED


class RandomSwapSpec(BaseModel):
    """Specification for one random-swap operation.

    Randomly selects ``num_swaps`` positions from the fixed positions of
    ``source_zone`` and spawns ``target_zone`` landmarks there, removing the
    corresponding ``source_zone`` landmarks.
    """

    source_zone: str
    target_zone: str
    num_swaps: int = 1


class FixedRandomSwapSpawnConfig(BaseModel):
    """Configuration for fixed spawning with random zone swaps and overlap removal.

    Extends fixed spawning by:
    1. Randomly placing ``target_zone`` landmarks at positions drawn from
       ``source_zone``'s fixed locations (and removing the displaced source landmarks).
    2. Removing any zone of type ``a`` that overlaps with a zone of type ``b``
       for each ``(a, b)`` pair in ``remove_overlapping``.
    """

    mode: Literal[SpawnMode.FIXED_RANDOM_SWAP] = SpawnMode.FIXED_RANDOM_SWAP
    swaps: list[RandomSwapSpec] = Field(default_factory=list)


# Union type for all spawn method configs with discriminator
SpawnMethodConfig = Annotated[
    GaussianSpawnConfig
    | UniformRandomConfig
    | FixedSpawnConfig
    | FixedRandomSwapSpawnConfig,
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

        # Use zone_size BaseModel class attribute matching obstacle_type
        obstacle_radius = getattr(scenario.zone_sizes, obstacle_type)
        agent_radius = scenario.config.spawn_config.agent_size
        min_agent_distance = obstacle_radius + agent_radius

        for i, landmark in enumerate(obstacle_landmarks[:num_obstacles]):
            has_fixed_obstacle_pos = False
            if obstacle_configs is not None and i < len(obstacle_configs):
                if isinstance(obstacle_configs[i].pos, tuple):
                    has_fixed_obstacle_pos = True
            elif isinstance(landmark.reset_config.spawn_pos, tuple):
                has_fixed_obstacle_pos = True

            has_fixed_agent_pos = scenario.config.spawn_config.agent is not None
            skip_agent_check = has_fixed_agent_pos and has_fixed_obstacle_pos

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

                if agent_pos is not None and not skip_agent_check:
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

    @staticmethod
    def _valid_position_array(pos: NDArray[np.float64]) -> NDArray[np.float64] | None:
        pos_array = np.asarray(pos, dtype=np.float64)
        if pos_array.shape != (2,) or not np.isfinite(pos_array).all():
            return None
        return pos_array

    def _get_existing_zone_positions(
        self,
        obstacle_type: str,
        scenario: "ZoneScenario",
    ) -> list[NDArray[np.float64]]:
        """Return positions of already-spawned colors for cross-color spacing checks."""
        spawn_order = ("yellow", "red", "white", "black")
        if obstacle_type not in spawn_order:
            return []

        landmarks_by_type = {
            "yellow": scenario.yellow,
            "red": scenario.red,
            "white": scenario.white,
            "black": scenario.black,
        }
        spawned_types = spawn_order[: spawn_order.index(obstacle_type)]

        existing_positions: list[NDArray[np.float64]] = []
        for zone_type in spawned_types:
            for landmark in landmarks_by_type[zone_type]:
                pos_array = self._valid_position_array(landmark.state.pos)
                if pos_array is not None:
                    existing_positions.append(pos_array)

        return existing_positions

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
        position_arrays: list[NDArray[np.float64]] = []

        obstacle_radius = getattr(scenario.zone_sizes, obstacle_type)
        agent_radius = scenario.config.spawn_config.agent_size
        min_agent_distance = obstacle_radius + agent_radius
        existing_zone_positions = self._get_existing_zone_positions(
            obstacle_type, scenario
        )

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
                    np.linalg.norm(candidate - pos_array) < self.config.min_spacing
                    for pos_array in position_arrays
                ):
                    continue

                if any(
                    np.linalg.norm(candidate - existing_pos) < self.config.min_spacing
                    for existing_pos in existing_zone_positions
                ):
                    continue

                positions.append((float(candidate[0]), float(candidate[1])))
                position_arrays.append(candidate)
                found_valid_position = True
                break

            if not found_valid_position:
                break

        return positions


class GaussianSpawnStrategy(SpawnStrategy):
    """Spawn obstacles from a Gaussian distribution centered on the map."""

    def __init__(self, config: GaussianSpawnConfig):
        self.config = config

    @staticmethod
    def _valid_position_array(pos: NDArray[np.float64]) -> NDArray[np.float64] | None:
        pos_array = np.asarray(pos, dtype=np.float64)
        if pos_array.shape != (2,) or not np.isfinite(pos_array).all():
            return None
        return pos_array

    def _get_existing_zone_positions(
        self,
        obstacle_type: str,
        scenario: "ZoneScenario",
    ) -> list[NDArray[np.float64]]:
        """Return positions of already-spawned zones (all colors) for cross-color spacing checks."""
        all_spawned = getattr(scenario, "_all_spawned_positions", None)
        if isinstance(all_spawned, list):
            return all_spawned
        return []

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
        """Spawn obstacles with a Gaussian bias toward the center of the map."""
        positions: list[Position] = []
        position_arrays: list[NDArray[np.float64]] = []

        obstacle_radius = getattr(scenario.zone_sizes, obstacle_type)
        agent_radius = scenario.config.spawn_config.agent_size
        min_agent_distance = obstacle_radius + agent_radius
        existing_zone_positions = self._get_existing_zone_positions(
            obstacle_type, scenario
        )

        wall_limits = world.grid.wall_limits
        center = np.array(
            [
                (wall_limits.min_x + wall_limits.max_x) / 2.0,
                (wall_limits.min_y + wall_limits.max_y) / 2.0,
            ],
            dtype=np.float64,
        )

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
                    np_random.normal(loc=center, scale=self.config.gaussian_std),
                    dtype=np.float64,
                )

                if candidate[0] < min_x or candidate[0] > max_x:
                    continue
                if candidate[1] < min_y or candidate[1] > max_y:
                    continue

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
                    np.linalg.norm(candidate - pos_array) < self.config.min_spacing
                    for pos_array in position_arrays
                ):
                    continue

                if any(
                    np.linalg.norm(candidate - existing_pos) < self.config.min_spacing
                    for existing_pos in existing_zone_positions
                ):
                    continue

                positions.append((float(candidate[0]), float(candidate[1])))
                position_arrays.append(candidate)
                found_valid_position = True
                break

            if not found_valid_position:
                break

        return positions


class FixedRandomSwapSpawnStrategy(SpawnStrategy):
    """Fixed spawning with random zone swaps and overlap removal.

    1. Delegates to ``FixedSpawnStrategy`` to place all zones at configured positions.
    2. For each :class:`RandomSwapSpec`: randomly selects source-zone positions,
       creates target-zone landmarks there, and removes the displaced source landmarks.

    """

    def __init__(self, config: FixedRandomSwapSpawnConfig) -> None:
        self.config = config
        self._fixed_strategy = FixedSpawnStrategy()

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
        """Spawn obstacles at fixed positions — swap & overlap logic is in post_spawn."""
        return self._fixed_strategy.spawn_obstacles(
            num_obstacles=num_obstacles,
            obstacle_type=obstacle_type,
            world=world,
            scenario=scenario,
            np_random=np_random,
            agent_pos=agent_pos,
            obstacle_configs=obstacle_configs,
        )

    # ------------------------------------------------------------------ #
    #  Post-spawn helpers called by SpawnManager after all colours placed #
    # ------------------------------------------------------------------ #

    def post_spawn(
        self,
        scenario: "ZoneScenario",  # type: ignore
        spawned_positions: dict[str, list[Position]],
        np_random: np.random.Generator,
    ) -> dict[str, list[Position]]:
        """Apply random swaps and overlap removal after fixed spawning."""
        zone_landmarks: dict[str, list[Any]] = {
            "yellow": list(scenario.yellow),
            "red": list(scenario.red),
            "white": list(scenario.white),
            "black": list(scenario.black),
        }

        # --- 1. Random swaps ---
        for swap in self.config.swaps:
            src = swap.source_zone
            tgt = swap.target_zone
            src_landmarks = zone_landmarks.get(src, [])
            tgt_landmarks = zone_landmarks.get(tgt, [])
            src_positions = spawned_positions.get(src, [])

            if not src_landmarks or swap.num_swaps <= 0:
                continue

            n = min(swap.num_swaps, len(src_landmarks))
            chosen_indices = list(
                np_random.choice(len(src_landmarks), size=n, replace=False)
            )
            # Sort descending so removals don't shift indices
            chosen_indices.sort(reverse=True)

            for idx in chosen_indices:
                src_lm = src_landmarks[idx]
                pos_array = src_lm.state.pos.copy()
                pos_tuple: Position = (float(pos_array[0]), float(pos_array[1]))

                # Reuse the source landmark as a target landmark (re-color it)
                tgt_size = getattr(scenario.zone_sizes, tgt, 0.5)
                src_lm.name = f"{tgt}_{len(tgt_landmarks)}"
                src_lm.size = tgt_size
                from contgrid.core import Color

                src_lm.color = Color[tgt.upper()]
                tgt_landmarks.append(src_lm)

                if tgt not in spawned_positions:
                    spawned_positions[tgt] = []
                spawned_positions[tgt].append(pos_tuple)

                # Remove the source landmark
                src_landmarks.pop(idx)
                if idx < len(src_positions):
                    src_positions.pop(idx)

            zone_landmarks[src] = src_landmarks
            zone_landmarks[tgt] = tgt_landmarks
            spawned_positions[src] = src_positions

        # --- 3. Sync scenario landmark lists ---
        scenario.yellow = zone_landmarks["yellow"]
        scenario.red = zone_landmarks["red"]
        scenario.white = zone_landmarks["white"]
        scenario.black = zone_landmarks["black"]

        return spawned_positions
