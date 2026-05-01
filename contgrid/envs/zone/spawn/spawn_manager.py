"""Spawn manager for the Zone environment.

Encapsulates all obstacle spawning orchestration logic,
delegating to individual SpawnStrategy implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from contgrid.core import Landmark, SpawnPos, World
from contgrid.core.typing import CellPosition, Position

from .spawn_strategies import (
    FixedSpawnConfig,
    FixedSpawnStrategy,
    GaussianSpawnConfig,
    GaussianSpawnStrategy,
    SpawnStrategy,
    UniformRandomConfig,
    UniformRandomSpawnStrategy,
)

if TYPE_CHECKING:
    from ..configs import SpawnConfig


class SpawnManager:
    """Manages obstacle spawning for the Zone environment.

    Owns the spawn strategy and provides methods for position formatting,
    position selection, and full obstacle spawning orchestration.
    """

    def __init__(self, spawn_config: SpawnConfig) -> None:
        self.spawn_config = spawn_config
        self.spawn_strategy = self._create_spawn_strategy()

    def _create_spawn_strategy(self) -> SpawnStrategy:
        """Factory method to create appropriate spawn strategy."""
        method_config = self.spawn_config.spawn_method

        strategy_map = {
            FixedSpawnConfig: FixedSpawnStrategy,
            GaussianSpawnConfig: GaussianSpawnStrategy,
            UniformRandomConfig: UniformRandomSpawnStrategy,
        }

        for config_type, strategy_class in strategy_map.items():
            if isinstance(method_config, config_type):
                return (
                    strategy_class()
                    if config_type == FixedSpawnConfig
                    else strategy_class(method_config)
                )

        raise ValueError(f"Unknown spawn method config: {type(method_config)}")

    @staticmethod
    def format_spawn_pos(
        spawn_pos: SpawnPos,
    ) -> Position | list[Position] | None:
        """Format a SpawnPos into Position, list of Positions, or None."""
        match spawn_pos:
            case list():
                return [(float(pos[0]), float(pos[1])) for pos in spawn_pos]
            case tuple():
                return (float(spawn_pos[0]), float(spawn_pos[1]))
            case None:
                return None
            case _:
                raise ValueError("Invalid spawn_pos type")

    @classmethod
    def choose_new_pos(
        cls,
        spawn_pos: SpawnPos,
        free_cells: list[CellPosition],
        np_random: np.random.Generator,
    ) -> Position:
        """Choose a new position from the spawn specification and available free cells."""
        new_pos: Position
        match spawn_pos:
            case tuple() as pos:
                new_pos = (pos[0], pos[1])
            case list() as pos_list if pos_list:
                cell_pos_list: list[CellPosition] = pos_list  # type: ignore
                available_pos: list[CellPosition] = list(
                    set(cell_pos_list) & set(free_cells)
                )
                if available_pos:
                    chose_cell_idx = np_random.choice(len(available_pos))
                    new_pos = available_pos[chose_cell_idx]
                else:
                    chose_cell_idx = np_random.choice(len(free_cells))
                    new_pos = free_cells[chose_cell_idx]
            case None:
                chose_cell_idx = np_random.choice(len(free_cells))
                new_pos = free_cells[chose_cell_idx]
            case _:
                raise ValueError("Invalid spawn_pos type")
        return new_pos

    def spawn_obstacles(
        self,
        scenario: "ZoneScenario",  # type: ignore  # noqa: F821
        world: World,
        np_random: np.random.Generator,
        agent_pos: NDArray[np.float64] | None,
    ) -> dict[str, NDArray[np.float64]]:
        """Spawn all obstacles (zones) using the configured strategy.

        Updates ``scenario._spawned_zone_positions`` and
        ``scenario._all_spawned_positions`` during spawning for
        cross-zone spacing checks.

        Returns a dict mapping zone color to an array of spawned positions.
        """
        max_spawn_attempts = 100

        spawn_specs = [
            ("yellow", scenario.yellow, scenario.config.spawn_config.yellow_zone),
            ("red", scenario.red, scenario.config.spawn_config.red_zone),
            ("white", scenario.white, scenario.config.spawn_config.white_zone),
            ("black", scenario.black, scenario.config.spawn_config.black_zone),
        ]

        if isinstance(self.spawn_strategy, GaussianSpawnStrategy):
            flat_zones = []
            for color_type, obstacles, configs in spawn_specs:
                for i, (landmark, config) in enumerate(zip(obstacles, configs)):
                    flat_zones.append((color_type, landmark, config))
            flat_zones_shuffled = [
                flat_zones[idx] for idx in np_random.permutation(len(flat_zones))
            ]
            spawned_positions: dict[str, list[Position]] = {
                "yellow": [],
                "red": [],
                "white": [],
                "black": [],
            }
            for color_type, landmark, config in flat_zones_shuffled:
                positions = self.spawn_strategy.spawn_obstacles(
                    num_obstacles=1,
                    obstacle_type=color_type,
                    world=world,
                    scenario=scenario,
                    np_random=np_random,
                    agent_pos=agent_pos,
                    obstacle_configs=[config],
                )
                if positions:
                    pos_array = np.array(positions[0], dtype=np.float64)
                    landmark.state.pos = pos_array
                    spawned_positions[color_type].append(positions[0])
                    scenario._spawned_zone_positions[color_type].append(pos_array)
                    scenario._all_spawned_positions.append(pos_array)
        else:
            spawned_positions: dict[str, list[Position]] = {}
            for obstacle_type, obstacles, obstacle_configs in spawn_specs:
                positions = self._spawn_obstacle_type(
                    scenario=scenario,
                    obstacle_type=obstacle_type,
                    obstacles=obstacles,
                    obstacle_configs=obstacle_configs,
                    world=world,
                    np_random=np_random,
                    agent_pos=agent_pos,
                    max_attempts=max_spawn_attempts,
                )
                spawned_positions[obstacle_type] = positions
                scenario._spawned_zone_positions[obstacle_type] = [
                    np.array(pos, dtype=np.float64) for pos in positions
                ]
                scenario._all_spawned_positions.extend(
                    [np.array(pos, dtype=np.float64) for pos in positions]
                )

        return {
            color: np.array(pos_list, dtype=np.float64)
            for color, pos_list in spawned_positions.items()
        }

    def _spawn_obstacle_type(
        self,
        scenario: "ZoneScenario",  # type: ignore  # noqa: F821
        obstacle_type: str,
        obstacles: list[Landmark],
        obstacle_configs: list,
        world: World,
        np_random: np.random.Generator,
        agent_pos: NDArray[np.float64] | None,
        max_attempts: int,
    ) -> list[Position]:
        """Spawn a specific type of obstacle with fallback strategy."""
        # Try spawning with configured strategy
        positions: list[Position] = []
        for _ in range(max_attempts):
            positions = self.spawn_strategy.spawn_obstacles(
                num_obstacles=len(obstacles),
                obstacle_type=obstacle_type,
                world=world,
                scenario=scenario,
                np_random=np_random,
                agent_pos=agent_pos,
                obstacle_configs=obstacle_configs,
            )
            if len(obstacles) == len(positions):
                break

        # Fallback to uniform random sampling if needed
        if len(positions) < len(obstacles):
            fallback_strategy = UniformRandomSpawnStrategy(
                UniformRandomConfig(min_spacing=0.9)
            )
            remaining_configs = obstacle_configs[len(positions) :]
            additional_positions = fallback_strategy.spawn_obstacles(
                num_obstacles=len(obstacles) - len(positions),
                obstacle_type=obstacle_type,
                world=world,
                scenario=scenario,
                np_random=np_random,
                agent_pos=agent_pos,
                obstacle_configs=remaining_configs,
            )
            positions.extend(additional_positions)

        # Update obstacle positions with safety check for agent distance
        obstacle_radius = self.spawn_config.zone_size
        agent_radius = self.spawn_config.agent_size
        min_agent_distance = obstacle_radius + agent_radius

        valid_positions: list[Position] = []
        for i, pos in enumerate(positions):
            if i < len(obstacles):
                pos_array = np.array(pos, dtype=np.float64)
                position_is_valid = True

                # Verify distance from agent and re-spawn if too close
                if agent_pos is not None:
                    dist = np.linalg.norm(pos_array - agent_pos)
                    if dist < min_agent_distance:
                        # Find a position that's far enough from agent
                        found_valid_cell = False
                        for cell in scenario.free_cells:
                            cell_dist = np.linalg.norm(np.array(cell) - agent_pos)
                            if cell_dist >= min_agent_distance:
                                pos_array = np.array(cell, dtype=np.float64)
                                if cell in scenario.free_cells:
                                    scenario.free_cells.remove(cell)
                                found_valid_cell = True
                                break

                        if not found_valid_cell:
                            # Could not find a valid position - skip this obstacle
                            position_is_valid = False

                if position_is_valid:
                    obstacles[i].state.pos = pos_array
                    valid_positions.append(tuple(pos_array))

        return valid_positions
