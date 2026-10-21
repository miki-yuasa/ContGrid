"""Fixed and uniform obstacle spawning strategies for Rooms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from contgrid.core import World
from contgrid.core.typing import Position
from contgrid.envs.rooms.topology import RoomTopology

if TYPE_CHECKING:
    from contgrid.core.entities import Landmark
    from contgrid.envs.rooms.configs import ObjConfig
    from contgrid.envs.rooms.scenario import RoomsScenario

    from .strategies import UniformRandomConfig


class FixedSpawnStrategy:
    """Fixed-position obstacle spawning strategy."""

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
        """Spawn obstacles at configured positions or random free cells."""
        positions: list[Position] = []
        if scenario.config is None:
            raise ValueError("Scenario config must not be None.")

        landmarks = (
            scenario.lavas if obstacle_type == "lava" else scenario.holes
        )
        sc = scenario.config.spawn_config
        r = sc.lava_size if obstacle_type == "lava" else sc.hole_size
        min_dist = r + sc.agent_size

        for lm in landmarks[:num_obstacles]:
            pos = _find_fixed_cell(lm, scenario, np_random, agent_pos, min_dist)
            if pos is not None:
                positions.append(pos)
                cell = (int(pos[0]), int(pos[1]))
                if cell in scenario.free_cells:
                    scenario.free_cells.remove(cell)

        return positions


def _find_fixed_cell(
    lm: Landmark,
    scenario: RoomsScenario,
    np_random: np.random.Generator,
    agent_pos: NDArray[np.float64] | None,
    min_dist: float,
) -> Position | None:
    """Find a valid spawn cell for a fixed-position landmark."""
    for _ in range(100):
        pos = scenario._choose_new_pos(
            lm.reset_config.spawn_pos, scenario.free_cells, np_random
        )
        if agent_pos is not None:
            dist = float(np.linalg.norm(np.array(pos) - agent_pos))
            if dist < min_dist:
                continue
        return pos
    return None


class UniformRandomSpawnStrategy:
    """Uniform-random obstacle spawning strategy."""

    def __init__(self, config: UniformRandomConfig) -> None:
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
        """Spawn obstacles uniformly in free cells."""
        positions: list[Position] = []
        if scenario.config is None:
            raise ValueError("Scenario config must not be None.")
        if self.topology is None:
            self.topology = RoomTopology(scenario.config.spawn_config.doorways)

        sc = scenario.config.spawn_config
        r = sc.lava_size if obstacle_type == "lava" else sc.hole_size
        min_dist = r + sc.agent_size

        for idx in range(num_obstacles):
            req = (
                obstacle_configs[idx].room
                if obstacle_configs and idx < len(obstacle_configs)
                else None
            )
            avail = [
                c
                for c in scenario.free_cells
                if req is None or self.topology.get_room(c) == req
            ]
            if agent_pos is not None:
                avail = [
                    c
                    for c in avail
                    if float(np.linalg.norm(np.array(c) - agent_pos))
                    >= min_dist
                ]
            if avail:
                chosen = avail[int(np_random.choice(len(avail)))]
                positions.append(chosen)
                if chosen in scenario.free_cells:
                    scenario.free_cells.remove(chosen)

        return positions
