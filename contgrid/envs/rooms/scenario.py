"""RoomsScenario implementation."""

from typing import Any, Literal

import numpy as np
from gymnasium import spaces
from matplotlib.pylab import Generator
from numpy.typing import NDArray
from pydantic import BaseModel

from contgrid.core import (
    Agent,
    BaseScenario,
    Color,
    EntityState,
    Landmark,
    ResetConfig,
    SpawnPos,
    World,
    WorldConfig,
    rc2cell_pos,
)
from contgrid.core.typing import CellPosition, Position

from .spawn_strategies import (
    FixedSpawnConfig,
    FixedSpawnStrategy,
    PathGaussianConfig,
    PathGaussianSpawnStrategy,
    SpawnMethodConfig,
    SpawnStrategy,
    UniformRandomConfig,
    UniformRandomSpawnStrategy,
)


class RewardConfig(BaseModel):
    """Reward structure configuration."""

    step_penalty: float = 0.01
    sum_reward: bool = True


class ObjConfig(BaseModel):
    """Configuration for a single object (goal, lava, or hole)."""

    pos: Position | list[Position] | None = None
    reward: float = 0.0
    absorbing: bool = False
    room: Literal["top_left", "top_right", "bottom_left", "bottom_right"] | None = None


class SpawnConfig(BaseModel):
    """
    Configuration for spawning objects in the environment.

    Attributes
    ----------
    agent: Position | None
        The position of the agent or None for random spawning.
    goal: ObjConfig
        The configuration for the goal object.
    lavas: list[ObjConfig]
        A list of configurations for lava objects.
    holes: list[ObjConfig]
        A list of configurations for hole objects.
    doorways: dict[str, Position]
        A dictionary mapping doorway names to their positions.
    """

    agent: Position | None = None
    goal: ObjConfig = ObjConfig(pos=(9, 8), reward=1.0, absorbing=False)
    lavas: list[ObjConfig] = [
        ObjConfig(pos=(7, 8), reward=0.0, absorbing=False),
        ObjConfig(pos=(9, 9), reward=0.0, absorbing=False),
        ObjConfig(pos=(5, 10), reward=0.0, absorbing=False),
        ObjConfig(pos=(3, 7), reward=0.0, absorbing=False),
        ObjConfig(pos=(2, 4), reward=-1.0, absorbing=False),
        ObjConfig(pos=(3, 5), reward=-1.0, absorbing=False),
        ObjConfig(pos=(10, 4), reward=-1.0, absorbing=False),
        ObjConfig(pos=(8, 3), reward=-1.0, absorbing=False),
    ]
    holes: list[ObjConfig] = [
        ObjConfig(pos=(8, 9), reward=0.0, absorbing=False),
        ObjConfig(pos=(9, 7), reward=0.0, absorbing=False),
        ObjConfig(pos=(5, 8), reward=-1.0, absorbing=False),
        ObjConfig(pos=(4, 9), reward=-1.0, absorbing=False),
        ObjConfig(pos=(1, 5), reward=0.0, absorbing=False),
        ObjConfig(pos=(5, 3), reward=0.0, absorbing=False),
        ObjConfig(pos=(7, 4), reward=-1.0, absorbing=False),
        ObjConfig(pos=(9, 2), reward=-1.0, absorbing=False),
    ]
    doorways: dict[str, Position] = {
        "ld": (2, 6),
        "td": (6, 9),
        "rd": (9, 5),
        "bd": (6, 2),
    }
    agent_size: float = 0.25
    goal_size: float = 0.5
    lava_size: float = 0.5
    hole_size: float = 0.5
    goal_thr_dist: float | None = None
    lava_thr_dist: float | None = None
    hole_thr_dist: float | None = None
    agent_u_range: float = 10.0
    spawn_method: SpawnMethodConfig = FixedSpawnConfig()

    model_config = {"arbitrary_types_allowed": True}


class ObservationConfig(BaseModel):
    """Configuration for observations in the Rooms scenario."""

    lava_dist: Literal["closest", "all"] = "closest"
    hole_dist: Literal["closest", "all"] = "closest"


class RoomsScenarioConfig(BaseModel):
    """Configuration for the Rooms scenario."""

    spawn_config: SpawnConfig = SpawnConfig()
    reward_config: RewardConfig = RewardConfig(step_penalty=0.01)
    observation_config: ObservationConfig = ObservationConfig()


class RoomsScenario(BaseScenario[RoomsScenarioConfig, dict[str, NDArray[np.float64]]]):
    """Scenario for the Rooms environment with multiple rooms and doorways."""

    def __init__(
        self,
        config: RoomsScenarioConfig,
        world_config: WorldConfig,
    ) -> None:
        super().__init__(config, world_config)
        self.goal_thr_dist: float = (
            (config.spawn_config.goal_size + config.spawn_config.agent_size / 2)
            if config.spawn_config.goal_thr_dist is None
            else config.spawn_config.goal_thr_dist
        )
        self.lava_thr_dist: float = (
            (config.spawn_config.lava_size + config.spawn_config.agent_size / 2)
            if config.spawn_config.lava_thr_dist is None
            else config.spawn_config.lava_thr_dist
        )
        self.hole_thr_dist: float = (
            (config.spawn_config.hole_size + config.spawn_config.agent_size / 2)
            if config.spawn_config.hole_thr_dist is None
            else config.spawn_config.hole_thr_dist
        )

        self.doorways: dict[str, NDArray[np.float64]] = {
            name: np.array(pos, dtype=np.float64)
            for name, pos in config.spawn_config.doorways.items()
        }

        self.doorway_pos: NDArray[np.float64] = np.array(
            list(self.doorways.values()), dtype=np.float64
        )

        self.spawn_strategy = self._create_spawn_strategy()

        self.room_scale: float = max(
            self.world_config.grid.width, self.world_config.grid.height
        )

    def _create_spawn_strategy(self) -> SpawnStrategy:
        """Factory method to create appropriate spawn strategy."""
        assert self.config
        method_config = self.config.spawn_config.spawn_method

        if isinstance(method_config, FixedSpawnConfig):
            return FixedSpawnStrategy()
        elif isinstance(method_config, PathGaussianConfig):
            return PathGaussianSpawnStrategy(method_config)
        elif isinstance(method_config, UniformRandomConfig):
            return UniformRandomSpawnStrategy(method_config)
        else:
            raise ValueError(f"Unknown spawn method config: {type(method_config)}")

    def reset_world(self, world: World, np_random: np.random.Generator) -> None:
        """Reset world with agents first, then landmarks.

        This override ensures agents are positioned before obstacles are spawned,
        so that obstacle spawning can avoid overlapping with the agent.
        """
        self._pre_reset_world(world, np_random)
        # Reset agents FIRST so their positions are known
        world.agents = self.reset_agents(world, np_random)
        # Then reset landmarks, which can use agent positions to avoid overlaps
        world.landmarks = self.reset_landmarks(world, np_random)

    def init_agents(self, world: World, np_random=None) -> list[Agent]:
        assert self.config
        agent = Agent(
            name="agent_0",
            size=self.config.spawn_config.agent_size,
            color=Color.SKY_BLUE.name,
            u_range=self.config.spawn_config.agent_u_range,
        )
        return [agent]

    def init_landmarks(self, world: World, np_random=None) -> list[Landmark]:
        assert self.config
        self.init_free_cells: list[CellPosition] = [
            rc2cell_pos((p[0], p[1]), world.grid.height_cells)
            for p in np.argwhere(world.numeric_grid == 0).tolist()
        ]

        # Initialize the goal
        self.goal = Landmark(
            name="goal",
            size=self.config.spawn_config.goal_size,
            collide=False,
            color=Color.GREEN.name,
            state=EntityState(
                pos=np.array(self.config.spawn_config.goal.pos, dtype=np.float64)
            ),
            reset_config=ResetConfig(
                spawn_pos=self._format_spawn_pos(self.config.spawn_config.goal.pos)
            ),
        )

        # Initialize lava landmarks
        self.lavas = [
            Landmark(
                name=f"lava_{i}",
                size=self.config.spawn_config.lava_size,
                collide=False,
                color=Color.ORANGE.name,
                state=EntityState(pos=np.array(config.pos, dtype=np.float64)),
                hatch="//" if config.reward < 0 else "",
                reset_config=ResetConfig(spawn_pos=self._format_spawn_pos(config.pos)),
            )
            for i, config in enumerate(self.config.spawn_config.lavas)
        ]

        # Initialize holes
        self.holes = [
            Landmark(
                name=f"hole_{i}",
                size=self.config.spawn_config.hole_size,
                color=Color.PURPLE.name,
                collide=False,
                state=EntityState(pos=np.array(config.pos, dtype=np.float64)),
                hatch="//" if config.reward < 0 else "",
                reset_config=ResetConfig(spawn_pos=self._format_spawn_pos(config.pos)),
            )
            for i, config in enumerate(self.config.spawn_config.holes)
        ]

        return [self.goal] + self.lavas + self.holes

    def _format_spawn_pos(
        self, spawn_pos: SpawnPos
    ) -> CellPosition | list[CellPosition] | None:
        match spawn_pos:
            case list():
                return [(int(pos[0]), int(pos[1])) for pos in spawn_pos]
            case tuple():
                return (int(spawn_pos[0]), int(spawn_pos[1]))
            case None:
                return None
            case _:
                raise ValueError("Invalid spawn_pos type")

    def _pre_reset_world(self, world: World, np_random: Generator) -> None:
        """Pre-reset initialization.

        Initialize free_cells and remove fixed obstacle positions to prevent
        agent from spawning on top of obstacles.
        """
        assert self.config
        self.free_cells: list[CellPosition] = self.init_free_cells.copy()

        # Remove fixed obstacle positions from free_cells to prevent agent overlap
        # This must happen before agent spawning if we reset agents before obstacles

        # Remove fixed goal position
        goal_spawn_pos = self._format_spawn_pos(self.goal.reset_config.spawn_pos)
        if goal_spawn_pos is not None and not isinstance(goal_spawn_pos, list):
            if goal_spawn_pos in self.free_cells:
                self.free_cells.remove(goal_spawn_pos)

        # Remove fixed lava positions
        for lava in self.lavas:
            lava_spawn_pos = self._format_spawn_pos(lava.reset_config.spawn_pos)
            if lava_spawn_pos is not None:
                if isinstance(lava_spawn_pos, list):
                    for pos in lava_spawn_pos:
                        if pos in self.free_cells:
                            self.free_cells.remove(pos)
                else:
                    if lava_spawn_pos in self.free_cells:
                        self.free_cells.remove(lava_spawn_pos)

        # Remove fixed hole positions
        for hole in self.holes:
            hole_spawn_pos = self._format_spawn_pos(hole.reset_config.spawn_pos)
            if hole_spawn_pos is not None:
                if isinstance(hole_spawn_pos, list):
                    for pos in hole_spawn_pos:
                        if pos in self.free_cells:
                            self.free_cells.remove(pos)
                else:
                    if hole_spawn_pos in self.free_cells:
                        self.free_cells.remove(hole_spawn_pos)

    def reset_agents(self, world: World, np_random: np.random.Generator) -> list[Agent]:
        assert self.config
        for agent in world.agents:
            agent.reset(np_random)
            if self.config.spawn_config.agent is not None:
                agent.state.pos = np.array(
                    self.config.spawn_config.agent, dtype=np.float64
                )
            else:
                chose_cell_idx = np_random.choice(len(self.free_cells))
                new_pos: CellPosition = self.free_cells[chose_cell_idx]
                agent.state.pos = np.array([new_pos[0], new_pos[1]], dtype=np.float64)
                self.free_cells.pop(chose_cell_idx)

        return world.agents

    def reset_landmarks(
        self, world: World, np_random: np.random.Generator
    ) -> list[Landmark]:
        """Reset landmarks using the configured spawn strategy."""
        # Reset goal first
        goal_pos: CellPosition = self._choose_new_pos(
            self.goal.reset_config.spawn_pos, self.free_cells, np_random
        )
        if goal_pos in self.free_cells:
            self.free_cells.remove(goal_pos)

        self.goal_pos = np.array((goal_pos[0], goal_pos[1]), dtype=np.float64)
        self.goal.state.pos = self.goal_pos.copy()

        agent_pos = world.agents[0].state.pos if world.agents else None

        # Use strategy pattern for obstacles
        assert self.config
        max_spawn_attempts = 100

        # Spawn lavas
        for attempt in range(max_spawn_attempts):
            lava_positions = self.spawn_strategy.spawn_obstacles(
                num_obstacles=len(self.lavas),
                obstacle_type="lava",
                world=world,
                scenario=self,
                np_random=np_random,
                agent_pos=agent_pos,
                obstacle_configs=self.config.spawn_config.lavas,
            )
            if len(self.lavas) == len(lava_positions):
                break
        else:
            # Fallback to uniform random sampling for remaining lavas
            if len(lava_positions) < len(self.lavas):
                fallback_strategy = UniformRandomSpawnStrategy(
                    UniformRandomConfig(min_spacing=0.9)
                )
                remaining_configs = self.config.spawn_config.lavas[
                    len(lava_positions) :
                ]
                additional_positions = fallback_strategy.spawn_obstacles(
                    num_obstacles=len(self.lavas) - len(lava_positions),
                    obstacle_type="lava",
                    world=world,
                    scenario=self,
                    np_random=np_random,
                    agent_pos=agent_pos,
                    obstacle_configs=remaining_configs,
                )
                lava_positions.extend(additional_positions)

        for i, pos in enumerate(lava_positions):
            if i < len(self.lavas):
                self.lavas[i].state.pos = np.array(pos, dtype=np.float64)

        # Spawn holes
        for attempt in range(max_spawn_attempts):
            hole_positions = self.spawn_strategy.spawn_obstacles(
                num_obstacles=len(self.holes),
                obstacle_type="hole",
                world=world,
                scenario=self,
                np_random=np_random,
                agent_pos=agent_pos,
                obstacle_configs=self.config.spawn_config.holes,
            )
            if len(self.holes) == len(hole_positions):
                break
        else:
            # Fallback to uniform random sampling for remaining holes
            if len(hole_positions) < len(self.holes):
                fallback_strategy = UniformRandomSpawnStrategy(
                    UniformRandomConfig(min_spacing=0.9)
                )
                remaining_configs = self.config.spawn_config.holes[
                    len(hole_positions) :
                ]
                additional_positions = fallback_strategy.spawn_obstacles(
                    num_obstacles=len(self.holes) - len(hole_positions),
                    obstacle_type="hole",
                    world=world,
                    scenario=self,
                    np_random=np_random,
                    agent_pos=agent_pos,
                    obstacle_configs=remaining_configs,
                )
                hole_positions.extend(additional_positions)

        for i, pos in enumerate(hole_positions):
            if i < len(self.holes):
                self.holes[i].state.pos = np.array(pos, dtype=np.float64)

        # Store for observation
        self.lava_pos = (
            np.array(lava_positions, dtype=np.float64)
            if lava_positions
            else np.array([], dtype=np.float64).reshape(0, 2)
        )
        self.hole_pos = (
            np.array(hole_positions, dtype=np.float64)
            if hole_positions
            else np.array([], dtype=np.float64).reshape(0, 2)
        )
        self.wall_pos = np.array(
            world.wall_collision_checker.wall_centers, dtype=np.float64
        )

        return world.landmarks

    @classmethod
    def _choose_new_pos(
        cls, spawn_pos: SpawnPos, free_cells: list[CellPosition], np_random: Generator
    ) -> CellPosition:
        new_pos: CellPosition
        match spawn_pos:
            case tuple() as pos:
                new_pos = (int(pos[0]), int(pos[1]))
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

    def get_closest(
        self, pos: NDArray[np.float64], objects: NDArray[np.float64]
    ) -> tuple[float, int]:
        """Return the distance to the closest object from the given position."""
        if len(objects) == 0:
            return np.inf, -1
        dists = np.linalg.norm(objects - pos, axis=1)
        return np.min(dists), np.argmin(dists, axis=0)

    def observation(self, agent: Agent, world: World) -> dict[str, NDArray[np.float64]]:
        obs = {}
        obs["agent_pos"] = agent.state.pos.copy() / self.room_scale
        obs["goal_pos"] = (
            self.goal_pos.copy() - agent.state.pos.copy()
        ) / self.room_scale
        obs["lava_pos"] = (
            self.lava_pos.copy() - agent.state.pos.copy()
        ) / self.room_scale
        obs["hole_pos"] = (
            self.hole_pos.copy() - agent.state.pos.copy()
        ) / self.room_scale
        obs["doorway_pos"] = (
            self.doorway_pos.copy() - agent.state.pos.copy()
        ) / self.room_scale
        obs["goal_dist"] = (
            np.array([np.linalg.norm(agent.state.pos - self.goal_pos)])
            / self.room_scale
        )
        assert self.config
        if self.config.observation_config.lava_dist == "all":
            lava_dists = (
                np.linalg.norm(self.lava_pos - agent.state.pos, axis=1)
                / self.room_scale
            )
            obs["lava_dist"] = lava_dists.astype(np.float64)
        else:
            lava_dist, _ = self.get_closest(agent.state.pos, self.lava_pos)
            obs["lava_dist"] = np.array([lava_dist], dtype=np.float64) / self.room_scale
        if self.config.observation_config.hole_dist == "all":
            hole_dists = (
                np.linalg.norm(self.hole_pos - agent.state.pos, axis=1)
                / self.room_scale
            )
            obs["hole_dist"] = hole_dists.astype(np.float64)
        else:
            hole_dist, _ = self.get_closest(agent.state.pos, self.hole_pos)
            obs["hole_dist"] = np.array([hole_dist], dtype=np.float64) / self.room_scale
        wall_dist, _ = self.get_closest(agent.state.pos, self.wall_pos)
        obs["wall_dist"] = np.array([wall_dist], dtype=np.float64) / self.room_scale
        obs["doorway_dist"] = self._get_doorway_distances(agent) / self.room_scale
        return obs

    def observation_space(self, agent: Agent, world: World) -> spaces.Space:
        wall_limits = world.grid.wall_limits
        low_bound = -np.array((wall_limits.max_x, wall_limits.max_y))
        high_bound = np.array((wall_limits.max_x, wall_limits.max_y))
        num_lavas = len(self.lavas)
        num_holes = len(self.holes)
        max_dist = np.linalg.norm(high_bound - low_bound)

        assert self.config
        lava_dist_shape = (
            (1,)
            if self.config.observation_config.lava_dist == "closest"
            else (num_lavas,)
        )
        hole_dist_shape = (
            (1,)
            if self.config.observation_config.hole_dist == "closest"
            else (num_holes,)
        )

        return spaces.Dict(
            {
                "agent_pos": spaces.Box(
                    low=low_bound, high=high_bound, shape=(2,), dtype=np.float64
                ),
                "goal_pos": spaces.Box(
                    low=low_bound, high=high_bound, shape=(2,), dtype=np.float64
                ),
                "lava_pos": spaces.Box(
                    low=np.stack([low_bound] * num_lavas)
                    if num_lavas > 0
                    else np.array([], dtype=np.float64),
                    high=np.stack([high_bound] * num_lavas)
                    if num_lavas > 0
                    else np.array([], dtype=np.float64),
                    dtype=np.float64,
                ),
                "hole_pos": spaces.Box(
                    low=np.stack([low_bound] * num_holes)
                    if num_holes > 0
                    else np.array([], dtype=np.float64),
                    high=np.stack([high_bound] * num_holes)
                    if num_holes > 0
                    else np.array([], dtype=np.float64),
                    dtype=np.float64,
                ),
                "doorway_pos": spaces.Box(
                    low=np.array([low_bound] * len(self.doorways))
                    if len(self.doorways) > 0
                    else np.array([], dtype=np.float64),
                    high=np.array([high_bound] * len(self.doorways))
                    if len(self.doorways) > 0
                    else np.array([], dtype=np.float64),
                    dtype=np.float64,
                ),
                "goal_dist": spaces.Box(
                    low=0.0, high=max_dist, shape=(1,), dtype=np.float64
                ),
                "lava_dist": spaces.Box(
                    low=0.0, high=max_dist, shape=lava_dist_shape, dtype=np.float64
                ),
                "hole_dist": spaces.Box(
                    low=0.0, high=max_dist, shape=hole_dist_shape, dtype=np.float64
                ),
                "wall_dist": spaces.Box(
                    low=0.0, high=max_dist, shape=(1,), dtype=np.float64
                ),
                "doorway_dist": spaces.Box(
                    low=0.0,
                    high=max_dist,
                    shape=(len(self.doorways),),
                    dtype=np.float64,
                ),
            }
        )

    def reward(self, agent: Agent, world: World) -> float:
        lava_min_dist, lava_idx = self.get_closest(agent.state.pos, self.lava_pos)
        hole_min_dist, hole_idx = self.get_closest(agent.state.pos, self.hole_pos)
        goal_dist = np.linalg.norm(agent.state.pos - self.goal_pos)

        assert self.config

        if agent.terminated:
            reward = 0.0
        elif goal_dist < self.goal_thr_dist:
            reward = self.config.spawn_config.goal.reward
            agent.terminated = self.config.spawn_config.goal.absorbing
        elif lava_min_dist < self.lava_thr_dist and lava_idx != -1:
            reward = self.config.spawn_config.lavas[lava_idx].reward
            agent.terminated = self.config.spawn_config.lavas[lava_idx].absorbing
        elif hole_min_dist < self.hole_thr_dist and hole_idx != -1:
            reward = self.config.spawn_config.holes[hole_idx].reward
            agent.terminated = self.config.spawn_config.holes[hole_idx].absorbing
        else:
            reward = 0.0

        if not agent.terminated:
            reward -= self.config.reward_config.step_penalty

        return reward

    def info(self, agent: Agent, world: World) -> dict:
        info_dict: dict[str, Any] = {}
        info_dict["terminated"] = agent.terminated

        d_lv, _ = self.get_closest(agent.state.pos, self.lava_pos)
        d_hl, _ = self.get_closest(agent.state.pos, self.hole_pos)
        d_gl = np.linalg.norm(agent.state.pos - self.goal_pos)

        doorway_distances = {
            name: np.linalg.norm(agent.state.pos - pos)
            for name, pos in self.doorways.items()
        }

        info_dict["distances"] = {
            "lava": d_lv,
            "hole": d_hl,
            "goal": d_gl,
        } | doorway_distances

        info_dict["is_success"] = bool(d_gl < self.goal_thr_dist)
        info_dict["thresholds"] = {
            "goal": self.goal_thr_dist,
            "lava": self.lava_thr_dist,
            "hole": self.hole_thr_dist,
        }

        return info_dict

    def _get_doorway_distances(self, agent: Agent) -> NDArray[np.float64]:
        return np.array(
            [np.linalg.norm(agent.state.pos - pos) for pos in self.doorways.values()],
            dtype=np.float64,
        )
