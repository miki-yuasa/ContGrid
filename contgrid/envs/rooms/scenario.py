"""RoomsScenario implementation."""

from typing import Any

import numpy as np
from gymnasium import spaces
from matplotlib.pylab import Generator
from numpy.typing import NDArray

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

from .configs import RoomsScenarioConfig
from .observations import (
    ClosestObsPosObsFactory,
    DoorwayDistObsFactory,
    GoalDistObsFactory,
    ObstacleDistObsFactory,
)
from .spawn_strategies import (
    FixedSpawnConfig,
    FixedSpawnStrategy,
    PathGaussianConfig,
    PathGaussianSpawnStrategy,
    SpawnStrategy,
    UniformRandomConfig,
    UniformRandomSpawnStrategy,
)


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

        self.goal_dist_obs_factory = GoalDistObsFactory(
            room_scale=self.room_scale,
            dist_mode=config.observation_config.goal_dist,
        )
        self.doorway_dist_obs_factory = DoorwayDistObsFactory(
            room_scale=self.room_scale,
            dist_mode=config.observation_config.doorway_dist,
        )
        self.lava_dist_obs_factory = ObstacleDistObsFactory(
            room_scale=self.room_scale,
            dist_mode=config.observation_config.obs_dist,
            name="lava_dist",
        )
        self.hole_dist_obs_factory = ObstacleDistObsFactory(
            room_scale=self.room_scale,
            dist_mode=config.observation_config.obs_dist,
            name="hole_dist",
        )
        self.closest_obs_pos_obs_factory = ClosestObsPosObsFactory(
            room_scale=self.room_scale,
            enabled=config.observation_config.closest_obs_pos,
        )

    def _create_spawn_strategy(self) -> SpawnStrategy:
        """Factory method to create appropriate spawn strategy."""
        assert self.config
        method_config = self.config.spawn_config.spawn_method

        strategy_map = {
            FixedSpawnConfig: FixedSpawnStrategy,
            PathGaussianConfig: PathGaussianSpawnStrategy,
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
        """Pre-reset initialization: prepare free_cells list and remove fixed obstacle positions."""
        assert self.config
        self.free_cells: list[CellPosition] = self.init_free_cells.copy()

        # Remove all fixed obstacle positions from free_cells to prevent agent overlap
        self._remove_fixed_positions_from_free_cells()

    def _remove_fixed_positions_from_free_cells(self) -> None:
        """Remove fixed obstacle positions from free_cells list."""
        # Goal position
        self._remove_spawn_pos_from_free_cells(self.goal.reset_config.spawn_pos)

        # Lava positions
        for lava in self.lavas:
            self._remove_spawn_pos_from_free_cells(lava.reset_config.spawn_pos)

        # Hole positions
        for hole in self.holes:
            self._remove_spawn_pos_from_free_cells(hole.reset_config.spawn_pos)

    def _remove_spawn_pos_from_free_cells(self, spawn_pos: SpawnPos) -> None:
        """Remove a spawn position (or list of positions) from free_cells."""
        formatted_pos = self._format_spawn_pos(spawn_pos)
        if formatted_pos is None:
            return

        positions_to_remove = (
            formatted_pos if isinstance(formatted_pos, list) else [formatted_pos]
        )
        for pos in positions_to_remove:
            if pos in self.free_cells:
                self.free_cells.remove(pos)

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
        self._reset_goal(world, np_random)
        agent_pos = world.agents[0].state.pos if world.agents else None

        # Spawn obstacles using strategy pattern
        self._spawn_obstacles(world, np_random, agent_pos)

        # Store positions for observations
        self.wall_pos = np.array(
            world.wall_collision_checker.wall_centers, dtype=np.float64
        )

        return world.landmarks

    def _reset_goal(self, world: World, np_random: np.random.Generator) -> None:
        """Reset the goal landmark to a new position."""
        goal_pos: CellPosition = self._choose_new_pos(
            self.goal.reset_config.spawn_pos, self.free_cells, np_random
        )
        if goal_pos in self.free_cells:
            self.free_cells.remove(goal_pos)

        self.goal_pos = np.array((goal_pos[0], goal_pos[1]), dtype=np.float64)
        self.goal.state.pos = self.goal_pos.copy()

    def _spawn_obstacles(
        self,
        world: World,
        np_random: np.random.Generator,
        agent_pos: NDArray[np.float64] | None,
    ) -> None:
        """Spawn all obstacles (lavas and holes) using the configured strategy."""
        assert self.config
        max_spawn_attempts = 100

        # Spawn lavas
        lava_positions = self._spawn_obstacle_type(
            obstacle_type="lava",
            obstacles=self.lavas,
            obstacle_configs=self.config.spawn_config.lavas,
            world=world,
            np_random=np_random,
            agent_pos=agent_pos,
            max_attempts=max_spawn_attempts,
        )

        # Spawn holes
        hole_positions = self._spawn_obstacle_type(
            obstacle_type="hole",
            obstacles=self.holes,
            obstacle_configs=self.config.spawn_config.holes,
            world=world,
            np_random=np_random,
            agent_pos=agent_pos,
            max_attempts=max_spawn_attempts,
        )

        # Store for observations
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

    def _spawn_obstacle_type(
        self,
        obstacle_type: str,
        obstacles: list[Landmark],
        obstacle_configs: list,
        world: World,
        np_random: np.random.Generator,
        agent_pos: NDArray[np.float64] | None,
        max_attempts: int,
    ) -> list[Position]:
        """Spawn a specific type of obstacle with fallback strategy."""
        assert self.config

        # Try spawning with configured strategy
        positions: list[Position] = []
        for _ in range(max_attempts):
            positions = self.spawn_strategy.spawn_obstacles(
                num_obstacles=len(obstacles),
                obstacle_type=obstacle_type,
                world=world,
                scenario=self,
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
                scenario=self,
                np_random=np_random,
                agent_pos=agent_pos,
                obstacle_configs=remaining_configs,
            )
            positions.extend(additional_positions)

        # Update obstacle positions
        for i, pos in enumerate(positions):
            if i < len(obstacles):
                obstacles[i].state.pos = np.array(pos, dtype=np.float64)

        return positions

    def _post_reset_world(self, world: World, np_random: Generator) -> None:
        """Post-reset: find and cache closest obstacles to the agent."""
        agent = world.agents[0]
        self.closest_lava_pos = self._find_closest_obstacle(
            agent.state.pos, self.lava_pos
        )
        self.closest_hole_pos = self._find_closest_obstacle(
            agent.state.pos, self.hole_pos
        )

    def _find_closest_obstacle(
        self, agent_pos: NDArray[np.float64], obstacle_positions: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Find the position of the closest obstacle to the agent."""
        if len(obstacle_positions) == 0:
            return np.array([], dtype=np.float64)

        distances = np.linalg.norm(obstacle_positions - agent_pos, axis=1)
        closest_idx = int(np.argmin(distances))
        return obstacle_positions[closest_idx]

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
        wall_dist, _ = self.get_closest(agent.state.pos, self.wall_pos)
        obs["wall_dist"] = np.array([wall_dist], dtype=np.float64) / self.room_scale
        obs |= self.goal_dist_obs_factory.observation(agent, self.goal_pos)
        obs |= self.lava_dist_obs_factory.observation(agent, self.lava_pos)
        obs |= self.hole_dist_obs_factory.observation(agent, self.hole_pos)
        obs |= self.doorway_dist_obs_factory.observation(agent, self.doorway_pos)
        obs |= self.closest_obs_pos_obs_factory.observation(
            agent, self.lava_pos, self.hole_pos
        )
        return obs

    def observation_space(self, agent: Agent, world: World) -> spaces.Space:
        wall_limits = world.grid.wall_limits
        low_bound = np.array((wall_limits.min_x, wall_limits.min_y))
        high_bound = np.array((wall_limits.max_x, wall_limits.max_y))
        if self.room_scale is not None:
            low_bound = np.array((-1.0, -1.0))
            high_bound = np.array((1.0, 1.0))
        num_lavas = len(self.lavas)
        num_holes = len(self.holes)
        max_dist = float(np.linalg.norm(high_bound - low_bound))

        obs_space_dict: dict[str, spaces.Space] = {
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
            "wall_dist": spaces.Box(
                low=0.0, high=max_dist, shape=(1,), dtype=np.float64
            ),
        }
        obs_space_dict |= self.goal_dist_obs_factory.obs_space_dict(max_dist)
        obs_space_dict |= self.lava_dist_obs_factory.obs_space_dict(num_lavas, max_dist)
        obs_space_dict |= self.hole_dist_obs_factory.obs_space_dict(num_holes, max_dist)
        obs_space_dict |= self.doorway_dist_obs_factory.obs_space_dict(
            len(self.doorways), max_dist
        )
        obs_space_dict |= self.closest_obs_pos_obs_factory.obs_space_dict()
        return spaces.Dict(obs_space_dict)

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

        # Handle cases where there are no obstacles
        d_lava_closest = (
            np.linalg.norm(agent.state.pos - self.closest_lava_pos)
            if len(self.closest_lava_pos) > 0
            else np.inf
        )
        d_hole_closest = (
            np.linalg.norm(agent.state.pos - self.closest_hole_pos)
            if len(self.closest_hole_pos) > 0
            else np.inf
        )

        doorway_distances = {
            name: np.linalg.norm(agent.state.pos - pos)
            for name, pos in self.doorways.items()
        }

        info_dict["distances"] = {
            "lava": d_lv,
            "hole": d_hl,
            "goal": d_gl,
            "closest_lava": d_lava_closest,
            "closest_hole": d_hole_closest,
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
