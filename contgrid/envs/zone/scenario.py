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

from .configs import ZoneScenarioConfig, ZoneType
from .observations import (
    SubtaskObsFactory,
    VisitCountObsFactory,
    ZoneDistObsFactory,
)
from .spawn_strategies import (
    FixedSpawnConfig,
    FixedSpawnStrategy,
    GaussianSpawnConfig,
    GaussianSpawnStrategy,
    SpawnStrategy,
    UniformRandomConfig,
    UniformRandomSpawnStrategy,
)


class ZoneScenario(BaseScenario[ZoneScenarioConfig, dict[str, NDArray[np.float64]]]):
    """Scenario for the Zones environment with multiple zones and doorways."""

    def __init__(
        self,
        config: ZoneScenarioConfig,
        world_config: WorldConfig,
    ) -> None:
        super().__init__(config, world_config)
        self.zone_thr_dist: float = (
            (config.spawn_config.zone_size + config.spawn_config.agent_size / 2)
            if config.spawn_config.zone_thr_dist is None
            else config.spawn_config.zone_thr_dist
        )

        self.spawn_strategy = self._create_spawn_strategy()

        # self.room_scale: float = max(
        #     self.world_config.grid.width, self.world_config.grid.height
        # )
        self.room_scale: float = 1.0

        self.yellow_dist_obs_factory = ZoneDistObsFactory(
            room_scale=self.room_scale, name="yellow_dist"
        )
        self.red_dist_obs_factory = ZoneDistObsFactory(
            room_scale=self.room_scale, name="red_dist"
        )
        self.white_dist_obs_factory = ZoneDistObsFactory(
            room_scale=self.room_scale, name="white_dist"
        )
        self.black_dist_obs_factory = ZoneDistObsFactory(
            room_scale=self.room_scale, name="black_dist"
        )
        self.visit_count_obs_factory = VisitCountObsFactory()
        self.subtask_obs_factory = SubtaskObsFactory()

        self.yellow_visit_count: int = 0
        self.red_visit_count: int = 0
        self.white_visit_count: int = 0
        self.black_visit_count: int = 0

        self._yellow_zone_active: bool = False
        self._red_zone_active: bool = False
        self._white_zone_active: bool = False
        self._black_zone_active: bool = False

        self.current_subtask_idx: int = 0
        self.is_success: bool = len(config.spawn_config.subtask_seq) == 0

    @staticmethod
    def _to_position_tuple(pos: NDArray[np.float64]) -> Position:
        return (float(pos[0]), float(pos[1]))

    def export_spawned_config(self, world: World) -> ZoneScenarioConfig:
        """Export current world state as a ZoneScenarioConfig.

        Captures current spawned positions for the agent and all zone landmarks,
        while preserving all other configuration fields.
        """
        assert self.config

        config = self.config.model_copy(deep=True)

        config.spawn_config.spawn_method = (
            FixedSpawnConfig()
        )  # Override to fixed since we're exporting specific positions

        if world.agents:
            config.spawn_config.agent = self._to_position_tuple(
                world.agents[0].state.pos
            )

        for zone_cfg, zone_landmark in zip(
            config.spawn_config.yellow_zone,
            self.yellow,
        ):
            zone_cfg.pos = self._to_position_tuple(zone_landmark.state.pos)

        for zone_cfg, zone_landmark in zip(
            config.spawn_config.red_zone,
            self.red,
        ):
            zone_cfg.pos = self._to_position_tuple(zone_landmark.state.pos)

        for zone_cfg, zone_landmark in zip(
            config.spawn_config.white_zone,
            self.white,
        ):
            zone_cfg.pos = self._to_position_tuple(zone_landmark.state.pos)

        for zone_cfg, zone_landmark in zip(
            config.spawn_config.black_zone,
            self.black,
        ):
            zone_cfg.pos = self._to_position_tuple(zone_landmark.state.pos)

        return config

    def _create_spawn_strategy(self) -> SpawnStrategy:
        """Factory method to create appropriate spawn strategy."""
        assert self.config
        method_config = self.config.spawn_config.spawn_method

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

        # Initialize yellow, red, white, and black zones with default position [0, 0]
        # These will be updated during spawning
        default_pos = np.array([0.0, 0.0], dtype=np.float64)
        self.yellow = [
            Landmark(
                name=f"yellow_{i}",
                size=self.config.spawn_config.zone_size,
                collide=False,
                color=Color.YELLOW.name,
                state=EntityState(
                    pos=default_pos
                    if config.pos is None
                    else np.array(config.pos, dtype=np.float64)
                ),
                reset_config=ResetConfig(spawn_pos=self._format_spawn_pos(config.pos)),
            )
            for i, config in enumerate(self.config.spawn_config.yellow_zone)
        ]
        self.red = [
            Landmark(
                name=f"red_{i}",
                size=self.config.spawn_config.zone_size,
                collide=False,
                color=Color.RED.name,
                state=EntityState(
                    pos=default_pos
                    if config.pos is None
                    else np.array(config.pos, dtype=np.float64)
                ),
                reset_config=ResetConfig(spawn_pos=self._format_spawn_pos(config.pos)),
            )
            for i, config in enumerate(self.config.spawn_config.red_zone)
        ]
        self.white = [
            Landmark(
                name=f"white_{i}",
                size=self.config.spawn_config.zone_size,
                collide=False,
                color=Color.WHITE.name,
                state=EntityState(
                    pos=default_pos
                    if config.pos is None
                    else np.array(config.pos, dtype=np.float64)
                ),
                reset_config=ResetConfig(spawn_pos=self._format_spawn_pos(config.pos)),
            )
            for i, config in enumerate(self.config.spawn_config.white_zone)
        ]
        self.black = [
            Landmark(
                name=f"black_{i}",
                size=self.config.spawn_config.zone_size,
                collide=False,
                color=Color.BLACK.name,
                state=EntityState(
                    pos=default_pos
                    if config.pos is None
                    else np.array(config.pos, dtype=np.float64)
                ),
                reset_config=ResetConfig(spawn_pos=self._format_spawn_pos(config.pos)),
            )
            for i, config in enumerate(self.config.spawn_config.black_zone)
        ]

        return self.yellow + self.red + self.white + self.black

    def _format_spawn_pos(
        self, spawn_pos: SpawnPos
    ) -> Position | list[Position] | None:
        match spawn_pos:
            case list():
                return [(float(pos[0]), float(pos[1])) for pos in spawn_pos]
            case tuple():
                return (float(spawn_pos[0]), float(spawn_pos[1]))
            case None:
                return None
            case _:
                raise ValueError("Invalid spawn_pos type")

    def reset_world(self, world: World, np_random: Generator) -> None:
        """Reset the world, ensuring agents are reset before landmarks.

        This override is necessary because obstacle spawning strategies need
        to know the agent's position to avoid overlapping spawns.
        """
        self._pre_reset_world(world, np_random)
        # Reset agents FIRST so landmarks can use agent position for spawn validation
        world.agents = self.reset_agents(world, np_random)
        # Reset landmarks (obstacles) using agent position
        world.landmarks = self.reset_landmarks(world, np_random)
        self._post_reset_world(world, np_random)

    def _pre_reset_world(self, world: World, np_random: Generator) -> None:
        """Pre-reset initialization: prepare free_cells list and remove fixed obstacle positions."""
        assert self.config
        if not hasattr(self, "init_free_cells"):
            self.init_free_cells: list[CellPosition] = [
                rc2cell_pos((p[0], p[1]), world.grid.height_cells)
                for p in np.argwhere(world.numeric_grid == 0).tolist()
            ]
        self.free_cells: list[CellPosition] = self.init_free_cells.copy()
        self._spawned_zone_positions: dict[str, list[NDArray[np.float64]]] = {
            "yellow": [],
            "red": [],
            "white": [],
            "black": [],
        }
        self._all_spawned_positions: list[NDArray[np.float64]] = []
        self._reset_visit_counts()
        self.current_subtask_idx = 0
        self.is_success = len(self.config.spawn_config.subtask_seq) == 0

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
        agent_pos = world.agents[0].state.pos if world.agents else None

        # Spawn obstacles using strategy pattern
        self._spawn_obstacles(world, np_random, agent_pos)

        # Store positions for observations
        self.wall_pos = np.array(
            world.wall_collision_checker.wall_centers, dtype=np.float64
        )
        self.wall_bounds = world.wall_collision_checker.wall_bounds
        self.cell_size = world.grid.cell_size

        return world.landmarks

    def _spawn_obstacles(
        self,
        world: World,
        np_random: np.random.Generator,
        agent_pos: NDArray[np.float64] | None,
    ) -> None:
        """Spawn all obstacles (lavas and holes) using the configured strategy."""
        assert self.config
        max_spawn_attempts = 100

        spawn_specs = [
            ("yellow", self.yellow, self.config.spawn_config.yellow_zone),
            ("red", self.red, self.config.spawn_config.red_zone),
            ("white", self.white, self.config.spawn_config.white_zone),
            ("black", self.black, self.config.spawn_config.black_zone),
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
                    scenario=self,
                    np_random=np_random,
                    agent_pos=agent_pos,
                    obstacle_configs=[config],
                )
                if positions:
                    pos_array = np.array(positions[0], dtype=np.float64)
                    landmark.state.pos = pos_array
                    spawned_positions[color_type].append(positions[0])
                    self._spawned_zone_positions[color_type].append(pos_array)
                    self._all_spawned_positions.append(pos_array)
        else:
            spawned_positions: dict[str, list[Position]] = {}
            for obstacle_type, obstacles, obstacle_configs in spawn_specs:
                positions = self._spawn_obstacle_type(
                    obstacle_type=obstacle_type,
                    obstacles=obstacles,
                    obstacle_configs=obstacle_configs,
                    world=world,
                    np_random=np_random,
                    agent_pos=agent_pos,
                    max_attempts=max_spawn_attempts,
                )
                spawned_positions[obstacle_type] = positions
                self._spawned_zone_positions[obstacle_type] = [
                    np.array(pos, dtype=np.float64) for pos in positions
                ]
                self._all_spawned_positions.extend(
                    [np.array(pos, dtype=np.float64) for pos in positions]
                )

        # Store for observations
        self.yellow_pos = np.array(spawned_positions["yellow"], dtype=np.float64)
        self.red_pos = np.array(spawned_positions["red"], dtype=np.float64)
        self.white_pos = np.array(spawned_positions["white"], dtype=np.float64)
        self.black_pos = np.array(spawned_positions["black"], dtype=np.float64)

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

        # Update obstacle positions with safety check for agent distance
        obstacle_radius = self.config.spawn_config.zone_size
        agent_radius = self.config.spawn_config.agent_size
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
                        for cell in self.free_cells:
                            cell_dist = np.linalg.norm(np.array(cell) - agent_pos)
                            if cell_dist >= min_agent_distance:
                                pos_array = np.array(cell, dtype=np.float64)
                                if cell in self.free_cells:
                                    self.free_cells.remove(cell)
                                found_valid_cell = True
                                break

                        if not found_valid_cell:
                            # Could not find a valid position - skip this obstacle
                            position_is_valid = False

                if position_is_valid:
                    obstacles[i].state.pos = pos_array
                    valid_positions.append(tuple(pos_array))

        return valid_positions

    def _post_reset_world(self, world: World, np_random: Generator) -> None:
        """Post-reset: find and cache closest obstacles to the agent."""
        agent = world.agents[0]
        self.closest_yellow_pos = self._find_closest_obstacle(
            agent.state.pos, self.yellow_pos
        )
        self.closest_red_pos = self._find_closest_obstacle(
            agent.state.pos, self.red_pos
        )
        self.closest_white_pos = self._find_closest_obstacle(
            agent.state.pos, self.white_pos
        )
        self.closest_black_pos = self._find_closest_obstacle(
            agent.state.pos, self.black_pos
        )
        self._update_visit_counts(agent.state.pos)

    def _reset_visit_counts(self) -> None:
        self.yellow_visit_count = 0
        self.red_visit_count = 0
        self.white_visit_count = 0
        self.black_visit_count = 0

        self._yellow_zone_active = False
        self._red_zone_active = False
        self._white_zone_active = False
        self._black_zone_active = False

    def _is_inside_zone(
        self, agent_pos: NDArray[np.float64], zone_positions: NDArray[np.float64]
    ) -> bool:
        if len(zone_positions) == 0:
            return False
        distances = np.linalg.norm(zone_positions - agent_pos, axis=1)
        return bool(np.any(distances < self.zone_thr_dist))

    def _update_visit_counts(self, agent_pos: NDArray[np.float64]) -> None:
        in_yellow_zone = self._is_inside_zone(agent_pos, self.yellow_pos)
        in_red_zone = self._is_inside_zone(agent_pos, self.red_pos)
        in_white_zone = self._is_inside_zone(agent_pos, self.white_pos)
        in_black_zone = self._is_inside_zone(agent_pos, self.black_pos)

        if in_yellow_zone and not self._yellow_zone_active:
            self.yellow_visit_count += 1
        if in_red_zone and not self._red_zone_active:
            self.red_visit_count += 1
        if in_white_zone and not self._white_zone_active:
            self.white_visit_count += 1
        if in_black_zone and not self._black_zone_active:
            self.black_visit_count += 1

        self._yellow_zone_active = in_yellow_zone
        self._red_zone_active = in_red_zone
        self._white_zone_active = in_white_zone
        self._black_zone_active = in_black_zone

    def _get_visit_counts(self) -> NDArray[np.int32]:
        return np.array(
            [
                self.yellow_visit_count,
                self.red_visit_count,
                self.white_visit_count,
                self.black_visit_count,
            ],
            dtype=np.int32,
        )

    @staticmethod
    def _zone_type_to_visit_index(zone_type: ZoneType) -> int:
        zone_type_to_index = {
            ZoneType.YELLOW: 0,
            ZoneType.RED: 1,
            ZoneType.WHITE: 2,
            ZoneType.BLACK: 3,
        }
        return zone_type_to_index[zone_type]

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
    ) -> Position:
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
        obs["agent_vel"] = agent.state.vel.copy() / self.room_scale
        # Calculate wall distances in four cardinal directions: top, right, bottom, left
        wall_dists = self._get_wall_distances(agent.state.pos, self.wall_pos)
        obs["wall_dist"] = wall_dists
        obs |= self.yellow_dist_obs_factory.observation(agent, self.yellow_pos)
        obs |= self.red_dist_obs_factory.observation(agent, self.red_pos)
        obs |= self.white_dist_obs_factory.observation(agent, self.white_pos)
        obs |= self.black_dist_obs_factory.observation(agent, self.black_pos)
        obs |= self.visit_count_obs_factory.observation(agent, self._get_visit_counts())
        if self.config.obs_config.include_subtask:
            obs |= self.subtask_obs_factory.observation(
                agent, self.current_subtask_idx
            )
        return obs

    def observation_space(self, agent: Agent, world: World) -> spaces.Space:
        wall_limits = world.grid.wall_limits
        low_bound = np.array((wall_limits.min_x, wall_limits.min_y))
        high_bound = np.array((wall_limits.max_x, wall_limits.max_y))
        rel_low_bound = low_bound - high_bound
        rel_high_bound = high_bound - low_bound
        # if self.room_scale != 1:
        #     low_bound = np.array((-1.0, -1.0))
        #     high_bound = np.array((1.0, 1.0))
        max_dist = float(np.linalg.norm(high_bound - low_bound))

        obs_space_dict: dict[str, spaces.Space] = {
            "agent_pos": spaces.Box(
                low=low_bound, high=high_bound, shape=(2,), dtype=np.float64
            ),
            "agent_vel": spaces.Box(
                low=-max_dist, high=max_dist, shape=(2,), dtype=np.float64
            ),
            "wall_dist": spaces.Box(
                low=0.0, high=max_dist, shape=(4,), dtype=np.float64
            ),
        }
        obs_space_dict |= self.yellow_dist_obs_factory.obs_space_dict(
            num_zones=len(self.yellow),
            low_bound=rel_low_bound,
            high_bound=rel_high_bound,
        )
        obs_space_dict |= self.red_dist_obs_factory.obs_space_dict(
            num_zones=len(self.red), low_bound=rel_low_bound, high_bound=rel_high_bound
        )
        obs_space_dict |= self.white_dist_obs_factory.obs_space_dict(
            num_zones=len(self.white),
            low_bound=rel_low_bound,
            high_bound=rel_high_bound,
        )
        obs_space_dict |= self.black_dist_obs_factory.obs_space_dict(
            num_zones=len(self.black),
            low_bound=rel_low_bound,
            high_bound=rel_high_bound,
        )
        obs_space_dict |= self.visit_count_obs_factory.obs_space_dict()

        if self.config.obs_config.include_subtask:
            obs_space_dict |= self.subtask_obs_factory.obs_space_dict(
                num_subtasks=len(self.config.spawn_config.subtask_seq)
            )

        return spaces.Dict(obs_space_dict)

    def reward(self, agent: Agent, world: World) -> float:
        assert self.config
        reward: float = 0.0

        previous_visit_counts = self._get_visit_counts().copy()
        self._update_visit_counts(agent.state.pos)
        current_visit_counts = self._get_visit_counts()

        if not agent.terminated and self.current_subtask_idx < len(
            self.config.spawn_config.subtask_seq
        ):
            current_subtask = self.config.spawn_config.subtask_seq[
                self.current_subtask_idx
            ]

            if current_subtask.obstacle is not None:
                obstacle_idx = self._zone_type_to_visit_index(current_subtask.obstacle)
                obstacle_visited = (
                    current_visit_counts[obstacle_idx]
                    > previous_visit_counts[obstacle_idx]
                )
                if obstacle_visited:
                    reward += current_subtask.penalty
                    if current_subtask.obstacle_absorbing:
                        agent.terminated = True

            goal_idx = self._zone_type_to_visit_index(current_subtask.goal)
            goal_visited = (
                current_visit_counts[goal_idx] > previous_visit_counts[goal_idx]
            )
            if goal_visited and not agent.terminated:
                reward += current_subtask.reward
                is_last_subtask = self.current_subtask_idx == (
                    len(self.config.spawn_config.subtask_seq) - 1
                )
                if is_last_subtask:
                    self.is_success = True
                if current_subtask.goal_absorbing:
                    agent.terminated = True
                else:
                    self.current_subtask_idx += 1

        if not agent.terminated:
            reward -= self.config.reward_config.step_penalty

        return reward

    def info(self, agent: Agent, world: World) -> dict:
        info_dict: dict[str, Any] = {}
        info_dict["terminated"] = agent.terminated

        d_y, _ = self.get_closest(agent.state.pos, self.yellow_pos)
        d_r, _ = self.get_closest(agent.state.pos, self.red_pos)
        d_w, _ = self.get_closest(agent.state.pos, self.white_pos)
        d_b, _ = self.get_closest(agent.state.pos, self.black_pos)

        info_dict["distances"] = {"yellow": d_y, "red": d_r, "white": d_w, "black": d_b}

        subtask_seq = self.config.spawn_config.subtask_seq
        info_dict["is_success"] = self.is_success
        info_dict["visit_counts"] = {
            "yellow": self.yellow_visit_count,
            "red": self.red_visit_count,
            "white": self.white_visit_count,
            "black": self.black_visit_count,
        }
        if self.current_subtask_idx < len(subtask_seq):
            active_subtask = subtask_seq[self.current_subtask_idx]
            info_dict["current_subtask"] = {
                "idx": self.current_subtask_idx,
                "goal": active_subtask.goal,
                "obstacle": active_subtask.obstacle,
            }
        else:
            info_dict["current_subtask"] = {
                "idx": self.current_subtask_idx,
                "goal": None,
                "obstacle": None,
            }
        info_dict["thresholds"] = {"zone": self.zone_thr_dist}
        info_dict["prohibited_actions"] = self._get_prohibited_actions(
            agent, self.wall_pos
        )

        return info_dict

    def _get_wall_distances(
        self, agent_pos: NDArray[np.float64], wall_positions: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Calculate distances to the nearest wall in four cardinal directions.

        Only considers walls that are actually in the agent's path (i.e., whose
        perpendicular extent overlaps with the agent's position). This correctly
        handles internal walls in multi-room layouts by ignoring walls in other
        rooms that happen to be in the same cardinal direction but aren't blocking.

        Returns distances in order: [top, right, bottom, left].
        """
        if len(self.wall_bounds) == 0:
            return (
                np.array([np.inf, np.inf, np.inf, np.inf], dtype=np.float64)
                / self.room_scale
            )

        agent_x, agent_y = agent_pos[0], agent_pos[1]
        # wall_bounds columns: [min_x, max_x, min_y, max_y]
        w_min_x = self.wall_bounds[:, 0]
        w_max_x = self.wall_bounds[:, 1]
        w_min_y = self.wall_bounds[:, 2]
        w_max_y = self.wall_bounds[:, 3]

        # Walls aligned horizontally with the agent (agent's x is within wall's x-extent)
        x_aligned = (w_min_x <= agent_x) & (agent_x <= w_max_x)
        # Walls aligned vertically with the agent (agent's y is within wall's y-extent)
        y_aligned = (w_min_y <= agent_y) & (agent_y <= w_max_y)

        # Top: x-aligned walls whose bottom edge is above the agent
        top_mask = x_aligned & (w_min_y > agent_y)
        top_dist = (
            float(np.min(w_min_y[top_mask] - agent_y)) if np.any(top_mask) else np.inf
        )

        # Bottom: x-aligned walls whose top edge is below the agent
        bottom_mask = x_aligned & (w_max_y < agent_y)
        bottom_dist = (
            float(np.min(agent_y - w_max_y[bottom_mask]))
            if np.any(bottom_mask)
            else np.inf
        )

        # Right: y-aligned walls whose left edge is to the right of agent
        right_mask = y_aligned & (w_min_x > agent_x)
        right_dist = (
            float(np.min(w_min_x[right_mask] - agent_x))
            if np.any(right_mask)
            else np.inf
        )

        # Left: y-aligned walls whose right edge is to the left of agent
        left_mask = y_aligned & (w_max_x < agent_x)
        left_dist = (
            float(np.min(agent_x - w_max_x[left_mask])) if np.any(left_mask) else np.inf
        )

        wall_dists = np.array(
            [top_dist, right_dist, bottom_dist, left_dist], dtype=np.float64
        )
        return wall_dists / self.room_scale

    def _get_prohibited_actions(
        self, agent: Agent, wall_positions: NDArray[np.float64]
    ) -> list[int]:
        """Determine which actions are prohibited due to wall proximity.

        Returns a list of action indices that would cause collision with walls.
        For discrete action spaces with n actions (where n is divisible by 4),
        assumes actions are arranged in a circular pattern starting from top (0)
        going clockwise. For continuous action spaces, returns empty list.

        Action mapping for n actions (n % 4 == 0):
        - Top: index 0
        - Right: index n//4
        - Bottom: index n//2
        - Left: index 3*n//4

        Examples:
        - 4 actions: [0=top, 1=right, 2=bottom, 3=left]
        - 8 actions: [0=top, 2=right, 4=bottom, 6=left]
        - 16 actions: [0=top, 4=right, 8=bottom, 12=left]
        """
        assert self.config

        # Return empty list if no action space set or if continuous
        if self.action_space_ref is None or not isinstance(
            self.action_space_ref, spaces.Discrete
        ):
            return []

        n_actions = self.action_space_ref.n

        wall_dists = self._get_wall_distances(agent.state.pos, wall_positions)
        collision_threshold = self.config.spawn_config.agent_size / 2

        # Check which cardinal directions are blocked
        blocked_directions = {
            "top": wall_dists[0] < collision_threshold,
            "right": wall_dists[1] < collision_threshold,
            "bottom": wall_dists[2] < collision_threshold,
            "left": wall_dists[3] < collision_threshold,
        }

        prohibited = []

        # Map cardinal directions to action indices based on action space size
        # Assumes actions are arranged in a circular pattern starting from top (0) going clockwise
        # Pattern: top=0, right=n/4, bottom=n/2, left=3n/4
        if n_actions >= 4 and n_actions % 4 == 0:
            action_indices = {
                "right": 0,
                "top": n_actions // 4,
                "left": n_actions // 2,
                "bottom": (3 * n_actions) // 4,
            }

            for direction, is_blocked in blocked_directions.items():
                if is_blocked:
                    prohibited.append(action_indices[direction])
        # For action spaces not divisible by 4, return empty list as mapping is unclear

        return prohibited
