from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pprint import pprint
from typing import Any, Literal

import numpy as np
from gymnasium import spaces
from matplotlib.pylab import Generator
from numpy.typing import NDArray
from pydantic import BaseModel

from contgrid.contgrid import DEFAULT_RENDER_CONFIG, BaseGymEnv, RenderConfig
from contgrid.core import (
    DEFAULT_ACTION_CONFIG,
    ActionModeConfig,
    Agent,
    BaseScenario,
    Color,
    EntityState,
    Grid,
    Landmark,
    ResetConfig,
    SpawnPos,
    World,
    WorldConfig,
    rc2cell_pos,
)
from contgrid.core.typing import CellPosition, Position


class SpawnMode(str, Enum):
    """Enumeration of available obstacle spawning methods."""

    FIXED = "fixed"
    PATH_GAUSSIAN = "path_gaussian"
    UNIFORM_RANDOM = "uniform_random"


class PathGaussianConfig(BaseModel):
    """Configuration for path-based Gaussian spawning."""

    mode: Literal[SpawnMode.PATH_GAUSSIAN] = SpawnMode.PATH_GAUSSIAN
    gaussian_std: float = 0.5
    min_spacing: float = 0.5
    edge_buffer: float = 0.3
    include_agent_paths: bool = True


class UniformRandomConfig(BaseModel):
    """Configuration for uniform random spawning."""

    mode: Literal[SpawnMode.UNIFORM_RANDOM] = SpawnMode.UNIFORM_RANDOM
    min_spacing: float = 0.5


class FixedSpawnConfig(BaseModel):
    """Configuration for fixed position spawning."""

    mode: Literal[SpawnMode.FIXED] = SpawnMode.FIXED


# Union type for all spawn method configs
SpawnMethodConfig = PathGaussianConfig | UniformRandomConfig | FixedSpawnConfig


class RewardConfig(BaseModel):
    step_penalty: float = 0.01
    sum_reward: bool = True


class ObjConfig(BaseModel):
    pos: Position | list[Position] | None = (
        None  # Default position indicating no specific position
    )
    reward: float = 0.0
    absorbing: bool = False


class SpawnConfig(BaseModel):
    """
    Configuration for spawning objects in the environment.

    Attributes
    ----------
    agent: Position | tuple[Position, Size] | None
        The position or size of the agent object.
    goal: ObjConfig
        The configuration for the goal object.
    lavas: list[ObjConfig]
        A list of configurations for lava objects.
    holes: list[ObjConfig]
        A list of configurations for hole objects.
    doorways: dict[str, Position]
        A dictionary mapping doorway names to their positions.
    """

    agent: Position | None = None  # (3, 3)
    goal: ObjConfig = ObjConfig(
        pos=(9, 8), reward=1.0, absorbing=False
    )  # List of goal objects, if any
    lavas: list[ObjConfig] = [
        ObjConfig(pos=(7, 8), reward=0.0, absorbing=False),
        ObjConfig(pos=(9, 9), reward=0.0, absorbing=False),
        ObjConfig(pos=(5, 10), reward=0.0, absorbing=False),
        ObjConfig(pos=(3, 7), reward=0.0, absorbing=False),
        ObjConfig(pos=(2, 4), reward=-1.0, absorbing=False),
        ObjConfig(pos=(3, 5), reward=-1.0, absorbing=False),
        ObjConfig(pos=(10, 4), reward=-1.0, absorbing=False),
        ObjConfig(pos=(8, 3), reward=-1.0, absorbing=False),
    ]  # List of lava objects, if any
    holes: list[ObjConfig] = [
        ObjConfig(pos=(8, 9), reward=0.0, absorbing=False),
        ObjConfig(pos=(9, 7), reward=0.0, absorbing=False),
        ObjConfig(pos=(5, 8), reward=-1.0, absorbing=False),
        ObjConfig(pos=(4, 9), reward=-1.0, absorbing=False),
        ObjConfig(pos=(1, 5), reward=0.0, absorbing=False),
        ObjConfig(pos=(5, 3), reward=0.0, absorbing=False),
        ObjConfig(pos=(7, 4), reward=-1.0, absorbing=False),
        ObjConfig(pos=(9, 2), reward=-1.0, absorbing=False),
    ]  # List of hole objects, if any
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


class RoomsScenarioConfig(BaseModel):
    spawn_config: SpawnConfig = SpawnConfig()
    reward_config: RewardConfig = RewardConfig(step_penalty=0.01)


DEFAULT_ROOMS_SCENARIO_CONFIG = RoomsScenarioConfig()
DEFAULT_WORLD_CONFIG = WorldConfig(
    grid=Grid(
        layout=[
            "#############",
            "#     #     #",
            "#     #     #",
            "#           #",
            "#     #     #",
            "#     #     #",
            "## ####     #",
            "#     ### ###",
            "#     #     #",
            "#     #     #",
            "#           #",
            "#     #     #",
            "#############",
        ]
    )
)


@dataclass
class LineSegment:
    """Represents a straight line segment in continuous space."""

    start: NDArray[np.float64]
    end: NDArray[np.float64]

    def sample_point(self, t: float) -> NDArray[np.float64]:
        """Sample a point at parameter t ∈ [0, 1] along the line."""
        return self.start + t * (self.end - self.start)

    def length(self) -> float:
        """Get the length of the line segment."""
        return float(np.linalg.norm(self.end - self.start))


class RoomTopology:
    """Defines the room structure and doorway connections."""

    def __init__(self, doorways: dict[str, Position]):
        self.doorways = doorways
        # Define which doorways are neighbors (connected by a room)
        self.neighbor_map: dict[str, list[str]] = {
            "ld": ["td", "bd"],  # left doorway connects to top-left and bottom-left
            "td": ["ld", "rd"],  # top doorway connects to top-left and top-right
            "rd": ["td", "bd"],  # right doorway connects to top-right and bottom-right
            "bd": [
                "ld",
                "rd",
            ],  # bottom doorway connects to bottom-left and bottom-right
        }

    def get_neighbor_doorways(self, doorway_name: str) -> list[str]:
        """Get names of doorways that are neighbors to the given doorway."""
        return self.neighbor_map.get(doorway_name, [])

    def get_doorways_in_room(self, position: Position, grid: Grid) -> list[str]:
        """
        Determine which doorways belong to the room containing the given position.
        Returns up to 2 doorway names.
        """
        # Simple approach: find the two closest doorways
        # This assumes the agent is in a room bounded by those doorways
        distances = {
            name: np.linalg.norm(np.array(position) - np.array(pos))
            for name, pos in self.doorways.items()
        }
        # Sort by distance and take the two closest
        sorted_doorways = sorted(distances.items(), key=lambda x: x[1])

        # Return the two closest doorways (they should be neighbors)
        closest_two = [name for name, _ in sorted_doorways[:2]]

        # Verify they are actually neighbors
        if len(closest_two) == 2:
            if closest_two[1] in self.neighbor_map.get(closest_two[0], []):
                return closest_two

        # If not neighbors, return the closest one and its neighbors
        closest = closest_two[0]
        return [closest] + self.neighbor_map.get(closest, [])[:1]


def get_relevant_path_segments(
    agent_pos: NDArray[np.float64],
    goal_pos: NDArray[np.float64],
    doorways: dict[str, NDArray[np.float64]],
    topology: RoomTopology,
    grid: Grid,
) -> list[LineSegment]:
    """
    Get relevant path segments based on agent position.
    Only includes paths between neighboring doorways and agent to its room's doorways.

    Parameters
    ----------
    agent_pos : Agent's current position
    goal_pos : Goal position
    doorways : Dictionary of doorway positions
    topology : Room topology information
    grid : Grid for spatial queries

    Returns
    -------
    List of LineSegment objects representing relevant paths
    """
    segments = []

    # 1. Find which room the agent is in
    agent_room_doorways = topology.get_doorways_in_room(tuple(agent_pos), grid)

    # 2. Add segments from agent to its room's doorways
    for doorway_name in agent_room_doorways:
        if doorway_name in doorways:
            segments.append(
                LineSegment(agent_pos.copy(), doorways[doorway_name].copy())
            )

    # 3. Add segments between neighboring doorways only
    added_pairs = set()
    for doorway_name, doorway_pos in doorways.items():
        neighbors = topology.get_neighbor_doorways(doorway_name)
        for neighbor_name in neighbors:
            if neighbor_name in doorways:
                # Avoid duplicates by sorting names
                pair = tuple(sorted([doorway_name, neighbor_name]))
                if pair not in added_pairs:
                    segments.append(
                        LineSegment(doorway_pos.copy(), doorways[neighbor_name].copy())
                    )
                    added_pairs.add(pair)

    # 4. Find which doorways are closest to the goal and add those paths
    goal_distances = {
        name: np.linalg.norm(goal_pos - pos) for name, pos in doorways.items()
    }
    closest_to_goal = min(goal_distances.items(), key=lambda x: x[1])[0]

    # Add path from closest doorway to goal
    if closest_to_goal in doorways:
        segments.append(LineSegment(doorways[closest_to_goal].copy(), goal_pos.copy()))

    return segments


class SpawnStrategy(ABC):
    """Abstract base class for object spawning strategies."""

    @abstractmethod
    def spawn_obstacles(
        self,
        num_obstacles: int,
        obstacle_type: str,
        world: World,
        scenario: "RoomsScenario",
        np_random: np.random.Generator,
        agent_pos: NDArray[np.float64] | None = None,
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
        scenario: "RoomsScenario",
        np_random: np.random.Generator,
        agent_pos: NDArray[np.float64] | None = None,
    ) -> list[Position]:
        """Spawn obstacles at configured positions or random free cells."""
        positions = []

        obstacle_landmarks = (
            scenario.lavas if obstacle_type == "lava" else scenario.holes
        )

        for landmark in obstacle_landmarks[:num_obstacles]:
            new_pos = scenario._choose_new_pos(
                landmark.reset_config.spawn_pos, scenario.free_cells, np_random
            )
            positions.append(new_pos)
            if new_pos in scenario.free_cells:
                scenario.free_cells.remove(new_pos)

        return positions


class PathGaussianSpawnStrategy(SpawnStrategy):
    """Spawn obstacles near shortest paths with Gaussian noise."""

    def __init__(self, config: PathGaussianConfig):
        self.config = config
        self.topology: RoomTopology | None = None

    def spawn_obstacles(
        self,
        num_obstacles: int,
        obstacle_type: str,
        world: World,
        scenario: "RoomsScenario",
        np_random: np.random.Generator,
        agent_pos: NDArray[np.float64] | None = None,
    ) -> list[Position]:
        """Spawn obstacles along relevant paths with Gaussian perturbation."""
        positions = []

        # Initialize topology if not done
        if self.topology is None:
            assert scenario.config
            self.topology = RoomTopology(scenario.config.spawn_config.doorways)

        # Get relevant path segments (requires agent position)
        if agent_pos is None and self.config.include_agent_paths:
            # Fallback: use all neighboring doorway pairs
            segments = self._get_doorway_segments_only(scenario)
        else:
            segments = get_relevant_path_segments(
                agent_pos if agent_pos is not None else scenario.goal_pos,
                scenario.goal_pos,
                scenario.doorways,
                self.topology,
                world.grid,
            )

        if not segments:
            # Fallback to fixed strategy
            return FixedSpawnStrategy().spawn_obstacles(
                num_obstacles, obstacle_type, world, scenario, np_random, agent_pos
            )

        # Weight segments by length
        lengths = np.array([seg.length() for seg in segments])
        if lengths.sum() == 0:
            return []

        probabilities = lengths / lengths.sum()

        max_attempts = 100
        for _ in range(num_obstacles):
            for attempt in range(max_attempts):
                # Sample a segment (weighted by length)
                segment_idx = np_random.choice(len(segments), p=probabilities)
                segment = segments[segment_idx]

                # Sample point along segment
                t = np_random.uniform(0, 1)
                base_pos = segment.sample_point(t)

                # Add Gaussian noise perpendicular and parallel to path
                direction = segment.end - segment.start
                norm = np.linalg.norm(direction)
                if norm > 1e-6:
                    direction = direction / norm
                    perpendicular = np.array([-direction[1], direction[0]])
                else:
                    direction = np.array([1.0, 0.0])
                    perpendicular = np.array([0.0, 1.0])

                parallel_noise = np_random.normal(0, self.config.gaussian_std)
                perp_noise = np_random.normal(0, self.config.gaussian_std)

                perturbed_pos = (
                    base_pos + parallel_noise * direction + perp_noise * perpendicular
                )

                # Validate position
                if self._is_valid_position(
                    perturbed_pos, world, positions, scenario, obstacle_type
                ):
                    positions.append(tuple(perturbed_pos))
                    break

        return positions

    def _get_doorway_segments_only(
        self, scenario: "RoomsScenario"
    ) -> list[LineSegment]:
        """Get segments between neighboring doorways and to goal."""
        segments = []

        # Neighboring doorway pairs
        neighbor_pairs = [("ld", "td"), ("ld", "bd"), ("td", "rd"), ("rd", "bd")]

        for d1, d2 in neighbor_pairs:
            if d1 in scenario.doorways and d2 in scenario.doorways:
                segments.append(
                    LineSegment(
                        scenario.doorways[d1].copy(), scenario.doorways[d2].copy()
                    )
                )

        # Add paths from doorways closest to goal
        goal_distances = {
            name: np.linalg.norm(scenario.goal_pos - pos)
            for name, pos in scenario.doorways.items()
        }
        closest_doorway = min(goal_distances.items(), key=lambda x: x[1])[0]
        if closest_doorway in scenario.doorways:
            segments.append(
                LineSegment(
                    scenario.doorways[closest_doorway].copy(), scenario.goal_pos.copy()
                )
            )

        return segments

    def _is_valid_position(
        self,
        pos: NDArray[np.float64],
        world: World,
        existing_positions: list[Position],
        scenario: "RoomsScenario",
        obstacle_type: str,
    ) -> bool:
        """Check if position is valid for obstacle spawning."""
        assert scenario.config
        limits = world.grid.wall_limits

        # Get obstacle size based on type
        obstacle_radius = (
            scenario.config.spawn_config.lava_size
            if obstacle_type == "lava"
            else scenario.config.spawn_config.hole_size
        )

        # Check bounds with buffer
        if not (
            limits.min_x + self.config.edge_buffer
            <= pos[0]
            <= limits.max_x - self.config.edge_buffer
            and limits.min_y + self.config.edge_buffer
            <= pos[1]
            <= limits.max_y - self.config.edge_buffer
        ):
            return False

        # Check wall collision using actual obstacle radius
        is_collided, _ = world.wall_collision_checker.is_collision(
            R=obstacle_radius,
            C=0.0,
            robot_pos=(float(pos[0]), float(pos[1])),
            collision_force=1.0,
            contact_margin=0.1,
        )
        if is_collided:
            return False

        # Check spacing from existing obstacles
        for existing_pos in existing_positions:
            if np.linalg.norm(pos - np.array(existing_pos)) < self.config.min_spacing:
                return False

        # Check not too close to goal
        if np.linalg.norm(pos - scenario.goal_pos) < scenario.goal_thr_dist + 0.5:
            return False

        return True


class UniformRandomSpawnStrategy(SpawnStrategy):
    """Spawn obstacles uniformly at random in free space."""

    def __init__(self, config: UniformRandomConfig):
        self.config = config

    def spawn_obstacles(
        self,
        num_obstacles: int,
        obstacle_type: str,
        world: World,
        scenario: "RoomsScenario",
        np_random: np.random.Generator,
        agent_pos: NDArray[np.float64] | None = None,
    ) -> list[Position]:
        """Spawn obstacles uniformly in free cells."""
        positions = []

        for _ in range(num_obstacles):
            if scenario.free_cells:
                chosen_idx = np_random.choice(len(scenario.free_cells))
                new_pos = scenario.free_cells[chosen_idx]
                positions.append(new_pos)
                scenario.free_cells.pop(chosen_idx)

        return positions


class RoomsScenario(BaseScenario[RoomsScenarioConfig, dict[str, NDArray[np.float64]]]):
    def __init__(
        self,
        config: RoomsScenarioConfig = DEFAULT_ROOMS_SCENARIO_CONFIG,
        world_config: WorldConfig = DEFAULT_WORLD_CONFIG,
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

        # Initialize spawn strategy
        self.spawn_strategy = self._create_spawn_strategy()

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
        landmarks = [self.goal] + self.lavas + self.holes

        return landmarks

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
        self.free_cells: list[CellPosition] = self.init_free_cells.copy()

    def reset_agents(self, world: World, np_random: np.random.Generator) -> list[Agent]:
        assert self.config
        for agent in world.agents:
            agent.reset(np_random)
            if self.config.spawn_config.agent is not None:
                agent.state.pos = np.array(
                    self.config.spawn_config.agent, dtype=np.float64
                )
            else:
                # Ensure the new position is not inside other landmarks
                chose_cell_idx = np_random.choice(len(self.free_cells))
                new_pos: CellPosition = self.free_cells[chose_cell_idx]

                agent.state.pos = np.array([new_pos[0], new_pos[1]], dtype=np.float64)

                # Remove the chosen cell from free cells
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

        # Get agent position (must be set before calling this)
        agent_pos = world.agents[0].state.pos if world.agents else None

        # Use strategy pattern for obstacles
        lava_positions = self.spawn_strategy.spawn_obstacles(
            num_obstacles=len(self.lavas),
            obstacle_type="lava",
            world=world,
            scenario=self,
            np_random=np_random,
            agent_pos=agent_pos,
        )

        hole_positions = self.spawn_strategy.spawn_obstacles(
            num_obstacles=len(self.holes),
            obstacle_type="hole",
            world=world,
            scenario=self,
            np_random=np_random,
            agent_pos=agent_pos,
        )

        # Update landmark positions
        for i, pos in enumerate(lava_positions):
            if i < len(self.lavas):
                self.lavas[i].state.pos = np.array(pos, dtype=np.float64)

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
                # Get the available positions that are still free
                available_pos: list[CellPosition] = list(
                    set(cell_pos_list) & set(free_cells)
                )
                if available_pos:
                    chose_cell_idx = np_random.choice(len(available_pos))
                    new_pos = available_pos[chose_cell_idx]

                else:  # Choose from a free cell
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
        """
        Return the distance to the closest object from the given position.
        If there are no objects, return infinity.

        Parameters
        ----------
        pos : NDArray[np.float64]
            The position to measure from.
        objects : NDArray[np.float64]
            The positions of the objects.

        Returns
        -------
        min_dist : float
            The distance to the closest object, or infinity if no objects exist.
        index : int
            The index of the closest object, or -1 if no objects exist.
        """
        if len(objects) == 0:
            return np.inf, -1
        dists = np.linalg.norm(objects - pos, axis=1)
        return np.min(dists), np.argmin(dists, axis=0)

    def observation(self, agent: Agent, world: World) -> dict[str, NDArray[np.float64]]:
        obs = {}
        # Agent's own position
        obs["agent_pos"] = agent.state.pos.copy()
        # Goal position
        obs["goal_pos"] = self.goal_pos.copy() - agent.state.pos.copy()
        obs["lava_pos"] = self.lava_pos.copy() - agent.state.pos.copy()
        obs["hole_pos"] = self.hole_pos.copy() - agent.state.pos.copy()
        # obs["wall_pos"] = self.wall_pos.copy() - agent.state.pos.copy()
        obs["doorway_pos"] = self.doorway_pos.copy() - agent.state.pos.copy()
        # Distance to the goal
        obs["goal_dist"] = np.array([np.linalg.norm(agent.state.pos - self.goal_pos)])
        # Distance to the closest lava
        lava_dist, _ = self.get_closest(agent.state.pos, self.lava_pos)
        obs["lava_dist"] = np.array([lava_dist], dtype=np.float64)
        # Distance to the closest hole
        hole_dist, _ = self.get_closest(agent.state.pos, self.hole_pos)
        obs["hole_dist"] = np.array([hole_dist], dtype=np.float64)
        wall_dist, _ = self.get_closest(agent.state.pos, self.wall_pos)
        obs["wall_dist"] = np.array([wall_dist], dtype=np.float64)
        obs["doorway_dist"] = self._get_doorway_distances(agent)

        return obs

    def observation_space(self, agent: Agent, world: World) -> spaces.Space:
        wall_limits = world.grid.wall_limits
        low_bound = -np.array(
            (wall_limits.max_x, wall_limits.max_y)
        )  # np.array((wall_limits.min_x, wall_limits.min_y))
        high_bound = np.array((wall_limits.max_x, wall_limits.max_y))
        num_lavas = len(self.lavas)
        num_holes = len(self.holes)

        max_dist = np.linalg.norm(high_bound - low_bound)
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
                # "wall_pos": spaces.Box(
                #     low=np.array(
                #         [low_bound] * len(world.wall_collision_checker.wall_centers)
                #     )
                #     if len(world.wall_collision_checker.wall_centers) > 0
                #     else np.array([], dtype=np.float64),
                #     high=np.array(
                #         [high_bound] * len(world.wall_collision_checker.wall_centers)
                #     )
                #     if len(world.wall_collision_checker.wall_centers) > 0
                #     else np.array([], dtype=np.float64),
                #     dtype=np.float64,
                # ),
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
                    low=0.0, high=max_dist, shape=(1,), dtype=np.float64
                ),
                "hole_dist": spaces.Box(
                    low=0.0, high=max_dist, shape=(1,), dtype=np.float64
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

        reward: float
        # Check if the agent has already terminated
        if agent.terminated:
            reward = 0.0
        # Reward for reaching the goal
        elif goal_dist < self.goal_thr_dist:
            reward = self.config.spawn_config.goal.reward
            agent.terminated = self.config.spawn_config.goal.absorbing
        # Penalty for falling into lava
        elif lava_min_dist < self.lava_thr_dist and lava_idx != -1:
            reward = self.config.spawn_config.lavas[lava_idx].reward
            agent.terminated = self.config.spawn_config.lavas[lava_idx].absorbing
        # Penalty for falling into a hole
        elif hole_min_dist < self.hole_thr_dist and hole_idx != -1:
            reward = self.config.spawn_config.holes[hole_idx].reward
            agent.terminated = self.config.spawn_config.holes[hole_idx].absorbing
        else:
            reward = 0.0

        # Step penalty
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

        # Success info
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


class RoomsEnvConfig(BaseModel):
    scenario_config: RoomsScenarioConfig = DEFAULT_ROOMS_SCENARIO_CONFIG
    action_config: ActionModeConfig = DEFAULT_ACTION_CONFIG
    world_config: WorldConfig = DEFAULT_WORLD_CONFIG
    render_config: RenderConfig = DEFAULT_RENDER_CONFIG


class RoomsEnv(
    BaseGymEnv[dict[str, NDArray[np.float64]], NDArray[np.float64], RoomsScenarioConfig]
):
    """
    Continuous Grid World with Rooms Environment

    This environment is a continuous 2D grid world where an agent must navigate through rooms to reach a goal while avoiding obstacles like lava and holes.

    Observation:
        Type: Dict
        {
            "agent_pos": Box(2,)  # Agent's position (x, y)
            "goal_pos": Box(2,)   # Goal's position (x, y)
            "lava_pos": Box(2 * num_lavas,)  # Positions of lava objects
            "hole_pos": Box(2 * num_holes,)  # Positions of hole objects
            "goal_dist": Box(1,)  # Distance to the goal
            "lava_dist": Box(1,)  # Distance to the closest lava
            "hole_dist": Box(1,)  # Distance to the closest hole
        }

    Actions:
        Type: Box(2,)
        Num     Action
        0       Move in x direction (-1.0 to 1.0)
        1       Move in y direction (-1.0 to 1.0)

    Reward:
        - Step penalty: -config.reward_config.step_penalty per step
        - Reaching the goal: +config.spawn_config.goal.reward
        - Falling into lava or hole: config.spawn_config.lavas[i].reward or config.spawn_config.holes[i].reward
    Starting State:
        - Agent starts at config.spawn_config.agent position or random free cell
        - Goal, lava, and holes are placed at their configured positions
    Episode Termination:
        - Agent reaches the goal
        - Agent falls into an absorbing lava or hole
        - Max episode steps reached
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        scenario_config: RoomsScenarioConfig = DEFAULT_ROOMS_SCENARIO_CONFIG,
        world_config: WorldConfig = DEFAULT_WORLD_CONFIG,
        render_config: RenderConfig = DEFAULT_RENDER_CONFIG,
        render_mode: str | None = None,
        action_config: ActionModeConfig = DEFAULT_ACTION_CONFIG,
        verbose: bool = False,
    ) -> None:
        if isinstance(scenario_config, dict):
            scenario_config = RoomsScenarioConfig(**scenario_config)
        if isinstance(world_config, dict):
            world_config = WorldConfig(**world_config)
        if isinstance(render_config, dict):
            render_config = RenderConfig(**render_config)
        scenario = RoomsScenario(scenario_config, world_config)
        super().__init__(
            scenario,
            render_config=render_config,
            render_mode=render_mode,
            action_config=action_config,
            local_ratio=None,
            verbose=verbose,
        )


if __name__ == "__main__":
    pprint(DEFAULT_ROOMS_SCENARIO_CONFIG.model_dump())
    pprint(DEFAULT_WORLD_CONFIG.model_dump())
    pprint(DEFAULT_RENDER_CONFIG.model_dump())
    pprint(RoomsEnvConfig().model_dump())
