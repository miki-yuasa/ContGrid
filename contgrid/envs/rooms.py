from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from pydantic import BaseModel

from contgrid.core import (
    DEFAULT_RENDER_CONFIG,
    Agent,
    BaseGymEnv,
    BaseScenario,
    Color,
    EntityState,
    Grid,
    Landmark,
    RenderConfig,
    World,
    WorldConfig,
)

Position = tuple[float, float]


class RewardConfig(BaseModel):
    step_penalty: float = 0.01
    sum_reward: bool = True


class ObjConfig(BaseModel):
    pos: Position | None = None  # Default position indicating no specific position
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

    agent: Position | None = (3, 3)
    goal: ObjConfig = ObjConfig(
        pos=(9, 8), reward=1.0, absorbing=True
    )  # List of goal objects, if any
    lavas: list[ObjConfig] = [
        ObjConfig(pos=(7, 8), reward=0.0, absorbing=False),
        ObjConfig(pos=(9, 9), reward=0.0, absorbing=False),
        ObjConfig(pos=(5, 10), reward=0.0, absorbing=False),
        ObjConfig(pos=(3, 7), reward=0.0, absorbing=False),
        ObjConfig(pos=(2, 4), reward=-1.0, absorbing=True),
        ObjConfig(pos=(3, 5), reward=-1.0, absorbing=True),
        ObjConfig(pos=(10, 4), reward=-1.0, absorbing=True),
        ObjConfig(pos=(8, 3), reward=-1.0, absorbing=True),
    ]  # List of lava objects, if any
    holes: list[ObjConfig] = [
        ObjConfig(pos=(8, 9), reward=0.0, absorbing=False),
        ObjConfig(pos=(9, 7), reward=0.0, absorbing=False),
        ObjConfig(pos=(5, 8), reward=-1.0, absorbing=True),
        ObjConfig(pos=(4, 9), reward=-1.0, absorbing=True),
        ObjConfig(pos=(1, 5), reward=0.0, absorbing=True),
        ObjConfig(pos=(5, 3), reward=0.0, absorbing=True),
        ObjConfig(pos=(7, 4), reward=-1.0, absorbing=True),
        ObjConfig(pos=(9, 2), reward=-1.0, absorbing=True),
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


class RoomsScenario(BaseScenario[RoomsScenarioConfig, dict[str, NDArray[np.float64]]]):
    def __init__(
        self,
        config: RoomsScenarioConfig = DEFAULT_ROOMS_SCENARIO_CONFIG,
        world_config: WorldConfig = DEFAULT_WORLD_CONFIG,
    ) -> None:
        super().__init__(config, world_config)
        self.goal_thr_dist: float = (
            config.spawn_config.goal_size + config.spawn_config.agent_size
        )
        self.lava_thr_dist: float = (
            config.spawn_config.lava_size + config.spawn_config.agent_size
        )
        self.hole_thr_dist: float = (
            config.spawn_config.hole_size + config.spawn_config.agent_size
        )
        self.doorways: dict[str, NDArray[np.float64]] = {
            name: np.array(pos, dtype=np.float64)
            for name, pos in config.spawn_config.doorways.items()
        }

    def init_agents(self, world: World, np_random=None) -> list[Agent]:
        assert self.config
        agent = Agent(
            name="agent_0",
            size=self.config.spawn_config.agent_size,
            color=Color.SKY_BLUE.name,
        )
        return [agent]

    def init_landmarks(self, world: World, np_random=None) -> list[Landmark]:
        assert self.config
        # Initialize the goal
        self.goal = Landmark(
            name="goal",
            size=self.config.spawn_config.goal_size,
            color=Color.GREEN.name,
            state=EntityState(
                pos=np.array(self.config.spawn_config.goal.pos, dtype=np.float64)
            ),
        )

        # Initialize lava landmarks
        self.lavas = [
            Landmark(
                name=f"lava_{i}",
                size=self.config.spawn_config.lava_size,
                color=Color.ORANGE.name,
                state=EntityState(pos=np.array(config.pos, dtype=np.float64)),
            )
            for i, config in enumerate(self.config.spawn_config.lavas)
        ]

        # Initialize holes
        self.holes = [
            Landmark(
                name=f"hole_{i}",
                size=self.config.spawn_config.hole_size,
                color=Color.PURPLE.name,
                state=EntityState(pos=np.array(config.pos, dtype=np.float64)),
            )
            for i, config in enumerate(self.config.spawn_config.holes)
        ]
        landmarks = [self.goal] + self.lavas + self.holes
        return landmarks

    def reset_agents(self, world: World, np_random) -> list[Agent]:
        assert self.config
        for agent in world.agents:
            agent.terminated = False
            if self.config.spawn_config.agent is not None:
                agent.state.pos = np.array(
                    self.config.spawn_config.agent, dtype=np.float64
                )
            else:
                # Randomly place the agent in a free cell
                free_cells = np.argwhere(world.grid.layout == 0)
                chosen_cell = free_cells[np_random.choice(len(free_cells))]
                agent.state.pos = chosen_cell + 0.5  # Center of the cell
            agent.state.vel = np.array([0.0, 0.0], dtype=np.float64)
        return world.agents

    def reset_landmarks(self, world: World, np_random) -> list[Landmark]:
        # Landmarks are static; no need to reset positions
        lava_pos: list[Position] = [
            (lava.state.pos[0], lava.state.pos[1]) for lava in self.lavas
        ]
        hole_pos: list[Position] = [
            (hole.state.pos[0], hole.state.pos[1]) for hole in self.holes
        ]
        self.lava_pos: NDArray[np.float64] = np.array(lava_pos, dtype=np.float64)
        self.hole_pos: NDArray[np.float64] = np.array(hole_pos, dtype=np.float64)
        self.goal_pos: NDArray[np.float64] = np.array(
            (self.goal.state.pos[0], self.goal.state.pos[1]), dtype=np.float64
        )
        return world.landmarks

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
        obs["goal_pos"] = self.goal_pos.copy()
        obs["lava_pos"] = self.lava_pos.copy()
        obs["hole_pos"] = self.hole_pos.copy()
        # Distance to the goal
        obs["goal_dist"] = np.linalg.norm(agent.state.pos - self.goal_pos)
        # Distance to the closest lava
        obs["lava_dist"], _ = np.array(
            [self.get_closest(agent.state.pos, self.lava_pos)], dtype=np.float64
        )
        # Distance to the closest hole
        obs["hole_dist"], _ = np.array(
            [self.get_closest(agent.state.pos, self.hole_pos)], dtype=np.float64
        )
        return obs

    def observation_space(self, agent: Agent, world: World) -> spaces.Space:
        wall_limits = world.grid.wall_limits
        low_bound = np.array((wall_limits.min_x, wall_limits.min_y))
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
                    low=np.concat([low_bound] * num_lavas),
                    high=np.concat([high_bound] * num_lavas),
                    dtype=np.float64,
                ),
                "hole_pos": spaces.Box(
                    low=np.concat([low_bound] * num_holes),
                    high=np.concat([high_bound] * num_holes),
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
            agent.terminated = True
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
            reward = -self.config.reward_config.step_penalty

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

        return info_dict


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
        config: RoomsScenarioConfig = DEFAULT_ROOMS_SCENARIO_CONFIG,
        world_config: WorldConfig = DEFAULT_WORLD_CONFIG,
        render_config: RenderConfig = DEFAULT_RENDER_CONFIG,
        max_episode_steps: int = 100,
        verbose: bool = False,
    ) -> None:
        scenario = RoomsScenario(config, world_config)
        super().__init__(
            scenario,
            max_cycles=max_episode_steps,
            render_config=render_config,
            local_ratio=None,
            verbose=verbose,
        )
