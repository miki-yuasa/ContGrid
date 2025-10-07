from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray
from pydantic import BaseModel

from contgrid.core import (
    DEFAULT_RENDER_CONFIG,
    ActionMode,
    Agent,
    AgentState,
    BaseEnv,
    BaseScenario,
    Color,
    EntityState,
    Grid,
    Landmark,
    RenderConfig,
    ScenarioConfigT,
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
    absorbing: bool = True


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
    cleared_doorways: list[Position]
        A list of positions for cleared doorways, if any.
        When specified, there will be no negative reward objects placed next to these doorways.
    """

    agent: Position | None = None
    goal: ObjConfig  # List of goal objects, if any
    lavas: list[ObjConfig] = []  # List of lava objects, if any
    holes: list[ObjConfig] = []  # List of hole objects, if any

    model_config = {"arbitrary_types_allowed": True}


class RoomsScenarioConfig(BaseModel):
    spawn_config: SpawnConfig
    reward_config: RewardConfig = RewardConfig(
        step_penalty=0.01,
        sum_reward=True,
    )


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
        config: RoomsScenarioConfig,
        world_config: WorldConfig = DEFAULT_WORLD_CONFIG,
    ) -> None:
        super().__init__(config, world_config)

    def init_agents(self, world: World, np_random=None) -> list[Agent]:
        agent = Agent(
            name="agent_0",
            size=0.25,
            color=Color.SKY_BLUE.name,
        )
        return [agent]

    def init_landmarks(self, world: World, np_random=None) -> list[Landmark]:
        assert self.config
        # Initialize the goal
        self.goal = Landmark(
            name="goal",
            size=0.5,
            color=Color.GREEN.name,
            state=EntityState(
                pos=np.array(self.config.spawn_config.goal.pos, dtype=np.float64)
            ),
        )

        # Initialize lava landmarks
        self.lavas = [
            Landmark(
                name=f"lava_{i}",
                size=0.5,
                color=Color.ORANGE.name,
                state=EntityState(pos=np.array(config.pos, dtype=np.float64)),
            )
            for i, config in enumerate(self.config.spawn_config.lavas)
        ]

        # Initialize holes
        self.holes = [
            Landmark(
                name=f"hole_{i}",
                size=0.5,
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
    ) -> float:
        if len(objects) == 0:
            return np.inf
        dists = np.linalg.norm(objects - pos, axis=1)
        return np.min(dists)

    def observation(self, agent: Agent, world: World) -> dict[str, NDArray[np.float64]]:
        obs = {}
        # Agent's own position
        obs["agent_pos"] = agent.state.pos.copy()
        # Goal position
        obs["goal_pos"] = self.goal_pos.copy()
        obs["lava_pos"] = self.lava_pos.copy()
        obs["hole_pos"] = self.hole_pos.copy()
        # Distance to the goal
        obs["goal_dist"] = np.array(
            [self.get_closest(agent.state.pos, self.goal_pos)], dtype=np.float64
        )
        # Distance to the closest lava
        obs["lava_dist"] = np.array(
            [self.get_closest(agent.state.pos, self.lava_pos)], dtype=np.float64
        )
        # Distance to the closest hole
        obs["hole_dist"] = np.array(
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
