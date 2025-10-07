from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import Env
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


class RoomsScenario(BaseScenario[RoomsScenarioConfig, NDArray]):
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
        return world.landmarks
