from abc import ABC, abstractmethod
from typing import Any, Generic

import numpy as np
from gymnasium.core import ActType
from gymnasium.spaces import Box, MultiDiscrete, Space
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from .agent import Agent
from .world import World


class ActionMode(Generic[ActType], ABC):
    name: str  # unique name of the action mode

    @classmethod
    def cap_name(cls) -> str:
        return cls.name.capitalize()

    @abstractmethod
    def define_action_space(self, agent: Agent) -> Space[ActType]:
        pass

    @abstractmethod
    def update_agent_action(self, agent: Agent, action: ActType, world: World) -> None:
        pass

    def _zero_agent_actions(self, agent: Agent, world: World) -> None:
        agent.action.u = np.zeros(world.dim_p)
        agent.action.c = np.zeros(world.dim_c)

    def _apply_sensitivity(self, agent: Agent, sensitivity: float) -> None:
        agent.action.u *= sensitivity

    def dim(self, agent: Agent) -> int:
        space_shape: tuple[int, ...] | None = self.define_action_space(agent).shape
        return space_shape[0] if space_shape is not None else 0


class ActionModeConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    action_mode: type[ActionMode] | str = "continuous_minimal_velocity"
    action_mode_kwargs: dict[str, Any] = {}


DEFAULT_ACTION_MODE_CONFIG = ActionModeConfig()


class ContinuousMinimalVelocity(ActionMode[NDArray[np.float64]]):
    name: str = "continuous_minimal_velocity"

    def define_action_space(self, agent: Agent) -> Space[NDArray[np.float64]]:
        low_bound: list[float] = []
        high_bound: list[float] = []
        if agent.movable:
            low_bound += [-agent.u_range, -agent.u_range]
            high_bound += [agent.u_range, agent.u_range]
        if not agent.silent:
            low_bound += [0.0]
            high_bound += [1.0]
        return Box(
            low=np.array(low_bound, dtype=np.float64),
            high=np.array(high_bound, dtype=np.float64),
            dtype=np.float64,
        )

    def update_agent_action(
        self, agent: Agent, action: NDArray[np.float64], world: World
    ) -> None:
        self._zero_agent_actions(agent, world)

        if agent.movable:
            agent.action.u += action[0:2]

            self._apply_sensitivity(agent, agent.accel)

        if not agent.silent:
            agent.action.c = action[-1]


class DiscreteMinimalVelocity(ActionMode[NDArray[np.integer]]):
    """
    Discretize the v_x and v_y into num_discrete bins, and communication into num_discrete bins.
    The low/high bounds are [-u_range, u_range] for v_x and v_y, and [0, 1] for communication.
    """

    name: str = "discrete_minimal_velocity"

    def __init__(self, num_discrete: int) -> None:
        self.num_discrete = num_discrete

    def define_action_space(self, agent: Agent) -> Space[NDArray[np.integer]]:
        nvecs: list[int] = []
        if agent.movable:
            nvecs += [self.num_discrete, self.num_discrete]  # x and y velocity
        if not agent.silent:
            nvecs += [self.num_discrete]  # communication

        return MultiDiscrete(nvecs, dtype=np.int64)

    def update_agent_action(
        self, agent: Agent, action: NDArray[np.integer], world: World
    ) -> None:
        """
        Convert the discrete action into continuous action and update the agent's action.
        """
        # Convert the discrete action into continuous action
        self._zero_agent_actions(agent, world)

        if agent.movable:
            agent.action.u[0] = (
                action[0] / self.num_discrete
            ) * 2 * agent.u_range - agent.u_range
            agent.action.u[1] = (
                action[1] / self.num_discrete
            ) * 2 * agent.u_range - agent.u_range

            self._apply_sensitivity(agent, agent.accel)

        if not agent.silent:
            agent.action.c = action[-1] / self.num_discrete


class ContinuousFullVelocity(ActionMode[NDArray[np.float64]]):
    """
    (+/-)v_x and (+/-)v_y (i.e., 4 dimensions), and communication (i.e., 1 dimension).
    The bound is [0, u_range] for v_x and v_y, and [0, 1] for communication.
    """

    name: str = "continuous_full_velocity"

    def define_action_space(self, agent: Agent) -> Space[NDArray[np.float64]]:
        low_bound: list[float] = []
        high_bound: list[float] = []
        if agent.movable:
            low_bound += [0.0] * 4
            high_bound += [agent.u_range] * 4
        if not agent.silent:
            low_bound += [0.0]
            high_bound += [1.0]
        return Box(
            low=np.array(low_bound, dtype=np.float64),
            high=np.array(high_bound, dtype=np.float64),
            dtype=np.float64,
        )

    def update_agent_action(
        self, agent: Agent, action: NDArray[np.float64], world: World
    ) -> None:
        self._zero_agent_actions(agent, world)

        if agent.movable:
            agent.action.u[0] += action[0] - action[1]
            agent.action.u[1] += action[2] - action[3]

            self._apply_sensitivity(agent, agent.accel)

        if not agent.silent:
            agent.action.c = action[-1]


class DiscreteDirectionVelocity(ActionMode[NDArray[np.integer]]):
    """
    Add +1.0 velocity for a discrete direction (0: no, 1: up, 2: down, 3: left, 4: right) for velocity (i.e., 1 dimension), and communication (i.e., 1 dimension).
    """

    name: str = "discrete_direction_velocity"

    def define_action_space(self, agent: Agent) -> Space[NDArray[np.integer]]:
        nvecs: list[int] = []
        if agent.movable:
            nvecs += [5]  # 0: no, 1: up, 2: down, 3: left, 4: right
        if not agent.silent:
            nvecs += [2]  # communication: 0 or 1

        return MultiDiscrete(nvecs, dtype=np.int64)

    def update_agent_action(
        self, agent: Agent, action: NDArray[np.integer], world: World
    ) -> None:
        self._zero_agent_actions(agent, world)

        if agent.movable:
            if action[0] == 1:  # up
                agent.action.u[1] += 1.0
            elif action[0] == 2:  # down
                agent.action.u[1] -= 1.0
            elif action[0] == 3:  # left
                agent.action.u[0] -= 1.0
            elif action[0] == 4:  # right
                agent.action.u[0] += 1.0

            self._apply_sensitivity(agent, agent.accel)

        if not agent.silent:
            agent.action.c = action[-1]
