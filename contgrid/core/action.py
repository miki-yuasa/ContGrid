from abc import ABC, abstractmethod
from pprint import pprint
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

    def action2xy_vel(self, action: ActType, agent: Agent) -> tuple[float, float]:
        raise NotImplementedError

    def sum_actions(
        self, action_1: NDArray, action_2: NDArray, weight: float = 0.5
    ) -> NDArray:
        raise NotImplementedError


class ActionModeConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    action_mode: type[ActionMode] | str = "continuous_minimal_velocity"
    action_mode_kwargs: dict[str, Any] = {}


DEFAULT_ACTION_CONFIG = ActionModeConfig()


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
            x_vel, y_vel = self.action2xy_vel(action, agent)
            agent.action.u[0] += x_vel
            agent.action.u[1] += y_vel

            self._apply_sensitivity(agent, agent.accel)

        if not agent.silent:
            agent.action.c = action[-1]

    def action2xy_vel(
        self, action: NDArray[np.float64], agent: Agent
    ) -> tuple[float, float]:
        if len(action) >= 2:
            return float(action[0]), float(action[1])
        else:
            raise ValueError(
                f"Action length is less than 2: {len(action)}. Cannot extract x and y velocity."
            )


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
            x_vel, y_vel = self.action2xy_vel(action, agent)
            agent.action.u[0] += x_vel
            agent.action.u[1] += y_vel

            self._apply_sensitivity(agent, agent.accel)

        if not agent.silent:
            agent.action.c = action[-1] / self.num_discrete

            agent.action.u += action[0:2]

    def action2xy_vel(
        self, action: NDArray[np.integer], agent: Agent
    ) -> tuple[float, float]:
        if len(action) >= 2:
            x_vel = (action[0] / self.num_discrete) * 2 * agent.u_range - agent.u_range
            y_vel = (action[1] / self.num_discrete) * 2 * agent.u_range - agent.u_range
            return float(x_vel), float(y_vel)
        else:
            raise ValueError(
                f"Action length is less than 2: {len(action)}. Cannot extract x and y velocity."
            )

    def sum_actions(
        self,
        action_1: NDArray[np.integer],
        action_2: NDArray[np.integer],
        weight: float = 0.5,
    ) -> NDArray[np.integer]:
        summed_action = np.copy(action_1)
        for i in range(len(action_1)):
            summed_value = weight * action_1[i] + (1 - weight) * action_2[i]
            # Ensure the summed value is within valid range
            summed_value = max(0, min(self.num_discrete - 1, summed_value))
            summed_action[i] = summed_value
        return summed_action


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
            x_vel, y_vel = self.action2xy_vel(action, agent)
            agent.action.u[0] += x_vel
            agent.action.u[1] += y_vel

            self._apply_sensitivity(agent, agent.accel)

        if not agent.silent:
            agent.action.c = action[-1]

    def action2xy_vel(
        self, action: NDArray[np.float64], agent: Agent
    ) -> tuple[float, float]:
        if len(action) >= 4:
            x_vel = float(action[0] - action[1])
            y_vel = float(action[2] - action[3])
            return x_vel, y_vel
        else:
            raise ValueError(
                f"Action length is less than 4: {len(action)}. Cannot extract x and y velocity."
            )


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
            x_vel, y_vel = self.action2xy_vel(action, agent)
            agent.action.u[0] += x_vel
            agent.action.u[1] += y_vel

            self._apply_sensitivity(agent, agent.accel)

        if not agent.silent:
            agent.action.c = action[-1]

    def action2xy_vel(
        self, action: NDArray[np.integer], agent: Agent
    ) -> tuple[float, float]:
        if len(action) >= 1:
            x_vel = 0.0
            y_vel = 0.0
            if action[0] == 1:  # up
                y_vel += 1.0
            elif action[0] == 2:  # down
                y_vel -= 1.0
            elif action[0] == 3:  # left
                x_vel -= 1.0
            elif action[0] == 4:  # right
                x_vel += 1.0
            return x_vel, y_vel
        else:
            raise ValueError(
                f"Action length is less than 1: {len(action)}. Cannot extract x and y velocity."
            )


class DiscreteAngDirectional(ActionMode[NDArray[np.integer]]):
    """
    Discrete action for angle direction (0 to num_directions-1) and discrete action for velocity magnitude (0 to u_range) with num_vel_discrete bins,
    and communication (i.e., 1 dimension).
    """

    name: str = "discrete_ang_directional"

    def __init__(self, num_directions: int, num_vel_discrete: int) -> None:
        self.num_directions = num_directions
        self.num_vel_discrete = num_vel_discrete

    def define_action_space(self, agent: Agent) -> Space[NDArray[np.integer]]:
        nvecs: list[int] = []
        if agent.movable:
            nvecs += [self.num_directions]  # discrete directions
            nvecs += [self.num_vel_discrete]  # discrete velocity magnitude
        if not agent.silent:
            nvecs += [1]  # communication

        return MultiDiscrete(nvecs, dtype=np.int64)

    def update_agent_action(
        self, agent: Agent, action: NDArray[np.integer], world: World
    ) -> None:
        self._zero_agent_actions(agent, world)

        if agent.movable:
            x_vel, y_vel = self.action2xy_vel(action, agent)
            agent.action.u[0] += x_vel
            agent.action.u[1] += y_vel

            self._apply_sensitivity(agent, agent.accel)

        if not agent.silent:
            agent.action.c = action[-1]

    def action2xy_vel(
        self, action: NDArray[np.integer], agent: Agent
    ) -> tuple[float, float]:
        if len(action) >= 2:
            direction_idx = int(action[0])
            vel_idx = int(action[1])

            angle = (direction_idx / self.num_directions) * 2 * np.pi
            magnitude = (vel_idx / self.num_vel_discrete) * agent.u_range

            x_vel = magnitude * np.cos(angle)
            y_vel = magnitude * np.sin(angle)

            return x_vel, y_vel
        else:
            raise ValueError(
                f"Action length is less than 2: {len(action)}. Cannot extract x and y velocity."
            )

    def sum_actions(
        self,
        action_1: NDArray[np.integer],
        action_2: NDArray[np.integer],
        weight: float = 0.5,
    ) -> NDArray[np.integer]:
        """
        Sum two discrete ang-directional actions using weighted average for direction and velocity.

        Parameters
        ----------
        action_1 : NDArray[np.integer]
            The first action to be summed. Shape should be (n_envs, 2) where the first element is direction index and the second is velocity index.
        action_2 : NDArray[np.integer]
            The second action to be summed. Shape should be (n_envs, 2) where the first element is direction index and the second is velocity index.
        weight : float, optional
            The weight for the first action in the summation, by default 0.5.

        Returns
        -------
        summed_action: NDArray[np.integer]
            The resulting summed action. Shape is (n_envs, 2) where the first element is direction index and the second is velocity index.

        """
        # Extract direction and velocity indices for both actions
        # Shape: (n_envs,)
        dir_1 = action_1[:, 0]
        vel_1 = action_1[:, 1]
        dir_2 = action_2[:, 0]
        vel_2 = action_2[:, 1]

        # Convert direction indices to angles
        angle_1 = (dir_1 / self.num_directions) * 2 * np.pi
        angle_2 = (dir_2 / self.num_directions) * 2 * np.pi

        # Convert velocity indices to magnitudes (normalized to [0, 1])
        mag_1 = vel_1 / self.num_vel_discrete
        mag_2 = vel_2 / self.num_vel_discrete

        # Convert to Cartesian coordinates
        x_1 = mag_1 * np.cos(angle_1)
        y_1 = mag_1 * np.sin(angle_1)
        x_2 = mag_2 * np.cos(angle_2)
        y_2 = mag_2 * np.sin(angle_2)

        # Weighted sum in Cartesian space
        x_sum = weight * x_1 + (1 - weight) * x_2
        y_sum = weight * y_1 + (1 - weight) * y_2

        # Convert back to polar coordinates
        mag_sum = np.sqrt(x_sum**2 + y_sum**2)
        angle_sum = np.arctan2(y_sum, x_sum)

        # Normalize angle to [0, 2*pi]
        angle_sum = np.mod(angle_sum, 2 * np.pi)

        # Convert back to discrete indices
        dir_sum = (angle_sum / (2 * np.pi)) * self.num_directions
        # Handle wrap-around for direction
        dir_sum = np.mod(dir_sum, self.num_directions)

        vel_sum = mag_sum * self.num_vel_discrete
        # Clip velocity to valid range
        vel_sum = np.clip(vel_sum, 0, self.num_vel_discrete - 1)

        # Stack to create output shape (n_envs, 2)
        summed_action = np.stack([dir_sum, vel_sum], axis=1)

        return summed_action
