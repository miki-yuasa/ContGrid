from typing import Any, Callable, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from .const import Color
from .entities import Entity, EntityShape, EntityState, ResetConfig

CallbackT = TypeVar("CallbackT")


class AgentState(
    EntityState
):  # state of agents (including communication and internal/mental state)
    def __init__(
        self,
        pos: NDArray[np.float64] | None = None,
        vel: NDArray[np.float64] | None = None,
        rot: float = 0,
        ang_vel: float = 0,
        c: NDArray[np.float64] | None = None,
    ) -> None:
        super().__init__(pos, vel, rot, ang_vel)
        # communication utterance
        self.c: NDArray[np.float64] = (
            c if c is not None else np.array(0.0, dtype=np.float64)
        )


class Action:  # action of the agent
    def __init__(self):
        # physical action
        self.u: NDArray[np.float64] = np.array([0.0, 0.0], dtype=np.float64)
        # communication action
        self.c: NDArray[np.float64] = np.array(0.0, dtype=np.float64)


class AgentConfig(BaseModel, Generic[CallbackT]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = ""
    size: float = 0.25
    shape: EntityShape = EntityShape.CIRCLE
    movable: bool = True
    rotatable: bool = False
    collide: bool = True
    density: float = 25
    color: str = Color.SKY_BLUE.name
    max_speed: float | None = None
    accel: float = 5.0
    state: AgentState = AgentState()
    initial_mass: float = 1
    silent: bool = True
    blind: bool = False
    u_noise: float | None = None
    c_noise: float | None = None
    u_range: float = 10.0
    action: Action = Action()
    action_callback: Callable[["Agent", CallbackT], Action] | None = None


class Agent(Entity[AgentState], Generic[CallbackT]):  # properties of agent entities
    silent: bool
    blind: bool
    u_noise: float | None
    c_noise: float | None
    u_range: float
    action: Action
    action_callback: Callable[["Agent", CallbackT], Action] | None
    terminated: bool  # whether the agent has finished its task

    def __init__(
        self,
        name: str = "",
        size: float = 0.25,
        shape: EntityShape = EntityShape.CIRCLE,
        movable: bool = True,
        rotatable: bool = False,
        collide: bool = True,
        density: float = 25,
        color: str = Color.SKY_BLUE.name,
        max_speed: float | None = None,
        accel: float = 5.0,
        state: AgentState = AgentState(),
        initial_mass: float = 1,
        reset_config: ResetConfig = ResetConfig(),
        silent: bool = True,
        blind: bool = False,
        u_noise: float | None = None,
        c_noise: float | None = None,
        u_range: float = 10.0,
        action: Action = Action(),
        action_callback: Callable[["Agent", CallbackT], Action] | None = None,
        draw_pos_offset: NDArray[np.float64] = np.array([0.0, 0.0], dtype=np.float64),
    ):
        super().__init__(
            name,
            size,
            shape,
            movable,
            rotatable,
            collide,
            density,
            color,
            max_speed,
            accel,
            state,
            initial_mass,
            reset_config=reset_config,
            draw_pos_offset=draw_pos_offset,
        )
        # agents are movable by default
        # cannot send communication signals
        self.silent = silent
        # cannot observe the world
        self.blind = blind
        # physical motor noise amount
        self.u_noise = u_noise
        # communication noise amount
        self.c_noise = c_noise
        # control range
        self.u_range = u_range
        # action
        self.action = action
        # script behavior to execute
        self.action_callback = action_callback

        # whether the agent has finished its task
        self.terminated = False

    def reset(
        self, np_random: np.random.Generator, options: dict[str, Any] = {}
    ) -> None:
        super().reset(np_random)
        self.terminated = False
