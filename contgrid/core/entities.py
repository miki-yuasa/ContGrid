from enum import Enum, auto
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from .const import Color
from .typing import Position


class EntityState:  # physical/external base state of all entities
    pos: NDArray[np.float64]
    vel: NDArray[np.float64]
    rot: float
    ang_vel: float

    def __init__(
        self,
        pos: NDArray[np.float64] | None = None,
        vel: NDArray[np.float64] | None = None,
        rot: float = 0.0,
        ang_vel: float = 0.0,
    ) -> None:
        # physical position
        self.pos: NDArray[np.float64] = (
            pos if pos is not None else np.array([np.nan, np.nan], dtype=np.float64)
        )
        # physical velocity
        self.vel: NDArray[np.float64] = (
            vel if vel is not None else np.array([0.0, 0.0], dtype=np.float64)
        )

        # To be implemented in future versions
        # physical rotation -- from -pi to pi, 0 means facing right, pi/2 means facing up
        self.rot: float = rot
        # angular velocity
        self.ang_vel: float = ang_vel


EntityStateT = TypeVar("EntityStateT", bound=EntityState, covariant=True)


class EntityShape(Enum):
    CIRCLE = auto()
    SQUARE = auto()


class ResetConfig(BaseModel):
    spawn_positions: Position | list[Position] = []

    model_config = ConfigDict(arbitrary_types_allowed=True)


class Entity(Generic[EntityStateT]):  # properties and state of physical world entity
    """
    Properties and state of physical world entity

    Attributes
    ----------
    name : str
        Name of the entity
    size : float
        Size of the entity
    movable : bool
        Whether the entity can move or be pushed
    collide : bool
        Whether the entity collides with others
    density : float
        Material density (affects mass)
    color : str | None
        Color of the entity
    max_speed : float | None
        Maximum speed of the entity
    accel : float | None
        Acceleration of the entity
    state : EntityStateT
        State of the entity
    initial_mass : float
        Initial mass of the entity
    mass : float
        Mass of the entity (property)
    """

    name: str
    size: float
    shape: EntityShape
    movable: bool
    rotatable: bool
    collide: bool
    density: float
    color: Color = Color.WHITE
    max_speed: float | None
    accel: float
    state: EntityStateT
    initial_mass: float
    hatch: str
    reset_config: ResetConfig

    def __init__(
        self,
        name: str = "",
        size: float = 0.050,
        shape: EntityShape = EntityShape.CIRCLE,
        movable: bool = False,
        rotatable: bool = False,
        collide: bool = True,
        density: float = 25.0,
        color: str = Color.WHITE.name,
        max_speed: float | None = None,
        accel: float = 0.0,
        state: EntityStateT = EntityState(),
        initial_mass: float = 1.0,
        hatch: str = "",
        reset_config: ResetConfig = ResetConfig(),
    ):
        # name
        self.name = name
        # properties:
        self.size = size
        # shape
        self.shape = shape
        # entity can move / be pushed
        self.movable = movable
        # entity can rotate
        self.rotatable = rotatable
        # entity collides with others
        self.collide = collide
        # material density (affects mass)
        self.density = density
        # color
        self.color = Color[color]
        # max speed and accel
        self.max_speed = max_speed
        self.accel = accel
        # state
        self.state = state
        # mass
        self.initial_mass = initial_mass
        # hatch pattern for rendering
        self.hatch = hatch

        # reset configuration with some randomization options
        self.reset_config = reset_config

    @property
    def mass(self):
        return self.initial_mass

    def reset(self, np_random: np.random.Generator) -> None:
        """Reset the entity state based on reset configuration."""
        spawn_positions = self.reset_config.spawn_positions
        if isinstance(spawn_positions, list) and spawn_positions:
            chosen_pos = np_random.choice(len(spawn_positions))
            pos = spawn_positions[chosen_pos]
            self.state.pos = np.array([pos[0], pos[1]], dtype=np.float64)
        elif isinstance(spawn_positions, tuple) and spawn_positions:
            self.state.pos = np.array(
                [spawn_positions[0], spawn_positions[1]], dtype=np.float64
            )
        # Reset velocity and angular velocity
        self.state.vel = np.array([0.0, 0.0], dtype=np.float64)
        self.state.ang_vel = 0.0
        # Reset rotation
        self.state.rot = 0.0


class Landmark(Entity[EntityState]):  # properties of landmark entities
    def __init__(
        self,
        name: str = "",
        size: float = 0.5,
        shape: EntityShape = EntityShape.CIRCLE,
        movable: bool = False,
        rotatable: bool = False,
        collide: bool = False,
        density: float = 25,
        color: str = Color.GREEN.name,
        max_speed: float | None = None,
        accel: float = 0.0,
        state: EntityState = EntityState(),
        initial_mass: float = 1,
        hatch: str = "",
        reset_config: ResetConfig = ResetConfig(),
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
            hatch,
            reset_config,
        )
