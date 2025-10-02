from enum import Enum, auto
from typing import Generic, TypeVar

import numpy as np
from numpy.typing import NDArray


class EntityState:  # physical/external base state of all entities
    pos: NDArray[np.float64]
    vel: NDArray[np.float64]
    rot: float
    ang_vel: float

    def __init__(self):
        # physical position
        self.pos: NDArray[np.float64] = np.array([np.nan, np.nan], dtype=np.float64)
        # physical velocity
        self.vel: NDArray[np.float64] = np.array([0.0, 0.0], dtype=np.float64)

        # To be implemented in future versions
        # physical rotation -- from -pi to pi, 0 means facing right, pi/2 means facing up
        self.rot: float = 0.0
        # angular velocity
        self.ang_vel: float = 0.0


EntityStateT = TypeVar("EntityStateT", bound=EntityState, covariant=True)


class EntityShape(Enum):
    CIRCLE = auto()
    SQUARE = auto()


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
    color: str | None
    max_speed: float | None
    accel: float | None
    state: EntityStateT
    initial_mass: float

    def __init__(
        self,
        name: str = "",
        size: float = 0.050,
        shape: EntityShape = EntityShape.CIRCLE,
        movable: bool = False,
        rotatable: bool = False,
        collide: bool = True,
        density: float = 25.0,
        color: str | None = None,
        max_speed: float | None = None,
        accel: float | None = None,
        state: EntityStateT = EntityState(),
        initial_mass: float = 1.0,
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
        self.color = color
        # max speed and accel
        self.max_speed = max_speed
        self.accel = accel
        # state
        self.state = state
        # mass
        self.initial_mass = initial_mass

    @property
    def mass(self):
        return self.initial_mass


class Landmark(Entity[EntityState]):  # properties of landmark entities
    def __init__(
        self,
        name: str = "",
        size: float = 0.05,
        shape: EntityShape = EntityShape.CIRCLE,
        movable: bool = False,
        collide: bool = True,
        density: float = 25,
        color: str | None = None,
        max_speed: float | None = None,
        accel: float | None = None,
        state: EntityState = EntityState(),
        initial_mass: float = 1,
    ):
        super().__init__(
            name,
            size,
            shape,
            movable,
            collide,
            density,
            color,
            max_speed,
            accel,
            state,
            initial_mass,
        )
