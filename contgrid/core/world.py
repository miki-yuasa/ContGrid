from typing import Callable, Generic, TypeVar

import numpy as np
from numpy.typing import NDArray


class EntityState:  # physical/external base state of all entities
    p_pos: NDArray[np.float64]
    p_vel: NDArray[np.float64]

    def __init__(self):
        # physical position
        self.p_pos: NDArray[np.float64] = np.array([np.nan, np.nan], dtype=np.float64)
        self.p_vel: NDArray[np.float64] = np.array([0.0, 0.0], dtype=np.float64)


EntityStateT = TypeVar("EntityStateT", bound=EntityState, covariant=True)


class AgentState(
    EntityState
):  # state of agents (including communication and internal/mental state)
    def __init__(self):
        super().__init__()
        # communication utterance
        self.c: NDArray[np.float64] = np.array(0.0, dtype=np.float64)


class Action:  # action of the agent
    def __init__(self):
        # physical action
        self.u: NDArray[np.float64] = np.array([0.0, 0.0], dtype=np.float64)
        # communication action
        self.c: NDArray[np.float64] = np.array(0.0, dtype=np.float64)


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
    movable: bool
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
        movable: bool = False,
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
        # entity can move / be pushed
        self.movable = movable
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
    def __init__(self):
        super().__init__()


class Agent(Entity[AgentState]):  # properties of agent entities
    silent: bool
    blind: bool
    u_noise: float | None
    c_noise: float | None
    u_range: float
    action: Action
    action_callback: Callable[["Agent", "World"], Action] | None

    def __init__(
        self,
        name: str = "",
        size: float = 0.05,
        movable: bool = True,
        collide: bool = True,
        density: float = 25,
        color: str | None = None,
        max_speed: float | None = None,
        accel: float | None = None,
        state: AgentState = AgentState(),
        initial_mass: float = 1,
        silent: bool = False,
        blind: bool = False,
        u_noise: float | None = None,
        c_noise: float | None = None,
        u_range: float = 1.0,
        action: Action = Action(),
        action_callback: Callable[["Agent", "World"], Action] | None = None,
    ):
        super().__init__(
            name,
            size,
            movable,
            collide,
            density,
            color,
            max_speed,
            accel,
            state,
            initial_mass,
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
        self.action = Action()
        # script behavior to execute
        self.action_callback = action_callback


class World:  # multi-agent world
    def __init__(self):
        # list of agents and entities (can change at execution-time!)
        self.agents: list[Agent] = []
        self.landmarks: list[Landmark] = []
        # communication channel dimensionality
        self.dim_c: int = 0
        # position dimensionality
        self.dim_p: int = 2
        # color dimensionality
        self.dim_color: int = 3
        # simulation timestep
        self.dt: float = 0.1
        # physical damping
        self.damping: float = 0.25
        # contact response parameters
        self.contact_force: float = 1e2
        self.contact_margin: float = 1e-3

    # return all entities in the world
    @property
    def entities(self) -> list[Agent | Landmark]:
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self) -> list[Agent]:
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self) -> list[Agent]:
        return [agent for agent in self.agents if agent.action_callback is not None]

    # update state of the world
    def step(self) -> None:
        # set actions for scripted agents
        for agent in self.scripted_agents:
            if agent.action_callback:
                agent.action = agent.action_callback(agent, self)

        # gather forces applied to entities
        p_force = [None] * len(self.entities)
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(
        self, p_force: list[None]
    ) -> list[NDArray[np.float64] | None]:
        # set applied forces
        new_p_force: list[None | NDArray[np.float64]] = [p for p in p_force]
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = (
                    np.random.randn(*agent.action.u.shape) * agent.u_noise
                    if agent.u_noise
                    else 0.0
                )
                new_p_force[i] = agent.action.u + noise
        return new_p_force

    # gather physical forces acting on entities
    def apply_environment_force(
        self, p_force: list[None | NDArray[np.float64]]
    ) -> list[None | NDArray[np.float64]]:
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if f_a is not None:
                    if p_force[a] is None:
                        p_force[a] = np.array(0.0, dtype=np.float64)
                    p_force[a] = f_a + p_force[a]
                if f_b is not None:
                    if p_force[b] is None:
                        p_force[b] = np.array(0.0, dtype=np.float64)
                    p_force[b] = f_b + p_force[b]
        return p_force

    # integrate physical state
    def integrate_state(self, p_force: list[None | NDArray[np.float64]]):
        for i, entity in enumerate(self.entities):
            if not entity.movable:
                continue
            entity.state.p_pos += entity.state.p_vel * self.dt
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(
                    np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1])
                )
                if speed > entity.max_speed:
                    entity.state.p_vel = (
                        entity.state.p_vel
                        / np.sqrt(
                            np.square(entity.state.p_vel[0])
                            + np.square(entity.state.p_vel[1])
                        )
                        * entity.max_speed
                    )

    def update_agent_state(self, agent: Agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = (
                np.random.randn(*agent.action.c.shape) * agent.c_noise
                if agent.c_noise
                else 0.0
            )
            agent.state.c = agent.action.c + noise

    # get collision forces for any contact between two entities
    def get_collision_force(
        self, entity_a: Entity[EntityStateT], entity_b: Entity[EntityStateT]
    ) -> list[NDArray[np.float64] | None]:
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos: NDArray[np.float64] = entity_a.state.p_pos - entity_b.state.p_pos
        dist: float = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min: float = entity_a.size + entity_b.size
        # softmax penetration
        k: float = self.contact_margin
        penetration: float = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.contact_force * delta_pos / dist * penetration
        force_a: NDArray[np.float64] | None = +force if entity_a.movable else None
        force_b: NDArray[np.float64] | None = -force if entity_b.movable else None
        return [force_a, force_b]
