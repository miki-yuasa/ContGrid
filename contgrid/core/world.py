from typing import TypeVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from .agent import Agent
from .const import COLLISION_FORCE, CONTACT_MARGIN, DRAG, Color
from .entities import Entity, EntityShape, EntityState, EntityStateT, Landmark
from .grid import DEFAULT_GRID, Grid

# Try to import the accelerated version, fall back to pure Python
try:
    from .grid_rust import WallCollisionCheckerAccelerated as WallCollisionChecker
except ImportError:
    from .grid import WallCollisionChecker  # type: ignore


class WorldConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    grid: Grid = DEFAULT_GRID
    dt: float = 0.1
    dim_c: int = 0
    drag: float = DRAG
    collision_force: float = COLLISION_FORCE
    contact_margin: float = CONTACT_MARGIN


DEFAULT_WORLD_CONFIG = WorldConfig()

WorldConfigT = TypeVar("WorldConfigT", bound=BaseModel)


class World:  # multi-agent world
    def __init__(
        self,
        grid: Grid = DEFAULT_GRID,
        dt: float = 0.1,
        dim_c: int = 0,
        drag: float = DRAG,
        collision_force: float = COLLISION_FORCE,
        contact_margin: float = CONTACT_MARGIN,
        verbose: bool = False,
    ):
        # list of agents and entities (can change at execution-time!)
        self.agents: list[Agent["World"]] = []
        self.landmarks: list[Landmark] = []
        # grid
        self.grid: Grid = grid
        # communication channel dimensionality
        self.dim_c: int = dim_c
        # position dimensionality
        self.dim_p: int = 2
        # color dimensionality
        self.dim_color: int = 3
        # simulation timestep
        self.dt: float = dt
        # physical drag
        self.drag: float = drag
        # contact response parameters
        self.collision_force: float = collision_force
        self.contact_margin: float = contact_margin

        # grid size
        self.grid_size: float = self.grid.cell_size

        # verbosity
        self.verbose: bool = verbose

        # initialize wall collision checker and wall entities
        self.wall_collision_checker: WallCollisionChecker
        self.walls: list[Landmark]
        self.wall_collision_checker, self.walls = self._init_walls(self.grid)

        # numeric grid representation
        self.numeric_grid: NDArray[np.int_] = self.get_numeric_grid()

    def _init_walls(self, grid: Grid) -> tuple[WallCollisionChecker, list[Landmark]]:
        wall_collision_checker = WallCollisionChecker(
            grid.layout, grid.cell_size, self.verbose
        )

        walls: list[Landmark] = []
        for i, (x, y) in enumerate(wall_collision_checker.wall_anchors):
            wall = Landmark(
                name=f"wall_{i}",
                size=self.grid.cell_size,
                shape=EntityShape.SQUARE,
                movable=False,
                rotatable=False,
                collide=True,
                density=1000,
                color=Color.GREY.name,
                max_speed=None,
                accel=0.0,
                state=EntityState(
                    pos=np.array([x, y], dtype=np.float64),
                    vel=None,
                    rot=0.0,
                    ang_vel=0.0,
                ),
                initial_mass=1000,
            )
            walls.append(wall)

        return wall_collision_checker, walls

    def get_numeric_grid(self) -> NDArray[np.int_]:
        """
        Replace "#" with 1 and " " with 0 in the grid layout.
        """
        new_grid: list[list[int]] = []
        for row in self.grid.layout:
            new_row: list[int] = []
            for cell in row:
                if cell == "#":
                    new_row.append(1)
                elif cell == " ":
                    new_row.append(0)
                else:
                    new_row.append(int(cell))
            new_grid.append(new_row)
        numeric_grid = np.array(new_grid, dtype=np.int_)
        return numeric_grid

    # return all entities in the world
    @property
    def entities(self) -> list[Agent | Landmark]:
        return self.agents + self.landmarks

    @property
    def all_entities(self) -> list[Agent | Landmark]:
        return self.walls + self.landmarks + self.agents

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

        # gather forces and torques applied to entities
        forces: list[None | NDArray[np.float64]] = [None] * len(self.entities)
        # apply agent physical controls
        forces = self.apply_action_force(forces)
        # apply environment forces
        forces = self.apply_environment_force(forces)
        # apply wall collision forces
        forces = self.apply_wall_collision_forces(forces)
        if self.verbose:
            print("forces:", forces)
        # integrate physical state
        self.integrate_state(forces)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

    # gather agent action forces
    def apply_action_force(
        self, forces: list[None | NDArray[np.float64]]
    ) -> list[NDArray[np.float64] | None]:
        # set applied forces
        new_forces: list[None | NDArray[np.float64]] = [p for p in forces]
        for i, agent in enumerate(self.agents):
            if agent.movable:
                noise = (
                    np.random.randn(*agent.action.u.shape) * agent.u_noise
                    if agent.u_noise
                    else 0.0
                )
                new_forces[i] = agent.action.u + noise
        return new_forces

    # gather physical forces acting on entities
    def apply_environment_force(
        self, forces: list[None | NDArray[np.float64]]
    ) -> list[None | NDArray[np.float64]]:
        # simple (but inefficient) collision response
        new_forces: list[None | NDArray[np.float64]] = [p for p in forces]
        for a, entity_a in enumerate(self.entities):
            for b, entity_b in enumerate(self.entities):
                if b <= a:
                    continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if f_a is not None:
                    a_forces = new_forces[a]
                    if a_forces is None:
                        a_forces = np.array(0.0, dtype=np.float64)
                    new_forces[a] = f_a + a_forces
                if f_b is not None:
                    b_forces = new_forces[b]
                    if b_forces is None:
                        b_forces = np.array(0.0, dtype=np.float64)
                    new_forces[b] = f_b + b_forces
        return new_forces

    # apply wall collision forces
    def apply_wall_collision_forces(
        self, forces: list[None | NDArray[np.float64]]
    ) -> list[None | NDArray[np.float64]]:
        new_forces: list[None | NDArray[np.float64]] = [p for p in forces]
        for i, entity in enumerate(self.entities):
            if isinstance(entity, Agent):
                f = self.get_wall_collision_force(entity)
                if f is not None:
                    entity_forces = new_forces[i]
                    if entity_forces is None:
                        entity_forces = np.array(0.0, dtype=np.float64)
                    new_forces[i] = f + entity_forces
        return new_forces

    # integrate physical state
    def integrate_state(self, forces: list[None | NDArray[np.float64]]) -> None:
        for force, entity in zip(forces, self.entities):
            if not entity.movable:
                continue
            new_pos = entity.state.pos + entity.state.vel * self.dt
            if not self.grid.allow_wall_overlap:
                # Clip new position to avoid wall collisions
                clipped_pos = self.wall_collision_checker.clip_new_position(
                    entity.size,
                    self.contact_margin,
                    tuple(entity.state.pos),
                    tuple(new_pos),
                )
                if self.verbose:
                    print(
                        f"Entity {entity.name} "
                        f"Old pos: {entity.state.pos}, New pos: {new_pos}, Clipped pos: {clipped_pos}"
                    )
                new_pos = clipped_pos
            else:
                pass

            entity.state.pos = np.array(new_pos, dtype=np.float64)
            entity.state.vel = entity.state.vel * (1 - self.drag)
            if force is not None:
                entity.state.vel += (force / entity.mass) * self.dt
            if entity.max_speed:
                speed = np.sqrt(
                    np.square(entity.state.vel[0]) + np.square(entity.state.vel[1])
                )
                if speed > entity.max_speed:
                    entity.state.vel = (
                        entity.state.vel
                        / np.sqrt(
                            np.square(entity.state.vel[0])
                            + np.square(entity.state.vel[1])
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
        delta_pos: NDArray[np.float64] = entity_a.state.pos - entity_b.state.pos
        dist: float = np.sqrt(np.sum(np.square(delta_pos)))
        # minimum allowable distance
        dist_min: float = entity_a.size + entity_b.size
        # softmax penetration
        k: float = self.contact_margin
        penetration: float = np.logaddexp(0, -(dist - dist_min) / k) * k
        force = self.collision_force * delta_pos / dist * penetration
        force_a: NDArray[np.float64] | None = +force if entity_a.movable else None
        force_b: NDArray[np.float64] | None = -force if entity_b.movable else None
        return [force_a, force_b]

    def get_wall_collision_force(self, agent: Agent) -> NDArray[np.float64] | None:
        if not agent.collide or not agent.movable:
            return None  # not a collider

        # Check for collision using the WallCollisionChecker
        collided, force = self.wall_collision_checker.is_collision(
            agent.size,
            self.contact_margin,
            tuple(agent.state.pos),
            self.collision_force,
            self.contact_margin,
        )
        if collided:
            return force * self.collision_force
