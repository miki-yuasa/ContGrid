from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from .entities import Landmark
from .world import DEFAULT_WORLD_CONFIG, Agent, World, WorldConfig, WorldConfigT

ScenarioConfigT = TypeVar("ScenarioConfigT")


class BaseScenario(
    ABC, Generic[ScenarioConfigT]
):  # defines scenario upon which the world is built
    def __init__(self, config: ScenarioConfigT | None = None) -> None:
        self.config = config

    def make_world(
        self,
        world_config: WorldConfig = DEFAULT_WORLD_CONFIG,
        verbose: bool = False,
    ) -> World:  # create elements of the world
        world = World(
            grid=world_config.grid,
            dt=world_config.dt,
            dim_c=world_config.dim_c,
            contact_margin=world_config.contact_margin,
            collision_force=world_config.collision_force,
            drag=world_config.drag,
            verbose=verbose,
        )
        # add agents
        world.agents = self.init_agents(world)
        # add landmarks
        world.landmarks = self.init_landmarks(world)

        return world

    @abstractmethod
    def init_agents(
        self, world: World, np_random: np.random.Generator | None = None
    ) -> list[Agent]:
        pass

    @abstractmethod
    def init_landmarks(
        self, world: World, np_random: np.random.Generator | None = None
    ) -> list[Landmark]:
        pass

    def reset_world(
        self, world: World, np_random: np.random.Generator
    ) -> None:  # create initial conditions of the world
        # reset agents
        world.agents = self.reset_agents(world, np_random)
        # reset landmarks
        world.landmarks = self.reset_landmarks(world, np_random)

    @abstractmethod
    def reset_agents(self, world: World, np_random: np.random.Generator) -> list[Agent]:
        pass

    @abstractmethod
    def reset_landmarks(
        self, world: World, np_random: np.random.Generator
    ) -> list[Landmark]:
        pass

    @abstractmethod
    def observation(self, agent: Agent, world: World) -> NDArray[np.float64]:
        pass

    @abstractmethod
    def observation_space(self, agent: Agent, world: World) -> spaces.Space:
        pass

    def global_reward(self, world: World) -> float:
        return 0.0

    def reward(self, agent: Agent, world: World) -> float:
        return 0.0
