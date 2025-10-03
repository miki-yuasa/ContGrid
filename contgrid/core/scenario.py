from abc import ABC, abstractmethod
from typing import Generic, TypeVar

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from .world import DEFAULT_WORLD_CONFIG, Agent, World, WorldConfig, WorldConfigT

ScenarioConfigT = TypeVar("ScenarioConfigT")


class BaseScenario(
    ABC, Generic[WorldConfigT, ScenarioConfigT]
):  # defines scenario upon which the world is built
    @abstractmethod
    def make_world(
        self,
        world_config: WorldConfigT = DEFAULT_WORLD_CONFIG,
        config: ScenarioConfigT | None = None,
    ) -> World:  # create elements of the world
        pass

    @abstractmethod
    def reset_world(
        self, world: World, np_random: np.random.Generator
    ):  # create initial conditions of the world
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
