from abc import ABC, abstractmethod

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from .world import Agent, World


class BaseScenario(ABC):  # defines scenario upon which the world is built
    @abstractmethod
    def make_world(self):  # create elements of the world
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
