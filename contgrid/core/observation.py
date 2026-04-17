"""Observation factory classes for the Rooms environment."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from .agent import Agent


class BaseObsFactory(ABC):
    """Base class for observation factories."""

    def __init__(self, room_scale: float):
        self.room_scale = room_scale

    @abstractmethod
    def obs_space_dict(self, *args: Any, **kwargs: Any) -> dict[str, spaces.Space]:
        """Return observation space dictionary."""
        pass

    @abstractmethod
    def observation(
        self, agent: Agent, *args: Any, **kwargs: Any
    ) -> dict[str, NDArray[np.float64]]:
        """Return observation dictionary."""
        pass

    def _normalize_distance(
        self, distance: float | np.floating | NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Normalize distance by room scale."""
        if isinstance(distance, (int, float, np.floating)):
            return np.array([distance], dtype=np.float64) / self.room_scale
        return distance.astype(np.float64) / self.room_scale

    def _compute_distances(
        self, agent_pos: NDArray[np.float64], target_positions: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Compute distances from agent to target positions."""
        return np.linalg.norm(target_positions - agent_pos, axis=1)

    def _find_closest(
        self, agent_pos: NDArray[np.float64], positions: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """Find closest position from a set of positions."""
        if len(positions) == 0:
            return np.array([np.inf, np.inf], dtype=np.float64)
        distances = self._compute_distances(agent_pos, positions)
        return positions[int(np.argmin(distances))]
