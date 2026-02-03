"""Observation factory classes for the Rooms environment."""

from abc import ABC, abstractmethod
from typing import Any, Literal

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from contgrid.core import Agent


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


class GoalDistObsFactory(BaseObsFactory):
    """Factory for goal distance observations."""

    def __init__(self, room_scale: float, dist_mode: Literal["all", "none"]):
        super().__init__(room_scale)
        self.dist_mode = dist_mode

    def obs_space_dict(self, max_dist: float) -> dict[str, spaces.Space]:
        if self.dist_mode != "all":
            return {}
        return {
            "goal_dist": spaces.Box(
                low=0.0, high=max_dist, shape=(1,), dtype=np.float64
            )
        }

    def observation(
        self, agent: Agent, goal_pos: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        if self.dist_mode != "all":
            return {}
        goal_dist = np.linalg.norm(agent.state.pos - goal_pos)
        return {"goal_dist": self._normalize_distance(goal_dist)}


class DoorwayDistObsFactory(BaseObsFactory):
    """Factory for doorway distance observations."""

    def __init__(
        self,
        room_scale: float,
        dist_mode: Literal["all", "none"],
        name: str = "doorway_dist",
    ):
        super().__init__(room_scale)
        self.name = name
        self.dist_mode = dist_mode

    def obs_space_dict(
        self, num_doorways: int, max_dist: float
    ) -> dict[str, spaces.Space]:
        if self.dist_mode != "all":
            return {}
        return {
            self.name: spaces.Box(
                low=0.0, high=max_dist, shape=(num_doorways,), dtype=np.float64
            )
        }

    def observation(
        self, agent: Agent, doorway_pos: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        if self.dist_mode != "all":
            return {}
        distances = self._compute_distances(agent.state.pos, doorway_pos)
        return {self.name: self._normalize_distance(distances)}


class ObstacleDistObsFactory(BaseObsFactory):
    """Factory for obstacle distance observations (lavas and holes)."""

    def __init__(
        self,
        room_scale: float,
        dist_mode: Literal["closest", "all", "none"],
        name: str = "obstacle_dist",
    ):
        super().__init__(room_scale)
        self.name = name
        self.dist_mode = dist_mode

    def obs_space_dict(
        self, num_obstacles: int, max_dist: float
    ) -> dict[str, spaces.Space]:
        if self.dist_mode == "none":
            return {}
        shape = (num_obstacles,) if self.dist_mode == "all" else (1,)
        return {
            self.name: spaces.Box(low=0.0, high=max_dist, shape=shape, dtype=np.float64)
        }

    def observation(
        self, agent: Agent, obstacle_pos: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        if self.dist_mode == "none":
            return {}

        if len(obstacle_pos) == 0:
            return {self.name: np.array([np.inf], dtype=np.float64)}

        distances = self._compute_distances(agent.state.pos, obstacle_pos)

        if self.dist_mode == "all":
            return {self.name: self._normalize_distance(distances)}
        else:  # closest
            return {self.name: self._normalize_distance(np.min(distances))}


class ClosestObsPosObsFactory(BaseObsFactory):
    """Factory for closest obstacle position observations."""

    def __init__(self, room_scale: float, enabled: bool):
        super().__init__(room_scale)
        self.enabled = enabled

    def obs_space_dict(self) -> dict[str, spaces.Space]:
        if not self.enabled:
            return {}

        pos_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)
        return {
            "closest_lava_pos": pos_space,
            "closest_hole_pos": pos_space,
        }

    def observation(
        self, agent: Agent, lava_pos: NDArray[np.float64], hole_pos: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        if not self.enabled:
            return {}

        closest_lava = self._find_closest(agent.state.pos, lava_pos)
        closest_hole = self._find_closest(agent.state.pos, hole_pos)

        return {
            "closest_lava_pos": (closest_lava - agent.state.pos) / self.room_scale,
            "closest_hole_pos": (closest_hole - agent.state.pos) / self.room_scale,
        }
