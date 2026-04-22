"""Observation factory classes for the Rooms environment."""

import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from contgrid.core import Agent, BaseObsFactory


class ZoneDistObsFactory(BaseObsFactory):
    """Factory for zone distance observations (white, yellow, black, red)."""

    def __init__(
        self,
        room_scale: float,
        name: str = "zone_dist",
    ):
        self.room_scale = room_scale
        self.name = name

    def obs_space_dict(
        self,
        num_zones: int,
        low_bound: NDArray[np.float64],
        high_bound: NDArray[np.float64],
    ) -> dict[str, spaces.Space]:

        return {
            self.name: spaces.Box(
                low=np.stack([low_bound] * num_zones),
                high=np.stack([high_bound] * num_zones),
                dtype=np.float64,
            )
        }

    def observation(
        self, agent: Agent, zone_pos: NDArray[np.float64]
    ) -> dict[str, NDArray[np.float64]]:
        if len(zone_pos) == 0:
            return {self.name: np.array([], dtype=np.float64)}

        rel_pos = (zone_pos - agent.state.pos) / self.room_scale
        distances = self._compute_distances(agent.state.pos, zone_pos)
        sorted_indices = np.argsort(distances)
        sorted_rel_pos = rel_pos[sorted_indices]

        return {self.name: sorted_rel_pos}


class VisitCountObsFactory(BaseObsFactory):
    """Factory for visitation count observations for the four zones (yellow, red, white, black)."""

    def __init__(
        self,
        name: str = "zone_visits",
    ):
        self.name = name

    def obs_space_dict(self) -> dict[str, spaces.Space]:
        num_zones: int = 4
        # Binary visitation counts (0 or 1) for each zone, hence MultiDiscrete with 2 options per zone
        return {self.name: spaces.MultiDiscrete([2] * num_zones)}

    def observation(
        self, agent: Agent, visitation_counts: NDArray[np.int32]
    ) -> dict[str, NDArray[np.int32]]:
        if len(visitation_counts) == 0:
            return {self.name: np.array([0, 0, 0, 0], dtype=np.int32)}

        # Clip visitation counts to 1 to indicate whether the agent has visited each zone at least once
        visitation_counts = np.clip(visitation_counts, 0, 1).astype(np.int32)
        return {self.name: visitation_counts}
