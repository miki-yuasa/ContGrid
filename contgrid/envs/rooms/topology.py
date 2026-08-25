"""Room topology and path segment utilities."""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from contgrid.core import Grid
from contgrid.core.typing import Position


@dataclass
class LineSegment:
    """Represents a straight line segment in continuous space."""

    start: NDArray[np.float64]
    end: NDArray[np.float64]

    def sample_point(self, t: float) -> NDArray[np.float64]:
        """Sample a point at parameter t ∈ [0, 1] along the line."""
        return self.start + t * (self.end - self.start)

    def length(self) -> float:
        """Get the length of the line segment."""
        return float(np.linalg.norm(self.end - self.start))


# ── Four-room topology constants ──────────────────────────────────────────────

_FOUR_ROOM_NEIGHBOR_MAP: dict[str, list[str]] = {
    "ld": ["td", "bd"],
    "td": ["ld", "rd"],
    "rd": ["td", "bd"],
    "bd": ["ld", "rd"],
}

_FOUR_ROOM_BOUNDARIES: dict[str, dict[str, float]] = {
    "top_left": {"min_x": 1.0, "max_x": 5.0, "min_y": 7.0, "max_y": 11.0},
    "top_right": {"min_x": 7.0, "max_x": 11.5, "min_y": 6.0, "max_y": 11.0},
    "bottom_left": {"min_x": 1.0, "max_x": 5.0, "min_y": 1.0, "max_y": 5.0},
    "bottom_right": {"min_x": 7.0, "max_x": 11.5, "min_y": 1.0, "max_y": 4.0},
}

_FOUR_ROOM_CENTERS: dict[str, NDArray[np.float64]] = {
    "top_left": np.array([3.5, 9.0]),
    "top_right": np.array([9.0, 9.0]),
    "bottom_left": np.array([3.5, 4.0]),
    "bottom_right": np.array([9.0, 3.5]),
}

_FOUR_ROOM_DOORWAYS_MAP: dict[str, list[str]] = {
    "top_left": ["ld", "td"],
    "top_right": ["td", "rd"],
    "bottom_left": ["ld", "bd"],
    "bottom_right": ["rd", "bd"],
}

# ── Nine-room topology constants ─────────────────────────────────────────────

_NINE_ROOM_NEIGHBOR_MAP: dict[str, list[str]] = {
    # Horizontal doorways (connect left-right neighbors)
    "tl_tc": ["tl_ml", "tc_tr", "tc_mc"],
    "tc_tr": ["tl_tc", "tc_mc", "tr_mr"],
    "ml_mc": ["tl_ml", "ml_bl", "tc_mc", "mc_mr", "mc_bc"],
    "mc_mr": ["tc_mc", "ml_mc", "mc_bc", "tr_mr", "mr_br"],
    "bl_bc": ["ml_bl", "mc_bc", "bc_br"],
    "bc_br": ["bl_bc", "mc_bc", "mr_br"],
    # Vertical doorways (connect top-bottom neighbors)
    "tl_ml": ["tl_tc", "ml_mc", "ml_bl"],
    "tc_mc": ["tl_tc", "tc_tr", "ml_mc", "mc_mr", "mc_bc"],
    "tr_mr": ["tc_tr", "mc_mr", "mr_br"],
    "ml_bl": ["tl_ml", "ml_mc", "bl_bc"],
    "mc_bc": ["tc_mc", "ml_mc", "mc_mr", "bl_bc", "bc_br"],
    "mr_br": ["tr_mr", "mc_mr", "bc_br"],
}

_NINE_ROOM_BOUNDARIES: dict[str, dict[str, float]] = {
    "top_left": {"min_x": 1.0, "max_x": 5.0, "min_y": 13.0, "max_y": 17.0},
    "top_center": {"min_x": 7.0, "max_x": 11.0, "min_y": 13.0, "max_y": 17.0},
    "top_right": {"min_x": 13.0, "max_x": 17.0, "min_y": 13.0, "max_y": 17.0},
    "middle_left": {"min_x": 1.0, "max_x": 5.0, "min_y": 7.0, "max_y": 11.0},
    "middle_center": {"min_x": 7.0, "max_x": 11.0, "min_y": 7.0, "max_y": 11.0},
    "middle_right": {"min_x": 13.0, "max_x": 17.0, "min_y": 7.0, "max_y": 11.0},
    "bottom_left": {"min_x": 1.0, "max_x": 5.0, "min_y": 1.0, "max_y": 5.0},
    "bottom_center": {"min_x": 7.0, "max_x": 11.0, "min_y": 1.0, "max_y": 5.0},
    "bottom_right": {"min_x": 13.0, "max_x": 17.0, "min_y": 1.0, "max_y": 5.0},
}

_NINE_ROOM_CENTERS: dict[str, NDArray[np.float64]] = {
    "top_left": np.array([3.0, 15.0]),
    "top_center": np.array([9.0, 15.0]),
    "top_right": np.array([15.0, 15.0]),
    "middle_left": np.array([3.0, 9.0]),
    "middle_center": np.array([9.0, 9.0]),
    "middle_right": np.array([15.0, 9.0]),
    "bottom_left": np.array([3.0, 3.0]),
    "bottom_center": np.array([9.0, 3.0]),
    "bottom_right": np.array([15.0, 3.0]),
}

_NINE_ROOM_DOORWAYS_MAP: dict[str, list[str]] = {
    "top_left": ["tl_tc", "tl_ml"],
    "top_center": ["tl_tc", "tc_tr", "tc_mc"],
    "top_right": ["tc_tr", "tr_mr"],
    "middle_left": ["tl_ml", "ml_mc", "ml_bl"],
    "middle_center": ["tc_mc", "ml_mc", "mc_mr", "mc_bc"],
    "middle_right": ["tr_mr", "mc_mr", "mr_br"],
    "bottom_left": ["ml_bl", "bl_bc"],
    "bottom_center": ["mc_bc", "bl_bc", "bc_br"],
    "bottom_right": ["mr_br", "bc_br"],
}


class RoomTopology:
    """Defines the room structure and doorway connections."""

    def __init__(
        self,
        doorways: dict[str, Position],
        *,
        neighbor_map: dict[str, list[str]] | None = None,
        room_boundaries: dict[str, dict[str, float]] | None = None,
        room_centers: dict[str, NDArray[np.float64]] | None = None,
        room_doorways_map: dict[str, list[str]] | None = None,
    ):
        self.doorways = doorways

        is_nine_rooms = len(doorways) > 4 or any(
            k in _NINE_ROOM_NEIGHBOR_MAP for k in doorways
        )

        if neighbor_map is not None:
            self.neighbor_map = neighbor_map
        else:
            self.neighbor_map = (
                _NINE_ROOM_NEIGHBOR_MAP if is_nine_rooms else _FOUR_ROOM_NEIGHBOR_MAP
            )

        if room_boundaries is not None:
            self.room_boundaries = room_boundaries
        else:
            self.room_boundaries = (
                _NINE_ROOM_BOUNDARIES if is_nine_rooms else _FOUR_ROOM_BOUNDARIES
            )

        if room_centers is not None:
            self._room_centers = room_centers
        else:
            self._room_centers = (
                _NINE_ROOM_CENTERS if is_nine_rooms else _FOUR_ROOM_CENTERS
            )

        if room_doorways_map is not None:
            self.room_doorways_map = room_doorways_map
        else:
            self.room_doorways_map = (
                _NINE_ROOM_DOORWAYS_MAP
                if is_nine_rooms
                else _FOUR_ROOM_DOORWAYS_MAP
            )

    @classmethod
    def four_rooms(cls, doorways: dict[str, Position]) -> "RoomTopology":
        """Create a topology for the standard 4-room layout."""
        return cls(
            doorways,
            neighbor_map=_FOUR_ROOM_NEIGHBOR_MAP,
            room_boundaries=_FOUR_ROOM_BOUNDARIES,
            room_centers=_FOUR_ROOM_CENTERS,
            room_doorways_map=_FOUR_ROOM_DOORWAYS_MAP,
        )

    @classmethod
    def nine_rooms(cls, doorways: dict[str, Position]) -> "RoomTopology":
        """Create a topology for the 9-room (3×3) layout."""
        return cls(
            doorways,
            neighbor_map=_NINE_ROOM_NEIGHBOR_MAP,
            room_boundaries=_NINE_ROOM_BOUNDARIES,
            room_centers=_NINE_ROOM_CENTERS,
            room_doorways_map=_NINE_ROOM_DOORWAYS_MAP,
        )

    def get_room(self, position: Position | NDArray[np.float64]) -> str:
        """Determine which room a position belongs to based on boundaries."""
        pos = np.array(position, dtype=np.float64)
        x, y = float(pos[0]), float(pos[1])

        # Check each room's boundaries
        for room_name, bounds in self.room_boundaries.items():
            if (
                bounds["min_x"] <= x <= bounds["max_x"]
                and bounds["min_y"] <= y <= bounds["max_y"]
            ):
                return room_name

        # Fallback: if position is outside all boundaries (e.g., doorway), find closest room center
        min_dist = float("inf")
        closest_room = next(iter(self._room_centers))

        for room_name, center in self._room_centers.items():
            dist = np.linalg.norm(pos - center)
            if dist < min_dist:
                min_dist = dist
                closest_room = room_name

        return closest_room

    def get_neighbor_doorways(self, doorway_name: str) -> list[str]:
        """Get names of doorways that are neighbors to the given doorway."""
        return self.neighbor_map.get(doorway_name, [])

    def get_neighbor_pairs(self) -> list[tuple[str, str]]:
        """Get all unique neighbor doorway pairs derived from the neighbor map."""
        pairs: set[tuple[str, str]] = set()
        for doorway, neighbors in self.neighbor_map.items():
            for neighbor in neighbors:
                pair = tuple(sorted([doorway, neighbor]))
                pairs.add(pair)  # type: ignore[arg-type]
        return list(pairs)

    def get_doorways_in_room(
        self, position: Position, grid: Grid | None = None
    ) -> list[str]:
        """
        Determine which doorways belong to the room containing the given position.
        """
        if self.room_doorways_map:
            room = self.get_room(position)
            if room in self.room_doorways_map:
                doorways = [
                    d for d in self.room_doorways_map[room] if d in self.doorways
                ]
                if doorways:
                    return doorways

        distances = {
            name: np.linalg.norm(np.array(position) - np.array(pos))
            for name, pos in self.doorways.items()
        }
        sorted_doorways = sorted(distances.items(), key=lambda x: x[1])
        closest_two = [name for name, _ in sorted_doorways[:2]]

        if len(closest_two) == 2:
            if closest_two[1] in self.neighbor_map.get(closest_two[0], []):
                return closest_two

        closest = closest_two[0]
        return [closest] + self.neighbor_map.get(closest, [])[:1]


def get_relevant_path_segments(
    agent_pos: NDArray[np.float64],
    goal_pos: NDArray[np.float64],
    doorways: dict[str, NDArray[np.float64]],
    topology: RoomTopology,
    grid: Grid,
) -> list[LineSegment]:
    """
    Get relevant path segments based on agent position.
    Only includes paths between neighboring doorways and agent to its room's doorways.

    Parameters
    ----------
    agent_pos : Agent's current position
    goal_pos : Goal position
    doorways : Dictionary of doorway positions
    topology : Room topology information
    grid : Grid for spatial queries

    Returns
    -------
    List of LineSegment objects representing relevant paths
    """
    segments = []

    # 1. Find which room the agent is in
    agent_room_doorways = topology.get_doorways_in_room(tuple(agent_pos), grid)

    # 2. Add segments from agent to its room's doorways
    for doorway_name in agent_room_doorways:
        if doorway_name in doorways:
            segments.append(
                LineSegment(agent_pos.copy(), doorways[doorway_name].copy())
            )

    # 3. Add segments between neighboring doorways only
    added_pairs = set()
    for doorway_name, doorway_pos in doorways.items():
        neighbors = topology.get_neighbor_doorways(doorway_name)
        for neighbor_name in neighbors:
            if neighbor_name in doorways:
                pair = tuple(sorted([doorway_name, neighbor_name]))
                if pair not in added_pairs:
                    segments.append(
                        LineSegment(doorway_pos.copy(), doorways[neighbor_name].copy())
                    )
                    added_pairs.add(pair)

    # 4. Find which doorways are closest to the goal and add those paths
    goal_distances = {
        name: np.linalg.norm(goal_pos - pos) for name, pos in doorways.items()
    }
    closest_to_goal = min(goal_distances.items(), key=lambda x: x[1])[0]

    if closest_to_goal in doorways:
        segments.append(LineSegment(doorways[closest_to_goal].copy(), goal_pos.copy()))

    return segments
