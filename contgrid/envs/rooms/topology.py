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


class RoomTopology:
    """Defines the room structure and doorway connections."""

    def __init__(self, doorways: dict[str, Position]):
        self.doorways = doorways
        # Define which doorways are neighbors (connected by a room)
        self.neighbor_map: dict[str, list[str]] = {
            "ld": ["td", "bd"],
            "td": ["ld", "rd"],
            "rd": ["td", "bd"],
            "bd": ["ld", "rd"],
        }

        # Room boundaries defined by corners (min_x, max_x, min_y, max_y)
        self.room_boundaries = {
            "top_left": {"min_x": 1.0, "max_x": 5.0, "min_y": 7.0, "max_y": 11.0},
            "top_right": {"min_x": 7.0, "max_x": 11.5, "min_y": 6.0, "max_y": 11.0},
            "bottom_left": {"min_x": 1.0, "max_x": 5.0, "min_y": 1.0, "max_y": 5.0},
            "bottom_right": {"min_x": 7.0, "max_x": 11.5, "min_y": 1.0, "max_y": 4.0},
        }

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
        room_centers = {
            "top_left": np.array([3.5, 9.0]),
            "top_right": np.array([9.0, 9.0]),
            "bottom_left": np.array([3.5, 4.0]),
            "bottom_right": np.array([9.0, 3.5]),
        }

        min_dist = float("inf")
        closest_room = "top_left"

        for room_name, center in room_centers.items():
            dist = np.linalg.norm(pos - center)
            if dist < min_dist:
                min_dist = dist
                closest_room = room_name

        return closest_room

    def get_neighbor_doorways(self, doorway_name: str) -> list[str]:
        """Get names of doorways that are neighbors to the given doorway."""
        return self.neighbor_map.get(doorway_name, [])

    def get_doorways_in_room(self, position: Position, grid: Grid) -> list[str]:
        """
        Determine which doorways belong to the room containing the given position.
        Returns up to 2 doorway names.
        """
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
