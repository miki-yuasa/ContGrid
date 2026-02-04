"""Obstacle spawning strategies for the Rooms environment."""

from abc import ABC, abstractmethod
from enum import Enum
from typing import TYPE_CHECKING, Annotated, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, Discriminator

from contgrid.core import World
from contgrid.core.typing import Position

from .topology import LineSegment, RoomTopology, get_relevant_path_segments

if TYPE_CHECKING:
    from .scenario import ObjConfig


class SpawnMode(str, Enum):
    """Enumeration of available obstacle spawning methods."""

    FIXED = "fixed"
    PATH_GAUSSIAN = "path_gaussian"
    UNIFORM_RANDOM = "uniform_random"


class PathGaussianConfig(BaseModel):
    """Configuration for path-based Gaussian spawning."""

    mode: Literal[SpawnMode.PATH_GAUSSIAN] = SpawnMode.PATH_GAUSSIAN
    gaussian_std: float = 0.6
    min_spacing: float = 1.0
    edge_buffer: float = 0.05
    include_agent_paths: bool = True


class UniformRandomConfig(BaseModel):
    """Configuration for uniform random spawning."""

    mode: Literal[SpawnMode.UNIFORM_RANDOM] = SpawnMode.UNIFORM_RANDOM
    min_spacing: float = 0.5


class FixedSpawnConfig(BaseModel):
    """Configuration for fixed position spawning."""

    mode: Literal[SpawnMode.FIXED] = SpawnMode.FIXED


# Union type for all spawn method configs with discriminator
SpawnMethodConfig = Annotated[
    PathGaussianConfig | UniformRandomConfig | FixedSpawnConfig,
    Discriminator("mode"),
]


class SpawnStrategy(ABC):
    """Abstract base class for object spawning strategies."""

    @abstractmethod
    def spawn_obstacles(
        self,
        num_obstacles: int,
        obstacle_type: str,
        world: World,
        scenario: "RoomsScenario",  # type: ignore
        np_random: np.random.Generator,
        agent_pos: NDArray[np.float64] | None = None,
        obstacle_configs: list["ObjConfig"] | None = None,
    ) -> list[Position]:
        """Generate obstacle positions."""
        pass


class FixedSpawnStrategy(SpawnStrategy):
    """Original fixed-position spawning strategy."""

    def spawn_obstacles(
        self,
        num_obstacles: int,
        obstacle_type: str,
        world: World,
        scenario: "RoomsScenario",  # type: ignore
        np_random: np.random.Generator,
        agent_pos: NDArray[np.float64] | None = None,
        obstacle_configs: list["ObjConfig"] | None = None,
    ) -> list[Position]:
        """Spawn obstacles at configured positions or random free cells."""
        positions = []

        obstacle_landmarks = (
            scenario.lavas if obstacle_type == "lava" else scenario.holes
        )

        obstacle_radius = (
            scenario.config.spawn_config.lava_size
            if obstacle_type == "lava"
            else scenario.config.spawn_config.hole_size
        )
        agent_radius = scenario.config.spawn_config.agent_size
        min_agent_distance = obstacle_radius + agent_radius

        for landmark in obstacle_landmarks[:num_obstacles]:
            max_attempts = 100
            for attempt in range(max_attempts):
                new_pos = scenario._choose_new_pos(
                    landmark.reset_config.spawn_pos, scenario.free_cells, np_random
                )

                # Check if position is valid (not overlapping with agent)
                is_valid = True
                if agent_pos is not None:
                    if (
                        np.linalg.norm(np.array(new_pos) - agent_pos)
                        < min_agent_distance
                    ):
                        is_valid = False

                if is_valid:
                    positions.append(new_pos)
                    if new_pos in scenario.free_cells:
                        scenario.free_cells.remove(new_pos)
                    break

        return positions


class PathGaussianSpawnStrategy(SpawnStrategy):
    """Spawn obstacles near shortest paths with Gaussian noise."""

    def __init__(self, config: PathGaussianConfig):
        self.config = config
        self.topology: RoomTopology | None = None

    def spawn_obstacles(
        self,
        num_obstacles: int,
        obstacle_type: str,
        world: World,
        scenario: "RoomsScenario",  # type: ignore
        np_random: np.random.Generator,
        agent_pos: NDArray[np.float64] | None = None,
        obstacle_configs: list["ObjConfig"] | None = None,
    ) -> list[Position]:
        """Spawn obstacles along relevant paths with Gaussian perturbation."""
        positions: list[Position] = []

        # Get room constraints for each obstacle
        room_constraints = [None] * num_obstacles
        if obstacle_configs:
            room_constraints = [
                config.room for config in obstacle_configs[:num_obstacles]
            ]

        # Get all existing obstacles to check spacing
        existing_obstacles = self._get_existing_obstacles(scenario)

        # Initialize topology if not done
        if self.topology is None:
            assert scenario.config
            self.topology = RoomTopology(scenario.config.spawn_config.doorways)

        # Get and validate path segments
        valid_segments, valid_lengths, probabilities = self._get_valid_segments(
            scenario, world, agent_pos
        )

        if valid_segments is None:
            return FixedSpawnStrategy().spawn_obstacles(
                num_obstacles,
                obstacle_type,
                world,
                scenario,
                np_random,
                agent_pos,
                obstacle_configs,
            )

        # Track failed positions for spatial rejection sampling
        failed_regions: list[tuple[NDArray[np.float64], float]] = []

        for obstacle_idx in range(num_obstacles):
            required_room = room_constraints[obstacle_idx]

            # Filter segments for the required room
            room_segments, room_probs = self._filter_segments_for_room(
                valid_segments, valid_lengths, probabilities, required_room
            )

            # Try to find a valid position
            position = self._find_valid_position(
                room_segments,
                room_probs,
                positions,
                existing_obstacles,
                world,
                scenario,
                obstacle_type,
                required_room,
                agent_pos,
                np_random,
                failed_regions,
            )

            if position is not None:
                positions.append(position)

        return positions

    def _get_valid_segments(
        self,
        scenario: "RoomsScenario",  # type: ignore
        world: World,
        agent_pos: NDArray[np.float64] | None,
    ) -> tuple[
        list[LineSegment] | None, NDArray[np.float64] | None, NDArray[np.float64] | None
    ]:
        """Get valid path segments with their lengths and probabilities."""
        # Get relevant path segments
        if agent_pos is None and self.config.include_agent_paths:
            segments = self._get_doorway_segments_only(scenario)
        else:
            segments = get_relevant_path_segments(
                agent_pos if agent_pos is not None else scenario.goal_pos,
                scenario.goal_pos,
                scenario.doorways,
                self.topology,
                world.grid,
            )

        if not segments:
            return None, None, None

        # Filter out zero-length segments
        lengths = np.array([seg.length() for seg in segments])
        valid_mask = (lengths > 1e-6) & np.isfinite(lengths)

        if not valid_mask.any():
            return None, None, None

        valid_segments = [seg for seg, valid in zip(segments, valid_mask) if valid]
        valid_lengths = lengths[valid_mask]
        probabilities = valid_lengths / valid_lengths.sum()

        return valid_segments, valid_lengths, probabilities

    def _filter_segments_for_room(
        self,
        segments: list[LineSegment],
        lengths: NDArray[np.float64],
        probabilities: NDArray[np.float64],
        required_room: str | None,
    ) -> tuple[list[LineSegment], NDArray[np.float64]]:
        """Filter and clip segments to the required room boundaries.

        When a room constraint is provided, segments are clipped to the room
        boundaries so that samples are always on the segment within the room.
        """
        if required_room is None:
            return segments, probabilities

        # Clip each segment to the room boundaries
        clipped_segments = []
        for seg in segments:
            clipped = self._clip_segment_to_room(seg, required_room)
            if clipped is not None and clipped.length() > 1e-6:
                clipped_segments.append(clipped)

        if not clipped_segments:
            # No segments intersect the room, fall back to original segments
            return segments, probabilities

        # Recalculate probabilities based on clipped segment lengths
        clipped_lengths = np.array([seg.length() for seg in clipped_segments])
        clipped_probs = clipped_lengths / clipped_lengths.sum()

        return clipped_segments, clipped_probs

    def _sample_position_on_segment(
        self,
        segment: LineSegment,
        t: float,
        np_random: np.random.Generator,
        required_room: str | None,
    ) -> NDArray[np.float64]:
        """Sample a position on segment at parameter t, with optional Gaussian noise and room clipping."""
        base_pos = segment.sample_point(t)

        # Apply Gaussian noise if std > 0
        if self.config.gaussian_std > 0:
            direction = segment.end - segment.start
            norm = np.linalg.norm(direction)
            if norm > 1e-6:
                direction = direction / norm
                perpendicular = np.array([-direction[1], direction[0]])
            else:
                direction = np.array([1.0, 0.0])
                perpendicular = np.array([0.0, 1.0])

            parallel_noise = np_random.normal(0, self.config.gaussian_std)
            perp_noise = np_random.normal(0, self.config.gaussian_std)
            base_pos = (
                base_pos + parallel_noise * direction + perp_noise * perpendicular
            )

        # Clip to room boundaries if required
        if required_room is not None:
            base_pos = self._clip_to_room_bounds(base_pos, required_room)

        return base_pos

    def _find_valid_position(
        self,
        segments: list[LineSegment],
        probabilities: NDArray[np.float64],
        positions: list[Position],
        existing_obstacles: list[Position],
        world: World,
        scenario: "RoomsScenario",  # type: ignore
        obstacle_type: str,
        required_room: str | None,
        agent_pos: NDArray[np.float64] | None,
        np_random: np.random.Generator,
        failed_regions: list[tuple[NDArray[np.float64], float]],
    ) -> Position | None:
        """Try to find a valid position using random sampling, then systematic sampling."""
        max_attempts = 100
        max_failed_regions = 50

        # Try random sampling first
        for _ in range(max_attempts):
            segment_idx = np_random.choice(len(segments), p=probabilities)
            segment = segments[segment_idx]

            t = np_random.uniform(0, 1)
            candidate = self._sample_position_on_segment(
                segment, t, np_random, required_room
            )

            # Skip if near recent failures
            if self._is_near_failed_region(candidate, failed_regions):
                continue

            if self._is_valid_position(
                candidate,
                world,
                positions + existing_obstacles,
                scenario,
                obstacle_type,
                required_room,
                agent_pos,
            ):
                return tuple(candidate)
            else:
                if len(failed_regions) < max_failed_regions:
                    failed_regions.append((candidate.copy(), self.config.min_spacing))

        # Fallback to systematic sampling
        return self._systematic_segment_sampling(
            segments,
            positions,
            existing_obstacles,
            world,
            scenario,
            obstacle_type,
            required_room,
            agent_pos,
            np_random,
        )

    def _is_near_failed_region(
        self,
        pos: NDArray[np.float64],
        failed_regions: list[tuple[NDArray[np.float64], float]],
    ) -> bool:
        """Check if position is too close to recent failed positions."""
        if not failed_regions:
            return False

        recent = failed_regions[-min(10, len(failed_regions)) :]
        for center, radius in recent:
            if np.linalg.norm(pos - center) < radius:
                return True
        return False

    def _systematic_segment_sampling(
        self,
        segments: list[LineSegment],
        positions: list[Position],
        existing_obstacles: list[Position],
        world: World,
        scenario: "RoomsScenario",  # type: ignore
        obstacle_type: str,
        required_room: str | None,
        agent_pos: NDArray[np.float64] | None,
        np_random: np.random.Generator,
    ) -> Position | None:
        """Try systematic sampling along segments to find a valid position."""
        if not segments:
            return None

        segment_order = list(range(len(segments)))
        np_random.shuffle(segment_order)

        num_samples = 20
        t_values = np.linspace(0.05, 0.95, num_samples)

        for seg_idx in segment_order:
            segment = segments[seg_idx]
            if segment.length() < 1e-6:
                continue

            np_random.shuffle(t_values)

            for t in t_values:
                t_perturbed = np.clip(t + np_random.uniform(-0.02, 0.02), 0.01, 0.99)
                candidate = self._sample_position_on_segment(
                    segment, t_perturbed, np_random, required_room
                )

                if self._is_valid_position(
                    candidate,
                    world,
                    positions + existing_obstacles,
                    scenario,
                    obstacle_type,
                    required_room,
                    agent_pos,
                ):
                    return tuple(candidate)

        return None

    def _clip_segment_to_room(
        self, segment: LineSegment, room_name: str
    ) -> LineSegment | None:
        """
        Clip a segment to the room boundaries using Liang-Barsky algorithm.

        Returns a new segment that is entirely within the room (with edge buffer),
        or None if the segment doesn't intersect the room.
        """
        if self.topology is None:
            return segment

        if room_name not in self.topology.room_boundaries:
            return segment

        bounds = self.topology.room_boundaries[room_name]

        x1, y1 = float(segment.start[0]), float(segment.start[1])
        x2, y2 = float(segment.end[0]), float(segment.end[1])

        dx = x2 - x1
        dy = y2 - y1

        # Room bounds with edge buffer
        min_x = bounds["min_x"] + self.config.edge_buffer
        max_x = bounds["max_x"] - self.config.edge_buffer
        min_y = bounds["min_y"] + self.config.edge_buffer
        max_y = bounds["max_y"] - self.config.edge_buffer

        # Liang-Barsky line clipping algorithm
        t0, t1 = 0.0, 1.0

        # p and q for each boundary: left, right, bottom, top
        p = [-dx, dx, -dy, dy]
        q = [x1 - min_x, max_x - x1, y1 - min_y, max_y - y1]

        for i in range(4):
            if abs(p[i]) < 1e-10:  # Line parallel to boundary
                if q[i] < 0:
                    return None  # Line is outside and parallel
            else:
                t = q[i] / p[i]
                if p[i] < 0:  # Entering boundary
                    t0 = max(t0, t)
                else:  # Exiting boundary
                    t1 = min(t1, t)

        if t0 > t1:
            return None  # No intersection with room

        # Compute clipped endpoints
        clipped_start = np.array([x1 + t0 * dx, y1 + t0 * dy])
        clipped_end = np.array([x1 + t1 * dx, y1 + t1 * dy])

        return LineSegment(clipped_start, clipped_end)

    def _segment_intersects_room(self, segment: "LineSegment", room_name: str) -> bool:
        """
        Check if a segment passes through or intersects a room.

        This samples points along the segment and checks if any fall within the room.
        """
        if self.topology is None:
            return False

        # Check endpoints first
        if self._is_position_in_room(segment.start, room_name):
            return True
        if self._is_position_in_room(segment.end, room_name):
            return True

        # Sample points along the segment to check if it passes through the room
        num_samples = 10
        for i in range(1, num_samples):
            t = i / num_samples
            point = segment.sample_point(t)
            if self._is_position_in_room(point, room_name):
                return True

        return False

    def _is_position_in_room(
        self, pos: NDArray[np.float64] | Position, room_name: str
    ) -> bool:
        """
        Check if a position is strictly within the room boundaries.

        Uses explicit boundary checks for robustness.
        """
        if self.topology is None:
            return False

        # Get room boundaries
        if room_name not in self.topology.room_boundaries:
            return False

        bounds = self.topology.room_boundaries[room_name]
        x, y = float(pos[0]), float(pos[1])

        # Strict boundary check
        return (
            bounds["min_x"] <= x <= bounds["max_x"]
            and bounds["min_y"] <= y <= bounds["max_y"]
        )

    def _clip_to_room_bounds(
        self, pos: NDArray[np.float64], room_name: str
    ) -> NDArray[np.float64]:
        """
        Clip a position to be within the room boundaries with edge buffer.

        Returns a new position that is guaranteed to be within the room.
        """
        if self.topology is None:
            return pos

        if room_name not in self.topology.room_boundaries:
            return pos

        bounds = self.topology.room_boundaries[room_name]
        clipped = pos.copy()

        clipped[0] = np.clip(
            clipped[0],
            bounds["min_x"] + self.config.edge_buffer,
            bounds["max_x"] - self.config.edge_buffer,
        )
        clipped[1] = np.clip(
            clipped[1],
            bounds["min_y"] + self.config.edge_buffer,
            bounds["max_y"] - self.config.edge_buffer,
        )

        return clipped

    def _get_existing_obstacles(self, scenario: "RoomsScenario") -> list[Position]:  # type: ignore
        """Get all existing obstacle positions (lavas and holes already spawned)."""
        existing = []

        for lava in scenario.lavas:
            if hasattr(lava.state, "pos") and lava.state.pos is not None:
                if isinstance(lava.state.pos, np.ndarray) and lava.state.pos.shape == (
                    2,
                ):
                    pos_tuple = tuple(lava.state.pos)
                    if pos_tuple != (0.0, 0.0):
                        existing.append(pos_tuple)

        for hole in scenario.holes:
            if hasattr(hole.state, "pos") and hole.state.pos is not None:
                if isinstance(hole.state.pos, np.ndarray) and hole.state.pos.shape == (
                    2,
                ):
                    pos_tuple = tuple(hole.state.pos)
                    if pos_tuple != (0.0, 0.0):
                        existing.append(pos_tuple)

        return existing

    def _get_doorway_segments_only(
        self, scenario: "RoomsScenario"
    ) -> list[LineSegment]:  # type: ignore
        """Get segments between neighboring doorways and to goal."""
        segments = []

        neighbor_pairs = [("ld", "td"), ("ld", "bd"), ("td", "rd"), ("rd", "bd")]

        for d1, d2 in neighbor_pairs:
            if d1 in scenario.doorways and d2 in scenario.doorways:
                segments.append(
                    LineSegment(
                        scenario.doorways[d1].copy(), scenario.doorways[d2].copy()
                    )
                )

        goal_distances = {
            name: np.linalg.norm(scenario.goal_pos - pos)
            for name, pos in scenario.doorways.items()
        }
        closest_doorway = min(goal_distances.items(), key=lambda x: x[1])[0]
        if closest_doorway in scenario.doorways:
            segments.append(
                LineSegment(
                    scenario.doorways[closest_doorway].copy(), scenario.goal_pos.copy()
                )
            )

        return segments

    def _is_valid_position(
        self,
        pos: NDArray[np.float64],
        world: World,
        existing_positions: list[Position],
        scenario: "RoomsScenario",  # type: ignore
        obstacle_type: str,
        required_room: str | None = None,
        agent_pos: NDArray[np.float64] | None = None,
    ) -> bool:
        """Check if position is valid for obstacle spawning."""
        assert scenario.config
        limits = world.grid.wall_limits

        obstacle_radius = (
            scenario.config.spawn_config.lava_size
            if obstacle_type == "lava"
            else scenario.config.spawn_config.hole_size
        )
        agent_radius = scenario.config.spawn_config.agent_size
        min_agent_distance = obstacle_radius + agent_radius

        # Check if position is in the required room
        if required_room is not None:
            if not self._is_position_in_room(pos, required_room):
                return False

        # Check bounds with buffer
        if not (
            limits.min_x + self.config.edge_buffer
            <= pos[0]
            <= limits.max_x - self.config.edge_buffer
            and limits.min_y + self.config.edge_buffer
            <= pos[1]
            <= limits.max_y - self.config.edge_buffer
        ):
            return False

        # Check wall collision
        is_collided, _ = world.wall_collision_checker.is_collision(
            R=obstacle_radius,
            C=0.0,
            robot_pos=(float(pos[0]), float(pos[1])),
            collision_force=1.0,
            contact_margin=0.1,
        )
        if is_collided:
            return False

        # Check spacing from existing obstacles
        for existing_pos in existing_positions:
            if np.linalg.norm(pos - np.array(existing_pos)) < self.config.min_spacing:
                return False

        # Check not too close to goal
        if np.linalg.norm(pos - scenario.goal_pos) < scenario.goal_thr_dist + 0.5:
            return False

        # Check not overlapping with agent (use proper size-based distance)
        if agent_pos is not None:
            if np.linalg.norm(pos - agent_pos) < min_agent_distance:
                return False

        return True


class UniformRandomSpawnStrategy(SpawnStrategy):
    """Spawn obstacles uniformly at random in free space."""

    def __init__(self, config: UniformRandomConfig):
        self.config = config

    def spawn_obstacles(
        self,
        num_obstacles: int,
        obstacle_type: str,
        world: World,
        scenario: "RoomsScenario",  # type: ignore
        np_random: np.random.Generator,
        agent_pos: NDArray[np.float64] | None = None,
        obstacle_configs: list["ObjConfig"] | None = None,
    ) -> list[Position]:
        """Spawn obstacles uniformly in free cells."""
        positions = []

        if not hasattr(self, "topology") or self.topology is None:
            assert scenario.config
            self.topology = RoomTopology(scenario.config.spawn_config.doorways)

        room_constraints = [None] * num_obstacles
        if obstacle_configs:
            room_constraints = [
                config.room for config in obstacle_configs[:num_obstacles]
            ]

        obstacle_radius = (
            scenario.config.spawn_config.lava_size
            if obstacle_type == "lava"
            else scenario.config.spawn_config.hole_size
        )
        agent_radius = scenario.config.spawn_config.agent_size
        min_agent_distance = obstacle_radius + agent_radius

        for obstacle_idx in range(num_obstacles):
            required_room = room_constraints[obstacle_idx]

            if required_room is not None:
                available_cells = [
                    cell
                    for cell in scenario.free_cells
                    if self.topology.get_room(cell) == required_room
                ]
            else:
                available_cells = scenario.free_cells

            # Filter out cells that are too close to agent
            if agent_pos is not None:
                available_cells = [
                    cell
                    for cell in available_cells
                    if np.linalg.norm(np.array(cell) - agent_pos) >= min_agent_distance
                ]

            if available_cells:
                chosen_idx = np_random.choice(len(available_cells))
                new_pos = available_cells[chosen_idx]
                positions.append(new_pos)
                if new_pos in scenario.free_cells:
                    scenario.free_cells.remove(new_pos)

        return positions
