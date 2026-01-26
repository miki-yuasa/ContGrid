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
        positions = []

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

        # Get relevant path segments (requires agent position)
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
            return FixedSpawnStrategy().spawn_obstacles(
                num_obstacles,
                obstacle_type,
                world,
                scenario,
                np_random,
                agent_pos,
                obstacle_configs,
            )

        # Weight segments by length and filter out zero-length segments
        lengths = np.array([seg.length() for seg in segments])
        valid_mask = (lengths > 1e-6) & np.isfinite(lengths)

        if not valid_mask.any():
            return FixedSpawnStrategy().spawn_obstacles(
                num_obstacles,
                obstacle_type,
                world,
                scenario,
                np_random,
                agent_pos,
                obstacle_configs,
            )

        valid_segments = [seg for seg, valid in zip(segments, valid_mask) if valid]
        valid_lengths = lengths[valid_mask]
        probabilities = valid_lengths / valid_lengths.sum()

        # Track failed positions for spatial rejection sampling
        failed_regions: list[tuple[NDArray[np.float64], float]] = []
        max_failed_regions = 50  # Limit memory usage

        max_attempts = 100
        for obstacle_idx in range(num_obstacles):
            required_room = room_constraints[obstacle_idx]

            # Filter segments to only those that pass through the required room
            room_valid_segments = valid_segments
            room_probabilities = probabilities

            if required_room is not None:
                room_segment_indices = []
                for idx, seg in enumerate(valid_segments):
                    # Check if segment passes through the required room
                    if self._segment_intersects_room(seg, required_room):
                        room_segment_indices.append(idx)

                if room_segment_indices:
                    room_valid_segments = [
                        valid_segments[i] for i in room_segment_indices
                    ]
                    room_lengths = valid_lengths[room_segment_indices]
                    room_probabilities = room_lengths / room_lengths.sum()

            position_found = False
            for attempt in range(max_attempts):
                segment_idx = np_random.choice(
                    len(room_valid_segments), p=room_probabilities
                )
                segment = room_valid_segments[segment_idx]

                t = np_random.uniform(0, 1)
                base_pos = segment.sample_point(t)

                # Add Gaussian noise
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

                perturbed_pos = (
                    base_pos + parallel_noise * direction + perp_noise * perpendicular
                )

                # Clip position to room boundaries if required room is specified
                if required_room is not None:
                    perturbed_pos = self._clip_to_room_bounds(
                        perturbed_pos, required_room
                    )

                # Spatial rejection sampling: skip if near recent failures
                # Only check last 10 failures to keep it fast
                too_close_to_failure = False
                if len(failed_regions) > 0:
                    recent_failures = failed_regions[-min(10, len(failed_regions)) :]
                    for failure_center, failure_radius in recent_failures:
                        if (
                            np.linalg.norm(perturbed_pos - failure_center)
                            < failure_radius
                        ):
                            too_close_to_failure = True
                            break

                if too_close_to_failure:
                    continue

                if self._is_valid_position(
                    perturbed_pos,
                    world,
                    positions + existing_obstacles,
                    scenario,
                    obstacle_type,
                    required_room,
                    agent_pos,
                ):
                    positions.append(tuple(perturbed_pos))
                    position_found = True
                    break
                else:
                    # Track this failure region to avoid it in future attempts
                    if len(failed_regions) < max_failed_regions:
                        # Store center and radius (use min_spacing as exclusion radius)
                        failed_regions.append(
                            (perturbed_pos.copy(), self.config.min_spacing)
                        )

            # If no valid position found after max_attempts, try systematic segment sampling
            # This ensures we stay on segments rather than falling back to random room sampling
            if not position_found:
                position_found = self._try_systematic_segment_sampling(
                    room_valid_segments,
                    positions,
                    existing_obstacles,
                    world,
                    scenario,
                    obstacle_type,
                    required_room,
                    agent_pos,
                    np_random,
                )
                if position_found:
                    # _try_systematic_segment_sampling appends to positions directly
                    pass
                # If still no position found, skip this obstacle (don't use random fallback)

        return positions

    def _try_systematic_segment_sampling(
        self,
        segments: list["LineSegment"],
        positions: list[Position],
        existing_obstacles: list[Position],
        world: World,
        scenario: "RoomsScenario",  # type: ignore
        obstacle_type: str,
        required_room: str | None,
        agent_pos: NDArray[np.float64] | None,
        np_random: np.random.Generator,
    ) -> bool:
        """
        Try systematic sampling along segments to find a valid position.

        This method samples points at regular intervals along each segment,
        with small random offsets to avoid deterministic patterns while staying on segments.

        Returns True if a valid position was found and appended to positions.
        """
        from .topology import LineSegment

        if not segments:
            return False

        # Try each segment with systematic sampling
        # Shuffle segment order to add randomness
        segment_order = list(range(len(segments)))
        np_random.shuffle(segment_order)

        # Fewer samples for efficiency
        num_samples_per_segment = 20

        for seg_idx in segment_order:
            segment = segments[seg_idx]
            seg_length = segment.length()

            if seg_length < 1e-6:
                continue

            # Generate sample points along the segment
            # Use evenly spaced points with small random perturbation
            t_values = np.linspace(0.05, 0.95, num_samples_per_segment)
            np_random.shuffle(t_values)

            for t in t_values:
                # Add small perturbation to t (staying within segment)
                t_perturbed = t + np_random.uniform(-0.02, 0.02)
                t_perturbed = np.clip(t_perturbed, 0.01, 0.99)

                sample_pos = segment.sample_point(t_perturbed)

                # For zero gaussian_std, don't apply any noise - stay exactly on segment
                # For non-zero std, apply Gaussian noise
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
                    sample_pos = (
                        sample_pos
                        + parallel_noise * direction
                        + perp_noise * perpendicular
                    )

                # Clip position to room boundaries if required room is specified
                if required_room is not None:
                    sample_pos = self._clip_to_room_bounds(sample_pos, required_room)

                if self._is_valid_position(
                    sample_pos,
                    world,
                    positions + existing_obstacles,
                    scenario,
                    obstacle_type,
                    required_room,
                    agent_pos,
                ):
                    positions.append(tuple(sample_pos))
                    return True

        return False

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

        # Check not overlapping with agent
        if agent_pos is not None:
            if np.linalg.norm(pos - agent_pos) < self.config.min_spacing:
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
