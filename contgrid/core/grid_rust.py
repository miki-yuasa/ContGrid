"""Rust-accelerated grid collision detection.

This module provides a high-performance implementation of WallCollisionChecker
using Rust + PyO3 for critical performance bottlenecks.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray

from .grid import Layout

try:
    from contgrid.contgrid_rust import WallCollisionCheckerRust as _RustChecker

    HAS_RUST = True
except ImportError:
    HAS_RUST = False
    _RustChecker = None  # type: ignore


class WallCollisionCheckerAccelerated:
    """Wall collision checker with optional Rust acceleration.

    This wrapper automatically uses the Rust implementation if available,
    falling back to the pure Python version otherwise.
    """

    def __init__(self, layout: Layout, L: float, verbose: bool = False) -> None:
        """Initialize the collision checker.

        Args:
            layout: The grid map layout ('#'=wall, '0'=free).
            L: The width and height of a single grid cell.
            verbose: Whether to print debug information.
        """
        self.layout = layout
        self.L = L
        self.verbose = verbose

        if HAS_RUST:
            # Convert layout to format expected by Rust.
            layout_list = []
            for row in layout:
                if isinstance(row, str):
                    layout_list.append(list(row))
                else:
                    layout_list.append(row)

            self._checker = _RustChecker(layout_list, L, verbose)
            self._using_rust = True
            if verbose:
                print("Using Rust-accelerated WallCollisionChecker")
        else:
            # Fall back to Python implementation.
            from .grid import WallCollisionChecker

            self._checker = WallCollisionChecker(layout, L, verbose)
            self._using_rust = False
            if verbose:
                print("Using Python WallCollisionChecker fallback")

        # Pre-compute wall information.
        if not HAS_RUST:
            # These are already computed in the Python version.
            self.wall_bounds = self._checker.wall_bounds
            self.walls = self._checker.walls
            self.wall_anchors = self._checker.wall_anchors
            self.wall_cells = self._checker.wall_cells
            self.wall_centers = self._checker.wall_centers
        else:
            walls, anchors, cells, centers = _extract_wall_info(layout, L)
            self.walls = walls
            self.wall_anchors = anchors
            self.wall_cells = cells
            self.wall_centers = centers
            self.wall_bounds = np.array(walls, dtype=np.float64)

    @property
    def using_rust(self) -> bool:
        """Check if Rust acceleration is being used."""
        return self._using_rust

    def is_collision(
        self,
        R: float,
        C: float,
        robot_pos: tuple[float, float],
        collision_force: float,
        contact_margin: float,
    ) -> tuple[bool, NDArray[np.float64]]:
        """Check for collision at the given robot position.

        Args:
            R: The radius of the robot.
            C: The minimum required clearance.
            robot_pos: The (x, y) coordinates of the robot's center.
            collision_force: The collision force magnitude.
            contact_margin: The contact margin for soft collisions.

        Returns:
            A tuple of (collided, force) where collided is a boolean
            and force is a 2D numpy array.
        """
        if self._using_rust:
            collided, force = self._checker.is_collision(
                R, C, robot_pos, collision_force, contact_margin
            )
            return (collided, np.array(force, dtype=np.float64))
        return self._checker.is_collision(
            R, C, robot_pos, collision_force, contact_margin
        )

    def is_position_valid(
        self, R: float, allowed_overlap: float, pos: tuple[float, float]
    ) -> bool:
        """Check if a position maintains minimum distance from all walls.

        Args:
            R: The radius of the entity.
            allowed_overlap: The allowed overlap with walls.
            pos: The (x, y) position to check.

        Returns:
            True if the position is valid, False otherwise.
        """
        return self._checker.is_position_valid(R, allowed_overlap, pos)

    def clip_new_position(
        self,
        R: float,
        allowed_overlap: float,
        curr_pos: tuple[float, float],
        new_pos: tuple[float, float],
    ) -> tuple[float, float]:
        """Clip the new position to ensure no collision with walls occurs.

        Args:
            R: The radius of the entity.
            allowed_overlap: The allowed overlap with walls.
            curr_pos: The current position of the entity.
            new_pos: The proposed new position of the entity.

        Returns:
            The clipped position of the entity.
        """
        return self._checker.clip_new_position(
            R, allowed_overlap, curr_pos, new_pos
        )

    def sample_path_gaussian(
        self,
        segments: Sequence[Sequence[float]],
        num_obstacles: int,
        obstacle_radius: float,
        gaussian_std: float,
        min_spacing: float,
        edge_buffer: float,
        grid_bounds: Sequence[float],
        room_bounds: Sequence[Sequence[float] | None],
        existing_obstacles: Sequence[tuple[float, float]],
        goal_pos: tuple[float, float],
        goal_clearance: float,
        agent_pos: tuple[float, float] | None,
        min_agent_dist: float,
        seed: int,
    ) -> list[tuple[float, float]]:
        """Sample obstacle positions along relevant paths with Rust.

        Args:
            segments: Line segments as [x1, y1, x2, y2].
            num_obstacles: Number of obstacles to spawn.
            obstacle_radius: Radius of the obstacle.
            gaussian_std: Standard deviation of Gaussian perturbation.
            min_spacing: Minimum spacing between obstacles.
            edge_buffer: Edge buffer distance.
            grid_bounds: Continuous grid bounds [min_x, max_x, min_y, max_y].
            room_bounds: Per-obstacle room bounds or None.
            existing_obstacles: Already spawned obstacle positions.
            goal_pos: Goal position.
            goal_clearance: Minimum clearance from goal.
            agent_pos: Agent position or None.
            min_agent_dist: Minimum clearance from agent.
            seed: PRNG seed.

        Returns:
            List of spawned (x, y) obstacle positions.
        """
        return self._checker.sample_path_gaussian(
            segments,
            num_obstacles,
            obstacle_radius,
            gaussian_std,
            min_spacing,
            edge_buffer,
            grid_bounds,
            room_bounds,
            existing_obstacles,
            goal_pos,
            goal_clearance,
            agent_pos,
            min_agent_dist,
            seed,
        )


def _extract_wall_info(
    layout: Layout, L: float
) -> tuple[
    list[tuple[float, float, float, float]],
    list[tuple[float, float]],
    list[tuple[int, int]],
    list[tuple[float, float]],
]:
    """Compute wall coordinates and bounds from layout."""
    n_rows = len(layout)
    walls = []
    anchors = []
    cells = []
    centers = []

    for r, row in enumerate(layout):
        for c, char in enumerate(row):
            if char != "#":
                continue
            x = c * L
            y = (n_rows - 1 - r) * L
            anchors.append((x, y))
            walls.append((x - L / 2, x + L / 2, y - L / 2, y + L / 2))
            cells.append((r, c))
            centers.append((x, y))

    return walls, anchors, cells, centers
