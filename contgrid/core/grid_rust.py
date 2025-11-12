"""
Rust-accelerated grid collision detection.

This module provides a high-performance implementation of WallCollisionChecker
using Rust + PyO3 for critical performance bottlenecks.
"""

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
    """
    Wall collision checker with optional Rust acceleration.

    This wrapper automatically uses the Rust implementation if available,
    falling back to the pure Python version otherwise.
    """

    def __init__(self, layout: Layout, L: float, verbose: bool = False):
        """
        Initialize the collision checker.

        Args:
            layout: The grid map layout ('#'=wall, '0'=free).
            L: The width and height of a single grid cell.
            verbose: Whether to print debug information.
        """
        self.layout = layout
        self.L = L
        self.verbose = verbose

        if HAS_RUST:
            # Convert layout to format expected by Rust
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
            # Fall back to Python implementation
            from .grid import WallCollisionChecker

            self._checker = WallCollisionChecker(layout, L, verbose)
            self._using_rust = False
            if verbose:
                print("Rust extension not available, using Python WallCollisionChecker")

        # Pre-compute wall information
        if not HAS_RUST:
            # These are already computed in the Python version
            self.wall_bounds = self._checker.wall_bounds
            self.walls = self._checker.walls
            self.wall_anchors = self._checker.wall_anchors
            self.wall_cells = self._checker.wall_cells
            self.wall_centers = self._checker.wall_centers
        else:
            # Need to compute these for compatibility
            n_rows = len(layout)
            walls = []
            wall_anchors = []
            wall_cells = []
            wall_centers = []

            for r, row in enumerate(layout):
                for c, char in enumerate(row):
                    if char == "#":
                        # Anchor (bottom-left corner)
                        x = c * L
                        y = (n_rows - 1 - r) * L
                        wall_anchors.append((x, y))

                        # Wall boundaries
                        min_x = x - L / 2
                        max_x = x + L / 2
                        min_y = y - L / 2
                        max_y = y + L / 2
                        walls.append((min_x, max_x, min_y, max_y))

                        # Cell location
                        wall_cells.append((r, c))

                        # Center
                        wall_centers.append((x, y))

            self.walls = walls
            self.wall_anchors = wall_anchors
            self.wall_cells = wall_cells
            self.wall_centers = wall_centers
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
        """
        Check for collision at the given robot position.

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
        else:
            return self._checker.is_collision(
                R, C, robot_pos, collision_force, contact_margin
            )

    def is_position_valid(
        self, R: float, allowed_overlap: float, pos: tuple[float, float]
    ) -> bool:
        """
        Check if a position maintains minimum distance from all walls.

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
        """
        Clip the new position to ensure no collision with walls occurs.

        Args:
            R: The radius of the entity.
            allowed_overlap: The allowed overlap with walls.
            curr_pos: The current position of the entity.
            new_pos: The proposed new position of the entity.

        Returns:
            The clipped position of the entity.
        """
        return self._checker.clip_new_position(R, allowed_overlap, curr_pos, new_pos)
