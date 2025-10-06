import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

Layout = list[str] | list[list[str]]


class Grid(BaseModel):
    layout: Layout
    cell_size: float = 1.0
    allow_wall_overlap: bool = False

    @property
    def width(self) -> float:
        return self.width_cells * self.cell_size

    @property
    def height(self) -> float:
        return self.height_cells * self.cell_size

    @property
    def width_cells(self) -> int:
        return len(self.layout[0]) if self.layout else 0

    @property
    def height_cells(self) -> int:
        return len(self.layout)


DEFAULT_GRID = Grid(
    layout=["#####", "#000#", "#0#0#", "#000#", "#####"],
    cell_size=1,
)


class WallCollisionChecker:
    """
    An efficient, memoized collision checker for a static map.

    This class pre-processes the map layout upon initialization to build a
    NumPy array of wall coordinates. Subsequent collision checks are then
    performed with highly optimized vectorized operations.
    """

    def __init__(self, layout: Layout, L: float, verbose: bool = False):
        """
        Initializes the checker by pre-computing wall locations.

        Args:
            layout (Layout): The grid map layout ('#'=wall, '0'=free).
            L (float): The width and height of a single grid cell.
        """
        if not layout or not layout[0]:
            self.wall_bounds = np.empty((0, 4))
            return

        self.L = L
        self.verbose = verbose
        n_rows = len(layout)

        # Pre-compute wall boundaries once
        walls: list[tuple[float, float, float, float]] = []
        # wall anchors (bottom-left corner of each wall cell)
        wall_anchors: list[tuple[float, float]] = []
        # wall grid locations (row, col) of each wall cell
        wall_cells: list[tuple[int, int]] = []
        # center of each wall cell
        wall_centers: list[tuple[float, float]] = []
        for r, row_str in enumerate(layout):
            for c, char in enumerate(row_str):
                if char == "#":
                    # Calculate continuous coordinates for the wall square
                    wall_min_x = (c - 0.5) * L
                    wall_max_x = (c + 0.5) * L
                    wall_min_y = (n_rows - 1 - r - 0.5) * L
                    wall_max_y = (n_rows - r - 0.5) * L

                    walls.append((wall_min_x, wall_max_x, wall_min_y, wall_max_y))
                    wall_anchors.append((wall_min_x, wall_min_y))
                    wall_cells.append((r, c))
                    wall_centers.append((wall_min_x + L / 2, wall_min_y + L / 2))

        self.walls: list[tuple[float, float, float, float]] = walls
        self.wall_anchors: list[tuple[float, float]] = wall_anchors
        self.wall_cells: list[tuple[int, int]] = wall_cells
        self.wall_centers: list[tuple[float, float]] = wall_centers

        # Store as a NumPy array for fast vectorized access
        self.wall_bounds: NDArray[np.float64] = np.array(walls, dtype=np.float64)
        print(
            f"Memoized CollisionChecker: Pre-processed {len(self.wall_bounds)} wall segments."
        )

    def is_collision(
        self,
        R: float,
        C: float,
        robot_pos: tuple[float, float],
        collision_force: float,
        contact_margin: float,
    ) -> tuple[bool, NDArray[np.float64]]:
        """
        Checks for a collision at the given robot position. This method is highly optimized.

        Args:
            R (float): The radius of the robot.
            C (float): The minimum required clearance.
            robot_pos (tuple[float, float]): The (x, y) coordinates of the robot's center.

        Returns
        -------
            collided: bool
                True if a collision occurs, False otherwise.
            forces: NDArray[np.float64]
                The repulsive force vector if a collision occurs, else zero vector.
        """
        if self.wall_bounds.shape[0] == 0:
            return (False, np.zeros(2, dtype=np.float64))  # No walls in the map

        x, y = robot_pos
        R_eff = R + C
        R_eff_sq = R_eff**2

        # --- 1. VECTORIZED BROAD PHASE ---
        # Use NumPy's boolean indexing for a fast, vectorized filter.
        # This checks for any overlap between the robot's check-box and the wall boxes.
        candidate_walls: NDArray[np.float64] = self._get_collision_candidates(
            R_eff, robot_pos
        )
        # --- 2. NARROW PHASE ---
        # Now, perform the precise check only on the small subset of candidate walls.
        force: NDArray[np.float64] = np.zeros(2, dtype=np.float64)
        is_collided: bool = False
        for wall_min_x, wall_max_x, wall_min_y, wall_max_y in candidate_walls:
            closest_x = np.clip(x, wall_min_x, wall_max_x)
            closest_y = np.clip(y, wall_min_y, wall_max_y)

            delta_pos = np.array([x - closest_x, y - closest_y], dtype=np.float64)
            dist_sq = np.sum(np.square(delta_pos))

            if dist_sq < R_eff_sq:
                # Collision detected, compute repulsive force added to total force
                is_collided = True
                dist = np.sqrt(dist_sq) if dist_sq > 0 else 1e-6  # Avoid div by zero
                k: float = contact_margin
                penetration: float = np.logaddexp(0, -(dist - R_eff) / k) * k
                force += collision_force * delta_pos / dist * penetration

        return is_collided, force

    def _get_collision_candidates(
        self, R_eff: float, robot_pos: tuple[float, float]
    ) -> NDArray[np.float64]:
        """
        Returns the wall bounds that are candidates for collision with the robot.

        Args:
            R (float): The radius of the robot.
            C (float): The minimum required clearance.
            robot_pos (tuple[float, float]): The (x, y) coordinates of the robot's center.
        Returns
        -------
            candidates: NDArray[np.float64]
                The wall bounds that are candidates for collision.

        """
        if self.wall_bounds.shape[0] == 0:
            return np.empty((0, 4), dtype=np.float64)  # No walls in the map

        x, y = robot_pos

        # --- 1. VECTORIZED BROAD PHASE ---
        # Find all walls whose bounding boxes *could possibly* overlap with the robot's effective circle.
        # This is much faster than looping through grid cells.
        min_check_x = x - R_eff
        max_check_x = x + R_eff
        min_check_y = y - R_eff
        max_check_y = y + R_eff

        # Use NumPy's boolean indexing for a fast, vectorized filter.
        # This checks for any overlap between the robot's check-box and the wall boxes.
        candidate_walls: NDArray[np.float64] = self.wall_bounds[
            (self.wall_bounds[:, 0] < max_check_x)  # wall_min_x < max_check_x
            & (self.wall_bounds[:, 1] > min_check_x)  # wall_max_x > min_check_x
            & (self.wall_bounds[:, 2] < max_check_y)  # wall_min_y < max_check_y
            & (self.wall_bounds[:, 3] > min_check_y)  # wall_max_y > min_check_y
        ]

        return candidate_walls

    def clip_new_position(
        self,
        R: float,
        allowed_overlap: float,
        curr_pos: tuple[float, float],
        new_pos: tuple[float, float],
    ) -> tuple[float, float]:
        """
        Clips the new position to ensure no collision with walls occurs.
        If the new position is valid, it is returned as is.
        If not, the vector from curr_pos to new_pos is scaled down to a valid position.

        Parameters
        ----------
        R : float
            The radius of the entity.
        allowed_overlap : float
            The allowed overlap with walls.
        curr_pos : tuple[float, float]
            The current position of the entity.
        new_pos : tuple[float, float]
            The proposed new position of the entity.

        Returns
        -------
        clipped_new_pos: tuple[float, float]
            The clipped position of the entity.
        """
        # If the new position is valid, return it as is
        if self.is_position_valid(R, allowed_overlap, new_pos):
            if self.verbose:
                print(f"- New position {new_pos} is valid.")
            return new_pos

        # If current position is invalid, return it (don't make things worse)
        if not self.is_position_valid(R, allowed_overlap, curr_pos):
            if self.verbose:
                print(f"- Current position {curr_pos} is invalid, cannot move.")
            return curr_pos

        # Analytically find the maximum valid scaling factor
        curr_x, curr_y = curr_pos
        new_x, new_y = new_pos

        # Movement vector
        dx = new_x - curr_x
        dy = new_y - curr_y

        # If no movement, return current position
        if abs(dx) < 1e-10 and abs(dy) < 1e-10:
            return curr_pos

        # Find the minimum scaling factor that would cause collision
        min_collision_t = 1.0  # Start with full movement
        min_distance = R - allowed_overlap  # Required clearance from wall surface

        # Check collision with each wall
        for wall_min_x, wall_max_x, wall_min_y, wall_max_y in self.wall_bounds:
            # Find the parameter t where the robot would first touch this wall
            t = self._find_wall_collision_parameter(
                curr_x,
                curr_y,
                dx,
                dy,
                wall_min_x,
                wall_max_x,
                wall_min_y,
                wall_max_y,
                min_distance,
            )

            if t is not None and 0 <= t < min_collision_t:
                min_collision_t = t

        # Use a small safety margin to ensure we don't quite reach the collision point
        safety_margin = 1e-6
        safe_t = max(0.0, min_collision_t - safety_margin)

        final_x = curr_x + safe_t * dx
        final_y = curr_y + safe_t * dy

        return (final_x, final_y)

    def _find_wall_collision_parameter(
        self,
        start_x: float,
        start_y: float,
        dx: float,
        dy: float,
        wall_min_x: float,
        wall_max_x: float,
        wall_min_y: float,
        wall_max_y: float,
        min_distance: float,
    ) -> float | None:
        """
        Analytically find the parameter t where a moving circle would first collide with a wall.

        The robot moves along the ray: (start_x + t*dx, start_y + t*dy) for t >= 0
        The wall is the rectangle [wall_min_x, wall_max_x] × [wall_min_y, wall_max_y]
        The robot has an effective radius of min_distance from the wall surface.

        Returns the smallest t >= 0 where collision would occur, or None if no collision.
        """
        # Expand the wall by min_distance in all directions (Minkowski sum)
        expanded_min_x = wall_min_x - min_distance
        expanded_max_x = wall_max_x + min_distance
        expanded_min_y = wall_min_y - min_distance
        expanded_max_y = wall_max_y + min_distance

        # Find intersection of ray with expanded rectangle
        # Ray: (start_x + t*dx, start_y + t*dy)
        # Rectangle: [expanded_min_x, expanded_max_x] × [expanded_min_y, expanded_max_y]

        t_min = 0.0  # We only consider forward movement
        t_max = float("inf")

        # Check intersection with vertical walls (x-direction)
        if abs(dx) > 1e-10:  # Moving in x-direction
            t1 = (expanded_min_x - start_x) / dx
            t2 = (expanded_max_x - start_x) / dx

            # Ensure t1 <= t2
            if t1 > t2:
                t1, t2 = t2, t1

            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
        else:  # Not moving in x-direction
            if start_x < expanded_min_x or start_x > expanded_max_x:
                return None  # Ray doesn't intersect rectangle

        # Check intersection with horizontal walls (y-direction)
        if abs(dy) > 1e-10:  # Moving in y-direction
            t1 = (expanded_min_y - start_y) / dy
            t2 = (expanded_max_y - start_y) / dy

            # Ensure t1 <= t2
            if t1 > t2:
                t1, t2 = t2, t1

            t_min = max(t_min, t1)
            t_max = min(t_max, t2)
        else:  # Not moving in y-direction
            if start_y < expanded_min_y or start_y > expanded_max_y:
                return None  # Ray doesn't intersect rectangle

        # Check if there's a valid intersection
        if t_min <= t_max and t_max >= 0:
            return t_min if t_min >= 0 else 0.0
        else:
            return None

    def is_position_valid(
        self, R: float, allowed_overlap: float, pos: tuple[float, float]
    ) -> bool:
        """Check if a position maintains minimum distance from all walls"""
        test_x, test_y = pos
        min_distance = R - allowed_overlap  # Minimum distance from wall surface
        for wall_min_x, wall_max_x, wall_min_y, wall_max_y in self.wall_bounds:
            closest_x = np.clip(test_x, wall_min_x, wall_max_x)
            closest_y = np.clip(test_y, wall_min_y, wall_max_y)
            dx = test_x - closest_x
            dy = test_y - closest_y
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < min_distance:
                if self.verbose:
                    print(f"Distance to wall: {dist} < min_distance: {min_distance}")
                    print(
                        f"- Position {pos} is invalid due to wall at ({wall_min_x}, {wall_min_y}, {wall_max_x}, {wall_max_y})"
                    )
                return False
        return True
