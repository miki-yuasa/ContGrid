import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel

Layout = list[str] | list[list[str]] | list[tuple[int, int]]


class Grid(BaseModel):
    layout: Layout
    width_cells: int
    height_cells: int
    cell_size: float = 0.100

    @property
    def width(self) -> float:
        return self.width_cells * self.cell_size

    @property
    def height(self) -> float:
        return self.height_cells * self.cell_size


DEFAULT_GRID = Grid(
    layout=["#####", "#000#", "#000#", "#000#", "#####"],
    width_cells=10,
    height_cells=10,
    cell_size=0.100,
)


class WallCollisionChecker:
    """
    An efficient, memoized collision checker for a static map.

    This class pre-processes the map layout upon initialization to build a
    NumPy array of wall coordinates. Subsequent collision checks are then
    performed with highly optimized vectorized operations.
    """

    def __init__(self, layout: Layout, L: float):
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
                    wall_min_x = c * L
                    wall_max_x = (c + 1) * L
                    wall_min_y = (n_rows - 1 - r) * L
                    wall_max_y = (n_rows - r) * L

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
        # Find all walls whose bounding boxes *could possibly* overlap with the robot's effective circle.
        # This is much faster than looping through grid cells.
        min_check_x = x - R_eff
        max_check_x = x + R_eff
        min_check_y = y - R_eff
        max_check_y = y + R_eff

        # Use NumPy's boolean indexing for a fast, vectorized filter.
        # This checks for any overlap between the robot's check-box and the wall boxes.
        candidate_walls = self.wall_bounds[
            (self.wall_bounds[:, 0] < max_check_x)  # wall_min_x < max_check_x
            & (self.wall_bounds[:, 1] > min_check_x)  # wall_max_x > min_check_x
            & (self.wall_bounds[:, 2] < max_check_y)  # wall_min_y < max_check_y
            & (self.wall_bounds[:, 3] > min_check_y)  # wall_max_y > min_check_y
        ]

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
