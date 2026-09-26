"""Geometry and placement utilities for PreyPredEnv."""

from __future__ import annotations

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from contgrid.core import Color, EntityShape, EntityState, Landmark, World
from contgrid.core.typing import Position

from .configs import CircularOrbitTrajectoryConfig, RegionEntityConfig
from .trajectories import BaseTrajectory, create_trajectory

PREY_COLORS: list[Color] = [
    Color.GREEN,
    Color.SKY_BLUE,
    Color.YELLOW,
    Color.PURPLE,
]
PRED_COLORS: list[Color] = [
    Color.RED,
    Color.PINK,
    Color.GREY,
    Color.ORANGE,
]
PRED_HATCHES: list[str] = ["///", "\\\\\\", "xxx", "..."]


class TrajectoryRenderItem(BaseModel):
    """Render specification for an entity trajectory and its display color."""

    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    trajectory: BaseTrajectory
    color: Color


def _resolve_spawn_pos(
    spawn_pos: Position | Literal["center", "random"],
    reg_center: Position,
) -> Position:
    """Resolve configured spawn position to concrete 2D coordinates.

    Args:
        spawn_pos: Configured placement ("center", "random", or coordinate tuple).
        reg_center: Center position of the region.

    Returns:
        Concrete (x, y) position tuple.
    """
    match spawn_pos:
        case "center":
            return reg_center
        case "random":
            return reg_center
        case (x, y):
            return (float(x), float(y))


def build_region_entities(
    reg: RegionEntityConfig,
) -> tuple[Landmark, BaseTrajectory, Landmark, BaseTrajectory]:
    """Instantiate Prey, Predator, and their trajectories for a region.

    Args:
        reg: Region entity configuration.

    Returns:
        Tuple of (prey_landmark, prey_trajectory, pred_landmark, pred_trajectory).
    """
    rid = reg.region_id
    min_x, max_x, min_y, max_y = reg.bounds
    reg_center: Position = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)

    p_pos = _resolve_spawn_pos(reg.prey.spawn_pos, reg_center)
    p_traj = create_trajectory(reg.prey.trajectory, p_pos)
    prey = Landmark(
        name=f"prey_{rid}",
        size=reg.prey.size,
        shape=EntityShape.CIRCLE,
        collide=False,
        movable=False,
        color=PREY_COLORS[rid % 4].name,
        state=EntityState(pos=np.array(p_pos, dtype=np.float64)),
    )

    q_pos = _resolve_spawn_pos(reg.predator.spawn_pos, reg_center)
    orbit_center = reg_center
    if isinstance(reg.predator.trajectory, CircularOrbitTrajectoryConfig):
        if reg.predator.trajectory.orbit_center == "region_center":
            orbit_center = reg_center
        elif isinstance(reg.predator.trajectory.orbit_center, tuple):
            orbit_center = reg.predator.trajectory.orbit_center

    q_traj = create_trajectory(
        reg.predator.trajectory,
        orbit_center,
        partner_entity=prey,
    )
    pred = Landmark(
        name=f"predator_{rid}",
        size=reg.predator.size,
        shape=EntityShape.CIRCLE,
        collide=False,
        movable=False,
        color=PRED_COLORS[rid % 4].name,
        hatch=PRED_HATCHES[rid % 4],
        state=EntityState(pos=np.array(q_pos, dtype=np.float64)),
    )
    return prey, p_traj, pred, q_traj


def enforce_bounds_and_clearance(
    reg: RegionEntityConfig,
    p_pos: NDArray[np.float64],
    p_size: float,
    p_moving: bool,
    q_pos: NDArray[np.float64],
    q_size: float,
    q_moving: bool,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Clamp entities to region boundaries and maintain minimum safety clearance.

    Args:
        reg: Region configuration defining bounds and minimum clearance.
        p_pos: 2D coordinates of the prey.
        p_size: Radius of the prey.
        p_moving: Whether prey moves dynamically.
        q_pos: 2D coordinates of the predator.
        q_size: Radius of the predator.
        q_moving: Whether predator moves dynamically.

    Returns:
        Tuple of (adjusted_prey_pos, adjusted_predator_pos).
    """
    min_x, max_x, min_y, max_y = reg.bounds
    min_dist = p_size + q_size + reg.min_clearance

    diff = q_pos - p_pos
    dist = float(np.linalg.norm(diff))
    if dist < min_dist:
        unit = diff / dist if dist > 1e-6 else np.array([1.0, 0.0])
        correction = min_dist - dist
        if p_moving and not q_moving:
            p_pos = p_pos - unit * correction
        elif q_moving and not p_moving:
            q_pos = q_pos + unit * correction
        else:
            p_pos = p_pos - unit * (correction * 0.5)
            q_pos = q_pos + unit * (correction * 0.5)

    p_pos = np.clip(
        p_pos,
        [min_x + p_size, min_y + p_size],
        [max_x - p_size, max_y - p_size],
    )
    q_pos = np.clip(
        q_pos,
        [min_x + q_size, min_y + q_size],
        [max_x - q_size, max_y - q_size],
    )
    return p_pos, q_pos


def sample_neutral_corridor(
    world: World,
    agent_size: float,
    np_random: np.random.Generator,
    max_attempts: int = 200,
) -> Position:
    """Sample a valid collision-free spawn position in the central neutral corridors.

    Args:
        world: ContGrid world instance.
        agent_size: Radius of the agent.
        np_random: Random number generator.
        max_attempts: Maximum rejection sampling attempts.

    Returns:
        (x, y) coordinates for agent spawn.
    """
    for _ in range(max_attempts):
        if np_random.uniform() < 0.5:
            cx = float(np_random.uniform(6.0, 8.0))
            cy = float(np_random.uniform(2.5, 11.5))
        else:
            cx = float(np_random.uniform(2.5, 11.5))
            cy = float(np_random.uniform(6.0, 8.0))
        if world.wall_collision_checker.is_position_valid(
            agent_size, world.contact_margin, (cx, cy)
        ):
            return (cx, cy)
    return (7.0, 7.0)


def compute_wall_distances(
    agent_pos: NDArray[np.float64],
    wall_bounds: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Calculate distances to blocking walls in cardinal directions [top, right, bottom, left].

    Args:
        agent_pos: Current (x, y) position of the agent.
        wall_bounds: (N, 4) array with [min_x, max_x, min_y, max_y] per wall cell.

    Returns:
        1D array of 4 cardinal distances [top, right, bottom, left].
    """
    if len(wall_bounds) == 0:
        return np.full(4, np.inf, dtype=np.float64)

    ax, ay = agent_pos[0], agent_pos[1]
    w_min_x, w_max_x = wall_bounds[:, 0], wall_bounds[:, 1]
    w_min_y, w_max_y = wall_bounds[:, 2], wall_bounds[:, 3]

    x_al = (w_min_x <= ax) & (ax <= w_max_x)
    y_al = (w_min_y <= ay) & (ay <= w_max_y)

    top_m = x_al & (w_min_y > ay)
    top_d = float(np.min(w_min_y[top_m] - ay)) if np.any(top_m) else np.inf

    bot_m = x_al & (w_max_y < ay)
    bot_d = float(np.min(ay - w_max_y[bot_m])) if np.any(bot_m) else np.inf

    right_m = y_al & (w_min_x > ax)
    right_d = (
        float(np.min(w_min_x[right_m] - ax)) if np.any(right_m) else np.inf
    )

    left_m = y_al & (w_max_x < ax)
    left_d = float(np.min(ax - w_max_x[left_m])) if np.any(left_m) else np.inf

    return np.array([top_d, right_d, bot_d, left_d], dtype=np.float64)
