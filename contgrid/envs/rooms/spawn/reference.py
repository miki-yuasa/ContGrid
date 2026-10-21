"""Reference pure-Python implementation of path-Gaussian obstacle spawning."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from contgrid.core import World
from contgrid.core.typing import Position
from contgrid.envs.rooms.topology import LineSegment

if TYPE_CHECKING:
    from contgrid.envs.rooms.scenario import RoomsScenario

    from .strategies import PathGaussianConfig


def sample_path_gaussian_reference(
    config: PathGaussianConfig,
    num_obstacles: int,
    obstacle_type: str,
    world: World,
    scenario: RoomsScenario,
    np_random: np.random.Generator,
    agent_pos: NDArray[np.float64] | None,
    obstacle_rooms: Sequence[str | None],
    segments: Sequence[LineSegment],
    bounds_by_room: Mapping[str, Mapping[str, float]],
) -> list[Position]:
    """Sample obstacle positions in pure Python as a reference."""
    positions: list[Position] = []
    existing = _get_existing_obstacles(scenario)
    valid_segs = [s for s in segments if s.length() > 1e-6]
    if not valid_segs or num_obstacles == 0:
        return positions

    lengths = np.array([s.length() for s in valid_segs])
    probs = lengths / lengths.sum()

    for idx in range(num_obstacles):
        req_room = obstacle_rooms[idx] if idx < len(obstacle_rooms) else None
        room_segs, room_probs = _filter_segments_for_room(
            valid_segs, probs, req_room, bounds_by_room, config.edge_buffer
        )
        pos = _find_valid_position(
            config,
            room_segs,
            room_probs,
            list(positions) + existing,
            world,
            scenario,
            obstacle_type,
            req_room,
            agent_pos,
            np_random,
            bounds_by_room,
        )
        if pos is not None:
            positions.append(pos)
    return positions


def _filter_segments_for_room(
    segments: Sequence[LineSegment],
    probabilities: NDArray[np.float64],
    required_room: str | None,
    bounds_by_room: Mapping[str, Mapping[str, float]],
    edge_buffer: float,
) -> tuple[Sequence[LineSegment], NDArray[np.float64]]:
    """Filter and clip segments to the required room boundaries."""
    if required_room is None or required_room not in bounds_by_room:
        return segments, probabilities

    clipped = [
        cs
        for s in segments
        if (
            cs := _clip_segment_to_room(
                s, required_room, bounds_by_room, edge_buffer
            )
        )
        and cs.length() > 1e-6
    ]
    if not clipped:
        return segments, probabilities
    lengths = np.array([s.length() for s in clipped])
    return clipped, lengths / lengths.sum()


def _clip_segment_to_room(
    segment: LineSegment,
    room_name: str,
    bounds_by_room: Mapping[str, Mapping[str, float]],
    edge_buffer: float,
) -> LineSegment | None:
    """Clip a segment to room boundaries using the Liang-Barsky algorithm."""
    if room_name not in bounds_by_room:
        return segment

    b = bounds_by_room[room_name]
    x1, y1 = float(segment.start[0]), float(segment.start[1])
    x2, y2 = float(segment.end[0]), float(segment.end[1])
    dx, dy = x2 - x1, y2 - y1

    min_x, max_x = b["min_x"] + edge_buffer, b["max_x"] - edge_buffer
    min_y, max_y = b["min_y"] + edge_buffer, b["max_y"] - edge_buffer
    t0, t1 = 0.0, 1.0
    p = [-dx, dx, -dy, dy]
    q = [x1 - min_x, max_x - x1, y1 - min_y, max_y - y1]

    for pi, qi in zip(p, q, strict=False):
        if abs(pi) < 1e-10 and qi < 0.0:
            return None
        if abs(pi) >= 1e-10:
            t = qi / pi
            t0 = max(t0, t) if pi < 0.0 else t0
            t1 = min(t1, t) if pi >= 0.0 else t1

    if t0 > t1:
        return None
    return LineSegment(
        np.array([x1 + t0 * dx, y1 + t0 * dy]),
        np.array([x1 + t1 * dx, y1 + t1 * dy]),
    )


def _find_valid_position(
    config: PathGaussianConfig,
    segments: Sequence[LineSegment],
    probabilities: NDArray[np.float64],
    all_obs: Sequence[Position],
    world: World,
    scenario: RoomsScenario,
    obstacle_type: str,
    required_room: str | None,
    agent_pos: NDArray[np.float64] | None,
    np_random: np.random.Generator,
    bounds_by_room: Mapping[str, Mapping[str, float]],
) -> Position | None:
    """Search for a valid position using random then systematic sampling."""
    for _ in range(100):
        seg = segments[int(np_random.choice(len(segments), p=probabilities))]
        cand = _sample_position_on_segment(
            seg, float(np_random.uniform(0, 1)), config.gaussian_std, np_random
        )
        if required_room is not None:
            cand = _clip_to_room_bounds(
                cand, required_room, bounds_by_room, config.edge_buffer
            )

        if _is_valid_position(
            config,
            cand,
            world,
            all_obs,
            scenario,
            obstacle_type,
            required_room,
            agent_pos,
            bounds_by_room,
        ):
            return (float(cand[0]), float(cand[1]))

    return _fallback_sampling(
        config,
        segments,
        all_obs,
        world,
        scenario,
        obstacle_type,
        required_room,
        agent_pos,
        np_random,
        bounds_by_room,
    )


def _sample_position_on_segment(
    segment: LineSegment,
    t: float,
    gaussian_std: float,
    np_random: np.random.Generator,
) -> NDArray[np.float64]:
    """Sample a point on a segment with optional Gaussian noise."""
    base_pos = segment.sample_point(t)
    if gaussian_std <= 0:
        return base_pos

    d = segment.end - segment.start
    norm = float(np.linalg.norm(d))
    u = d / norm if norm > 1e-6 else np.array([1.0, 0.0])
    p = np.array([-u[1], u[0]])
    n = np_random.normal(0, gaussian_std, size=2)
    return base_pos + float(n[0]) * u + float(n[1]) * p


def _clip_to_room_bounds(
    pos: NDArray[np.float64],
    room_name: str,
    bounds_by_room: Mapping[str, Mapping[str, float]],
    edge_buffer: float,
) -> NDArray[np.float64]:
    """Clip coordinates to fall strictly inside room boundaries."""
    if room_name not in bounds_by_room:
        return pos
    b, eb = bounds_by_room[room_name], edge_buffer
    mins = [b["min_x"] + eb, b["min_y"] + eb]
    maxs = [b["max_x"] - eb, b["max_y"] - eb]
    return np.asarray(np.clip(pos, mins, maxs), dtype=np.float64)


def _fallback_sampling(
    config: PathGaussianConfig,
    segments: Sequence[LineSegment],
    all_obs: Sequence[Position],
    world: World,
    scenario: RoomsScenario,
    obstacle_type: str,
    required_room: str | None,
    agent_pos: NDArray[np.float64] | None,
    np_random: np.random.Generator,
    bounds_by_room: Mapping[str, Mapping[str, float]],
) -> Position | None:
    """Sample systematic candidates across all segments as a fallback."""
    if not segments:
        return None

    orders = list(range(len(segments)))
    np_random.shuffle(orders)
    t_vals = np.linspace(0.05, 0.95, 20)

    for seg_idx in orders:
        seg = segments[seg_idx]
        if seg.length() < 1e-6:
            continue
        np_random.shuffle(t_vals)
        for t in t_vals:
            t_pert = float(
                np.clip(t + np_random.uniform(-0.02, 0.02), 0.01, 0.99)
            )
            cand = _sample_position_on_segment(
                seg, t_pert, config.gaussian_std, np_random
            )
            if required_room is not None:
                cand = _clip_to_room_bounds(
                    cand, required_room, bounds_by_room, config.edge_buffer
                )
            if _is_valid_position(
                config,
                cand,
                world,
                all_obs,
                scenario,
                obstacle_type,
                required_room,
                agent_pos,
                bounds_by_room,
            ):
                return (float(cand[0]), float(cand[1]))
    return None


def _is_valid_position(
    config: PathGaussianConfig,
    pos: NDArray[np.float64],
    world: World,
    existing_positions: Sequence[Position],
    scenario: RoomsScenario,
    obstacle_type: str,
    required_room: str | None,
    agent_pos: NDArray[np.float64] | None,
    bounds_by_room: Mapping[str, Mapping[str, float]],
) -> bool:
    """Validate that candidate position satisfies all spatial constraints."""
    if scenario.config is None:
        raise ValueError("Scenario config must not be None.")
    if required_room is not None and required_room in bounds_by_room:
        b = bounds_by_room[required_room]
        if not (
            b["min_x"] <= pos[0] <= b["max_x"]
            and b["min_y"] <= pos[1] <= b["max_y"]
        ):
            return False

    lim, eb = world.grid.wall_limits, config.edge_buffer
    if not (
        lim.min_x + eb <= pos[0] <= lim.max_x - eb
        and lim.min_y + eb <= pos[1] <= lim.max_y - eb
    ):
        return False

    sc_spawn = scenario.config.spawn_config
    obs_r = (
        sc_spawn.lava_size if obstacle_type == "lava" else sc_spawn.hole_size
    )
    is_col, _ = world.wall_collision_checker.is_collision(
        R=obs_r,
        C=0.0,
        robot_pos=(float(pos[0]), float(pos[1])),
        collision_force=1.0,
        contact_margin=0.1,
    )
    if is_col:
        return False

    x, y = float(pos[0]), float(pos[1])
    min_spacing_sq = config.min_spacing**2
    if any(
        (x - ex) ** 2 + (y - ey) ** 2 < min_spacing_sq
        for ex, ey in existing_positions
    ):
        return False

    gx, gy = float(scenario.goal_pos[0]), float(scenario.goal_pos[1])
    if (x - gx) ** 2 + (y - gy) ** 2 < (scenario.goal_thr_dist + 0.5) ** 2:
        return False

    if agent_pos is not None:
        min_ag = obs_r + sc_spawn.agent_size
        ax, ay = float(agent_pos[0]), float(agent_pos[1])
        if (x - ax) ** 2 + (y - ay) ** 2 < min_ag**2:
            return False

    return True


def _get_existing_obstacles(scenario: RoomsScenario) -> list[Position]:
    """Extract all active obstacle positions currently in the scenario."""
    return [
        (float(p[0]), float(p[1]))
        for lm in list(scenario.lavas) + list(scenario.holes)
        if isinstance(p := getattr(lm.state, "pos", None), np.ndarray)
        and p.shape == (2,)
        and (p[0] != 0.0 or p[1] != 0.0)
    ]
