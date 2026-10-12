from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import numpy as np
from absl.testing import absltest

from contgrid.envs.rooms import (
    ObjConfig,
    PathGaussianConfig,
    RewardConfig,
    RoomsEnv,
    RoomsScenario,
    RoomsScenarioConfig,
    RoomTopology,
    SpawnConfig,
)
from contgrid.envs.rooms.topology import LineSegment, get_relevant_path_segments


def _point_to_segment_distance(
    point: np.ndarray, segment: LineSegment
) -> float:
    v = segment.end - segment.start
    w = point - segment.start
    c1 = np.dot(w, v)
    c2 = np.dot(v, v)
    if c2 < 1e-10:
        return float(np.linalg.norm(point - segment.start))
    t = max(0.0, min(1.0, float(c1 / c2)))
    closest = segment.start + t * v
    return float(np.linalg.norm(point - closest))


def _min_distance_to_any_segment(
    point: np.ndarray, segments: Sequence[LineSegment]
) -> float:
    if not segments:
        return float("inf")
    return min(_point_to_segment_distance(point, seg) for seg in segments)


def _clip_segment_to_room(
    seg: LineSegment, room_name: str, topology: RoomTopology
) -> LineSegment | None:
    bounds = topology.room_boundaries[room_name]
    edge_buffer = 0.05
    x1, y1 = float(seg.start[0]), float(seg.start[1])
    x2, y2 = float(seg.end[0]), float(seg.end[1])
    dx, dy = x2 - x1, y2 - y1
    min_x = bounds["min_x"] + edge_buffer
    max_x = bounds["max_x"] - edge_buffer
    min_y = bounds["min_y"] + edge_buffer
    max_y = bounds["max_y"] - edge_buffer
    t0, t1 = 0.0, 1.0
    p = [-dx, dx, -dy, dy]
    q = [x1 - min_x, max_x - x1, y1 - min_y, max_y - y1]
    for i in range(4):
        if abs(p[i]) < 1e-10:
            if q[i] < 0:
                return None
        else:
            t = q[i] / p[i]
            if p[i] < 0:
                t0 = max(t0, t)
            else:
                t1 = min(t1, t)
    if t0 > t1:
        return None
    return LineSegment(
        np.array([x1 + t0 * dx, y1 + t0 * dy]),
        np.array([x1 + t1 * dx, y1 + t1 * dy]),
    )


class TestRoomsEnv(absltest.TestCase):
    def test_goal_reaching_reward_and_termination(self) -> None:
        spawn_config = SpawnConfig(
            agent=(3.0, 3.0),
            goal=ObjConfig(pos=(3.0, 3.0), reward=10.0, absorbing=True),
            lavas=[],
            holes=[],
        )
        env = RoomsEnv(
            scenario_config=RoomsScenarioConfig(spawn_config=spawn_config)
        )
        self.addCleanup(env.close)
        env.reset(seed=42)

        _, reward, termination, _, info = env.step(np.array([0.01, 0.01]))
        self.assertEqual(reward, 10.0)
        self.assertTrue(termination and info.get("terminated"))

    def test_step_penalty(self) -> None:
        spawn_config = SpawnConfig(
            agent=(1.0, 1.0),
            goal=ObjConfig(pos=(10.0, 10.0), reward=1.0, absorbing=True),
            lavas=[],
            holes=[],
        )
        env = RoomsEnv(
            scenario_config=RoomsScenarioConfig(
                spawn_config=spawn_config,
                reward_config=RewardConfig(step_penalty=0.1),
            )
        )
        self.addCleanup(env.close)
        env.reset(seed=42)

        _, reward, termination, _, _ = env.step(np.array([0.1, 0.1]))
        self.assertAlmostEqual(float(reward), -0.1)
        self.assertFalse(termination)

    def test_doorway_distances_in_info(self) -> None:
        env = RoomsEnv()
        self.addCleanup(env.close)
        env.reset(seed=42)

        _, _, _, _, info = env.step(np.array([0.0, 0.0]))
        distances = info.get("distances", {})
        self.assertTrue({"ld", "td", "rd", "bd"}.issubset(distances.keys()))
        self.assertTrue(all(d >= 0 for d in distances.values()))

    def test_no_agent_obstacle_overlap_at_spawn(self) -> None:
        spawn_config = SpawnConfig(
            agent=None,
            goal=ObjConfig(pos=(9.0, 8.0), reward=1.0, absorbing=False),
            lavas=[ObjConfig(pos=None), ObjConfig(pos=None)],
            holes=[ObjConfig(pos=None), ObjConfig(pos=None)],
            spawn_method=PathGaussianConfig(
                gaussian_std=0.5,
                min_spacing=0.9,
                edge_buffer=0.3,
                include_agent_paths=True,
            ),
        )
        env = RoomsEnv(
            scenario_config=RoomsScenarioConfig(spawn_config=spawn_config)
        )
        self.addCleanup(env.close)
        env.reset(seed=42)

        scenario = cast(RoomsScenario, env.scenario)
        agent_pos = env.world.agents[0].state.pos
        min_lava_dist = spawn_config.agent_size + spawn_config.lava_size
        min_hole_dist = spawn_config.agent_size + spawn_config.hole_size
        lava_dists = [
            float(np.linalg.norm(pos - agent_pos)) for pos in scenario.lava_pos
        ]
        hole_dists = [
            float(np.linalg.norm(pos - agent_pos)) for pos in scenario.hole_pos
        ]

        self.assertTrue(all(d >= min_lava_dist for d in lava_dists))
        self.assertTrue(all(d >= min_hole_dist for d in hole_dists))

    def test_path_gaussian_spawn_zero_std_on_segments(self) -> None:
        spawn_config = SpawnConfig(
            agent=(3.0, 3.0),
            goal=ObjConfig(pos=(9.0, 8.0), reward=1.0, absorbing=False),
            lavas=[
                ObjConfig(pos=None, room="top_left"),
                ObjConfig(pos=None, room="bottom_left"),
            ],
            holes=[
                ObjConfig(pos=None, room="top_right"),
                ObjConfig(pos=None, room="bottom_right"),
            ],
            spawn_method=PathGaussianConfig(
                gaussian_std=0.0,
                min_spacing=0.3,
                edge_buffer=0.05,
                include_agent_paths=True,
            ),
            lava_size=0.2,
            hole_size=0.2,
        )
        config = RoomsScenarioConfig(spawn_config=spawn_config)
        env = RoomsEnv(scenario_config=config)
        self.addCleanup(env.close)
        env.reset(seed=42)

        scenario = cast(RoomsScenario, env.scenario)
        topology = RoomTopology(config.spawn_config.doorways)
        segments = get_relevant_path_segments(
            env.world.agents[0].state.pos,
            scenario.goal_pos,
            scenario.doorways,
            topology,
            env.world.grid,
        )

        for i, lava_pos in enumerate(scenario.lava_pos):
            room = config.spawn_config.lavas[i].room
            assert room is not None
            clipped_segs = [
                cs
                for seg in segments
                if (cs := _clip_segment_to_room(seg, room, topology))
                is not None
                and np.linalg.norm(cs.end - cs.start) > 1e-6
            ]
            self.assertLess(
                _min_distance_to_any_segment(lava_pos, clipped_segs), 1e-6
            )


if __name__ == "__main__":
    absltest.main()
