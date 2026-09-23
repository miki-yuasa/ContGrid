from __future__ import annotations

import tempfile
from pathlib import Path
from typing import cast

import numpy as np
import yaml
from absl.testing import absltest
from gymnasium import spaces

from contgrid.envs.rooms import (
    DEFAULT_ROOMS_SCENARIO_CONFIG,
    DEFAULT_WORLD_CONFIG,
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
    point: np.ndarray, segments: list[LineSegment]
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
    def test_env_initialization(self) -> None:
        env = RoomsEnv()
        self.addCleanup(env.close)

        world = env.world
        self.assertLen(world.agents, 1)
        self.assertEqual(world.agents[0].name, "agent_0")

        default_config = DEFAULT_ROOMS_SCENARIO_CONFIG
        expected_landmarks = (
            1
            + len(default_config.spawn_config.lavas)
            + len(default_config.spawn_config.holes)
        )
        self.assertLen(world.landmarks, expected_landmarks)

    def test_env_reset(self) -> None:
        env = RoomsEnv()
        self.addCleanup(env.close)

        observation, info = env.reset(seed=42)

        self.assertIsInstance(observation, dict)
        expected_keys = {
            "agent_pos",
            "agent_vel",
            "goal_pos",
            "lava_pos",
            "hole_pos",
            "doorway_pos",
            "goal_dist",
            "lava_dist",
            "hole_dist",
            "wall_dist",
            "doorway_dist",
        }
        self.assertSetEqual(set(observation.keys()), expected_keys)
        self.assertIsInstance(observation["agent_pos"], np.ndarray)
        self.assertEqual(observation["agent_pos"].shape, (2,))
        self.assertIsInstance(observation["goal_pos"], np.ndarray)
        self.assertEqual(observation["goal_pos"].shape, (2,))
        self.assertIsInstance(info, dict)

    def test_export_and_save_spawned_config(self) -> None:
        spawn_config = SpawnConfig(
            agent=None,
            goal=ObjConfig(pos=None, reward=1.0, absorbing=False),
            lavas=[
                ObjConfig(pos=None, reward=-1.0, absorbing=False),
                ObjConfig(pos=None, reward=-1.0, absorbing=False),
            ],
            holes=[
                ObjConfig(pos=None, reward=-1.0, absorbing=False),
                ObjConfig(pos=None, reward=-1.0, absorbing=False),
            ],
        )
        env = RoomsEnv(
            scenario_config=RoomsScenarioConfig(spawn_config=spawn_config)
        )
        self.addCleanup(env.close)
        env.reset(seed=123)

        exported = env.export_spawned_config()
        self.assertIsInstance(exported, RoomsScenarioConfig)

        scenario = cast(RoomsScenario, env.scenario)
        world_agent_pos = tuple(float(v) for v in env.world.agents[0].state.pos)
        world_goal_pos = tuple(float(v) for v in scenario.goal.state.pos)
        world_lava_pos = [
            tuple(float(v) for v in lava.state.pos) for lava in scenario.lavas
        ]
        world_hole_pos = [
            tuple(float(v) for v in hole.state.pos) for hole in scenario.holes
        ]

        self.assertEqual(exported.spawn_config.agent, world_agent_pos)
        self.assertEqual(exported.spawn_config.goal.pos, world_goal_pos)
        self.assertEqual(
            [obj.pos for obj in exported.spawn_config.lavas], world_lava_pos
        )
        self.assertEqual(
            [obj.pos for obj in exported.spawn_config.holes], world_hole_pos
        )

        tmp_dir = Path(self.enter_context(tempfile.TemporaryDirectory()))
        output_path = tmp_dir / "spawned_config.yaml"
        saved = env.save_spawned_config(output_path)
        self.assertTrue(output_path.exists())

        loaded_data = yaml.safe_load(output_path.read_text(encoding="utf-8"))
        loaded = RoomsScenarioConfig.model_validate(loaded_data)
        self.assertEqual(loaded.model_dump(), saved.model_dump())

    def test_observation_space(self) -> None:
        env = RoomsEnv()
        self.addCleanup(env.close)

        agent_name = env.possible_agents[0]
        obs_space = env.observation_spaces[agent_name]

        self.assertIsInstance(obs_space, spaces.Dict)
        assert isinstance(obs_space, spaces.Dict)
        expected_keys = {
            "agent_pos",
            "agent_vel",
            "goal_pos",
            "lava_pos",
            "hole_pos",
            "doorway_pos",
            "goal_dist",
            "lava_dist",
            "hole_dist",
            "wall_dist",
            "doorway_dist",
        }
        self.assertSetEqual(set(obs_space.spaces.keys()), expected_keys)
        self.assertIsInstance(obs_space["agent_pos"], spaces.Box)
        self.assertEqual(obs_space["agent_pos"].shape, (2,))
        self.assertIsInstance(obs_space["goal_pos"], spaces.Box)
        self.assertEqual(obs_space["goal_pos"].shape, (2,))

    def test_action_space(self) -> None:
        env = RoomsEnv()
        self.addCleanup(env.close)

        agent_name = env.possible_agents[0]
        action_space = env.action_spaces[agent_name]

        self.assertIsInstance(action_space, spaces.Box)
        assert isinstance(action_space, spaces.Box)
        self.assertEqual(action_space.shape, (2,))
        self.assertEqual(action_space.low.tolist(), [-10.0, -10.0])
        self.assertEqual(action_space.high.tolist(), [10.0, 10.0])

    def test_step_functionality(self) -> None:
        env = RoomsEnv()
        self.addCleanup(env.close)
        env.reset(seed=42)

        action = np.array([0.1, 0.1], dtype=np.float64)
        observation, reward, termination, truncation, info = env.step(action)

        self.assertIsInstance(observation, dict)
        self.assertIsInstance(reward, (int, float))
        self.assertIsInstance(termination, bool)
        self.assertIsInstance(truncation, bool)
        self.assertIsInstance(info, dict)
        self.assertIsNotNone(observation["agent_pos"])

    def test_reward_system_goal_reaching(self) -> None:
        spawn_config = SpawnConfig(
            agent=(3.0, 3.0),
            goal=ObjConfig(pos=(3.0, 3.0), reward=10.0, absorbing=True),
            lavas=[],
            holes=[],
        )
        config = RoomsScenarioConfig(spawn_config=spawn_config)
        env = RoomsEnv(scenario_config=config)
        self.addCleanup(env.close)

        observation, _ = env.reset(seed=42)
        goal_dist = observation["goal_dist"]
        self.assertLess(goal_dist, 1.0)

        action = np.array([0.01, 0.01], dtype=np.float64)
        _, reward, termination, _, info = env.step(action)

        self.assertEqual(reward, 10.0)
        self.assertTrue(termination)
        self.assertTrue(info["terminated"])

    def test_step_penalty(self) -> None:
        spawn_config = SpawnConfig(
            agent=(1.0, 1.0),
            goal=ObjConfig(pos=(10.0, 10.0), reward=1.0, absorbing=True),
            lavas=[],
            holes=[],
        )
        reward_config = RewardConfig(step_penalty=0.1)
        config = RoomsScenarioConfig(
            spawn_config=spawn_config, reward_config=reward_config
        )
        env = RoomsEnv(scenario_config=config)
        self.addCleanup(env.close)

        env.reset(seed=42)
        action = np.array([0.1, 0.1], dtype=np.float64)
        _, reward, termination, _, _ = env.step(action)

        self.assertAlmostEqual(float(reward), -0.1)
        self.assertFalse(termination)

    def test_custom_configuration(self) -> None:
        custom_spawn = SpawnConfig(
            agent=(2.0, 2.0),
            goal=ObjConfig(pos=(8.0, 8.0), reward=5.0, absorbing=True),
            lavas=[ObjConfig(pos=(4.0, 4.0), reward=-2.0, absorbing=True)],
            holes=[ObjConfig(pos=(6.0, 6.0), reward=-1.0, absorbing=False)],
        )
        custom_reward = RewardConfig(step_penalty=0.05)
        custom_config = RoomsScenarioConfig(
            spawn_config=custom_spawn, reward_config=custom_reward
        )

        env = RoomsEnv(scenario_config=custom_config)
        self.addCleanup(env.close)
        env.reset(seed=42)

        scenario = cast(RoomsScenario, env.scenario)
        agent_pos_world = env.world.agents[0].state.pos
        np.testing.assert_array_almost_equal(
            agent_pos_world, np.array([2.0, 2.0])
        )
        np.testing.assert_array_almost_equal(
            scenario.goal_pos, np.array([8.0, 8.0])
        )
        self.assertEqual(scenario.lava_pos.shape, (1, 2))
        self.assertEqual(scenario.hole_pos.shape, (1, 2))

    def test_rendering(self) -> None:
        env = RoomsEnv()
        self.addCleanup(env.close)
        env.reset(seed=42)

        rendered = env.render()
        self.assertIsNotNone(rendered)
        self.assertIsInstance(rendered, np.ndarray)
        self.assertEqual(rendered.ndim, 3)
        self.assertEqual(rendered.shape[2], 3)
        self.assertEqual(rendered.dtype, np.uint8)

    def test_multiple_episodes(self) -> None:
        env = RoomsEnv()
        self.addCleanup(env.close)

        for episode in range(3):
            env.reset(seed=42 + episode)
            done = False
            step_count = 0
            while not done and step_count < 10:
                action = env.action_spaces[env.agent_selection].sample()
                _, _, termination, truncation, _ = env.step(action)
                done = termination or truncation
                step_count += 1
            self.assertLessEqual(step_count, 10)

    def test_scenario_doorway_distances(self) -> None:
        env = RoomsEnv()
        self.addCleanup(env.close)
        env.reset(seed=42)

        action = np.array([0.0, 0.0], dtype=np.float64)
        _, _, _, _, info = env.step(action)

        self.assertIn("distances", info)
        distances = info["distances"]
        expected_doorways = {"ld", "td", "rd", "bd"}
        for doorway in expected_doorways:
            self.assertIn(doorway, distances)
            self.assertIsInstance(distances[doorway], (int, float))
            self.assertGreaterEqual(distances[doorway], 0)

    def test_environment_termination_conditions(self) -> None:
        spawn_config = SpawnConfig(
            agent=(3.0, 3.0),
            goal=ObjConfig(pos=(3.5, 3.0), reward=10.0, absorbing=True),
            lavas=[],
            holes=[],
        )
        config = RoomsScenarioConfig(spawn_config=spawn_config)
        env = RoomsEnv(scenario_config=config)
        self.addCleanup(env.close)

        env.reset(seed=42)
        terminated = False
        final_reward = None
        for _ in range(10):
            action = np.array([0.1, 0.0], dtype=np.float64)
            _, reward, termination, _, _ = env.step(action)
            if termination:
                terminated = True
                final_reward = reward
                break

        self.assertTrue(terminated)
        self.assertEqual(final_reward, 10.0)

    def test_no_agent_obstacle_overlap_at_spawn(self) -> None:
        for seed in [42, 123]:
            spawn_config = SpawnConfig(
                agent=None,
                goal=ObjConfig(pos=(9, 8), reward=1.0, absorbing=False),
                lavas=[
                    ObjConfig(pos=None, reward=-1.0, absorbing=False),
                    ObjConfig(pos=None, reward=-1.0, absorbing=False),
                ],
                holes=[
                    ObjConfig(pos=None, reward=-0.5, absorbing=False),
                    ObjConfig(pos=None, reward=-0.5, absorbing=False),
                ],
                spawn_method=PathGaussianConfig(
                    gaussian_std=0.5,
                    min_spacing=0.9,
                    edge_buffer=0.3,
                    include_agent_paths=True,
                ),
            )
            config = RoomsScenarioConfig(spawn_config=spawn_config)
            env = RoomsEnv(scenario_config=config)
            self.addCleanup(env.close)
            env.reset(seed=seed)

            scenario = cast(RoomsScenario, env.scenario)
            agent_pos_world = env.world.agents[0].state.pos
            agent_radius = scenario.config.spawn_config.agent_size
            lava_radius = scenario.config.spawn_config.lava_size
            hole_radius = scenario.config.spawn_config.hole_size

            if len(scenario.lava_pos) > 0:
                min_allowed_distance = agent_radius + lava_radius
                for i, lava_pos in enumerate(scenario.lava_pos):
                    distance = float(np.linalg.norm(lava_pos - agent_pos_world))
                    self.assertGreaterEqual(
                        distance,
                        min_allowed_distance,
                        msg=f"Seed {seed}: Lava {i} overlaps agent",
                    )

            if len(scenario.hole_pos) > 0:
                min_allowed_distance = agent_radius + hole_radius
                for i, hole_pos in enumerate(scenario.hole_pos):
                    distance = float(np.linalg.norm(hole_pos - agent_pos_world))
                    self.assertGreaterEqual(
                        distance,
                        min_allowed_distance,
                        msg=f"Seed {seed}: Hole {i} overlaps agent",
                    )


class TestRoomsScenario(absltest.TestCase):
    def test_scenario_initialization(self) -> None:
        scenario = RoomsScenario(
            config=DEFAULT_ROOMS_SCENARIO_CONFIG,
            world_config=DEFAULT_WORLD_CONFIG,
        )
        self.assertEqual(scenario.config, DEFAULT_ROOMS_SCENARIO_CONFIG)
        self.assertEqual(scenario.world_config, DEFAULT_WORLD_CONFIG)
        self.assertTrue(hasattr(scenario, "goal_thr_dist"))
        self.assertTrue(hasattr(scenario, "lava_thr_dist"))
        self.assertTrue(hasattr(scenario, "hole_thr_dist"))

    def test_get_closest_method(self) -> None:
        scenario = RoomsScenario(
            config=DEFAULT_ROOMS_SCENARIO_CONFIG,
            world_config=DEFAULT_WORLD_CONFIG,
        )

        pos = np.array([0.0, 0.0])
        objects = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]])
        min_dist, min_idx = scenario.get_closest(pos, objects)
        self.assertEqual(min_dist, 1.0)
        self.assertIn(min_idx, [0, 1])

        empty_objects = np.array([]).reshape(0, 2)
        min_dist, min_idx = scenario.get_closest(pos, empty_objects)
        self.assertEqual(min_dist, np.inf)
        self.assertEqual(min_idx, -1)

    def test_observation_contains_all_elements(self) -> None:
        scenario = RoomsScenario(
            config=DEFAULT_ROOMS_SCENARIO_CONFIG,
            world_config=DEFAULT_WORLD_CONFIG,
        )
        world = scenario.make_world()
        rng = np.random.default_rng(42)
        scenario._pre_reset_world(world, rng)
        scenario.reset_landmarks(world, rng)

        agent = world.agents[0]
        obs = scenario.observation(agent, world)

        expected_keys = {
            "agent_pos",
            "agent_vel",
            "goal_pos",
            "lava_pos",
            "hole_pos",
            "doorway_pos",
            "goal_dist",
            "lava_dist",
            "hole_dist",
            "wall_dist",
            "doorway_dist",
        }
        self.assertSetEqual(set(obs.keys()), expected_keys)
        self.assertEqual(obs["agent_pos"].shape, (2,))
        self.assertEqual(obs["goal_pos"].shape, (2,))

    def test_random_spawn(self) -> None:
        spawn_config = SpawnConfig(
            agent=(3.0, 3.0),
            goal=ObjConfig(pos=(9, 8), reward=1.0, absorbing=False),
            hole_size=0.4,
            lava_size=0.4,
            lavas=[
                ObjConfig(
                    pos=[(3, 5), (2, 4), (3, 4)],
                    reward=-1.0,
                    absorbing=False,
                ),
            ],
            holes=[
                ObjConfig(
                    pos=[(2, 5), (3, 5), (2, 4)],
                    reward=0.0,
                    absorbing=False,
                ),
            ],
            doorways={
                "ld": (2, 6),
                "td": (6, 9),
                "rd": (9, 5),
                "bd": (6, 2),
            },
        )
        config = RoomsScenarioConfig(spawn_config=spawn_config)
        env = RoomsEnv(scenario_config=config)
        self.addCleanup(env.close)
        env.reset()

        rendered = env.render()
        self.assertIsNotNone(rendered)
        self.assertEqual(rendered.ndim, 3)

    def test_path_gaussian_spawn_with_random_agent(self) -> None:
        spawn_config = SpawnConfig(
            agent=None,
            goal=ObjConfig(pos=(9, 8), reward=1.0, absorbing=False),
            lavas=[
                ObjConfig(
                    pos=None, reward=-1.0, absorbing=False, room="top_left"
                ),
                ObjConfig(
                    pos=None, reward=-1.0, absorbing=False, room="bottom_left"
                ),
                ObjConfig(
                    pos=None, reward=-1.0, absorbing=False, room="top_right"
                ),
                ObjConfig(
                    pos=None, reward=-1.0, absorbing=False, room="bottom_right"
                ),
            ],
            holes=[
                ObjConfig(
                    pos=None, reward=-1.0, absorbing=False, room="top_left"
                ),
                ObjConfig(
                    pos=None, reward=-1.0, absorbing=False, room="bottom_left"
                ),
                ObjConfig(
                    pos=None, reward=-1.0, absorbing=False, room="top_right"
                ),
                ObjConfig(
                    pos=None, reward=-1.0, absorbing=False, room="bottom_right"
                ),
            ],
            spawn_method=PathGaussianConfig(
                gaussian_std=0.6,
                min_spacing=1.0,
                edge_buffer=0.05,
                include_agent_paths=True,
            ),
            agent_size=0.1,
            goal_size=0.4,
            lava_size=0.4,
            hole_size=0.4,
        )
        config = RoomsScenarioConfig(spawn_config=spawn_config)
        env = RoomsEnv(scenario_config=config)
        self.addCleanup(env.close)
        env.reset()

        scenario = cast(RoomsScenario, env.scenario)
        self.assertLen(scenario.lavas, 4)
        self.assertLen(scenario.holes, 4)

        agent_pos_world = env.world.agents[0].state.pos
        lava_positions = scenario.lava_pos
        hole_positions = scenario.hole_pos

        min_agent_lava = spawn_config.agent_size + spawn_config.lava_size
        for i, lava_pos in enumerate(lava_positions):
            dist = float(np.linalg.norm(lava_pos - agent_pos_world))
            self.assertGreaterEqual(
                dist, min_agent_lava, msg=f"Lava {i} overlaps agent"
            )

        min_agent_hole = spawn_config.agent_size + spawn_config.hole_size
        for i, hole_pos in enumerate(hole_positions):
            dist = float(np.linalg.norm(hole_pos - agent_pos_world))
            self.assertGreaterEqual(
                dist, min_agent_hole, msg=f"Hole {i} overlaps agent"
            )

        topology = RoomTopology(config.spawn_config.doorways)
        for i, (lava_pos, lava_config) in enumerate(
            zip(lava_positions, config.spawn_config.lavas)
        ):
            if lava_config.room is not None:
                self.assertEqual(
                    topology.get_room(lava_pos),
                    lava_config.room,
                    msg=f"Lava {i} room mismatch",
                )

        for i, (hole_pos, hole_config) in enumerate(
            zip(hole_positions, config.spawn_config.holes)
        ):
            if hole_config.room is not None:
                self.assertEqual(
                    topology.get_room(hole_pos),
                    hole_config.room,
                    msg=f"Hole {i} room mismatch",
                )

    def test_path_gaussian_spawn_zero_std_on_segments(self) -> None:
        spawn_config = SpawnConfig(
            agent=(3.0, 3.0),
            goal=ObjConfig(pos=(9, 8), reward=1.0, absorbing=False),
            lavas=[
                ObjConfig(
                    pos=None, reward=-1.0, absorbing=False, room="top_left"
                ),
                ObjConfig(
                    pos=None, reward=-1.0, absorbing=False, room="bottom_left"
                ),
                ObjConfig(
                    pos=None, reward=-1.0, absorbing=False, room="top_right"
                ),
                ObjConfig(
                    pos=None, reward=-1.0, absorbing=False, room="bottom_right"
                ),
            ],
            holes=[
                ObjConfig(
                    pos=None, reward=-1.0, absorbing=False, room="top_left"
                ),
                ObjConfig(
                    pos=None, reward=-1.0, absorbing=False, room="bottom_left"
                ),
                ObjConfig(
                    pos=None, reward=-1.0, absorbing=False, room="top_right"
                ),
                ObjConfig(
                    pos=None, reward=-1.0, absorbing=False, room="bottom_right"
                ),
            ],
            spawn_method=PathGaussianConfig(
                gaussian_std=0.0,
                min_spacing=0.3,
                edge_buffer=0.05,
                include_agent_paths=True,
            ),
            agent_size=0.1,
            goal_size=0.4,
            lava_size=0.2,
            hole_size=0.2,
        )
        config = RoomsScenarioConfig(spawn_config=spawn_config)
        env = RoomsEnv(scenario_config=config)
        self.addCleanup(env.close)

        tolerance = 1e-6
        topology = RoomTopology(config.spawn_config.doorways)

        for seed in [42, 123]:
            env.reset(seed=seed)
            scenario = cast(RoomsScenario, env.scenario)
            agent_pos_world = env.world.agents[0].state.pos

            segments = get_relevant_path_segments(
                agent_pos_world,
                scenario.goal_pos,
                scenario.doorways,
                topology,
                env.world.grid,
            )

            for i, lava_pos in enumerate(scenario.lava_pos):
                room = config.spawn_config.lavas[i].room
                if room is not None:
                    clipped_segs = [
                        cs
                        for seg in segments
                        if (cs := _clip_segment_to_room(seg, room, topology))
                        is not None
                        and np.linalg.norm(cs.end - cs.start) > 1e-6
                    ]
                    dist = _min_distance_to_any_segment(lava_pos, clipped_segs)
                    self.assertLess(
                        dist,
                        tolerance,
                        msg=f"Seed {seed}: Lava {i} not on clipped segment",
                    )

            for i, hole_pos in enumerate(scenario.hole_pos):
                room = config.spawn_config.holes[i].room
                if room is not None:
                    clipped_segs = [
                        cs
                        for seg in segments
                        if (cs := _clip_segment_to_room(seg, room, topology))
                        is not None
                        and np.linalg.norm(cs.end - cs.start) > 1e-6
                    ]
                    dist = _min_distance_to_any_segment(hole_pos, clipped_segs)
                    self.assertLess(
                        dist,
                        tolerance,
                        msg=f"Seed {seed}: Hole {i} not on clipped segment",
                    )


if __name__ == "__main__":
    absltest.main()
