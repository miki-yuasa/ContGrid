from __future__ import annotations

import warnings
from typing import ClassVar, cast

import numpy as np
from absl.testing import absltest, parameterized

from contgrid.core import Grid, WorldConfig
from contgrid.envs.zone import (
    FixedRandomSwapSpawnConfig,
    FixedRandomSwapSpawnStrategy,
    FixedSpawnConfig,
    FixedSpawnStrategy,
    GaussianSpawnConfig,
    GaussianSpawnStrategy,
    ObjConfig,
    RandomSwapSpec,
    RewardConfig,
    SpawnConfig,
    SubtaskConfig,
    UniformRandomConfig,
    UniformRandomSpawnStrategy,
    ZoneEnv,
    ZoneScenario,
    ZoneScenarioConfig,
    ZoneSizeConfig,
    ZoneType,
)
from contgrid.envs.zone.configs import ObsConfig

_MAP_LAYOUT = [
    "############",
    "#          #",
    "#          #",
    "#          #",
    "#          #",
    "#          #",
    "#          #",
    "#          #",
    "#          #",
    "#          #",
    "#          #",
    "############",
]


class TestUniformRandomSpawnStrategy(absltest.TestCase):
    def test_uniform_random_spawn(self) -> None:
        min_spacing = 1.5
        num_zones = 6
        zone_size = 0.5
        agent_size = 0.1

        spawn_config = SpawnConfig(
            agent=None,
            subtask_seq=[],
            yellow_zone=[ObjConfig(pos=None) for _ in range(num_zones)],
            red_zone=[ObjConfig(pos=None) for _ in range(num_zones)],
            white_zone=[ObjConfig(pos=None) for _ in range(num_zones)],
            black_zone=[ObjConfig(pos=None) for _ in range(num_zones)],
            zone_size=zone_size,
            agent_size=agent_size,
            spawn_method=UniformRandomConfig(min_spacing=min_spacing),
        )

        scenario_config = ZoneScenarioConfig(spawn_config=spawn_config)
        world_config = WorldConfig(grid=Grid(layout=_MAP_LAYOUT))

        env = ZoneEnv(
            scenario_config=scenario_config, world_config=world_config
        )
        self.addCleanup(env.close)

        for seed in (7, 17, 27, 37):
            env.reset(seed=seed)
            rendered = env.render()
            self.assertIsNotNone(rendered)

            scenario = cast(ZoneScenario, env.scenario)
            strategy = scenario.spawn_manager.spawn_strategy
            self.assertIsInstance(strategy, UniformRandomSpawnStrategy)

            zone_positions_by_color = {
                "yellow": scenario.yellow_pos,
                "red": scenario.red_pos,
                "white": scenario.white_pos,
                "black": scenario.black_pos,
            }

            all_zone_positions: list[tuple[str, int, np.ndarray]] = []
            for color, color_positions in zone_positions_by_color.items():
                self.assertLen(color_positions, num_zones)
                for idx, pos in enumerate(color_positions):
                    all_zone_positions.append((color, idx, pos))

            agent_pos = env.world.agents[0].state.pos.copy()
            min_agent_distance = zone_size + agent_size
            for color, idx, pos in all_zone_positions:
                candidate = (float(pos[0]), float(pos[1]))
                self.assertTrue(
                    env.world.wall_collision_checker.is_position_valid(
                        zone_size,
                        env.world.contact_margin,
                        candidate,
                    ),
                    msg=f"{color} zone {idx} in invalid location: {candidate}",
                )

                dist_to_agent = float(np.linalg.norm(pos - agent_pos))
                self.assertGreaterEqual(
                    dist_to_agent,
                    min_agent_distance - 1e-9,
                    msg=f"{color} zone {idx} too close to agent",
                )

            for i in range(len(all_zone_positions)):
                color_i, idx_i, pos_i = all_zone_positions[i]
                for j in range(i + 1, len(all_zone_positions)):
                    color_j, idx_j, pos_j = all_zone_positions[j]
                    dist = float(np.linalg.norm(pos_i - pos_j))
                    self.assertGreaterEqual(
                        dist,
                        min_spacing - 1e-9,
                        msg=f"{color_i}[{idx_i}] and {color_j}[{idx_j}] too close",
                    )


class TestGaussianSpawnStrategy(absltest.TestCase):
    def test_gaussian_spawn(self) -> None:
        num_yellow_zones = 2
        num_red_zones = 8
        num_white_zones = 2
        num_black_zones = 2
        zone_size = 0.5
        agent_size = 0.1
        min_spacing = 1.1
        gaussian_std = 2

        spawn_config = SpawnConfig(
            agent=None,
            subtask_seq=[],
            yellow_zone=[ObjConfig(pos=None) for _ in range(num_yellow_zones)],
            red_zone=[ObjConfig(pos=None) for _ in range(num_red_zones)],
            white_zone=[ObjConfig(pos=None) for _ in range(num_white_zones)],
            black_zone=[ObjConfig(pos=None) for _ in range(num_black_zones)],
            zone_size=zone_size,
            agent_size=agent_size,
            spawn_method=GaussianSpawnConfig(
                gaussian_std=gaussian_std, min_spacing=min_spacing
            ),
        )

        scenario_config = ZoneScenarioConfig(spawn_config=spawn_config)
        world_config = WorldConfig(grid=Grid(layout=_MAP_LAYOUT))

        env = ZoneEnv(
            scenario_config=scenario_config, world_config=world_config
        )
        self.addCleanup(env.close)

        for seed in (7, 17, 27, 37):
            env.reset(seed=seed)
            rendered = env.render()
            self.assertIsNotNone(rendered)

            scenario = cast(ZoneScenario, env.scenario)
            strategy = scenario.spawn_manager.spawn_strategy
            self.assertIsInstance(strategy, GaussianSpawnStrategy)

            zone_positions_by_color = {
                "yellow": scenario.yellow_pos,
                "red": scenario.red_pos,
                "white": scenario.white_pos,
                "black": scenario.black_pos,
            }
            num_zones_list = [
                num_yellow_zones,
                num_red_zones,
                num_white_zones,
                num_black_zones,
            ]

            all_zone_positions: list[np.ndarray] = []
            for color, num_zones in zip(
                zone_positions_by_color.keys(), num_zones_list
            ):
                color_positions = zone_positions_by_color[color]
                self.assertLen(
                    color_positions,
                    num_zones,
                    msg=f"{color} has {len(color_positions)} zones",
                )
                all_zone_positions.extend(color_positions)

            agent_pos = env.world.agents[0].state.pos.copy()
            min_agent_distance = zone_size + agent_size
            center = np.array([6.0, 6.0], dtype=np.float64)
            distances_to_center: list[float] = []

            for pos in all_zone_positions:
                candidate = (float(pos[0]), float(pos[1]))
                self.assertTrue(
                    env.world.wall_collision_checker.is_position_valid(
                        zone_size,
                        env.world.contact_margin,
                        candidate,
                    )
                )

                dist_to_agent = float(np.linalg.norm(pos - agent_pos))
                self.assertGreaterEqual(
                    dist_to_agent, min_agent_distance - 1e-9
                )
                distances_to_center.append(float(np.linalg.norm(pos - center)))

            self.assertLess(float(np.mean(distances_to_center)), 4.0)

            for i in range(len(all_zone_positions)):
                for j in range(i + 1, len(all_zone_positions)):
                    dist = float(
                        np.linalg.norm(
                            all_zone_positions[i] - all_zone_positions[j]
                        )
                    )
                    self.assertGreaterEqual(
                        dist,
                        min_spacing - 1e-6,
                        msg=f"Zones {i} and {j} violate min_spacing",
                    )

    def test_random_action(self) -> None:
        min_spacing = 1.5
        num_zones = 2
        zone_size = 0.5
        agent_size = 0.1

        spawn_config = SpawnConfig(
            agent=None,
            subtask_seq=[],
            yellow_zone=[ObjConfig(pos=None) for _ in range(num_zones)],
            red_zone=[ObjConfig(pos=None) for _ in range(num_zones)],
            white_zone=[ObjConfig(pos=None) for _ in range(num_zones)],
            black_zone=[ObjConfig(pos=None) for _ in range(num_zones)],
            zone_size=zone_size,
            agent_size=agent_size,
            spawn_method=UniformRandomConfig(min_spacing=min_spacing),
        )

        scenario_config = ZoneScenarioConfig(spawn_config=spawn_config)
        world_config = WorldConfig(grid=Grid(layout=_MAP_LAYOUT))

        env = ZoneEnv(
            scenario_config=scenario_config, world_config=world_config
        )
        self.addCleanup(env.close)

        for seed in (7, 17, 37):
            env.reset(seed=seed)
            for _ in range(20):
                action = env.action_space.sample()
                obs, reward, terminated, truncated, _ = env.step(action)
                self.assertIsNotNone(obs)
                self.assertIsInstance(reward, (int, float))
                if terminated or truncated:
                    break

            rendered = env.render()
            self.assertIsNotNone(rendered)
            self.assertEqual(rendered.ndim, 3)

    def test_no_observation_warnings(self) -> None:
        spawn_config = SpawnConfig(
            agent=None,
            subtask_seq=[
                SubtaskConfig(
                    goal=ZoneType.YELLOW,
                    obstacle=ZoneType.WHITE,
                    reward=50.0,
                    penalty=-1.0,
                )
            ],
            yellow_zone=[ObjConfig(pos=None)],
            red_zone=[ObjConfig(pos=None)],
            white_zone=[ObjConfig(pos=None)],
            black_zone=[ObjConfig(pos=None)],
            zone_size=0.5,
            agent_size=0.1,
            spawn_method=UniformRandomConfig(min_spacing=1.5),
        )

        scenario_config = ZoneScenarioConfig(
            spawn_config=spawn_config,
            obs_config=ObsConfig(include_subtask=True),
        )
        world_config = WorldConfig(grid=Grid(layout=_MAP_LAYOUT))

        env = ZoneEnv(
            scenario_config=scenario_config, world_config=world_config
        )
        self.addCleanup(env.close)

        with warnings.catch_warnings(record=True) as captured_warnings:
            warnings.simplefilter("always")
            env.reset(seed=42)
            for _ in range(5):
                action = env.action_space.sample()
                env.step(action)

            for w in captured_warnings:
                self.assertNotIn(
                    "should be an int or np.int64, actual type: <class"
                    " 'numpy.ndarray'>",
                    str(w.message),
                )


class TestFixedSpawnStrategy(absltest.TestCase):
    def test_fixed_spawn(self) -> None:
        min_spacing = 1.5
        num_zones_yellow = 4
        num_zones_red = 5
        num_zones_white = 0
        num_zones_black = 0
        agent_size = 0.1

        spawn_config = SpawnConfig(
            agent=None,
            subtask_seq=[],
            yellow_zone=[
                ObjConfig(pos=(1.5, 5.5)),
                ObjConfig(pos=(9.5, 5.5)),
                ObjConfig(pos=(5.5, 1.5)),
                ObjConfig(pos=(5.5, 9.5)),
            ],
            red_zone=[
                ObjConfig(pos=(5.5, 5.5)),
                ObjConfig(pos=(3.5, 3.5)),
                ObjConfig(pos=(7.5, 3.5)),
                ObjConfig(pos=(7.5, 7.5)),
                ObjConfig(pos=(3.5, 7.5)),
            ],
            white_zone=[],
            black_zone=[],
            zone_size=ZoneSizeConfig(
                yellow=0.25,
                red=0.75,
                white=0.5,
                black=0.5,
            ),
            agent_size=agent_size,
            spawn_method=FixedSpawnConfig(),
            reset_agent_first=False,
        )

        scenario_config = ZoneScenarioConfig(spawn_config=spawn_config)
        world_config = WorldConfig(grid=Grid(layout=_MAP_LAYOUT))

        env = ZoneEnv(
            scenario_config=scenario_config, world_config=world_config
        )
        self.addCleanup(env.close)

        for seed in (7, 17, 27, 37):
            env.reset(seed=seed)
            rendered = env.render()
            self.assertIsNotNone(rendered)

            scenario = cast(ZoneScenario, env.scenario)
            strategy = scenario.spawn_manager.spawn_strategy
            self.assertIsInstance(strategy, FixedSpawnStrategy)

            zone_positions_by_color = {
                "yellow": scenario.yellow_pos,
                "red": scenario.red_pos,
                "white": scenario.white_pos,
                "black": scenario.black_pos,
            }

            all_zone_positions: list[tuple[str, int, np.ndarray]] = []
            for color, color_positions in zone_positions_by_color.items():
                if color == "yellow":
                    self.assertLen(color_positions, num_zones_yellow)
                elif color == "red":
                    self.assertLen(color_positions, num_zones_red)
                elif color == "white":
                    self.assertLen(color_positions, num_zones_white)
                elif color == "black":
                    self.assertLen(color_positions, num_zones_black)
                for idx, pos in enumerate(color_positions):
                    all_zone_positions.append((color, idx, pos))

            agent_pos = env.world.agents[0].state.pos.copy()
            for color, idx, pos in all_zone_positions:
                if color == "yellow":
                    current_zone_size = scenario.zone_sizes.yellow
                elif color == "red":
                    current_zone_size = scenario.zone_sizes.red
                elif color == "white":
                    current_zone_size = scenario.zone_sizes.white
                else:
                    current_zone_size = scenario.zone_sizes.black

                min_agent_distance = current_zone_size + agent_size
                candidate = (float(pos[0]), float(pos[1]))
                self.assertTrue(
                    env.world.wall_collision_checker.is_position_valid(
                        current_zone_size,
                        env.world.contact_margin,
                        candidate,
                    ),
                    msg=f"{color} zone {idx} in invalid location: {candidate}",
                )

                dist_to_agent = float(np.linalg.norm(pos - agent_pos))
                self.assertGreaterEqual(
                    dist_to_agent, min_agent_distance - 1e-9
                )

            for i in range(len(all_zone_positions)):
                color_i, idx_i, pos_i = all_zone_positions[i]
                for j in range(i + 1, len(all_zone_positions)):
                    color_j, idx_j, pos_j = all_zone_positions[j]
                    dist = float(np.linalg.norm(pos_i - pos_j))
                    self.assertGreaterEqual(
                        dist,
                        min_spacing - 1e-9,
                        msg=f"{color_i}[{idx_i}] and {color_j}[{idx_j}] too close",
                    )

    def test_fixed_spawn_with_agent_overlap(self) -> None:
        spawn_config = SpawnConfig(
            agent=(1.5, 5.5),
            subtask_seq=[],
            yellow_zone=[ObjConfig(pos=(1.5, 5.5))],
            red_zone=[],
            white_zone=[],
            black_zone=[],
            zone_size=0.5,
            agent_size=0.1,
            spawn_method=FixedSpawnConfig(),
            reset_agent_first=True,
        )

        scenario_config = ZoneScenarioConfig(spawn_config=spawn_config)
        world_config = WorldConfig(grid=Grid(layout=_MAP_LAYOUT))

        env = ZoneEnv(
            scenario_config=scenario_config, world_config=world_config
        )
        try:
            env.reset(seed=42)
            scenario = cast(ZoneScenario, env.scenario)

            self.assertLen(scenario.yellow_pos, 1)
            np.testing.assert_allclose(
                scenario.yellow_pos[0], [1.5, 5.5], atol=1e-5
            )

            agent_pos = env.world.agents[0].state.pos
            np.testing.assert_allclose(agent_pos, [1.5, 5.5], atol=1e-5)

            spawn_config.reset_agent_first = False
            env2 = ZoneEnv(
                scenario_config=scenario_config, world_config=world_config
            )
            try:
                env2.reset(seed=42)
                scenario2 = cast(ZoneScenario, env2.scenario)
                self.assertLen(scenario2.yellow_pos, 1)
                np.testing.assert_allclose(
                    scenario2.yellow_pos[0], [1.5, 5.5], atol=1e-5
                )
                np.testing.assert_allclose(
                    env2.world.agents[0].state.pos, [1.5, 5.5], atol=1e-5
                )
            finally:
                env2.close()
        finally:
            env.close()


class TestFixedRandomSwapSpawnStrategy(absltest.TestCase):
    YELLOW_POSITIONS: ClassVar[tuple[tuple[float, float], ...]] = (
        (1.5, 5.5),
        (9.5, 5.5),
        (5.5, 1.5),
        (5.5, 9.5),
    )

    def test_fixed_random_swap_spawn(self) -> None:
        agent_size = 0.1

        spawn_config = SpawnConfig(
            agent=None,
            subtask_seq=[],
            yellow_zone=[
                ObjConfig(pos=(1.5, 5.5)),
                ObjConfig(pos=(9.5, 5.5)),
                ObjConfig(pos=(5.5, 1.5)),
                ObjConfig(pos=(5.5, 9.5)),
            ],
            red_zone=[
                ObjConfig(pos=(5.5, 5.5)),
                ObjConfig(pos=(3.5, 3.5)),
                ObjConfig(pos=(7.5, 3.5)),
                ObjConfig(pos=(7.5, 7.5)),
                ObjConfig(pos=(3.5, 7.5)),
            ],
            white_zone=[],
            black_zone=[ObjConfig(pos=None), ObjConfig(pos=None)],
            zone_size=ZoneSizeConfig(
                yellow=0.25,
                red=0.75,
                white=0.25,
                black=0.25,
            ),
            agent_size=agent_size,
            spawn_method=FixedRandomSwapSpawnConfig(
                swaps=[
                    RandomSwapSpec(
                        source_zone="yellow",
                        target_zone="white",
                        num_swaps=1,
                    ),
                ],
            ),
            reset_agent_first=False,
        )

        scenario_config = ZoneScenarioConfig(spawn_config=spawn_config)
        world_config = WorldConfig(grid=Grid(layout=_MAP_LAYOUT))

        env = ZoneEnv(
            scenario_config=scenario_config, world_config=world_config
        )
        self.addCleanup(env.close)

        for seed in (7, 17, 27, 37):
            env.reset(seed=seed)
            rendered = env.render()
            self.assertIsNotNone(rendered)

            scenario = cast(ZoneScenario, env.scenario)
            strategy = scenario.spawn_manager.spawn_strategy
            self.assertIsInstance(strategy, FixedRandomSwapSpawnStrategy)

            self.assertLen(scenario.white_pos, 1)
            white_pos_tuple = (
                float(scenario.white_pos[0][0]),
                float(scenario.white_pos[0][1]),
            )
            self.assertIn(white_pos_tuple, self.YELLOW_POSITIONS)

            self.assertLen(scenario.red_pos, 5)
            self.assertLen(scenario.yellow_pos, 3)
            self.assertLen(scenario.white_pos, 1)
            self.assertLen(scenario.black_pos, 2)

            agent_pos = env.world.agents[0].state.pos.copy()
            for lm in (
                scenario.yellow + scenario.red + scenario.white + scenario.black
            ):
                dist_to_agent = float(np.linalg.norm(lm.state.pos - agent_pos))
                min_agent_distance = lm.size + agent_size
                self.assertGreaterEqual(
                    dist_to_agent,
                    min_agent_distance - 1e-9,
                    msg=f"{lm.name} too close to agent",
                )

            expected_total = 3 + 5 + 1 + 2
            self.assertEqual(
                len(scenario.yellow)
                + len(scenario.red)
                + len(scenario.white)
                + len(scenario.black),
                expected_total,
            )


class TestAgentSpawningPerturbation(absltest.TestCase):
    def _make_config(
        self, agent_perturbation: float
    ) -> tuple[ZoneScenarioConfig, WorldConfig]:
        spawn_config = SpawnConfig(
            agent=None,
            subtask_seq=[],
            yellow_zone=[ObjConfig(pos=None) for _ in range(2)],
            red_zone=[ObjConfig(pos=None) for _ in range(2)],
            white_zone=[ObjConfig(pos=None) for _ in range(2)],
            black_zone=[ObjConfig(pos=None) for _ in range(2)],
            zone_size=0.5,
            agent_size=0.1,
            agent_perturbation=agent_perturbation,
            spawn_method=UniformRandomConfig(min_spacing=1.5),
            reset_agent_first=False,
        )
        return (
            ZoneScenarioConfig(spawn_config=spawn_config),
            WorldConfig(grid=Grid(layout=_MAP_LAYOUT)),
        )

    def test_agent_spawning_with_perturbation(self) -> None:
        sc_config, w_config = self._make_config(agent_perturbation=0.25)
        env = ZoneEnv(scenario_config=sc_config, world_config=w_config)
        self.addCleanup(env.close)

        perturbed_count = 0
        for seed in range(5):
            env.reset(seed=seed)
            agent_pos = env.world.agents[0].state.pos.copy()

            self.assertTrue(
                env.world.wall_collision_checker.is_position_valid(
                    0.1,
                    env.world.contact_margin,
                    (float(agent_pos[0]), float(agent_pos[1])),
                )
            )

            scenario = cast(ZoneScenario, env.scenario)
            for lm in (
                scenario.yellow + scenario.red + scenario.white + scenario.black
            ):
                dist_to_agent = float(np.linalg.norm(lm.state.pos - agent_pos))
                min_agent_distance = lm.size + 0.1
                self.assertGreaterEqual(
                    dist_to_agent, min_agent_distance - 1e-9
                )

            frac_x = abs(agent_pos[0] - round(agent_pos[0]))
            frac_y = abs(agent_pos[1] - round(agent_pos[1]))
            if frac_x > 1e-6 or frac_y > 1e-6:
                perturbed_count += 1

        self.assertGreater(perturbed_count, 0)

    def test_agent_spawning_without_perturbation(self) -> None:
        sc_config, w_config = self._make_config(agent_perturbation=0.0)
        env = ZoneEnv(scenario_config=sc_config, world_config=w_config)
        self.addCleanup(env.close)

        for seed in range(5):
            env.reset(seed=seed)
            agent_pos = env.world.agents[0].state.pos.copy()

            frac_x = abs(agent_pos[0] - round(agent_pos[0]))
            frac_y = abs(agent_pos[1] - round(agent_pos[1]))
            self.assertLess(frac_x, 1e-6)
            self.assertLess(frac_y, 1e-6)


class TestSubtaskConfig(parameterized.TestCase):
    def test_single_obstacle(self) -> None:
        cfg = SubtaskConfig(goal=ZoneType.YELLOW, obstacle=ZoneType.WHITE)
        self.assertEqual(cfg.obstacle, [ZoneType.WHITE])

    def test_single_obstacle_str(self) -> None:
        cfg = SubtaskConfig(goal="yellow", obstacle="white")  # type: ignore[arg-type]
        self.assertEqual(cfg.obstacle, [ZoneType.WHITE])

    def test_multiple_obstacles(self) -> None:
        cfg = SubtaskConfig(
            goal=ZoneType.YELLOW,
            obstacle=[ZoneType.WHITE, ZoneType.BLACK],
        )
        self.assertEqual(cfg.obstacle, [ZoneType.WHITE, ZoneType.BLACK])

    def test_multiple_obstacles_str(self) -> None:
        cfg = SubtaskConfig(
            goal="yellow",  # type: ignore[arg-type]
            obstacle=["white", "black"],  # type: ignore[arg-type]
        )
        self.assertEqual(cfg.obstacle, [ZoneType.WHITE, ZoneType.BLACK])

    def test_no_obstacle(self) -> None:
        cfg_none = SubtaskConfig(goal=ZoneType.YELLOW)
        self.assertEqual(cfg_none.obstacle, [])

        cfg_explicit_none = SubtaskConfig(
            goal=ZoneType.YELLOW,
            obstacle=None,  # ty: ignore[invalid-argument-type]
        )
        self.assertEqual(cfg_explicit_none.obstacle, [])

        cfg_empty = SubtaskConfig(goal=ZoneType.YELLOW, obstacle=[])
        self.assertEqual(cfg_empty.obstacle, [])

    def test_goal_in_obstacles_raises(self) -> None:
        with self.assertRaises(ValueError):
            SubtaskConfig(goal=ZoneType.YELLOW, obstacle=ZoneType.YELLOW)

        with self.assertRaises(ValueError):
            SubtaskConfig(
                goal=ZoneType.YELLOW,
                obstacle=[ZoneType.WHITE, ZoneType.YELLOW],
            )

    def test_duplicate_obstacles_deduplicated(self) -> None:
        cfg = SubtaskConfig(
            goal=ZoneType.YELLOW,
            obstacle=[ZoneType.WHITE, ZoneType.WHITE],
        )
        self.assertEqual(cfg.obstacle, [ZoneType.WHITE])


class TestZoneScenarioMultipleObstacles(absltest.TestCase):
    def _create_env(
        self,
        subtask_seq: list[SubtaskConfig],
        agent_start: tuple[float, float] = (1.5, 1.5),
    ) -> ZoneEnv:
        spawn_config = SpawnConfig(
            agent=agent_start,
            subtask_seq=subtask_seq,
            yellow_zone=[ObjConfig(pos=(8.5, 1.5))],
            white_zone=[ObjConfig(pos=(4.5, 1.5))],
            black_zone=[ObjConfig(pos=(6.5, 1.5))],
            red_zone=[ObjConfig(pos=(1.5, 8.5))],
            zone_size=0.5,
            agent_size=0.1,
            spawn_method=FixedSpawnConfig(),
        )
        scenario_config = ZoneScenarioConfig(
            spawn_config=spawn_config,
            reward_config=RewardConfig(step_penalty=0.01),
        )
        world_config = WorldConfig(grid=Grid(layout=_MAP_LAYOUT))
        env = ZoneEnv(
            scenario_config=scenario_config, world_config=world_config
        )
        self.addCleanup(env.close)
        return env

    def test_multiple_obstacles_absorbing_first_obstacle(self) -> None:
        subtask = SubtaskConfig(
            goal=ZoneType.YELLOW,
            obstacle=[ZoneType.WHITE, ZoneType.BLACK],
            reward=50.0,
            penalty=-10.0,
            goal_absorbing=False,
            obstacle_absorbing=True,
        )
        env = self._create_env([subtask])
        env.reset(seed=42)

        agent = env.world.agents[0]
        # Move agent into White zone (first obstacle)
        agent.state.pos = np.array([4.5, 1.5], dtype=np.float64)
        reward = env.scenario.reward(agent, env.world)

        self.assertAlmostEqual(reward, -10.0)
        self.assertTrue(agent.terminated)

    def test_multiple_obstacles_absorbing_second_obstacle(self) -> None:
        subtask = SubtaskConfig(
            goal=ZoneType.YELLOW,
            obstacle=[ZoneType.WHITE, ZoneType.BLACK],
            reward=50.0,
            penalty=-10.0,
            goal_absorbing=False,
            obstacle_absorbing=True,
        )
        env = self._create_env([subtask])
        env.reset(seed=42)

        agent = env.world.agents[0]
        # Move agent into Black zone (second obstacle)
        agent.state.pos = np.array([6.5, 1.5], dtype=np.float64)
        reward = env.scenario.reward(agent, env.world)

        self.assertAlmostEqual(reward, -10.0)
        self.assertTrue(agent.terminated)

    def test_multiple_obstacles_non_absorbing_and_goal_reach(self) -> None:
        subtask = SubtaskConfig(
            goal=ZoneType.YELLOW,
            obstacle=[ZoneType.WHITE, ZoneType.BLACK],
            reward=50.0,
            penalty=-5.0,
            goal_absorbing=False,
            obstacle_absorbing=False,
        )
        env = self._create_env([subtask])
        env.reset(seed=42)

        agent = env.world.agents[0]

        # Enter White zone (obstacle 1)
        agent.state.pos = np.array([4.5, 1.5], dtype=np.float64)
        reward1 = env.scenario.reward(agent, env.world)
        self.assertAlmostEqual(reward1, -5.0 - 0.01)
        self.assertFalse(agent.terminated)

        # Move to neutral location
        agent.state.pos = np.array([5.5, 1.5], dtype=np.float64)
        reward2 = env.scenario.reward(agent, env.world)
        self.assertAlmostEqual(reward2, -0.01)
        self.assertFalse(agent.terminated)

        # Enter Black zone (obstacle 2)
        agent.state.pos = np.array([6.5, 1.5], dtype=np.float64)
        reward3 = env.scenario.reward(agent, env.world)
        self.assertAlmostEqual(reward3, -5.0 - 0.01)
        self.assertFalse(agent.terminated)

        # Move into Yellow zone (goal)
        agent.state.pos = np.array([8.5, 1.5], dtype=np.float64)
        reward4 = env.scenario.reward(agent, env.world)
        self.assertAlmostEqual(reward4, 50.0 - 0.01)
        scenario = cast(ZoneScenario, env.scenario)
        self.assertTrue(scenario.is_success)

    def test_info_contains_obstacle_list(self) -> None:
        subtask = SubtaskConfig(
            goal=ZoneType.YELLOW,
            obstacle=[ZoneType.WHITE, ZoneType.BLACK],
        )
        env = self._create_env([subtask])
        _, info = env.reset(seed=42)

        self.assertIn("current_subtask", info)
        self.assertEqual(info["current_subtask"]["idx"], 0)
        self.assertEqual(info["current_subtask"]["goal"], ZoneType.YELLOW)
        self.assertEqual(
            info["current_subtask"]["obstacle"],
            [ZoneType.WHITE, ZoneType.BLACK],
        )


if __name__ == "__main__":
    absltest.main()
