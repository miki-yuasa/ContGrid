from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar, cast

import numpy as np
from absl.testing import absltest

from contgrid.core import Grid, WorldConfig
from contgrid.envs.zone import (
    FixedRandomSwapSpawnConfig,
    FixedRandomSwapSpawnStrategy,
    FixedSpawnConfig,
    ObjConfig,
    RandomSwapSpec,
    RewardConfig,
    SpawnConfig,
    SubtaskConfig,
    ZoneEnv,
    ZoneScenario,
    ZoneScenarioConfig,
    ZoneSizeConfig,
    ZoneType,
)

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


def _create_zone_env(
    subtask_seq: Sequence[SubtaskConfig],
    agent_start: tuple[float, float] = (1.5, 1.5),
) -> ZoneEnv:
    spawn_config = SpawnConfig(
        agent=agent_start,
        subtask_seq=list(subtask_seq),
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
    return ZoneEnv(
        scenario_config=scenario_config, world_config=world_config
    )


class TestSubtaskConfig(absltest.TestCase):
    def test_goal_in_obstacles_raises(self) -> None:
        with self.assertRaises(ValueError):
            SubtaskConfig(goal=ZoneType.YELLOW, obstacle=ZoneType.YELLOW)


class TestZoneScenarioMultipleObstacles(absltest.TestCase):
    def test_absorbing_obstacle_terminates(self) -> None:
        subtask = SubtaskConfig(
            goal=ZoneType.YELLOW,
            obstacle=[ZoneType.WHITE, ZoneType.BLACK],
            reward=50.0,
            penalty=-10.0,
            goal_absorbing=False,
            obstacle_absorbing=True,
        )
        env = _create_zone_env([subtask])
        self.addCleanup(env.close)
        env.reset(seed=42)

        agent = env.world.agents[0]
        agent.state.pos = np.array([4.5, 1.5], dtype=np.float64)
        reward = env.scenario.reward(agent, env.world)

        self.assertAlmostEqual(reward, -10.0)
        self.assertTrue(agent.terminated)

    def test_non_absorbing_obstacle_and_goal_reach(self) -> None:
        subtask = SubtaskConfig(
            goal=ZoneType.YELLOW,
            obstacle=[ZoneType.WHITE],
            reward=50.0,
            penalty=-5.0,
            goal_absorbing=False,
            obstacle_absorbing=False,
        )
        env = _create_zone_env([subtask])
        self.addCleanup(env.close)
        env.reset(seed=42)

        agent = env.world.agents[0]
        agent.state.pos = np.array([4.5, 1.5], dtype=np.float64)
        pen_reward = env.scenario.reward(agent, env.world)
        self.assertAlmostEqual(pen_reward, -5.0 - 0.01)

        agent.state.pos = np.array([8.5, 1.5], dtype=np.float64)
        goal_reward = env.scenario.reward(agent, env.world)
        self.assertAlmostEqual(goal_reward, 50.0 - 0.01)
        self.assertTrue(cast(ZoneScenario, env.scenario).is_success)

    def test_info_contains_subtask(self) -> None:
        subtask = SubtaskConfig(
            goal=ZoneType.YELLOW,
            obstacle=[ZoneType.WHITE, ZoneType.BLACK],
        )
        env = _create_zone_env([subtask])
        self.addCleanup(env.close)
        _, info = env.reset(seed=42)

        self.assertIn("current_subtask", info)
        self.assertEqual(info["current_subtask"]["goal"], ZoneType.YELLOW)


class TestFixedRandomSwapSpawnStrategy(absltest.TestCase):
    YELLOW_POSITIONS: ClassVar[tuple[tuple[float, float], ...]] = (
        (1.5, 5.5),
        (9.5, 5.5),
        (5.5, 1.5),
        (5.5, 9.5),
    )

    def test_fixed_random_swap_spawn(self) -> None:
        spawn_config = SpawnConfig(
            agent=None,
            subtask_seq=[],
            yellow_zone=[
                ObjConfig(pos=(1.5, 5.5)),
                ObjConfig(pos=(9.5, 5.5)),
                ObjConfig(pos=(5.5, 1.5)),
                ObjConfig(pos=(5.5, 9.5)),
            ],
            red_zone=[],
            white_zone=[],
            black_zone=[],
            zone_size=ZoneSizeConfig(
                yellow=0.25, red=0.75, white=0.25, black=0.25
            ),
            agent_size=0.1,
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
        env = ZoneEnv(
            scenario_config=ZoneScenarioConfig(spawn_config=spawn_config),
            world_config=WorldConfig(grid=Grid(layout=_MAP_LAYOUT)),
        )
        self.addCleanup(env.close)
        env.reset(seed=42)

        scenario = cast(ZoneScenario, env.scenario)
        self.assertIsInstance(
            scenario.spawn_manager.spawn_strategy,
            FixedRandomSwapSpawnStrategy,
        )
        self.assertLen(scenario.white_pos, 1)
        white_pos = (
            float(scenario.white_pos[0][0]),
            float(scenario.white_pos[0][1]),
        )
        self.assertIn(white_pos, self.YELLOW_POSITIONS)


if __name__ == "__main__":
    absltest.main()
