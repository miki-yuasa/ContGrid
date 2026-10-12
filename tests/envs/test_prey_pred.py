from __future__ import annotations

import numpy as np
from absl.testing import absltest

from contgrid.envs.prey_pred import (
    AgentSpawnConfig,
    OrderedTaskConfig,
    PreyPredEnv,
    PreyPredScenarioConfig,
    PreyPredSubtaskConfig,
    UnorderedTaskConfig,
)


class TestPreyPredEnv(absltest.TestCase):
    def test_unordered_task_completion(self) -> None:
        task_cfg = UnorderedTaskConfig(
            required_preys=[0, 2],
            capture_rewards={0: 15.0, 2: 30.0},
            completion_reward=100.0,
            step_penalty=0.0,
        )
        env = PreyPredEnv(
            scenario_config=PreyPredScenarioConfig(task=task_cfg)
        )
        self.addCleanup(env.close)
        env.reset()

        agent = env.world.agents[0]
        agent.state.pos = env.scenario.preys[0].state.pos.copy()
        env.scenario.reward(agent, env.world)
        agent.state.pos = np.array([7.0, 7.0], dtype=np.float64)
        env.scenario.reward(agent, env.world)
        agent.state.pos = env.scenario.preys[2].state.pos.copy()
        reward = env.scenario.reward(agent, env.world)

        self.assertAlmostEqual(reward, 130.0)
        self.assertTrue(env.scenario.is_success)

    def test_unconfigured_prey_reward_raises_key_error(self) -> None:
        task_cfg = UnorderedTaskConfig(
            required_preys=[0, 1],
            capture_rewards={0: 10.0},
            completion_reward=50.0,
            step_penalty=0.0,
        )
        env = PreyPredEnv(
            scenario_config=PreyPredScenarioConfig(task=task_cfg)
        )
        self.addCleanup(env.close)
        env.reset()

        agent = env.world.agents[0]
        agent.state.pos = env.scenario.preys[1].state.pos.copy()
        with self.assertRaises(KeyError):
            env.scenario.reward(agent, env.world)

    def test_ordered_task_out_of_order(self) -> None:
        task_cfg = OrderedTaskConfig(
            subtask_seq=[
                PreyPredSubtaskConfig(target_prey=1, reward=20.0),
                PreyPredSubtaskConfig(target_prey=3, reward=40.0),
            ],
            completion_reward=50.0,
            step_penalty=0.0,
        )
        env = PreyPredEnv(
            scenario_config=PreyPredScenarioConfig(task=task_cfg)
        )
        self.addCleanup(env.close)
        env.reset()

        agent = env.world.agents[0]
        agent.state.pos = env.scenario.preys[3].state.pos.copy()
        reward = env.scenario.reward(agent, env.world)

        self.assertAlmostEqual(reward, 0.0)
        self.assertEqual(env.scenario.current_subtask_idx, 0)

    def test_ordered_task_progression_and_completion(self) -> None:
        task_cfg = OrderedTaskConfig(
            subtask_seq=[
                PreyPredSubtaskConfig(target_prey=1, reward=20.0),
                PreyPredSubtaskConfig(target_prey=3, reward=40.0),
            ],
            completion_reward=50.0,
            step_penalty=0.0,
        )
        env = PreyPredEnv(
            scenario_config=PreyPredScenarioConfig(task=task_cfg)
        )
        self.addCleanup(env.close)
        env.reset()

        agent = env.world.agents[0]
        agent.state.pos = env.scenario.preys[1].state.pos.copy()
        r1 = env.scenario.reward(agent, env.world)
        agent.state.pos = np.array([7.0, 7.0], dtype=np.float64)
        env.scenario.reward(agent, env.world)
        agent.state.pos = env.scenario.preys[3].state.pos.copy()
        r2 = env.scenario.reward(agent, env.world)

        self.assertAlmostEqual(r1, 20.0)
        self.assertAlmostEqual(r2, 90.0)
        self.assertTrue(env.scenario.is_success)

    def test_stationary_prey_remains_anchored(self) -> None:
        env = PreyPredEnv()
        self.addCleanup(env.close)
        env.reset(seed=42)

        initial_pos = env.scenario.preys[0].state.pos.copy()
        for _ in range(30):
            env.step(env.action_space.sample())

        np.testing.assert_allclose(
            env.scenario.preys[0].state.pos, initial_pos, atol=1e-5
        )

    def test_predator_absorbing_collision(self) -> None:
        env = PreyPredEnv()
        self.addCleanup(env.close)
        env.reset()

        agent = env.world.agents[0]
        agent.state.pos = env.scenario.predators[0].state.pos.copy()
        reward = env.scenario.reward(agent, env.world)

        self.assertLess(reward, 0.0)
        self.assertTrue(agent.terminated)

    def test_spawn_bounds_and_clearance(self) -> None:
        cfg = PreyPredScenarioConfig(
            agent_spawn=AgentSpawnConfig(mode="random", min_clearance=0.05)
        )
        env = PreyPredEnv(scenario_config=cfg)
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=42)

        ax, ay = obs["agent_pos"]
        self.assertTrue(0.6 <= ax <= 13.4 and 0.6 <= ay <= 13.4)

        min_dist = min(
            float(np.linalg.norm(pred.state.pos - prey.state.pos))
            for prey, pred in zip(
                env.scenario.preys, env.scenario.predators
            )
        )
        self.assertGreaterEqual(min_dist, 0.70 - 1e-4)


if __name__ == "__main__":
    absltest.main()
