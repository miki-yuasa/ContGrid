from __future__ import annotations

import math
import tempfile
from pathlib import Path

import gymnasium as gym
import numpy as np
from absl.testing import absltest, parameterized

import contgrid  # noqa: F401
from contgrid.envs.prey_pred import (
    AgentSpawnConfig,
    OrderedTaskConfig,
    PreyPredEnv,
    PreyPredScenarioConfig,
    PreyPredSubtaskConfig,
    TrajectoryRenderItem,
    UnorderedTaskConfig,
)


class TestPreyPredEnv(parameterized.TestCase):

    def test_gym_spec_and_action_space(self) -> None:
        env = gym.make("contgrid/PreyPred-v0")
        self.addCleanup(env.close)
        assert env.spec is not None
        self.assertEqual(env.spec.max_episode_steps, 300)
        self.assertEqual(env.action_space.shape, (2,))

    def test_gym_reset_observation_structure(self) -> None:
        env = gym.make("contgrid/PreyPred-v0")
        self.addCleanup(env.close)
        obs, info = env.reset()
        expected = {
            "agent_pos",
            "agent_vel",
            "wall_dist",
            "prey_rel_pos",
            "prey_rel_vel",
            "predator_rel_pos",
            "predator_rel_vel",
            "prey_capture_counts",
        }
        self.assertEqual(set(obs.keys()), expected)
        self.assertEqual(obs["prey_capture_counts"].shape, (4,))
        self.assertIn("is_success", info)

    @parameterized.parameters(42, 123)
    def test_random_agent_rollouts(self, seed: int) -> None:
        env = gym.make("contgrid/PreyPred-v0")
        self.addCleanup(env.close)
        obs, _ = env.reset(seed=seed)
        for _ in range(30):
            obs, reward, terminated, truncated, _ = env.step(
                env.action_space.sample()
            )
            self.assertFalse(math.isnan(float(reward)))
            self.assertTrue(env.observation_space.contains(obs))
            if terminated or truncated:
                break

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

    def test_region_clearance_preserved_during_rollout(self) -> None:
        env = PreyPredEnv()
        self.addCleanup(env.close)
        env.reset(seed=99)
        min_dist = float("inf")
        for _ in range(100):
            env.step(env.action_space.sample())
            for prey, pred in zip(
                env.scenario.preys, env.scenario.predators
            ):
                d = float(np.linalg.norm(pred.state.pos - prey.state.pos))
                min_dist = min(min_dist, d)
        self.assertGreaterEqual(min_dist, 0.70 - 1e-4)

    def test_stationary_prey_remains_anchored(self) -> None:
        env = PreyPredEnv()
        self.addCleanup(env.close)
        env.reset(seed=42)
        p0_initial = env.scenario.preys[0].state.pos.copy()
        for _ in range(50):
            env.step(env.action_space.sample())
        np.testing.assert_allclose(
            env.scenario.preys[0].state.pos, p0_initial, atol=1e-5
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
        self.assertGreater(env.scenario.predator_collisions, 0)

    def test_export_and_save_config(self) -> None:
        env = PreyPredEnv()
        self.addCleanup(env.close)
        env.reset()
        with tempfile.TemporaryDirectory() as tmp_dir:
            yaml_path = Path(tmp_dir) / "config.yaml"
            cfg_yaml = env.save_spawned_config(yaml_path, format="yaml")
            self.assertTrue(yaml_path.exists())
            self.assertEqual(len(cfg_yaml.regions), 4)

    def test_trajectory_render_items(self) -> None:
        env = PreyPredEnv()
        self.addCleanup(env.close)
        env.reset()
        items = env.scenario.trajectory_items
        self.assertEqual(len(items), 8)
        self.assertTrue(all(isinstance(item, TrajectoryRenderItem) for item in items))
        self.assertTrue(all(item.color.value.startswith("#") for item in items))

    def test_random_agent_spawn_mode_bounds_and_clearance(self) -> None:
        cfg = PreyPredScenarioConfig(
            agent_spawn=AgentSpawnConfig(mode="random", min_clearance=0.05)
        )
        env = PreyPredEnv(scenario_config=cfg)
        self.addCleanup(env.close)
        within_bounds = True
        no_overlaps = True
        for seed in range(20):
            obs, _ = env.reset(seed=seed)
            ax, ay = obs["agent_pos"]
            within_bounds = within_bounds and (0.6 <= ax <= 13.4 and 0.6 <= ay <= 13.4)
            entities = env.scenario.preys + env.scenario.predators
            for e in entities:
                d = math.hypot(ax - e.state.pos[0], ay - e.state.pos[1])
                if d < (0.1 + e.size + 0.05 - 1e-6):
                    no_overlaps = False
        self.assertTrue(within_bounds)
        self.assertTrue(no_overlaps)

    def test_fixed_agent_spawn_missing_pos_raises(self) -> None:
        cfg = PreyPredScenarioConfig(
            agent_spawn=AgentSpawnConfig(mode="fixed", fixed_pos=None)
        )
        with self.assertRaises(ValueError):
            PreyPredEnv(scenario_config=cfg)


if __name__ == "__main__":
    absltest.main()

