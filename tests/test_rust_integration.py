from __future__ import annotations

import gymnasium as gym
import numpy as np
from absl.testing import absltest

import contgrid  # noqa: F401


class RustIntegrationTest(absltest.TestCase):
    def test_rust_integration(self) -> None:
        env = gym.make("contgrid/Rooms-v0", max_episode_steps=100)
        self.addCleanup(env.close)

        observation, _ = env.reset(seed=42)
        for _ in range(100):
            action = env.action_space.sample()
            observation, _, terminated, truncated, _ = env.step(action)
            if terminated or truncated:
                break

        self.assertIsInstance(observation, dict)
        self.assertIn("agent_pos", observation)
        self.assertIn("goal_pos", observation)

        agent_pos = observation["agent_pos"]
        self.assertIsInstance(agent_pos, np.ndarray)
        self.assertEqual(agent_pos.shape, (2,))


if __name__ == "__main__":
    absltest.main()
