from __future__ import annotations

import gymnasium as gym
import numpy as np
from absl.testing import absltest, parameterized
from gymnasium import spaces
from gymnasium.utils.env_checker import check_env
from pydantic import BaseModel

import contgrid  # noqa: F401
from contgrid.contgrid import BaseEnv
from contgrid.core import Agent, AgentState, World
from contgrid.core.const import Color
from contgrid.core.entities import EntityState, Landmark
from contgrid.core.scenario import BaseScenario


class DummyScenarioConfig(BaseModel):
    pass


class SimpleScenario(BaseScenario[DummyScenarioConfig, np.ndarray]):
    def __init__(self) -> None:
        super().__init__(config=DummyScenarioConfig())

    def init_agents(
        self, world: World, np_random: np.random.Generator | None = None
    ) -> list[Agent]:
        agent = Agent(
            name="agent_0",
            size=0.25,
            color=Color.SKY_BLUE.name,
            state=AgentState(
                pos=np.array([10, 0.19], dtype=np.float64),
                vel=np.array([0.0, 0.0], dtype=np.float64),
            ),
        )
        return [agent]

    def init_landmarks(
        self, world: World, np_random: np.random.Generator | None = None
    ) -> list[Landmark]:
        landmark = Landmark(
            name="landmark_0",
            size=0.5,
            color=Color.GREEN.name,
            state=EntityState(
                pos=np.array([2, 3], dtype=np.float64),
                vel=np.array([0.0, 0.0], dtype=np.float64),
            ),
        )
        return [landmark]

    def reset_agents(
        self, world: World, np_random: np.random.Generator
    ) -> list[Agent]:
        for agent in world.agents:
            agent.state.pos = np.array([1, 1], dtype=np.float64)
            agent.state.vel = np.array([0.0, 0.0], dtype=np.float64)
        return world.agents

    def reset_landmarks(
        self, world: World, np_random: np.random.Generator
    ) -> list[Landmark]:
        for landmark in world.landmarks:
            landmark.state.pos = np.array([2, 3], dtype=np.float64)
            landmark.state.vel = np.array([0.0, 0.0], dtype=np.float64)
        return world.landmarks

    def observation(self, agent: Agent, world: World) -> np.ndarray:
        obs: list[float] = []
        obs.extend(agent.state.pos)
        if world.landmarks:
            obs.extend(world.landmarks[0].state.pos)
        else:
            obs.extend([0.0, 0.0])
        return np.array(obs, dtype=np.float64)

    def observation_space(self, agent: Agent, world: World) -> spaces.Space:
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32
        )


class TestEnvironmentRendering(absltest.TestCase):
    def test_render_rgb_array(self) -> None:
        scenario = SimpleScenario()
        env = BaseEnv(scenario=scenario, max_cycles=10)
        self.addCleanup(env.close)
        env.reset(seed=42)

        rendered_image = env.render()
        self.assertIsInstance(rendered_image, np.ndarray)
        assert rendered_image is not None
        self.assertEqual(rendered_image.shape[2], 3)


class TestGymApiConformance(parameterized.TestCase):
    @parameterized.parameters(
        "contgrid/Rooms-v0",
        "contgrid/NineRooms-v0",
        "contgrid/PreyPred-v0",
        "contgrid/Zone-v0",
    )
    def test_gym_check_env(self, env_id: str) -> None:
        env = gym.make(env_id)
        self.addCleanup(env.close)
        check_env(env.unwrapped)


class TestAllPossibleStates(absltest.TestCase):
    def test_all_possible_states_matches_free_cells(self) -> None:
        scenario = SimpleScenario()
        env = BaseEnv(scenario=scenario, max_cycles=10)
        env.reset(seed=42)

        states = env.all_possible_states()
        layout = env.grid.layout
        expected_free_cells = sum(cell != "#" for row in layout for cell in row)

        for agent_name in env.possible_agents:
            self.assertEqual(len(states[agent_name]), expected_free_cells)

        env.close()

    def test_all_possible_states_at_resolution_one_matches_legacy(self) -> None:
        scenario = SimpleScenario()
        env = BaseEnv(scenario=scenario, max_cycles=10)
        env.reset(seed=42)

        legacy_states = env.all_possible_states()
        sampled_states = env.all_possible_states_at_resolution(
            env.grid.cell_size
        )

        for agent_name in env.possible_agents:
            self.assertSetEqual(
                set(sampled_states[agent_name].keys()),
                set(legacy_states[agent_name].keys()),
            )

        env.close()

    def test_all_possible_states_at_finer_resolution_has_more_samples(
        self,
    ) -> None:
        scenario = SimpleScenario()
        env = BaseEnv(scenario=scenario, max_cycles=10)
        env.reset(seed=42)

        legacy_states = env.all_possible_states()
        finer_states = env.all_possible_states_at_resolution(
            env.grid.cell_size / 2
        )

        for agent_name in env.possible_agents:
            self.assertGreater(
                len(finer_states[agent_name]), len(legacy_states[agent_name])
            )

        env.close()

    def test_all_possible_states_at_resolution_restores_agent_position(
        self,
    ) -> None:
        scenario = SimpleScenario()
        env = BaseEnv(scenario=scenario, max_cycles=10)
        env.reset(seed=42)

        original_positions = {
            agent.name: agent.state.pos.copy() for agent in env.world.agents
        }
        env.all_possible_states_at_resolution((0.5, 1.0))

        for agent in env.world.agents:
            np.testing.assert_allclose(
                agent.state.pos, original_positions[agent.name]
            )

        env.close()

    def test_all_possible_states_at_resolution_rejects_non_positive_spacing(
        self,
    ) -> None:
        scenario = SimpleScenario()
        env = BaseEnv(scenario=scenario, max_cycles=10)
        env.reset(seed=42)

        with self.assertRaisesRegex(
            ValueError, "resolution steps must be positive"
        ):
            env.all_possible_states_at_resolution(0.0)

        with self.assertRaisesRegex(
            ValueError, "resolution steps must be positive"
        ):
            env.all_possible_states_at_resolution((0.5, -1.0))

        env.close()


if __name__ == "__main__":
    absltest.main()
