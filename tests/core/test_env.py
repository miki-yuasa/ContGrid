import os

import numpy as np
from gymnasium import spaces
from PIL import Image

from contgrid.core.const import Color
from contgrid.core.entities import EntityState, Landmark
from contgrid.core.env import BaseEnv
from contgrid.core.scenario import BaseScenario
from contgrid.core.world import Agent, AgentState, World, WorldConfig


class SimpleScenario(BaseScenario[None]):
    """A simple test scenario with one agent and one landmark"""

    def init_agents(self, world: World, np_random=None) -> list[Agent]:
        """Initialize a single agent in the world"""
        agent = Agent(
            name="agent_0",
            size=0.025,
            color=Color.SKY_BLUE,
            state=AgentState(
                pos=np.array([10, 0.19], dtype=np.float64),
                vel=np.array([0.0, 0.0], dtype=np.float64),
            ),
        )
        return [agent]

    def init_landmarks(self, world: World, np_random=None) -> list[Landmark]:
        """Initialize a single landmark in the world"""
        landmark = Landmark(
            name="landmark_0",
            size=0.03,
            color=Color.GREEN,
            state=EntityState(
                pos=np.array([0.2, 0.3], dtype=np.float64),
                vel=np.array([0.0, 0.0], dtype=np.float64),
            ),
        )
        return [landmark]

    def reset_agents(self, world: World, np_random) -> list[Agent]:
        """Reset agent positions"""
        for agent in world.agents:
            agent.state.pos = np.array([0.1, 0.1], dtype=np.float64)
            agent.state.vel = np.array([0.0, 0.0], dtype=np.float64)
        return world.agents

    def reset_landmarks(self, world: World, np_random) -> list[Landmark]:
        """Reset landmark positions"""
        for landmark in world.landmarks:
            landmark.state.pos = np.array([0.2, 0.3], dtype=np.float64)
            landmark.state.vel = np.array([0.0, 0.0], dtype=np.float64)
        return world.landmarks

    def observation(self, agent: Agent, world: World) -> np.ndarray:
        """Simple observation: agent position and landmark position"""
        obs = []
        # Agent's own position
        obs.extend(agent.state.pos)
        # Landmark position
        if world.landmarks:
            obs.extend(world.landmarks[0].state.pos)
        else:
            obs.extend([0.0, 0.0])
        return np.array(obs, dtype=np.float64)

    def observation_space(self, agent: Agent, world: World) -> spaces.Space:
        """Define observation space"""
        return spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)


class TestEnvironmentRendering:
    """Test class for environment rendering functionality"""

    def test_render_default_environment_and_save_png(self):
        """Test rendering the default environment and saving it as PNG"""
        # Create a simple scenario
        scenario = SimpleScenario()

        # Create environment with rgb_array render mode
        env = BaseEnv(scenario=scenario, render_mode="rgb_array", max_cycles=10)

        # Reset environment
        env.reset(seed=42)

        # Render the environment
        rendered_image = env.render()

        # Verify the rendered image is not None and has correct shape
        assert rendered_image is not None, "Rendered image should not be None"
        assert len(rendered_image.shape) == 3, (
            "Image should be 3D (height, width, channels)"
        )
        assert rendered_image.shape[2] == 3, "Image should have 3 color channels (RGB)"

        # Save numpy array as PNG using PIL
        # rendered_image is in format (height, width, channels)
        height, width, channels = rendered_image.shape

        # Create PIL Image from numpy array
        pil_image = Image.fromarray(rendered_image.astype(np.uint8))

        # Save as PNG
        output_path = os.path.join("tests", "out", "plots", "default_environment.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        pil_image.save(output_path)

        # Verify file was created
        assert os.path.exists(output_path), (
            f"PNG file should be created at {output_path}"
        )

        # Verify file is not empty
        assert os.path.getsize(output_path) > 0, "PNG file should not be empty"

        # Clean up
        env.close()

        print(f"Successfully rendered and saved environment to {output_path}")

    def test_render_with_multiple_steps(self):
        """Test rendering environment after taking a few steps"""
        scenario = SimpleScenario()
        env = BaseEnv(scenario=scenario, render_mode="rgb_array", max_cycles=10)

        # Reset environment
        env.reset(seed=42)

        # Take a few steps with random actions
        for step in range(3):
            if env.agents:
                # Get a valid action for the current agent
                agent_name = env.agent_selection
                action_space = env.action_space(agent_name)
                if hasattr(action_space, "sample"):
                    action = action_space.sample()
                    action = np.array([-0.5, 0.5])  # Simple movement

                else:
                    action = np.array([0.01, 0.01])  # Simple movement

                print(f"Step {step + 1}, Action: {action}")

                env.step(action)

        # Render the environment after steps
        rendered_image = env.render()

        # Verify rendering still works
        assert rendered_image is not None
        assert len(rendered_image.shape) == 3

        # Save stepped environment
        height, width, channels = rendered_image.shape
        pil_image = Image.fromarray(rendered_image.astype(np.uint8))

        output_path = os.path.join("tests", "out", "plots", "stepped_environment.png")
        pil_image.save(output_path)

        assert os.path.exists(output_path)
        env.close()

        print(f"Successfully rendered stepped environment to {output_path}")


if __name__ == "__main__":
    # Allow running the test directly
    test_instance = TestEnvironmentRendering()
    test_instance.test_render_default_environment_and_save_png()
    test_instance.test_render_with_multiple_steps()
