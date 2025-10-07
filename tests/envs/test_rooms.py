import numpy as np
import pytest
from gymnasium import spaces

from contgrid.envs.rooms import (
    DEFAULT_ROOMS_SCENARIO_CONFIG,
    DEFAULT_WORLD_CONFIG,
    ObjConfig,
    RewardConfig,
    RoomsEnv,
    RoomsScenario,
    RoomsScenarioConfig,
    SpawnConfig,
)


class TestRoomsEnv:
    """Test class for RoomsEnv environment"""

    def test_env_initialization(self):
        """Test that RoomsEnv initializes correctly with default configuration"""
        env = RoomsEnv()

        # Check that environment is created
        assert env is not None

        # Check that the world has the correct number of agents and landmarks
        world = env.world
        assert len(world.agents) == 1
        assert world.agents[0].name == "agent_0"

        # Check landmarks (goal + lavas + holes)
        default_config = DEFAULT_ROOMS_SCENARIO_CONFIG
        expected_landmarks = (
            1
            + len(default_config.spawn_config.lavas)
            + len(default_config.spawn_config.holes)
        )
        assert len(world.landmarks) == expected_landmarks

        env.close()

    def test_env_reset(self):
        """Test environment reset functionality"""
        env = RoomsEnv()

        # Reset environment
        observation, info = env.reset(seed=42)

        # Check observation structure
        assert isinstance(observation, dict)
        expected_keys = {
            "agent_pos",
            "goal_pos",
            "lava_pos",
            "hole_pos",
            "goal_dist",
            "lava_dist",
            "hole_dist",
        }
        assert set(observation.keys()) == expected_keys

        # Check observation types and shapes
        assert isinstance(observation["agent_pos"], np.ndarray)
        assert observation["agent_pos"].shape == (2,)
        assert isinstance(observation["goal_pos"], np.ndarray)
        assert observation["goal_pos"].shape == (2,)

        # Check info structure (info is populated during steps, not reset)
        assert isinstance(info, dict)
        # Info should be empty after reset
        assert len(info) == 0

        env.close()

    def test_observation_space(self):
        """Test observation space definition"""
        env = RoomsEnv()

        # Get observation space for first agent
        agent_name = env.possible_agents[0]
        obs_space = env.observation_spaces[agent_name]

        assert isinstance(obs_space, spaces.Dict)

        # Check that all expected keys are present
        expected_keys = {
            "agent_pos",
            "goal_pos",
            "lava_pos",
            "hole_pos",
            "goal_dist",
            "lava_dist",
            "hole_dist",
        }
        assert set(obs_space.spaces.keys()) == expected_keys

        # Check individual spaces
        assert isinstance(obs_space["agent_pos"], spaces.Box)
        assert obs_space["agent_pos"].shape == (2,)
        assert isinstance(obs_space["goal_pos"], spaces.Box)
        assert obs_space["goal_pos"].shape == (2,)

        env.close()

    def test_action_space(self):
        """Test action space definition"""
        env = RoomsEnv()

        # Get action space for first agent
        agent_name = env.possible_agents[0]
        action_space = env.action_spaces[agent_name]

        assert isinstance(action_space, spaces.Box)
        assert action_space.shape == (2,)  # 2D movement
        assert action_space.low.tolist() == [-10.0, -10.0]
        assert action_space.high.tolist() == [10.0, 10.0]

        env.close()

    def test_step_functionality(self):
        """Test stepping through the environment"""
        env = RoomsEnv()
        observation, info = env.reset(seed=42)

        # Take a step with a valid action
        action = np.array([0.1, 0.1])

        observation, reward, termination, truncation, info = env.step(action)

        # Check return types
        assert isinstance(observation, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(termination, bool)
        assert isinstance(truncation, bool)
        assert isinstance(info, dict)

        # Check that agent moved (position should be different)
        assert observation["agent_pos"] is not None

        env.close()

    def test_reward_system_goal_reaching(self):
        """Test reward system when agent reaches the goal"""
        # Create custom config with goal at agent's starting position
        spawn_config = SpawnConfig(
            agent=(3.0, 3.0),
            goal=ObjConfig(pos=(3.0, 3.0), reward=10.0, absorbing=True),
            lavas=[],  # No lavas for this test
            holes=[],  # No holes for this test
        )
        config = RoomsScenarioConfig(spawn_config=spawn_config)
        env = RoomsEnv(config=config)

        observation, info = env.reset(seed=42)

        # Agent should be very close to or at the goal
        goal_dist = observation["goal_dist"]
        assert goal_dist < 1.0  # Should be close to goal

        # Take a small step (should trigger goal reward)
        action = np.array([0.01, 0.01])
        observation, reward, termination, truncation, info = env.step(action)

        # Should receive goal reward and terminate
        assert reward == 10.0
        assert termination is True
        assert info["terminated"] is True

        env.close()

    def test_step_penalty(self):
        """Test step penalty when agent doesn't reach goal or obstacles"""
        # Create config with goal far away and no obstacles
        spawn_config = SpawnConfig(
            agent=(1.0, 1.0),
            goal=ObjConfig(pos=(10.0, 10.0), reward=1.0, absorbing=True),
            lavas=[],  # No lavas
            holes=[],  # No holes
        )
        reward_config = RewardConfig(step_penalty=0.1)
        config = RoomsScenarioConfig(
            spawn_config=spawn_config, reward_config=reward_config
        )
        env = RoomsEnv(config=config)

        observation, info = env.reset(seed=42)

        # Take a step that doesn't reach goal or obstacles
        action = np.array([0.1, 0.1])
        observation, reward, termination, truncation, info = env.step(action)

        # Should receive step penalty
        assert reward == -0.1
        assert termination is False

        env.close()

    def test_custom_configuration(self):
        """Test environment with custom configuration"""
        # Create custom spawn configuration
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

        env = RoomsEnv(config=custom_config)
        observation, info = env.reset(seed=42)

        # Check that agent is at expected position
        expected_pos = np.array([2.0, 2.0])
        np.testing.assert_array_almost_equal(observation["agent_pos"], expected_pos)

        # Check that goal is at expected position
        expected_goal = np.array([8.0, 8.0])
        np.testing.assert_array_almost_equal(observation["goal_pos"], expected_goal)

        # Check that we have the right number of obstacles
        assert observation["lava_pos"].shape == (1, 2)  # 1 lava with (x, y) coordinates
        assert observation["hole_pos"].shape == (1, 2)  # 1 hole with (x, y) coordinates

        env.close()

    def test_rendering(self):
        """Test that environment can be rendered without errors"""
        env = RoomsEnv()
        observation, info = env.reset(seed=42)

        # Test rgb_array rendering
        rendered = env.render()

        assert rendered is not None
        assert isinstance(rendered, np.ndarray)
        assert len(rendered.shape) == 3  # Height, width, channels
        assert rendered.shape[2] == 3  # RGB channels

        env.close()

    def test_multiple_episodes(self):
        """Test running multiple episodes"""
        env = RoomsEnv(max_episode_steps=10)

        for episode in range(3):
            observation, info = env.reset(seed=42 + episode)
            done = False
            step_count = 0

            while not done and step_count < 10:
                action = env.action_spaces[env.agent_selection].sample()
                observation, reward, termination, truncation, info = env.step(action)
                done = termination or truncation
                step_count += 1

            # Episode should complete without errors
            assert step_count <= 10

        env.close()

    def test_scenario_doorway_distances(self):
        """Test that doorway distances are calculated correctly in info"""
        env = RoomsEnv()
        observation, info = env.reset(seed=42)

        # Take a step to populate info
        action = np.array([0.0, 0.0])
        observation, reward, termination, truncation, info = env.step(action)

        # Check that doorway distances are in info
        assert "distances" in info
        distances = info["distances"]  # Default doorways should be present
        expected_doorways = {"ld", "td", "rd", "bd"}
        for doorway in expected_doorways:
            assert doorway in distances
            assert isinstance(distances[doorway], (int, float))
            assert distances[doorway] >= 0

        env.close()

    def test_environment_termination_conditions(self):
        """Test different termination conditions"""
        # Test with absorbing lava
        spawn_config = SpawnConfig(
            agent=(3.0, 3.0),
            goal=ObjConfig(pos=(10.0, 10.0), reward=1.0, absorbing=True),
            lavas=[ObjConfig(pos=(3.0, 3.0), reward=-5.0, absorbing=True)],
            holes=[],
        )
        config = RoomsScenarioConfig(spawn_config=spawn_config)
        env = RoomsEnv(config=config)

        observation, info = env.reset(seed=42)

        # Agent should be at lava position, causing termination
        action = np.array([0.01, 0.01])
        observation, reward, termination, truncation, info = env.step(action)

        # Should terminate due to lava
        assert termination is True
        assert reward == -5.0

        env.close()


class TestRoomsScenario:
    """Test class for RoomsScenario specifically"""

    def test_scenario_initialization(self):
        """Test RoomsScenario initialization"""
        scenario = RoomsScenario()

        assert scenario.config == DEFAULT_ROOMS_SCENARIO_CONFIG
        assert scenario.world_config == DEFAULT_WORLD_CONFIG

        # Check threshold distances are set
        assert hasattr(scenario, "goal_thr_dist")
        assert hasattr(scenario, "lava_thr_dist")
        assert hasattr(scenario, "hole_thr_dist")

    def test_get_closest_method(self):
        """Test the get_closest utility method"""
        scenario = RoomsScenario()

        # Test with some objects
        pos = np.array([0.0, 0.0])
        objects = np.array([[1.0, 0.0], [0.0, 1.0], [3.0, 4.0]])

        min_dist, min_idx = scenario.get_closest(pos, objects)

        assert min_dist == 1.0  # Distance to closest object
        assert min_idx in [0, 1]  # Either first or second object

        # Test with no objects
        empty_objects = np.array([]).reshape(0, 2)
        min_dist, min_idx = scenario.get_closest(pos, empty_objects)

        assert min_dist == np.inf
        assert min_idx == -1

    def test_observation_contains_all_elements(self):
        """Test that observation contains all expected elements"""
        scenario = RoomsScenario()
        world = scenario.make_world()
        scenario.reset_landmarks(world, np.random.RandomState(42))

        agent = world.agents[0]
        obs = scenario.observation(agent, world)

        # Check all expected keys are present
        expected_keys = {
            "agent_pos",
            "goal_pos",
            "lava_pos",
            "hole_pos",
            "goal_dist",
            "lava_dist",
            "hole_dist",
        }
        assert set(obs.keys()) == expected_keys

        # Check types and shapes
        assert isinstance(obs["agent_pos"], np.ndarray)
        assert obs["agent_pos"].shape == (2,)
        assert isinstance(obs["goal_pos"], np.ndarray)
        assert obs["goal_pos"].shape == (2,)

    def test_animation_random_actions(self):
        """Test creating and saving an animation of random actions"""
        import os

        import imageio

        env = RoomsEnv(max_episode_steps=100)
        observation, info = env.reset(seed=42)

        frames = []
        rewards = []
        positions = []

        # Collect frames for animation
        num_steps = 100
        for step in range(num_steps):
            # Render current state
            rendered = env.render()
            assert rendered is not None
            frames.append(rendered.astype(np.uint8))  # Ensure uint8 for imageio

            # Store position and reward for later analysis
            positions.append(observation["agent_pos"].copy())

            # Take random action
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            rewards.append(reward)

            if terminated or truncated:
                break

        # Create output directory
        output_dir = os.path.join("tests", "out", "animations")
        os.makedirs(output_dir, exist_ok=True)

        # Save individual frames as images for verification
        frame_dir = os.path.join(output_dir, "frames")
        os.makedirs(frame_dir, exist_ok=True)

        for i, frame in enumerate(frames):
            # Save individual frame using imageio
            frame_path = os.path.join(frame_dir, f"frame_{i:03d}.png")
            imageio.imwrite(frame_path, frame)

        # Create and save GIF animation using imageio
        if len(frames) > 1:
            gif_path = os.path.join(output_dir, "random_actions_animation.gif")
            imageio.mimsave(
                gif_path,
                frames,
                duration=0.2,  # 0.2 seconds per frame (5 FPS)
                loop=0,  # Loop forever
            )

            # Verify GIF was created
            assert os.path.exists(gif_path)
            assert os.path.getsize(gif_path) > 0
            print(f"Animation saved to: {gif_path}")

            # Also save as MP4 for better quality and smaller file size
            mp4_path = os.path.join(output_dir, "random_actions_animation.mp4")
            try:
                imageio.mimsave(
                    mp4_path,
                    frames,
                    fps=5,  # 5 frames per second
                    quality=8,  # Good quality
                )
                print(f"MP4 animation saved to: {mp4_path}")
            except Exception as e:
                print(f"Could not save MP4 (missing codec?): {e}")

        # Verify we have collected data
        assert len(frames) > 0
        assert len(rewards) > 0
        assert len(positions) > 0
        assert len(frames) == len(rewards)
        assert len(frames) == len(positions)

        # Verify frame properties
        for frame in frames:
            assert isinstance(frame, np.ndarray)
            assert len(frame.shape) == 3
            assert frame.shape[2] == 3  # RGB
            assert frame.dtype == np.uint8  # Correct dtype for imageio

        # Create summary statistics
        total_reward = sum(rewards)
        avg_reward = total_reward / len(rewards)

        print(f"Animation created with {len(frames)} frames")
        print(f"Total reward: {total_reward:.3f}")
        print(f"Average reward per step: {avg_reward:.3f}")
        print(f"Final position: {positions[-1]}")
        print(f"Position change: {np.linalg.norm(positions[-1] - positions[0]):.3f}")

        env.close()


if __name__ == "__main__":
    # Allow running tests directly
    import sys

    # Run all tests in this file
    pytest.main([__file__] + sys.argv[1:])
