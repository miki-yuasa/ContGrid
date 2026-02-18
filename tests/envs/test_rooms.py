import os
from pprint import pprint

import imageio
import numpy as np
import pytest
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
            "doorway_pos",
            "goal_dist",
            "lava_dist",
            "hole_dist",
            "wall_dist",
            "doorway_dist",
        }
        assert set(observation.keys()) == expected_keys

        # Check observation types and shapes
        assert isinstance(observation["agent_pos"], np.ndarray)
        assert observation["agent_pos"].shape == (2,)
        assert isinstance(observation["goal_pos"], np.ndarray)
        assert observation["goal_pos"].shape == (2,)

        # Check info structure (info is populated during steps, not reset)
        assert isinstance(info, dict)
        # Info is populated after reset with distances and other metadata

        env.close()

    def test_export_and_save_spawned_config(self, tmp_path):
        """Export and save current random spawns as RoomsScenarioConfig."""
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
        env = RoomsEnv(scenario_config=RoomsScenarioConfig(spawn_config=spawn_config))
        env.reset(seed=123)

        exported = env.export_spawned_config()
        assert isinstance(exported, RoomsScenarioConfig)

        world_agent_pos = tuple(float(v) for v in env.world.agents[0].state.pos)
        world_goal_pos = tuple(float(v) for v in env.scenario.goal.state.pos)
        world_lava_pos = [tuple(float(v) for v in lava.state.pos) for lava in env.scenario.lavas]
        world_hole_pos = [tuple(float(v) for v in hole.state.pos) for hole in env.scenario.holes]

        assert exported.spawn_config.agent == world_agent_pos
        assert exported.spawn_config.goal.pos == world_goal_pos
        assert [obj.pos for obj in exported.spawn_config.lavas] == world_lava_pos
        assert [obj.pos for obj in exported.spawn_config.holes] == world_hole_pos

        output_path = tmp_path / "spawned_config.json"
        saved = env.save_spawned_config(output_path)
        assert output_path.exists()

        loaded = RoomsScenarioConfig.model_validate_json(output_path.read_text())
        assert loaded.model_dump() == saved.model_dump()

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
            "doorway_pos",
            "goal_dist",
            "lava_dist",
            "hole_dist",
            "wall_dist",
            "doorway_dist",
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
        env = RoomsEnv(scenario_config=config)

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
        env = RoomsEnv(scenario_config=config)

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

        env = RoomsEnv(scenario_config=custom_config)
        observation, info = env.reset(seed=42)

        # Check that agent is at expected position (use world coordinates)
        expected_pos = np.array([2.0, 2.0])
        agent_pos_world = env.world.agents[0].state.pos
        np.testing.assert_array_almost_equal(agent_pos_world, expected_pos)

        # Check that goal is at expected position (use world coordinates)
        expected_goal = np.array([8.0, 8.0])
        np.testing.assert_array_almost_equal(env.scenario.goal_pos, expected_goal)

        # Check that we have the right number of obstacles
        assert env.scenario.lava_pos.shape == (1, 2)  # 1 lava with (x, y) coordinates
        assert env.scenario.hole_pos.shape == (1, 2)  # 1 hole with (x, y) coordinates

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
        env = RoomsEnv()

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
        # Test that agent can reach goal and terminate
        # Place goal close to agent
        spawn_config = SpawnConfig(
            agent=(3.0, 3.0),
            goal=ObjConfig(
                pos=(3.5, 3.0), reward=10.0, absorbing=True
            ),  # Very close to agent
            lavas=[],
            holes=[],
        )
        config = RoomsScenarioConfig(spawn_config=spawn_config)
        env = RoomsEnv(scenario_config=config)

        observation, info = env.reset(seed=42)

        # Move agent towards goal to trigger termination
        terminated = False
        final_reward = None
        for _ in range(10):
            action = np.array([0.1, 0.0])  # Move right towards goal
            observation, reward, termination, truncation, info = env.step(action)
            if termination:
                terminated = True
                final_reward = reward
                break

        # Should terminate due to reaching goal
        assert terminated is True, "Agent should have reached goal and terminated"
        assert final_reward == 10.0, f"Expected goal reward 10.0, got {final_reward}"

        env.close()

    def test_no_agent_obstacle_overlap_at_spawn(self):
        """Test that agent and obstacles don't overlap when spawned"""
        # Test with various spawn configurations
        for seed in [42, 123, 456]:
            spawn_config = SpawnConfig(
                agent=None,  # Random agent position
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

            observation, info = env.reset(seed=seed)

            # Use world coordinates for distance calculations
            agent_pos_world = env.world.agents[0].state.pos
            agent_radius = env.scenario.config.spawn_config.agent_size
            lava_radius = env.scenario.config.spawn_config.lava_size
            hole_radius = env.scenario.config.spawn_config.hole_size

            # Check no overlap with lavas
            if len(env.scenario.lava_pos) > 0:
                min_allowed_distance = agent_radius + lava_radius
                for i, lava_pos in enumerate(env.scenario.lava_pos):
                    distance = np.linalg.norm(lava_pos - agent_pos_world)
                    assert distance >= min_allowed_distance, (
                        f"Seed {seed}: Lava {i} overlaps with agent: distance={distance:.3f} < {min_allowed_distance:.3f}"
                    )

            # Check no overlap with holes
            if len(env.scenario.hole_pos) > 0:
                min_allowed_distance = agent_radius + hole_radius
                for i, hole_pos in enumerate(env.scenario.hole_pos):
                    distance = np.linalg.norm(hole_pos - agent_pos_world)
                    assert distance >= min_allowed_distance, (
                        f"Seed {seed}: Hole {i} overlaps with agent: distance={distance:.3f} < {min_allowed_distance:.3f}"
                    )

            env.close()


class TestRoomsScenario:
    """Test class for RoomsScenario specifically"""

    def test_scenario_initialization(self):
        """Test RoomsScenario initialization"""
        scenario = RoomsScenario(
            config=DEFAULT_ROOMS_SCENARIO_CONFIG, world_config=DEFAULT_WORLD_CONFIG
        )

        assert scenario.config == DEFAULT_ROOMS_SCENARIO_CONFIG
        assert scenario.world_config == DEFAULT_WORLD_CONFIG

        # Check threshold distances are set
        assert hasattr(scenario, "goal_thr_dist")
        assert hasattr(scenario, "lava_thr_dist")
        assert hasattr(scenario, "hole_thr_dist")

    def test_get_closest_method(self):
        """Test the get_closest utility method"""
        scenario = RoomsScenario(
            config=DEFAULT_ROOMS_SCENARIO_CONFIG, world_config=DEFAULT_WORLD_CONFIG
        )

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
        scenario = RoomsScenario(
            config=DEFAULT_ROOMS_SCENARIO_CONFIG, world_config=DEFAULT_WORLD_CONFIG
        )
        world = scenario.make_world()
        np_random = np.random.RandomState(42)
        scenario._pre_reset_world(world, np_random)  # Initialize free_cells
        scenario.reset_landmarks(world, np_random)

        agent = world.agents[0]
        obs = scenario.observation(agent, world)

        # Check all expected keys are present
        expected_keys = {
            "agent_pos",
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
        assert set(obs.keys()) == expected_keys

        # Check types and shapes
        assert isinstance(obs["agent_pos"], np.ndarray)
        assert obs["agent_pos"].shape == (2,)
        assert isinstance(obs["goal_pos"], np.ndarray)
        assert obs["goal_pos"].shape == (2,)

    # def test_animation_random_actions(self):
    #     """Test creating and saving an animation of random actions"""

    #     env = RoomsEnv()
    #     observation, info = env.reset(seed=42)

    #     frames = []
    #     rewards = []
    #     positions = []

    #     # Collect frames for animation
    #     num_steps = 100
    #     for step in range(num_steps):
    #         # Render current state
    #         rendered = env.render()
    #         assert rendered is not None
    #         frames.append(rendered.astype(np.uint8))  # Ensure uint8 for imageio

    #         # Store position and reward for later analysis
    #         positions.append(observation["agent_pos"].copy())

    #         # Take random action
    #         action = env.action_space.sample()
    #         observation, reward, terminated, truncated, info = env.step(action)
    #         rewards.append(reward)

    #         if terminated or truncated:
    #             break

    #     # Create output directory
    #     output_dir = os.path.join("tests", "out", "animations")
    #     os.makedirs(output_dir, exist_ok=True)

    #     # Save individual frames as images for verification
    #     frame_dir = os.path.join(output_dir, "frames")
    #     os.makedirs(frame_dir, exist_ok=True)

    #     for i, frame in enumerate(frames):
    #         # Save individual frame using imageio
    #         frame_path = os.path.join(frame_dir, f"frame_{i:03d}.png")
    #         imageio.imwrite(frame_path, frame)

    #     # Create and save GIF animation using imageio
    #     if len(frames) > 1:
    #         gif_path = os.path.join(output_dir, "random_actions_animation.gif")
    #         imageio.mimsave(
    #             gif_path,
    #             frames,
    #             duration=0.2,  # 0.2 seconds per frame (5 FPS)
    #             loop=0,  # Loop forever
    #         )

    #         # Verify GIF was created
    #         assert os.path.exists(gif_path)
    #         assert os.path.getsize(gif_path) > 0
    #         print(f"Animation saved to: {gif_path}")

    #         # Also save as MP4 for better quality and smaller file size
    #         mp4_path = os.path.join(output_dir, "random_actions_animation.mp4")
    #         try:
    #             imageio.mimsave(
    #                 mp4_path,
    #                 frames,
    #                 fps=5,  # 5 frames per second
    #                 quality=8,  # Good quality
    #             )
    #             print(f"MP4 animation saved to: {mp4_path}")
    #         except Exception as e:
    #             print(f"Could not save MP4 (missing codec?): {e}")

    #     # Verify we have collected data
    #     assert len(frames) > 0
    #     assert len(rewards) > 0
    #     assert len(positions) > 0
    #     assert len(frames) == len(rewards)
    #     assert len(frames) == len(positions)

    #     # Verify frame properties
    #     for frame in frames:
    #         assert isinstance(frame, np.ndarray)
    #         assert len(frame.shape) == 3
    #         assert frame.shape[2] == 3  # RGB
    #         assert frame.dtype == np.uint8  # Correct dtype for imageio

    #     # Create summary statistics
    #     total_reward = sum(rewards)
    #     avg_reward = total_reward / len(rewards)

    #     print(f"Animation created with {len(frames)} frames")
    #     print(f"Total reward: {total_reward:.3f}")
    #     print(f"Average reward per step: {avg_reward:.3f}")
    #     print(f"Final position: {positions[-1]}")
    #     print(f"Position change: {np.linalg.norm(positions[-1] - positions[0]):.3f}")

    #     env.close()

    def test_random_spawn(self):
        """Test that agent and objects spawn within valid room boundaries"""
        plot_save_path: str = "tests/out/rooms_random_spawn.png"
        spawn_config: SpawnConfig = SpawnConfig(
            agent=(3.0, 3.0),
            goal=ObjConfig(pos=(9, 8), reward=1.0, absorbing=False),
            lavas=[
                ObjConfig(
                    pos=[
                        (3, 5),
                        (2, 4),
                        (3, 4),
                        (4, 3),
                        (5, 3),
                        (4, 2),
                    ],
                    reward=-1.0,
                    absorbing=False,
                ),
                ObjConfig(
                    pos=[
                        (3, 5),
                        (2, 4),
                        (3, 4),
                        (4, 3),
                        (5, 3),
                        (4, 2),
                    ],
                    reward=-1.0,
                    absorbing=False,
                ),
                ObjConfig(
                    pos=[
                        (2, 7),
                        (3, 7),
                        (3, 8),
                        (4, 8),
                        (5, 8),
                        (3, 9),
                        (4, 9),
                        (5, 9),
                    ],
                    reward=0.0,
                    absorbing=False,
                ),
                ObjConfig(
                    pos=[
                        (2, 7),
                        (3, 7),
                        (3, 8),
                        (4, 8),
                        (5, 8),
                        (3, 9),
                        (4, 9),
                        (5, 9),
                    ],
                    reward=0.0,
                    absorbing=False,
                ),
                ObjConfig(
                    pos=[
                        (7, 9),
                        (8, 9),
                        (9, 9),
                        (7, 8),
                        (8, 8),
                        (8, 7),
                        (9, 7),
                        (8, 6),
                        (9, 6),
                    ],
                    reward=0.0,
                    absorbing=False,
                ),
                ObjConfig(
                    pos=[
                        (7, 9),
                        (8, 9),
                        (9, 9),
                        (7, 8),
                        (8, 8),
                        (8, 7),
                        (9, 7),
                        (8, 6),
                        (9, 6),
                    ],
                    reward=0.0,
                    absorbing=False,
                ),
                ObjConfig(
                    pos=[
                        (8, 2),
                        (9, 2),
                        (7, 3),
                        (8, 3),
                        (9, 3),
                        (8, 4),
                    ],
                    reward=-1.0,
                    absorbing=False,
                ),
                ObjConfig(
                    pos=[
                        (8, 2),
                        (9, 2),
                        (7, 3),
                        (8, 3),
                        (9, 3),
                        (8, 4),
                    ],
                    reward=-1.0,
                    absorbing=False,
                ),
            ],
            holes=[
                ObjConfig(
                    pos=[
                        (2, 5),
                        (3, 5),
                        (2, 4),
                        (3, 4),
                        (4, 3),
                        (5, 3),
                        (4, 2),
                        (5, 2),
                    ],
                    reward=0.0,
                    absorbing=False,
                ),
                ObjConfig(
                    pos=[
                        (2, 5),
                        (3, 5),
                        (2, 4),
                        (3, 4),
                        (4, 3),
                        (5, 3),
                        (4, 2),
                        (5, 2),
                    ],
                    reward=0.0,
                    absorbing=False,
                ),
                ObjConfig(
                    pos=[
                        (3, 7),
                        (3, 8),
                        (4, 8),
                        (5, 8),
                        (3, 9),
                        (4, 9),
                    ],
                    reward=-1.0,
                    absorbing=False,
                ),
                ObjConfig(
                    pos=[
                        (3, 7),
                        (3, 8),
                        (4, 8),
                        (5, 8),
                        (3, 9),
                        (4, 9),
                    ],
                    reward=-1.0,
                    absorbing=False,
                ),
                ObjConfig(
                    pos=[
                        (7, 9),
                        (8, 9),
                        (9, 9),
                        (7, 8),
                        (8, 8),
                        (8, 7),
                        (9, 7),
                        (8, 6),
                        (9, 6),
                    ],
                    reward=0.0,
                    absorbing=False,
                ),
                ObjConfig(
                    pos=[
                        (7, 9),
                        (8, 9),
                        (9, 9),
                        (7, 8),
                        (8, 8),
                        (8, 7),
                        (9, 7),
                        (8, 6),
                        (9, 6),
                    ],
                    reward=0.0,
                    absorbing=False,
                ),
                ObjConfig(
                    pos=[
                        (8, 2),
                        (9, 2),
                        (7, 3),
                        (8, 3),
                        (9, 3),
                        (8, 4),
                    ],
                    reward=-1.0,
                    absorbing=False,
                ),
                ObjConfig(
                    pos=[
                        (8, 2),
                        (9, 2),
                        (7, 3),
                        (8, 3),
                        (9, 3),
                        (8, 4),
                    ],
                    reward=-1.0,
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
        config: RoomsScenarioConfig = RoomsScenarioConfig(spawn_config=spawn_config)
        env = RoomsEnv(scenario_config=config)
        observation, info = env.reset()
        rendered = env.render()
        os.makedirs("tests/out", exist_ok=True)

        imageio.imwrite(plot_save_path, rendered)
        print(f"Saved random spawn render to: {plot_save_path}")
        env.close()

        # Print spawn configuration in dict
        pprint(spawn_config.model_dump())

        assert os.path.exists(plot_save_path)

    def test_path_gaussian_spawn_with_random_agent(self):
        """Test path-based Gaussian obstacle spawning with random agent position"""
        output_dir = os.path.join("tests", "out", "path_gaussian_spawn")
        os.makedirs(output_dir, exist_ok=True)

        # Create config with path-based Gaussian spawning
        spawn_config = SpawnConfig(
            agent=None,  # Random agent spawning
            goal=ObjConfig(pos=(9, 8), reward=1.0, absorbing=False),
            lavas=[
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="top_left"),
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="bottom_left"),
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="top_right"),
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="bottom_right"),
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="top_left"),
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="bottom_left"),
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="top_right"),
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="bottom_right"),
            ],
            holes=[
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="top_left"),
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="bottom_left"),
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="top_right"),
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="bottom_right"),
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="top_left"),
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="bottom_left"),
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="top_right"),
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="bottom_right"),
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

        # Save multiple resets to show different configurations
        num_samples = 5
        all_frames = []

        for sample_idx in range(num_samples):
            observation, info = env.reset()
            rendered = env.render()

            # Save individual frame
            frame_path = os.path.join(
                output_dir, f"path_gaussian_spawn_sample_{sample_idx}.png"
            )
            imageio.imwrite(frame_path, rendered)
            print(f"Sample {sample_idx}: Agent at {observation['agent_pos']}")
            print(f"  Goal at {observation['agent_pos'] + observation['goal_pos']}")
            print(
                f"  Lavas at: {observation['agent_pos'] + observation['lava_pos'][:3]}..."
            )
            print(f"  Saved to: {frame_path}")

            all_frames.append(rendered.astype(np.uint8))

            # Verify obstacles are placed
            assert len(env.scenario.lavas) == 8  # 8 lavas configured
            assert len(env.scenario.holes) == 8  # 8 holes configured
            assert observation["lava_pos"].shape[0] <= 8
            assert observation["hole_pos"].shape[0] <= 8

            # Verify minimum spacing between obstacles
            min_spacing = 0.9

            # Get absolute positions in world coordinates
            agent_pos_world = env.world.agents[0].state.pos
            lava_positions = env.scenario.lava_pos
            hole_positions = env.scenario.hole_pos

            # Check agent doesn't overlap with lavas
            if len(lava_positions) > 0:
                agent_lava_min_distance = (
                    spawn_config.agent_size + spawn_config.lava_size
                )
                for i, lava_pos in enumerate(lava_positions):
                    distance = np.linalg.norm(lava_pos - agent_pos_world)
                    assert distance >= agent_lava_min_distance, (
                        f"Sample {sample_idx}: Lava {i} overlaps with agent: "
                        f"distance={distance:.3f} < {agent_lava_min_distance:.3f}"
                    )

            # Check agent doesn't overlap with holes
            if len(hole_positions) > 0:
                agent_hole_min_distance = (
                    spawn_config.agent_size + spawn_config.hole_size
                )
                for i, hole_pos in enumerate(hole_positions):
                    distance = np.linalg.norm(hole_pos - agent_pos_world)
                    assert distance >= agent_hole_min_distance, (
                        f"Sample {sample_idx}: Hole {i} overlaps with agent: "
                        f"distance={distance:.3f} < {agent_hole_min_distance:.3f}"
                    )

            # Check spacing between lavas
            if len(lava_positions) > 1:
                for i in range(len(lava_positions)):
                    for j in range(i + 1, len(lava_positions)):
                        dist = np.linalg.norm(lava_positions[i] - lava_positions[j])
                        assert dist >= min_spacing, (
                            f"Lavas {i} and {j} too close: {dist:.3f} < {min_spacing}"
                        )

            # Check spacing between holes
            if len(hole_positions) > 1:
                for i in range(len(hole_positions)):
                    for j in range(i + 1, len(hole_positions)):
                        dist = np.linalg.norm(hole_positions[i] - hole_positions[j])
                        assert dist >= min_spacing, (
                            f"Holes {i} and {j} too close: {dist:.3f} < {min_spacing}"
                        )

            # Check spacing between lavas and holes
            for i, lava_pos in enumerate(lava_positions):
                for j, hole_pos in enumerate(hole_positions):
                    dist = np.linalg.norm(lava_pos - hole_pos)
                    assert dist >= min_spacing, (
                        f"Lava {i} and hole {j} too close: {dist:.3f} < {min_spacing}"
                    )

            # Verify obstacles spawned in their designated rooms

            topology = RoomTopology(config.spawn_config.doorways)

            # Check each lava is in its designated room
            for i, (lava_pos, lava_config) in enumerate(
                zip(lava_positions, config.spawn_config.lavas)
            ):
                if lava_config.room is not None:
                    actual_room = topology.get_room(lava_pos)
                    assert actual_room == lava_config.room, (
                        f"Lava {i} spawned in room '{actual_room}' but expected '{lava_config.room}' at position {lava_pos}"
                    )

            # Check each hole is in its designated room
            for i, (hole_pos, hole_config) in enumerate(
                zip(hole_positions, config.spawn_config.holes)
            ):
                if hole_config.room is not None:
                    actual_room = topology.get_room(hole_pos)
                    assert actual_room == hole_config.room, (
                        f"Hole {i} spawned in room '{actual_room}' but expected '{hole_config.room}' at position {hole_pos}"
                    )

    def test_path_gaussian_spawn_zero_std_on_segments(self):
        """Test that when gaussian_std=0, obstacles are spawned exactly on path segments."""
        from contgrid.envs.rooms.topology import LineSegment, get_relevant_path_segments

        def point_to_segment_distance(point: np.ndarray, segment: LineSegment) -> float:
            """Calculate minimum distance from a point to a line segment."""
            # Vector from start to end
            v = segment.end - segment.start
            # Vector from start to point
            w = point - segment.start

            # Length squared of segment
            c1 = np.dot(w, v)
            c2 = np.dot(v, v)

            if c2 < 1e-10:  # Degenerate segment (point)
                return float(np.linalg.norm(point - segment.start))

            # Parameter t for closest point on line
            t = c1 / c2

            # Clamp t to [0, 1] to stay on segment
            t = max(0.0, min(1.0, t))

            # Closest point on segment
            closest = segment.start + t * v

            return float(np.linalg.norm(point - closest))

        def min_distance_to_any_segment(
            point: np.ndarray, segments: list[LineSegment]
        ) -> float:
            """Find minimum distance from point to any segment in the list."""
            if not segments:
                return float("inf")
            return min(point_to_segment_distance(point, seg) for seg in segments)

        # Create config with gaussian_std=0 (no perturbation)
        # Note: min_spacing is reduced to 0.3 to allow obstacles to fit on shorter
        # room-clipped segments (e.g., bottom_right has a ~1.27 unit segment)
        spawn_config = SpawnConfig(
            agent=(3.0, 3.0),  # Fixed agent position for reproducibility
            goal=ObjConfig(pos=(9, 8), reward=1.0, absorbing=False),
            lavas=[
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="top_left"),
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="bottom_left"),
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="top_right"),
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="bottom_right"),
            ],
            holes=[
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="top_left"),
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="bottom_left"),
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="top_right"),
                ObjConfig(pos=None, reward=-1.0, absorbing=False, room="bottom_right"),
            ],
            spawn_method=PathGaussianConfig(
                gaussian_std=0.0,  # Zero std - should spawn exactly on segments
                min_spacing=0.3,  # Reduced from 0.5 to fit on short segments
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

        # Test with multiple seeds
        # With segment clipping to room boundaries, obstacles should be exactly on
        # the clipped segments when gaussian_std=0 (within floating point tolerance)
        tolerance = 1e-6  # Floating point tolerance

        for seed in [42, 123, 456, 789, 1000]:
            observation, info = env.reset(seed=seed)

            # Get the path segments that were used for spawning (use world coordinates)
            topology = RoomTopology(config.spawn_config.doorways)
            agent_pos_world = env.world.agents[0].state.pos

            segments = get_relevant_path_segments(
                agent_pos_world,
                env.scenario.goal_pos,
                env.scenario.doorways,
                topology,
                env.world.grid,
            )

            # Get absolute positions of obstacles in world coordinates
            lava_positions = env.scenario.lava_pos
            hole_positions = env.scenario.hole_pos

            # Helper to clip segment to room bounds (matches spawn_strategies logic)
            def clip_segment_to_room(
                seg: LineSegment, room_name: str
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

            # Check each lava is on a clipped segment for its room
            for i, lava_pos in enumerate(lava_positions):
                room = config.spawn_config.lavas[i].room
                clipped_segs = [
                    cs
                    for seg in segments
                    if (cs := clip_segment_to_room(seg, room)) is not None
                    and np.linalg.norm(cs.end - cs.start) > 1e-6
                ]
                dist = min_distance_to_any_segment(lava_pos, clipped_segs)
                assert dist < tolerance, (
                    f"Seed {seed}: Lava {i} at {lava_pos} is not on any clipped segment. "
                    f"Min distance to segments: {dist:.8f}"
                )

            # Check each hole is on a clipped segment for its room
            for i, hole_pos in enumerate(hole_positions):
                room = config.spawn_config.holes[i].room
                clipped_segs = [
                    cs
                    for seg in segments
                    if (cs := clip_segment_to_room(seg, room)) is not None
                    and np.linalg.norm(cs.end - cs.start) > 1e-6
                ]
                dist = min_distance_to_any_segment(hole_pos, clipped_segs)
                assert dist < tolerance, (
                    f"Seed {seed}: Hole {i} at {hole_pos} is not on any clipped segment. "
                    f"Min distance to segments: {dist:.8f}"
                )

        env.close()


if __name__ == "__main__":
    # Allow running tests directly
    import sys

    # Run all tests in this file
    pytest.main([__file__] + sys.argv[1:])
