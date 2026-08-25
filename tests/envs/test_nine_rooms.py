import os

import imageio
import numpy as np
import pytest
import yaml
from gymnasium import spaces

from contgrid.envs.rooms import (
    DEFAULT_NINE_ROOMS_SCENARIO_CONFIG,
    NineRoomsEnv,
    ObjConfig,
    PathGaussianConfig,
    RewardConfig,
    RoomsScenarioConfig,
    RoomTopology,
    SpawnConfig,
)


class TestNineRoomsEnv:
    """Test class for NineRoomsEnv environment."""

    def test_env_initialization(self):
        """Test that NineRoomsEnv initializes correctly with default configuration."""
        env = NineRoomsEnv()

        # Check that environment is created
        assert env is not None

        # Check that the world has the correct number of agents and landmarks
        world = env.world
        assert len(world.agents) == 1
        assert world.agents[0].name == "agent_0"

        # Check landmarks (goal + lavas + holes = 1 + 18 + 18 = 37)
        default_config = DEFAULT_NINE_ROOMS_SCENARIO_CONFIG
        expected_landmarks = (
            1
            + len(default_config.spawn_config.lavas)
            + len(default_config.spawn_config.holes)
        )
        assert expected_landmarks == 37
        assert len(world.landmarks) == expected_landmarks

        env.close()

    def test_env_reset(self):
        """Test environment reset functionality."""
        env = NineRoomsEnv()

        # Reset environment
        observation, info = env.reset(seed=42)

        # Check observation structure
        assert isinstance(observation, dict)
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
        assert set(observation.keys()) == expected_keys

        # Check observation types and shapes
        assert isinstance(observation["agent_pos"], np.ndarray)
        assert observation["agent_pos"].shape == (2,)
        assert isinstance(observation["goal_pos"], np.ndarray)
        assert observation["goal_pos"].shape == (2,)

        # Check doorway observations — should have 12 doorways
        assert observation["doorway_pos"].shape == (12, 2)
        assert observation["doorway_dist"].shape == (12,)

        # Check info structure
        assert isinstance(info, dict)

        env.close()

    def test_observation_space(self):
        """Test observation space definition."""
        env = NineRoomsEnv()

        agent_name = env.possible_agents[0]
        obs_space = env.observation_spaces[agent_name]

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
        assert set(obs_space.spaces.keys()) == expected_keys

        assert isinstance(obs_space["agent_pos"], spaces.Box)
        assert obs_space["agent_pos"].shape == (2,)
        assert isinstance(obs_space["goal_pos"], spaces.Box)
        assert obs_space["goal_pos"].shape == (2,)
        assert isinstance(obs_space["doorway_dist"], spaces.Box)
        assert obs_space["doorway_dist"].shape == (12,)

        env.close()

    def test_action_space(self):
        """Test action space definition."""
        env = NineRoomsEnv()

        agent_name = env.possible_agents[0]
        action_space = env.action_spaces[agent_name]

        assert isinstance(action_space, spaces.Box)
        assert action_space.shape == (2,)

        env.close()

    def test_step_functionality(self):
        """Test stepping through the environment."""
        env = NineRoomsEnv()
        observation, info = env.reset(seed=42)

        action = np.array([0.1, 0.1])
        observation, reward, termination, truncation, info = env.step(action)

        assert isinstance(observation, dict)
        assert isinstance(reward, (int, float))
        assert isinstance(termination, bool)
        assert isinstance(truncation, bool)
        assert isinstance(info, dict)
        assert observation["agent_pos"] is not None

        env.close()

    def test_reward_system_goal_reaching(self):
        """Test reward system when agent reaches the goal."""
        spawn_config = SpawnConfig(
            agent=(15.0, 15.0),
            goal=ObjConfig(pos=(15.0, 15.0), reward=10.0, absorbing=True),
            lavas=[],
            holes=[],
            doorways=DEFAULT_NINE_ROOMS_SCENARIO_CONFIG.spawn_config.doorways,
        )
        config = RoomsScenarioConfig(spawn_config=spawn_config)
        env = NineRoomsEnv(scenario_config=config)

        observation, info = env.reset(seed=42)

        goal_dist = observation["goal_dist"]
        assert goal_dist < 1.0

        action = np.array([0.01, 0.01])
        observation, reward, termination, truncation, info = env.step(action)

        assert reward == 10.0
        assert termination is True
        assert info["terminated"] is True

        env.close()

    def test_step_penalty(self):
        """Test step penalty when agent doesn't reach goal or obstacles."""
        spawn_config = SpawnConfig(
            agent=(1.0, 1.0),
            goal=ObjConfig(pos=(17.0, 17.0), reward=1.0, absorbing=True),
            lavas=[],
            holes=[],
            doorways=DEFAULT_NINE_ROOMS_SCENARIO_CONFIG.spawn_config.doorways,
        )
        reward_config = RewardConfig(step_penalty=0.1)
        config = RoomsScenarioConfig(
            spawn_config=spawn_config, reward_config=reward_config
        )
        env = NineRoomsEnv(scenario_config=config)

        observation, info = env.reset(seed=42)

        action = np.array([0.1, 0.1])
        observation, reward, termination, truncation, info = env.step(action)

        assert reward == -0.1
        assert termination is False

        env.close()

    def test_rendering(self):
        """Test that environment can be rendered without errors."""
        env = NineRoomsEnv()
        observation, info = env.reset(seed=42)

        rendered = env.render()

        assert rendered is not None
        assert isinstance(rendered, np.ndarray)
        assert len(rendered.shape) == 3
        assert rendered.shape[2] == 3

        env.close()

    def test_multiple_episodes(self):
        """Test running multiple episodes."""
        env = NineRoomsEnv()

        for episode in range(3):
            observation, info = env.reset(seed=42 + episode)
            done = False
            step_count = 0

            while not done and step_count < 10:
                action = env.action_spaces[env.agent_selection].sample()
                observation, reward, termination, truncation, info = env.step(action)
                done = termination or truncation
                step_count += 1

            assert step_count <= 10

        env.close()

    def test_scenario_doorway_distances(self):
        """Test that doorway distances are calculated correctly in info."""
        env = NineRoomsEnv()
        observation, info = env.reset(seed=42)

        action = np.array([0.0, 0.0])
        observation, reward, termination, truncation, info = env.step(action)

        assert "distances" in info
        distances = info["distances"]
        expected_doorways = {
            "tl_tc",
            "tc_tr",
            "ml_mc",
            "mc_mr",
            "bl_bc",
            "bc_br",
            "tl_ml",
            "tc_mc",
            "tr_mr",
            "ml_bl",
            "mc_bc",
            "mr_br",
        }
        for doorway in expected_doorways:
            assert doorway in distances
            assert isinstance(distances[doorway], (int, float))
            assert distances[doorway] >= 0

        env.close()

    def test_nine_room_grid_dimensions(self):
        """Test that the 9-room grid has 19x19 dimensions."""
        env = NineRoomsEnv()

        assert env.world.grid.width_cells == 19
        assert env.world.grid.height_cells == 19

        env.close()

    def test_twelve_doorways(self):
        """Test that the environment has exactly 12 doorways."""
        config = DEFAULT_NINE_ROOMS_SCENARIO_CONFIG
        assert len(config.spawn_config.doorways) == 12

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
            doorways=DEFAULT_NINE_ROOMS_SCENARIO_CONFIG.spawn_config.doorways,
        )
        env = NineRoomsEnv(
            scenario_config=RoomsScenarioConfig(spawn_config=spawn_config)
        )
        env.reset(seed=123)

        exported = env.export_spawned_config()
        assert isinstance(exported, RoomsScenarioConfig)

        world_agent_pos = tuple(float(v) for v in env.world.agents[0].state.pos)
        world_goal_pos = tuple(float(v) for v in env.scenario.goal.state.pos)

        assert exported.spawn_config.agent == world_agent_pos
        assert exported.spawn_config.goal.pos == world_goal_pos

        output_path = tmp_path / "spawned_nine_rooms_config.yaml"
        saved = env.save_spawned_config(output_path)
        assert output_path.exists()

        loaded_data = yaml.safe_load(output_path.read_text())
        loaded = RoomsScenarioConfig.model_validate(loaded_data)
        assert loaded.model_dump() == saved.model_dump()

        env.close()

    def test_path_gaussian_spawn_nine_rooms(self):
        """Test PathGaussian obstacle spawning across all 9 rooms."""
        rooms = [
            "top_left",
            "top_center",
            "top_right",
            "middle_left",
            "middle_center",
            "middle_right",
            "bottom_left",
            "bottom_center",
            "bottom_right",
        ]
        lavas = [
            ObjConfig(pos=None, reward=-1.0, absorbing=False, room=r) for r in rooms
        ]
        holes = [
            ObjConfig(pos=None, reward=-1.0, absorbing=False, room=r) for r in rooms
        ]

        spawn_config = SpawnConfig(
            agent=None,
            goal=ObjConfig(pos=(15.0, 15.0), reward=1.0, absorbing=False),
            lavas=lavas,
            holes=holes,
            doorways=DEFAULT_NINE_ROOMS_SCENARIO_CONFIG.spawn_config.doorways,
            spawn_method=PathGaussianConfig(
                gaussian_std=0.5,
                min_spacing=0.8,
                edge_buffer=0.05,
                include_agent_paths=True,
            ),
            agent_size=0.1,
            goal_size=0.4,
            lava_size=0.4,
            hole_size=0.4,
        )
        config = RoomsScenarioConfig(spawn_config=spawn_config)
        env = NineRoomsEnv(scenario_config=config)
        env.reset(seed=42)

        topology = RoomTopology.nine_rooms(config.spawn_config.doorways)
        assert len(env.scenario.lava_pos) == 9
        assert len(env.scenario.hole_pos) == 9

        for i, lava_pos in enumerate(env.scenario.lava_pos):
            expected_room = rooms[i]
            actual_room = topology.get_room(lava_pos)
            assert actual_room == expected_room, (
                f"Lava {i} in '{actual_room}', expected '{expected_room}'"
            )

        for i, hole_pos in enumerate(env.scenario.hole_pos):
            expected_room = rooms[i]
            actual_room = topology.get_room(hole_pos)
            assert actual_room == expected_room, (
                f"Hole {i} in '{actual_room}', expected '{expected_room}'"
            )

        env.close()

    def test_save_rendering(self):
        """Test rendering and saving an image of the 9-room environment."""
        output_dir = os.path.join("tests", "out")
        os.makedirs(output_dir, exist_ok=True)

        env = NineRoomsEnv()
        observation, info = env.reset(seed=42)
        rendered = env.render()

        save_path = os.path.join(output_dir, "nine_rooms_render.png")
        imageio.imwrite(save_path, rendered)
        print(f"Saved nine-rooms render to: {save_path}")

        assert os.path.exists(save_path)
        env.close()


class TestNineRoomTopology:
    """Test class for 9-room topology."""

    def test_nine_room_topology_creation(self):
        """Test RoomTopology.nine_rooms factory method."""
        doorways = DEFAULT_NINE_ROOMS_SCENARIO_CONFIG.spawn_config.doorways
        topology = RoomTopology.nine_rooms(doorways)

        assert len(topology.room_boundaries) == 9
        assert len(topology.neighbor_map) == 12

    def test_nine_room_get_room(self):
        """Test room identification for positions in each of the 9 rooms."""
        doorways = DEFAULT_NINE_ROOMS_SCENARIO_CONFIG.spawn_config.doorways
        topology = RoomTopology.nine_rooms(doorways)

        assert topology.get_room((3.0, 15.0)) == "top_left"
        assert topology.get_room((9.0, 15.0)) == "top_center"
        assert topology.get_room((15.0, 15.0)) == "top_right"
        assert topology.get_room((3.0, 9.0)) == "middle_left"
        assert topology.get_room((9.0, 9.0)) == "middle_center"
        assert topology.get_room((15.0, 9.0)) == "middle_right"
        assert topology.get_room((3.0, 3.0)) == "bottom_left"
        assert topology.get_room((9.0, 3.0)) == "bottom_center"
        assert topology.get_room((15.0, 3.0)) == "bottom_right"

    def test_four_room_topology_backward_compat(self):
        """Test that default RoomTopology constructor still works for 4-room layout."""
        doorways = {"ld": (2, 6), "td": (6, 9), "rd": (9, 5), "bd": (6, 2)}
        topology = RoomTopology(doorways)

        assert len(topology.room_boundaries) == 4
        assert "top_left" in topology.room_boundaries
        assert "ld" in topology.neighbor_map

    def test_neighbor_pairs(self):
        """Test that get_neighbor_pairs returns correct pairs for 9-room layout."""
        doorways = DEFAULT_NINE_ROOMS_SCENARIO_CONFIG.spawn_config.doorways
        topology = RoomTopology.nine_rooms(doorways)

        pairs = topology.get_neighbor_pairs()
        assert len(pairs) > 0
        for d1, d2 in pairs:
            assert d1 != d2
            assert d1 in topology.neighbor_map or d2 in topology.neighbor_map


if __name__ == "__main__":
    import sys

    pytest.main([__file__] + sys.argv[1:])
