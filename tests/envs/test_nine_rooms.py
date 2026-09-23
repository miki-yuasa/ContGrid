from __future__ import annotations

from typing import cast

from absl.testing import absltest
from gymnasium import spaces

from contgrid.envs.rooms import (
    DEFAULT_NINE_ROOMS_SCENARIO_CONFIG,
    NineRoomsEnv,
    ObjConfig,
    PathGaussianConfig,
    RoomsScenario,
    RoomsScenarioConfig,
    RoomTopology,
    SpawnConfig,
)

_NINE_ROOM_NAMES = [
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


class TestNineRoomsEnv(absltest.TestCase):
    def test_env_initialization(self) -> None:
        env = NineRoomsEnv()
        self.addCleanup(env.close)

        world = env.world
        self.assertLen(world.agents, 1)
        self.assertEqual(world.agents[0].name, "agent_0")

        default_config = DEFAULT_NINE_ROOMS_SCENARIO_CONFIG
        expected_landmarks = (
            1
            + len(default_config.spawn_config.lavas)
            + len(default_config.spawn_config.holes)
        )
        self.assertEqual(expected_landmarks, 37)
        self.assertLen(world.landmarks, expected_landmarks)

    def test_nine_room_grid_dimensions(self) -> None:
        env = NineRoomsEnv()
        self.addCleanup(env.close)

        self.assertEqual(env.world.grid.width_cells, 19)
        self.assertEqual(env.world.grid.height_cells, 19)

    def test_twelve_doorways(self) -> None:
        config = DEFAULT_NINE_ROOMS_SCENARIO_CONFIG
        self.assertLen(config.spawn_config.doorways, 12)

    def test_observation_space_doorways(self) -> None:
        env = NineRoomsEnv()
        self.addCleanup(env.close)

        agent_name = env.possible_agents[0]
        obs_space = env.observation_spaces[agent_name]

        self.assertIsInstance(obs_space, spaces.Dict)
        assert isinstance(obs_space, spaces.Dict)
        self.assertIsInstance(obs_space["doorway_dist"], spaces.Box)
        self.assertEqual(obs_space["doorway_dist"].shape, (12,))
        self.assertIsInstance(obs_space["doorway_pos"], spaces.Box)
        self.assertEqual(obs_space["doorway_pos"].shape, (12, 2))

    def test_scenario_doorway_distances(self) -> None:
        env = NineRoomsEnv()
        self.addCleanup(env.close)
        env.reset(seed=42)

        import numpy as np

        action = np.array([0.0, 0.0], dtype=np.float64)
        _, _, _, _, info = env.step(action)

        self.assertIn("distances", info)
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
            self.assertIn(doorway, distances)
            self.assertIsInstance(distances[doorway], (int, float))
            self.assertGreaterEqual(distances[doorway], 0)

    def test_path_gaussian_spawn_nine_rooms(self) -> None:
        lavas = [
            ObjConfig(pos=None, reward=-1.0, absorbing=False, room=r)
            for r in _NINE_ROOM_NAMES + _NINE_ROOM_NAMES
        ]
        holes = [
            ObjConfig(pos=None, reward=-1.0, absorbing=False, room=r)
            for r in _NINE_ROOM_NAMES + _NINE_ROOM_NAMES
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
        self.addCleanup(env.close)
        env.reset(seed=42)

        topology = RoomTopology.nine_rooms(config.spawn_config.doorways)
        scenario = cast(RoomsScenario, env.scenario)
        self.assertLen(scenario.lava_pos, 18)
        self.assertLen(scenario.hole_pos, 18)

        expected_rooms = _NINE_ROOM_NAMES + _NINE_ROOM_NAMES
        for i, lava_pos in enumerate(scenario.lava_pos):
            expected_room = expected_rooms[i]
            actual_room = topology.get_room(lava_pos)
            self.assertEqual(
                actual_room,
                expected_room,
                msg=f"Lava {i} in '{actual_room}', expected '{expected_room}'",
            )

        for i, hole_pos in enumerate(scenario.hole_pos):
            expected_room = expected_rooms[i]
            actual_room = topology.get_room(hole_pos)
            self.assertEqual(
                actual_room,
                expected_room,
                msg=f"Hole {i} in '{actual_room}', expected '{expected_room}'",
            )


class TestNineRoomTopology(absltest.TestCase):
    def test_nine_room_topology_creation(self) -> None:
        doorways = DEFAULT_NINE_ROOMS_SCENARIO_CONFIG.spawn_config.doorways
        topology = RoomTopology.nine_rooms(doorways)

        self.assertLen(topology.room_boundaries, 9)
        self.assertLen(topology.neighbor_map, 12)

    def test_nine_room_get_room(self) -> None:
        doorways = DEFAULT_NINE_ROOMS_SCENARIO_CONFIG.spawn_config.doorways
        topology = RoomTopology.nine_rooms(doorways)

        expected_positions = {
            "top_left": (3.0, 15.0),
            "top_center": (9.0, 15.0),
            "top_right": (15.0, 15.0),
            "middle_left": (3.0, 9.0),
            "middle_center": (9.0, 9.0),
            "middle_right": (15.0, 9.0),
            "bottom_left": (3.0, 3.0),
            "bottom_center": (9.0, 3.0),
            "bottom_right": (15.0, 3.0),
        }
        for expected_room, pos in expected_positions.items():
            self.assertEqual(topology.get_room(pos), expected_room)

    def test_neighbor_pairs(self) -> None:
        doorways = DEFAULT_NINE_ROOMS_SCENARIO_CONFIG.spawn_config.doorways
        topology = RoomTopology.nine_rooms(doorways)

        pairs = topology.get_neighbor_pairs()
        self.assertNotEmpty(pairs)
        for d1, d2 in pairs:
            self.assertNotEqual(d1, d2)
            self.assertTrue(
                d1 in topology.neighbor_map or d2 in topology.neighbor_map
            )


if __name__ == "__main__":
    absltest.main()
