from __future__ import annotations

from typing import cast

import numpy as np
from absl.testing import absltest, parameterized

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
    def test_doorway_distances_contains_twelve_doorways(self) -> None:
        env = NineRoomsEnv()
        self.addCleanup(env.close)
        env.reset(seed=42)

        _, _, _, _, info = env.step(np.array([0.0, 0.0], dtype=np.float64))
        distances = info.get("distances", {})
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
        self.assertTrue(expected_doorways.issubset(distances.keys()))
        self.assertTrue(all(distances[d] >= 0 for d in expected_doorways))

    def test_path_gaussian_spawn_nine_rooms(self) -> None:
        lavas = [
            ObjConfig(pos=None, reward=-1.0, absorbing=False, room=r)
            for r in _NINE_ROOM_NAMES
        ]
        spawn_config = SpawnConfig(
            agent=None,
            goal=ObjConfig(pos=(15.0, 15.0), reward=1.0, absorbing=False),
            lavas=lavas,
            holes=[],
            doorways=DEFAULT_NINE_ROOMS_SCENARIO_CONFIG.spawn_config.doorways,
            spawn_method=PathGaussianConfig(
                gaussian_std=0.5,
                min_spacing=0.8,
                edge_buffer=0.05,
                include_agent_paths=True,
            ),
        )
        env = NineRoomsEnv(
            scenario_config=RoomsScenarioConfig(spawn_config=spawn_config)
        )
        self.addCleanup(env.close)
        env.reset(seed=42)

        topology = RoomTopology.nine_rooms(spawn_config.doorways)
        scenario = cast(RoomsScenario, env.scenario)
        actual_rooms = [topology.get_room(pos) for pos in scenario.lava_pos]
        self.assertEqual(actual_rooms, _NINE_ROOM_NAMES)


class TestNineRoomTopology(parameterized.TestCase):
    def test_topology_structure(self) -> None:
        doorways = DEFAULT_NINE_ROOMS_SCENARIO_CONFIG.spawn_config.doorways
        topology = RoomTopology.nine_rooms(doorways)
        self.assertLen(topology.room_boundaries, 9)
        self.assertLen(topology.neighbor_map, 12)

    @parameterized.named_parameters(
        ("tl", (3.0, 15.0), "top_left"),
        ("tc", (9.0, 15.0), "top_center"),
        ("tr", (15.0, 15.0), "top_right"),
        ("ml", (3.0, 9.0), "middle_left"),
        ("mc", (9.0, 9.0), "middle_center"),
        ("mr", (15.0, 9.0), "middle_right"),
        ("bl", (3.0, 3.0), "bottom_left"),
        ("bc", (9.0, 3.0), "bottom_center"),
        ("br", (15.0, 3.0), "bottom_right"),
    )
    def test_nine_room_get_room(
        self, pos: tuple[float, float], expected_room: str
    ) -> None:
        doorways = DEFAULT_NINE_ROOMS_SCENARIO_CONFIG.spawn_config.doorways
        topology = RoomTopology.nine_rooms(doorways)
        self.assertEqual(topology.get_room(pos), expected_room)


if __name__ == "__main__":
    absltest.main()
