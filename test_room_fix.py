#!/usr/bin/env python
"""Quick test to verify room constraint fix"""
import numpy as np
from contgrid.envs.rooms import (
    RoomsEnv,
    RoomsScenarioConfig,
    SpawnConfig,
    ObjConfig,
    PathGaussianConfig,
    RoomTopology,
)

spawn_config = SpawnConfig(
    agent=(3.0, 3.0),
    goal=ObjConfig(pos=(9, 8), reward=1.0, absorbing=False),
    lavas=[
        ObjConfig(pos=None, reward=-1.0, absorbing=False, room="top_left"),
        ObjConfig(pos=None, reward=-1.0, absorbing=False, room="top_right"),
        ObjConfig(pos=None, reward=-1.0, absorbing=False, room="bottom_left"),
        ObjConfig(pos=None, reward=-1.0, absorbing=False, room="bottom_right"),
    ],
    holes=[],
    spawn_method=PathGaussianConfig(
        gaussian_std=0.6,
        min_spacing=0.9,
        edge_buffer=0.4,
        include_agent_paths=True,
    ),
    goal_size=0.4,
    lava_size=0.4,
    hole_size=0.4,
)
config = RoomsScenarioConfig(spawn_config=spawn_config)
env = RoomsEnv(scenario_config=config)

print("Testing room constraints with 3 different seeds...")
for seed in [42, 43, 44]:
    print(f"\n=== Seed {seed} ===")
    observation, info = env.reset(seed=seed)
    
    topology = RoomTopology(config.spawn_config.doorways)
    lava_positions = observation["agent_pos"] + observation["lava_pos"]
    
    all_match = True
    for i, (lava_pos, lava_config) in enumerate(
        zip(lava_positions, config.spawn_config.lavas)
    ):
        actual_room = topology.get_room(lava_pos)
        expected_room = lava_config.room
        match = actual_room == expected_room
        all_match = all_match and match
        print(
            f"  Lava {i}: pos={lava_pos}, expected={expected_room}, actual={actual_room}, {'✓' if match else '✗ MISMATCH'}"
        )
    
    if all_match:
        print("  ✓ All obstacles in correct rooms!")
    else:
        print("  ✗ Some obstacles in wrong rooms!")

env.close()
print("\nTest complete!")
