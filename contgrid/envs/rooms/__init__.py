"""Rooms environment - a continuous grid world with multiple rooms and doorways."""

from .env import (
    DEFAULT_ROOMS_SCENARIO_CONFIG,
    DEFAULT_WORLD_CONFIG,
    RoomsEnv,
    RoomsEnvConfig,
)
from .scenario import (
    ObjConfig,
    RewardConfig,
    RoomsScenario,
    RoomsScenarioConfig,
    SpawnConfig,
)
from .spawn_strategies import (
    FixedSpawnConfig,
    FixedSpawnStrategy,
    PathGaussianConfig,
    PathGaussianSpawnStrategy,
    SpawnMode,
    SpawnStrategy,
    UniformRandomConfig,
    UniformRandomSpawnStrategy,
)
from .topology import LineSegment, RoomTopology, get_relevant_path_segments

__all__ = [
    # Config classes
    "FixedSpawnConfig",
    "ObjConfig",
    "PathGaussianConfig",
    "RewardConfig",
    "RoomsScenarioConfig",
    "SpawnConfig",
    "SpawnMode",
    "UniformRandomConfig",
    # Environment
    "RoomsEnv",
    "RoomsEnvConfig",
    "DEFAULT_ROOMS_SCENARIO_CONFIG",
    "DEFAULT_WORLD_CONFIG",
    # Scenario
    "RoomsScenario",
    # Spawn strategies
    "SpawnStrategy",
    "FixedSpawnStrategy",
    "PathGaussianSpawnStrategy",
    "UniformRandomSpawnStrategy",
    # Topology
    "LineSegment",
    "RoomTopology",
    "get_relevant_path_segments",
]
