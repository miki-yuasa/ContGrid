import gymnasium as gym

from .contgrid import (
    DEFAULT_RENDER_CONFIG,
    ActionOption,
    BaseEnv,
    BaseGymEnv,
    EnvConfig,
    RenderConfig,
)
from .core import (
    DEFAULT_WORLD_CONFIG,
    Agent,
    AgentState,
    BaseScenario,
    Color,
    Entity,
    EntityShape,
    Grid,
    Landmark,
    ScenarioConfigT,
    World,
    WorldConfig,
)

__all__ = [
    "ActionOption",
    "Agent",
    "AgentState",
    "BaseEnv",
    "BaseGymEnv",
    "BaseScenario",
    "Color",
    "DEFAULT_RENDER_CONFIG",
    "DEFAULT_WORLD_CONFIG",
    "Entity",
    "EntityShape",
    "EnvConfig",
    "Grid",
    "Landmark",
    "RenderConfig",
    "ScenarioConfigT",
    "World",
    "WorldConfig",
    "DEFAULT_WORLD_CONFIG",
]

# Register custom gymnasium environments

### Rooms Environment ###
gym.register(
    id="contgrid/Rooms-v0",
    entry_point="contgrid.envs.rooms:RoomsEnv",
    max_episode_steps=500,
)
