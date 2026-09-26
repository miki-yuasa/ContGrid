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
    "DEFAULT_RENDER_CONFIG",
    "DEFAULT_WORLD_CONFIG",
    "ActionOption",
    "Agent",
    "AgentState",
    "BaseEnv",
    "BaseGymEnv",
    "BaseScenario",
    "Color",
    "Entity",
    "EntityShape",
    "EnvConfig",
    "Grid",
    "Landmark",
    "RenderConfig",
    "ScenarioConfigT",
    "World",
    "WorldConfig",
]

# Register custom gymnasium environments

### Rooms Environment ###
gym.register(
    id="contgrid/Rooms-v0",
    entry_point="contgrid.envs.rooms:RoomsEnv",
    max_episode_steps=250,
)
gym.register(
    id="contgrid/NineRooms-v0",
    entry_point="contgrid.envs.rooms:NineRoomsEnv",
    max_episode_steps=500,
)


### Zone Environment ###
gym.register(
    id="contgrid/Zone-v0",
    entry_point="contgrid.envs.zone:ZoneEnv",
    max_episode_steps=250,
)


### Prey-Predator Environment ###
gym.register(
    id="contgrid/PreyPred-v0",
    entry_point="contgrid.envs.prey_pred:PreyPredEnv",
    max_episode_steps=300,
)
