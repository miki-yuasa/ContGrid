"""Configuration classes for the Rooms environment."""

from enum import Enum

from pydantic import BaseModel, Field

from contgrid.core.typing import Position

from .spawn_strategies import FixedSpawnConfig, SpawnMethodConfig


class RewardConfig(BaseModel):
    """Reward structure configuration."""

    step_penalty: float = 0.01
    sum_reward: bool = True


class ObjConfig(BaseModel):
    """Configuration for a single object (goal, lava, or hole)."""

    pos: Position | list[Position] | None = None


class ZoneType(str, Enum):
    """Enumeration of zone types."""

    YELLOW = "yellow"
    RED = "red"
    BLUE = "blue"
    BLACK = "black"


class SubtaskConfig(BaseModel):
    """Configuration for a single subtask (zone)."""

    goal: ZoneType
    obstacle: ZoneType | None = None
    reward: float = 0.0
    penalty: float = 0.0
    goal_absorbing: bool = False
    obstacle_absorbing: bool = False


class SpawnConfig(BaseModel):
    """
    Configuration for spawning objects in the environment.

    Attributes
    ----------
    agent: Position | None
        The position of the agent or None for random spawning.
    goal: ObjConfig
        The configuration for the goal object.
    lavas: list[ObjConfig]
        A list of configurations for lava objects.
    holes: list[ObjConfig]
        A list of configurations for hole objects.
    doorways: dict[str, Position]
        A dictionary mapping doorway names to their positions.
    """

    agent: Position | None = None
    subtask_seq: list[SubtaskConfig] = Field(
        default_factory=lambda: [
            SubtaskConfig(
                goal=ZoneType.YELLOW,
                obstacle=ZoneType.BLUE,
                reward=50.0,
                penalty=-1.0,
                goal_absorbing=False,
                obstacle_absorbing=True,
            ),
            SubtaskConfig(
                goal=ZoneType.RED,
                obstacle=ZoneType.BLUE,
                reward=50.0,
                penalty=-1.0,
                goal_absorbing=False,
                obstacle_absorbing=True,
            ),
        ]
    )
    yellow_zone: list[ObjConfig] = Field(default_factory=lambda: [ObjConfig(pos=None)])
    red_zone: list[ObjConfig] = Field(default_factory=lambda: [ObjConfig(pos=None)])
    blue_zone: list[ObjConfig] = Field(default_factory=lambda: [ObjConfig(pos=None)])
    black_zone: list[ObjConfig] = Field(default_factory=lambda: [ObjConfig(pos=None)])
    agent_size: float = 0.1
    zone_size: float = 0.35
    zone_thr_dist: float | None = None
    agent_u_range: float = 5.0
    spawn_method: SpawnMethodConfig = FixedSpawnConfig()

    model_config = {"arbitrary_types_allowed": True}


class ZoneScenarioConfig(BaseModel):
    """Configuration for the Rooms scenario."""

    spawn_config: SpawnConfig = SpawnConfig()
    reward_config: RewardConfig = RewardConfig(step_penalty=0.01)
