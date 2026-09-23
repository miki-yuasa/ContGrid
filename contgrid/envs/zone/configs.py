from __future__ import annotations

from enum import Enum
from typing import Annotated, Any

from pydantic import BaseModel, BeforeValidator, Field, model_validator

from contgrid.core.typing import Position

from .spawn import SpawnMethodConfig, UniformRandomConfig


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
    WHITE = "white"
    BLACK = "black"


def _normalize_obstacle(v: Any) -> list[Any]:
    """Normalize single ZoneType, str, None, or sequence into a list."""
    if v is None:
        return []
    if isinstance(v, (ZoneType, str)):
        return [v]
    if isinstance(v, (list, tuple, set)):
        return list(dict.fromkeys(v))
    return [v]


class SubtaskConfig(BaseModel):
    """Configuration for a single subtask (zone).

    Attributes:
        goal: The target zone type that the agent must reach.
        obstacle: The obstacle zone(s) for this subtask (normalized to a list
            of ZoneType).
        reward: The reward given when the goal zone is reached.
        penalty: The non-positive penalty applied when an obstacle zone is entered.
        goal_absorbing: Whether entering the goal terminates the episode.
        obstacle_absorbing: Whether entering any obstacle terminates the episode.
    """

    goal: ZoneType
    obstacle: Annotated[
        list[ZoneType],
        BeforeValidator(_normalize_obstacle),
    ] = Field(default_factory=list)
    reward: float = 0.0
    penalty: float = Field(default=0.0, le=0.0)
    goal_absorbing: bool = False
    obstacle_absorbing: bool = False

    @model_validator(mode="after")
    def _validate_goal_obstacle_overlap(self) -> SubtaskConfig:
        """Ensure that the goal zone is not designated as an obstacle."""
        if self.goal in self.obstacle:
            raise ValueError(
                f"Goal zone '{self.goal.value}' cannot also be an obstacle zone."
            )
        return self


class ZoneSizeConfig(BaseModel):
    yellow: float = 0.5
    red: float = 0.5
    white: float = 0.5
    black: float = 0.5


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
                obstacle=ZoneType.WHITE,
                reward=50.0,
                penalty=-1.0,
                goal_absorbing=False,
                obstacle_absorbing=True,
            ),
            SubtaskConfig(
                goal=ZoneType.RED,
                obstacle=ZoneType.WHITE,
                reward=50.0,
                penalty=-1.0,
                goal_absorbing=False,
                obstacle_absorbing=True,
            ),
        ]
    )
    yellow_zone: list[ObjConfig] = Field(
        default_factory=lambda: [ObjConfig(pos=None)]
    )
    red_zone: list[ObjConfig] = Field(
        default_factory=lambda: [ObjConfig(pos=None)]
    )
    white_zone: list[ObjConfig] = Field(
        default_factory=lambda: [ObjConfig(pos=None)]
    )
    black_zone: list[ObjConfig] = Field(
        default_factory=lambda: [ObjConfig(pos=None)]
    )
    agent_size: float = 0.1
    agent_perturbation: float = 0.25
    zone_size: float | ZoneSizeConfig = 0.5
    zone_thr_dist: float | None = None
    agent_u_range: float = 5.0
    spawn_method: SpawnMethodConfig = UniformRandomConfig()
    reset_agent_first: bool = True

    model_config = {"arbitrary_types_allowed": True}


class ObsConfig(BaseModel):
    """Configuration for observations."""

    include_subtask: bool = False


class ZoneScenarioConfig(BaseModel):
    """Configuration for the Rooms scenario."""

    spawn_config: SpawnConfig = SpawnConfig()
    reward_config: RewardConfig = RewardConfig(step_penalty=0.01)
    obs_config: ObsConfig = ObsConfig()
