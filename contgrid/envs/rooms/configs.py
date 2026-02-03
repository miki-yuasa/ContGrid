"""Configuration classes for the Rooms environment."""

from typing import Literal

from pydantic import BaseModel

from contgrid.core.typing import Position

from .spawn_strategies import FixedSpawnConfig, SpawnMethodConfig


class RewardConfig(BaseModel):
    """Reward structure configuration."""

    step_penalty: float = 0.01
    sum_reward: bool = True


class ObjConfig(BaseModel):
    """Configuration for a single object (goal, lava, or hole)."""

    pos: Position | list[Position] | None = None
    reward: float = 0.0
    absorbing: bool = False
    room: Literal["top_left", "top_right", "bottom_left", "bottom_right"] | None = None


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
    goal: ObjConfig = ObjConfig(pos=(9, 8), reward=1.0, absorbing=False)
    lavas: list[ObjConfig] = [
        ObjConfig(pos=(7, 8), reward=0.0, absorbing=False),
        ObjConfig(pos=(9, 9), reward=0.0, absorbing=False),
        ObjConfig(pos=(5, 10), reward=0.0, absorbing=False),
        ObjConfig(pos=(3, 7), reward=0.0, absorbing=False),
        ObjConfig(pos=(2, 4), reward=-1.0, absorbing=False),
        ObjConfig(pos=(3, 5), reward=-1.0, absorbing=False),
        ObjConfig(pos=(10, 4), reward=-1.0, absorbing=False),
        ObjConfig(pos=(8, 3), reward=-1.0, absorbing=False),
    ]
    holes: list[ObjConfig] = [
        ObjConfig(pos=(8, 9), reward=0.0, absorbing=False),
        ObjConfig(pos=(9, 7), reward=0.0, absorbing=False),
        ObjConfig(pos=(5, 8), reward=-1.0, absorbing=False),
        ObjConfig(pos=(4, 9), reward=-1.0, absorbing=False),
        ObjConfig(pos=(1, 5), reward=0.0, absorbing=False),
        ObjConfig(pos=(5, 3), reward=0.0, absorbing=False),
        ObjConfig(pos=(7, 4), reward=-1.0, absorbing=False),
        ObjConfig(pos=(9, 2), reward=-1.0, absorbing=False),
    ]
    doorways: dict[str, Position] = {
        "ld": (2, 6),
        "td": (6, 9),
        "rd": (9, 5),
        "bd": (6, 2),
    }
    agent_size: float = 0.25
    goal_size: float = 0.5
    lava_size: float = 0.5
    hole_size: float = 0.5
    goal_thr_dist: float | None = None
    lava_thr_dist: float | None = None
    hole_thr_dist: float | None = None
    agent_u_range: float = 10.0
    spawn_method: SpawnMethodConfig = FixedSpawnConfig()

    model_config = {"arbitrary_types_allowed": True}


class ObservationConfig(BaseModel):
    """Configuration for observations in the Rooms scenario."""

    obs_dist: Literal["closest", "all", "none"] = "closest"
    goal_dist: Literal["all", "none"] = "all"
    doorway_dist: Literal["all", "none"] = "all"
    closest_obs_pos: bool = False


class RoomsScenarioConfig(BaseModel):
    """Configuration for the Rooms scenario."""

    spawn_config: SpawnConfig = SpawnConfig()
    reward_config: RewardConfig = RewardConfig(step_penalty=0.01)
    observation_config: ObservationConfig = ObservationConfig()
