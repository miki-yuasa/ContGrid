"""Configuration models for the Prey-Predator (prey_pred) environment."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from contgrid.core import ActionModeConfig, Grid, WorldConfig
from contgrid.core.typing import Position

_DEFAULT_15X15_LAYOUT: list[str] = [
    "###############",
    "#             #",
    "#             #",
    "#             #",
    "#             #",
    "#             #",
    "#             #",
    "#             #",
    "#             #",
    "#             #",
    "#             #",
    "#             #",
    "#             #",
    "#             #",
    "###############",
]


class StationaryTrajectoryConfig(BaseModel):
    """Configuration for a stationary entity."""

    type: Literal["stationary"] = "stationary"


class LinearPatrolTrajectoryConfig(BaseModel):
    """Configuration for an oscillating linear patrol trajectory."""

    type: Literal["linear_patrol"] = "linear_patrol"
    axis: Literal["horizontal", "vertical", "diagonal", "reverse_diagonal"] = (
        "horizontal"
    )
    speed: float = 0.75
    amplitude: float = 1.2
    phase: float = 0.0


class CircularOrbitTrajectoryConfig(BaseModel):
    """Configuration for a circular orbit trajectory."""

    type: Literal["circular_orbit"] = "circular_orbit"
    orbit_center: Position | Literal["region_center", "partner_entity"] = (
        "region_center"
    )
    radius: float = 1.0
    speed: float = 0.75
    clockwise: bool = False
    start_angle: float = 0.0


class WaypointPatrolTrajectoryConfig(BaseModel):
    """Configuration for a closed-loop waypoint patrol trajectory."""

    type: Literal["waypoint_patrol"] = "waypoint_patrol"
    waypoints: list[Position] = Field(default_factory=list)
    speed: float = 0.75
    loop: bool = True


class LissajousTrajectoryConfig(BaseModel):
    """Configuration for a figure-8 / harmonic curve patrol."""

    type: Literal["lissajous"] = "lissajous"
    freq_x: float = 1.0
    freq_y: float = 2.0
    amp_x: float = 1.2
    amp_y: float = 1.0
    phase_delta: float = 1.5707963267948966  # pi / 2
    speed: float = 0.75


TrajectoryConfig = Annotated[
    StationaryTrajectoryConfig
    | LinearPatrolTrajectoryConfig
    | CircularOrbitTrajectoryConfig
    | WaypointPatrolTrajectoryConfig
    | LissajousTrajectoryConfig,
    Field(discriminator="type"),
]


class EntityPlacementConfig(BaseModel):
    """Placement and movement specification for a single prey or predator."""

    spawn_pos: Position | Literal["center", "random"] = "random"
    trajectory: TrajectoryConfig = Field(
        default_factory=StationaryTrajectoryConfig
    )
    size: float = 0.25


class RegionEntityConfig(BaseModel):
    """Configuration of entities inside a 4x4 region."""

    region_id: int
    bounds: tuple[float, float, float, float]
    prey: EntityPlacementConfig
    predator: EntityPlacementConfig
    min_clearance: float = 0.2


def _get_default_regions() -> list[RegionEntityConfig]:
    """Build the standard default 4-region configurations."""
    return [
        RegionEntityConfig(
            region_id=0,
            bounds=(1.5, 5.5, 1.5, 5.5),
            prey=EntityPlacementConfig(
                spawn_pos=(2.5, 2.5),
                trajectory=StationaryTrajectoryConfig(),
                size=0.25,
            ),
            predator=EntityPlacementConfig(
                spawn_pos=(4.5, 4.5),
                trajectory=LinearPatrolTrajectoryConfig(
                    axis="reverse_diagonal", speed=1, amplitude=2.0
                ),
                size=0.25,
            ),
            min_clearance=0.2,
        ),
        RegionEntityConfig(
            region_id=1,
            bounds=(8.5, 12.5, 1.5, 5.5),
            prey=EntityPlacementConfig(
                spawn_pos="center",
                trajectory=StationaryTrajectoryConfig(),
                size=0.25,
            ),
            predator=EntityPlacementConfig(
                spawn_pos=(10.5, 2.2),
                trajectory=CircularOrbitTrajectoryConfig(
                    orbit_center="region_center",
                    radius=1.5,
                    speed=0.75,
                    clockwise=False,
                ),
                size=0.25,
            ),
            min_clearance=0.2,
        ),
        RegionEntityConfig(
            region_id=2,
            bounds=(1.5, 5.5, 8.5, 12.5),
            prey=EntityPlacementConfig(
                spawn_pos=(2.0, 9.0),
                trajectory=WaypointPatrolTrajectoryConfig(
                    waypoints=[
                        (2.0, 9.0),
                        (5.0, 9.0),
                        (5.0, 12.0),
                        (2.0, 12.0),
                    ],
                    speed=1.0,
                    loop=True,
                ),
                size=0.25,
            ),
            predator=EntityPlacementConfig(
                spawn_pos="center",
                trajectory=StationaryTrajectoryConfig(),
                size=0.25,
            ),
            min_clearance=0.2,
        ),
        RegionEntityConfig(
            region_id=3,
            bounds=(8.5, 12.5, 8.5, 12.5),
            prey=EntityPlacementConfig(
                spawn_pos=(11.4, 10.5),
                trajectory=StationaryTrajectoryConfig(),
                size=0.25,
            ),
            predator=EntityPlacementConfig(
                spawn_pos=(10.5, 9.6),
                trajectory=LissajousTrajectoryConfig(
                    freq_x=1.0,
                    freq_y=2.0,
                    amp_x=1.7,
                    amp_y=1.5,
                    speed=0.75,
                ),
                size=0.25,
            ),
            min_clearance=0.2,
        ),
    ]


class AgentSpawnConfig(BaseModel):
    """Configuration for agent spawning."""

    mode: Literal["neutral_corridor", "fixed", "random"] = "neutral_corridor"
    fixed_pos: Position | None = None
    min_clearance: float = 0.0
    perturbation: float = 0.25
    size: float = 0.1
    u_range: float = 5.0


class BaseTaskConfig(BaseModel):
    """Base configuration shared by all task modes."""

    step_penalty: float = 0.005
    predator_penalty: float = Field(default=-10.0, le=0.0)
    predator_absorbing: bool = True


class UnorderedTaskConfig(BaseTaskConfig):
    """Configuration for set-based prey capture where order does not matter."""

    type: Literal["unordered"] = "unordered"
    required_preys: list[int] = Field(default_factory=lambda: [0, 1, 2, 3])
    completion_reward: float = 100.0
    capture_rewards: dict[int, float] = Field(
        default_factory=lambda: {0: 10.0, 1: 10.0, 2: 10.0, 3: 10.0}
    )


class PreyPredSubtaskConfig(BaseModel):
    """Configuration for a single subtask step in an ordered sequence."""

    target_prey: int
    avoid_predators: list[int] = Field(default_factory=list)
    reward: float = 10.0
    penalty: float = Field(default=-1.0, le=0.0)
    prey_absorbing: bool = False
    predator_absorbing: bool = True


class OrderedTaskConfig(BaseTaskConfig):
    """Configuration for a strict sequential subtask progression."""

    type: Literal["ordered"] = "ordered"
    subtask_seq: list[PreyPredSubtaskConfig] = Field(default_factory=list)
    completion_reward: float = 100.0


TaskConfig = Annotated[
    UnorderedTaskConfig | OrderedTaskConfig,
    Field(discriminator="type"),
]


class PreyPredRenderConfig(BaseModel):
    """Configuration for PreyPred rendering options."""

    show_trajectories: bool = True
    show_hud: bool = False
    show_region_boxes: bool = True


class PreyPredScenarioConfig(BaseModel):
    """Configuration for the PreyPredScenario."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    agent_spawn: AgentSpawnConfig = Field(default_factory=AgentSpawnConfig)
    regions: list[RegionEntityConfig] = Field(
        default_factory=_get_default_regions
    )
    task: TaskConfig = Field(default_factory=UnorderedTaskConfig)
    render_options: PreyPredRenderConfig = Field(
        default_factory=PreyPredRenderConfig
    )


DEFAULT_WORLD_CONFIG = WorldConfig(grid=Grid(layout=_DEFAULT_15X15_LAYOUT))

DEFAULT_ACTION_CONFIG = ActionModeConfig(
    action_mode="discrete_ang_directional",
    action_mode_kwargs={"num_directions": 8, "num_vel_discrete": 6},
)

DEFAULT_SCENARIO_CONFIG = PreyPredScenarioConfig()
