"""NineRoomsEnv implementation — a 3×3 room variant of RoomsEnv."""

from pathlib import Path
from typing import Any, Literal

import yaml
from numpy.typing import NDArray
from pydantic import BaseModel

from contgrid.contgrid import DEFAULT_RENDER_CONFIG, BaseGymEnv, RenderConfig
from contgrid.core import DEFAULT_ACTION_CONFIG, ActionModeConfig, Grid, WorldConfig

from .configs import ObjConfig, RoomsScenarioConfig, SpawnConfig
from .scenario import RoomsScenario

# ── Default configs for 9-room layout ────────────────────────────────────────

DEFAULT_NINE_ROOMS_SCENARIO_CONFIG = RoomsScenarioConfig(
    spawn_config=SpawnConfig(
        agent=None,
        goal=ObjConfig(pos=(15, 15), reward=1.0, absorbing=False),
        lavas=[
            # 2 lavas per room (18 total)
            ObjConfig(pos=(3, 15), reward=-1.0, absorbing=False, room="top_left"),
            ObjConfig(pos=(4, 16), reward=-1.0, absorbing=False, room="top_left"),
            ObjConfig(pos=(9, 15), reward=-1.0, absorbing=False, room="top_center"),
            ObjConfig(pos=(8, 14), reward=0.0, absorbing=False, room="top_center"),
            ObjConfig(pos=(14, 14), reward=0.0, absorbing=False, room="top_right"),
            ObjConfig(pos=(16, 16), reward=-1.0, absorbing=False, room="top_right"),
            ObjConfig(pos=(3, 9), reward=-1.0, absorbing=False, room="middle_left"),
            ObjConfig(pos=(2, 8), reward=0.0, absorbing=False, room="middle_left"),
            ObjConfig(pos=(9, 9), reward=-1.0, absorbing=False, room="middle_center"),
            ObjConfig(pos=(10, 10), reward=-1.0, absorbing=False, room="middle_center"),
            ObjConfig(pos=(15, 9), reward=-1.0, absorbing=False, room="middle_right"),
            ObjConfig(pos=(14, 8), reward=0.0, absorbing=False, room="middle_right"),
            ObjConfig(pos=(3, 3), reward=-1.0, absorbing=False, room="bottom_left"),
            ObjConfig(pos=(2, 2), reward=0.0, absorbing=False, room="bottom_left"),
            ObjConfig(pos=(9, 3), reward=-1.0, absorbing=False, room="bottom_center"),
            ObjConfig(pos=(8, 2), reward=0.0, absorbing=False, room="bottom_center"),
            ObjConfig(pos=(15, 3), reward=-1.0, absorbing=False, room="bottom_right"),
            ObjConfig(pos=(14, 2), reward=0.0, absorbing=False, room="bottom_right"),
        ],
        holes=[
            # 2 holes per room (18 total)
            ObjConfig(pos=(4, 15), reward=-1.0, absorbing=False, room="top_left"),
            ObjConfig(pos=(3, 14), reward=0.0, absorbing=False, room="top_left"),
            ObjConfig(pos=(10, 15), reward=-1.0, absorbing=False, room="top_center"),
            ObjConfig(pos=(9, 14), reward=0.0, absorbing=False, room="top_center"),
            ObjConfig(pos=(14, 15), reward=-1.0, absorbing=False, room="top_right"),
            ObjConfig(pos=(15, 14), reward=0.0, absorbing=False, room="top_right"),
            ObjConfig(pos=(4, 9), reward=-1.0, absorbing=False, room="middle_left"),
            ObjConfig(pos=(3, 8), reward=0.0, absorbing=False, room="middle_left"),
            ObjConfig(pos=(8, 8), reward=-1.0, absorbing=False, room="middle_center"),
            ObjConfig(pos=(10, 8), reward=-1.0, absorbing=False, room="middle_center"),
            ObjConfig(pos=(14, 9), reward=-1.0, absorbing=False, room="middle_right"),
            ObjConfig(pos=(15, 8), reward=0.0, absorbing=False, room="middle_right"),
            ObjConfig(pos=(4, 3), reward=-1.0, absorbing=False, room="bottom_left"),
            ObjConfig(pos=(3, 2), reward=0.0, absorbing=False, room="bottom_left"),
            ObjConfig(pos=(10, 3), reward=-1.0, absorbing=False, room="bottom_center"),
            ObjConfig(pos=(9, 2), reward=0.0, absorbing=False, room="bottom_center"),
            ObjConfig(pos=(14, 3), reward=-1.0, absorbing=False, room="bottom_right"),
            ObjConfig(pos=(15, 2), reward=0.0, absorbing=False, room="bottom_right"),
        ],
        doorways={
            # Horizontal doorways (gaps in vertical walls at col 6 and col 12)
            "tl_tc": (6, 16),
            "tc_tr": (12, 14),
            "ml_mc": (6, 10),
            "mc_mr": (12, 8),
            "bl_bc": (6, 4),
            "bc_br": (12, 2),
            # Vertical doorways (gaps in horizontal walls at row 6 and row 12)
            "tl_ml": (2, 12),
            "tc_mc": (8, 12),
            "tr_mr": (16, 12),
            "ml_bl": (4, 6),
            "mc_bc": (10, 6),
            "mr_br": (14, 6),
        },
    )
)

DEFAULT_NINE_ROOMS_WORLD_CONFIG = WorldConfig(
    grid=Grid(
        layout=[
            "###################",
            "#     #     #     #",
            "#           #     #",
            "#     #     #     #",
            "#     #           #",
            "#     #     #     #",
            "## ##### ####### ##",
            "#     #     #     #",
            "#           #     #",
            "#     #     #     #",
            "#     #           #",
            "#     #     #     #",
            "#### ##### ### ####",
            "#     #     #     #",
            "#           #     #",
            "#     #     #     #",
            "#     #           #",
            "#     #     #     #",
            "###################",
        ]
    )
)


class NineRoomsEnvConfig(BaseModel):
    """Configuration for the NineRoomsEnv."""

    scenario_config: RoomsScenarioConfig = DEFAULT_NINE_ROOMS_SCENARIO_CONFIG
    action_config: ActionModeConfig = DEFAULT_ACTION_CONFIG
    world_config: WorldConfig = DEFAULT_NINE_ROOMS_WORLD_CONFIG
    render_config: RenderConfig = DEFAULT_RENDER_CONFIG


class NineRoomsEnv(BaseGymEnv[dict[str, NDArray], NDArray, RoomsScenarioConfig]):
    """
    Continuous Grid World with Nine Rooms Environment

    This environment is a continuous 2D grid world arranged as a 3×3 grid of
    rooms. An agent must navigate through rooms via doorways to reach a goal
    while avoiding obstacles like lava and holes.

    Room layout (3×3):
        top_left    | top_center    | top_right
        middle_left | middle_center | middle_right
        bottom_left | bottom_center | bottom_right

    12 doorways connect adjacent rooms (6 horizontal + 6 vertical).
    Doorways are staggered (not aligned) to create more interesting navigation.

    Observation:
        Type: Dict
        {
            "agent_pos": Box(2,)  # Agent's position (x, y)
            "agent_vel": Box(2,)  # Agent's velocity
            "goal_pos": Box(2,)   # Goal's position (x, y)
            "lava_pos": Box(2 * num_lavas,)  # Positions of lava objects
            "hole_pos": Box(2 * num_holes,)  # Positions of hole objects
            "doorway_pos": Box(2 * num_doorways,)  # Positions of doorways
            "doorway_dist": Box(num_doorways,)  # Distances to doorways
            "goal_dist": Box(1,)  # Distance to the goal
            "lava_dist": Box(1,)  # Distance to the closest lava
            "hole_dist": Box(1,)  # Distance to the closest hole
            "wall_dist": Box(4,)  # Distances to walls in cardinal directions
        }

    Actions:
        Type: Box(2,)
        Num     Action
        0       Move in x direction (-1.0 to 1.0)
        1       Move in y direction (-1.0 to 1.0)

    Reward:
        - Step penalty: -config.reward_config.step_penalty per step
        - Reaching the goal: +config.spawn_config.goal.reward
        - Falling into lava or hole: config.spawn_config.lavas[i].reward or
          config.spawn_config.holes[i].reward

    Starting State:
        - Agent starts at config.spawn_config.agent position or random free cell
        - Goal, lava, and holes are placed at their configured positions

    Episode Termination:
        - Agent reaches the goal
        - Agent falls into an absorbing lava or hole
        - Max episode steps reached
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        scenario_config: RoomsScenarioConfig = DEFAULT_NINE_ROOMS_SCENARIO_CONFIG,
        world_config: WorldConfig = DEFAULT_NINE_ROOMS_WORLD_CONFIG,
        render_config: RenderConfig = DEFAULT_RENDER_CONFIG,
        render_mode: str | None = None,
        action_config: ActionModeConfig = DEFAULT_ACTION_CONFIG,
        verbose: bool = False,
    ) -> None:
        if isinstance(scenario_config, dict):
            scenario_config = RoomsScenarioConfig(**scenario_config)
        if isinstance(world_config, dict):
            world_config = WorldConfig(**world_config)
        if isinstance(render_config, dict):
            render_config = RenderConfig(**render_config)

        scenario = RoomsScenario(scenario_config, world_config)
        super().__init__(
            scenario,
            render_config=render_config,
            render_mode=render_mode,
            action_config=action_config,
            local_ratio=None,
            verbose=verbose,
        )

    def export_spawned_config(self) -> RoomsScenarioConfig:
        """Export current environment state as a RoomsScenarioConfig."""
        scenario = self.scenario

        return scenario.export_spawned_config(self.world)

    def save_spawned_config(
        self,
        output_path: str | Path,
        *,
        format: Literal["yaml", "json"] = "yaml",
        indent: int = 2,
    ) -> RoomsScenarioConfig:
        """Save current environment state as a RoomsScenarioConfig file."""
        config = self.export_spawned_config()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        json_dump: str = config.model_dump_json(indent=indent)
        if format == "json":
            output.write_text(json_dump, encoding="utf-8")
        else:
            # Use JSON dump to preserve field order, then convert to YAML
            json_dict: dict[str, Any] = yaml.safe_load(json_dump)
            yaml_text = yaml.dump(json_dict, sort_keys=False, indent=indent)
            output.write_text(yaml_text, encoding="utf-8")
        return config
