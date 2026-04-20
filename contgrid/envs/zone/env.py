"""RoomsEnv implementation."""

from pathlib import Path
from typing import Any, Literal

import yaml
from numpy.typing import NDArray
from pydantic import BaseModel

from contgrid.contgrid import DEFAULT_RENDER_CONFIG, BaseGymEnv, RenderConfig
from contgrid.core import DEFAULT_ACTION_CONFIG, ActionModeConfig, Grid, WorldConfig

from .scenario import ZoneScenario, ZoneScenarioConfig

DEFAULT_SCENARIO_CONFIG = ZoneScenarioConfig()
DEFAULT_WORLD_CONFIG = WorldConfig(
    grid=Grid(
        layout=[
            "#############",
            "#     #     #",
            "#     #     #",
            "#           #",
            "#     #     #",
            "#     #     #",
            "## ####     #",
            "#     ### ###",
            "#     #     #",
            "#     #     #",
            "#           #",
            "#     #     #",
            "#############",
        ]
    )
)


class ZoneEnvConfig(BaseModel):
    """Configuration for the ZoneEnv."""

    scenario_config: ZoneScenarioConfig = DEFAULT_SCENARIO_CONFIG
    action_config: ActionModeConfig = DEFAULT_ACTION_CONFIG
    world_config: WorldConfig = DEFAULT_WORLD_CONFIG
    render_config: RenderConfig = DEFAULT_RENDER_CONFIG


class ZoneEnv(BaseGymEnv[dict[str, NDArray], NDArray, ZoneScenarioConfig]):
    """
    Continuous Grid World with Zone Environment

    This environment is a continuous 2D grid world where an agent must navigate
    through zones to reach a goal while avoiding obstacles like lava and holes.

    Observation:
        Type: Dict
        {
            "agent_pos": Box(2,)  # Agent's position (x, y)
            "goal_pos": Box(2,)   # Goal's position (x, y)
            "lava_pos": Box(2 * num_lavas,)  # Positions of lava objects
            "hole_pos": Box(2 * num_holes,)  # Positions of hole objects
            "goal_dist": Box(1,)  # Distance to the goal
            "lava_dist": Box(1,)  # Distance to the closest lava
            "hole_dist": Box(1,)  # Distance to the closest hole
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
        scenario_config: ZoneScenarioConfig = DEFAULT_SCENARIO_CONFIG,
        world_config: WorldConfig = DEFAULT_WORLD_CONFIG,
        render_config: RenderConfig = DEFAULT_RENDER_CONFIG,
        render_mode: str | None = None,
        action_config: ActionModeConfig = DEFAULT_ACTION_CONFIG,
        verbose: bool = False,
    ) -> None:
        if isinstance(scenario_config, dict):
            scenario_config = ZoneScenarioConfig(**scenario_config)
        if isinstance(world_config, dict):
            world_config = WorldConfig(**world_config)
        if isinstance(render_config, dict):
            render_config = RenderConfig(**render_config)

        scenario = ZoneScenario(scenario_config, world_config)
        super().__init__(
            scenario,
            render_config=render_config,
            render_mode=render_mode,
            action_config=action_config,
            local_ratio=None,
            verbose=verbose,
        )

    def export_spawned_config(self) -> ZoneScenarioConfig:
        """Export current environment state as a ZoneScenarioConfig."""
        scenario = self.scenario

        return scenario.export_spawned_config(self.world)

    def save_spawned_config(
        self,
        output_path: str | Path,
        *,
        format: Literal["yaml", "json"] = "yaml",
        indent: int = 2,
    ) -> ZoneScenarioConfig:
        """Save current environment state as a ZoneScenarioConfig file."""
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
