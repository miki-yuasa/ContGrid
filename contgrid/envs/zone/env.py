"""RoomsEnv implementation."""

from pathlib import Path
from typing import Any, Literal

import yaml
from numpy.typing import NDArray
from pydantic import BaseModel

from contgrid.contgrid import DEFAULT_RENDER_CONFIG, BaseGymEnv, RenderConfig
from contgrid.core import ActionModeConfig, Grid, WorldConfig

from .scenario import ZoneScenario, ZoneScenarioConfig

DEFAULT_SCENARIO_CONFIG = ZoneScenarioConfig()
DEFAULT_WORLD_CONFIG = WorldConfig(
    grid=Grid(
        layout=[
            "#############",
            "#           #",
            "#           #",
            "#           #",
            "#           #",
            "#           #",
            "#           #",
            "#           #",
            "#           #",
            "#           #",
            "#           #",
            "#############",
        ]
    )
)

DEFAULT_ACTION_CONFIG = ActionModeConfig(
    action_mode="discrete_ang_directional",
    action_mode_kwargs={"num_directions": 8, "num_vel_discrete": 6},
)


class ZoneEnvConfig(BaseModel):
    """Configuration for the ZoneEnv."""

    scenario_config: ZoneScenarioConfig = DEFAULT_SCENARIO_CONFIG
    action_config: ActionModeConfig = DEFAULT_ACTION_CONFIG
    world_config: WorldConfig = DEFAULT_WORLD_CONFIG
    render_config: RenderConfig = DEFAULT_RENDER_CONFIG


class ZoneEnv(BaseGymEnv[dict[str, NDArray], NDArray, ZoneScenarioConfig]):
    """
    Continuous multi-zone navigation environment.

    The agent moves in a walled 2D world that contains color-coded zones
    (yellow, red, white, black). Tasks are defined as an ordered sequence in
    ``spawn_config.subtask_seq`` where each subtask specifies:
    - a goal zone that grants reward when visited,
    - an optional obstacle zone that applies penalty when visited,
    - whether goal/obstacle visits are absorbing.

    Observation:
        A dict containing agent kinematics, wall distances, relative zone
        positions (``yellow_dist``, ``red_dist``, ``white_dist``, ``black_dist``),
        and zone visitation counts (``zone_visits``).

    Actions:
        Controlled by ``action_config``. The default mode is 2D continuous motion.

    Reward:
        - Step penalty while the agent remains active.
        - Subtask goal reward on entering the active goal zone.
        - Subtask obstacle penalty on entering the active obstacle zone.

    Episode Termination:
        - Agent enters an absorbing goal/obstacle zone.
        - Max episode steps reached.
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
