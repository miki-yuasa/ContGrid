"""PreyPredEnv Gymnasium environment implementation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import yaml
from numpy.typing import NDArray
from pydantic import BaseModel

from contgrid.contgrid import DEFAULT_RENDER_CONFIG, BaseGymEnv, RenderConfig
from contgrid.core import ActionModeConfig, WorldConfig

from .configs import (
    DEFAULT_ACTION_CONFIG,
    DEFAULT_SCENARIO_CONFIG,
    DEFAULT_WORLD_CONFIG,
    PreyPredScenarioConfig,
)
from .renderer import PreyPredRenderer
from .scenario import PreyPredScenario


class PreyPredEnvConfig(BaseModel):
    """Overall environment configuration for PreyPredEnv."""

    scenario_config: PreyPredScenarioConfig = DEFAULT_SCENARIO_CONFIG
    action_config: ActionModeConfig = DEFAULT_ACTION_CONFIG
    world_config: WorldConfig = DEFAULT_WORLD_CONFIG
    render_config: RenderConfig = DEFAULT_RENDER_CONFIG


class PreyPredEnv(
    BaseGymEnv[dict[str, NDArray], NDArray, PreyPredScenarioConfig]
):
    """Continuous multi-region prey-predator navigation environment.

    The agent navigates a 15x15 continuous grid world containing four 4x4
    regions. Each region contains moving and stationary preys (goals) and
    predators (hazards).

    Tasks can be configured in unordered mode (capturing required preys in
    any order with per-prey milestone rewards) or ordered reach-avoid mode.

    Observation:
        Dict containing pure spatial-kinematic vectors:
        - `agent_pos`: (2,) absolute position
        - `agent_vel`: (2,) velocity vector
        - `wall_dist`: (4,) cardinal distances to blocking walls
        - `prey_rel_pos`: (4, 2) relative positions of preys
        - `prey_rel_vel`: (4, 2) relative velocities of preys
        - `predator_rel_pos`: (4, 2) relative positions of predators
        - `predator_rel_vel`: (4, 2) relative velocities of predators
        - `prey_capture_counts`: (4,) edge-triggered capture counts

    Actions:
        Configured via `action_config`. Defaults to `discrete_ang_directional`
        (8 radial directions, 6 discrete speeds).

    Reward:
        - Step penalty (-0.005 by default).
        - Milestone capture reward upon entering an uncaptured prey.
        - Predator penalty on touching a predator (absorbing by default).
        - Completion reward (+100.0) upon satisfying task objectives.
    """

    metadata = {  # noqa: RUF012
        "render_modes": ["human", "rgb_array"],
        "render_fps": 10,
    }

    def __init__(
        self,
        scenario_config: PreyPredScenarioConfig
        | Mapping[str, Any] = DEFAULT_SCENARIO_CONFIG,
        world_config: WorldConfig | Mapping[str, Any] = DEFAULT_WORLD_CONFIG,
        render_config: RenderConfig | Mapping[str, Any] = DEFAULT_RENDER_CONFIG,
        render_mode: str | None = None,
        action_config: ActionModeConfig | Mapping[str, Any] = DEFAULT_ACTION_CONFIG,
        verbose: bool = False,
    ) -> None:
        if isinstance(scenario_config, Mapping):
            scenario_config = PreyPredScenarioConfig(**scenario_config)
        if isinstance(world_config, Mapping):
            world_config = WorldConfig(**world_config)
        if isinstance(render_config, Mapping):
            render_config = RenderConfig(**render_config)
        if isinstance(action_config, Mapping):
            action_config = ActionModeConfig(**action_config)

        scenario = PreyPredScenario(scenario_config, world_config)

        super().__init__(
            scenario=scenario,
            render_config=render_config,
            render_mode=render_mode,
            action_config=action_config,
            local_ratio=None,
            verbose=verbose,
        )
        self.scenario: PreyPredScenario = scenario

        effective_rc = render_config
        if render_mode is not None:
            effective_rc = render_config.model_copy(
                update={"render_mode": render_mode}
            )
        self.env.renderers.append(PreyPredRenderer(scenario, effective_rc))

    def step(
        self, action: NDArray
    ) -> tuple[dict[str, NDArray], float, bool, bool, dict[str, Any]]:
        """Advance moving entities and perform one agent action step."""
        self.scenario.step_entities(self.world)
        self.env.step(action)

        cur_agent = self.env.agent_selection
        obs: dict[str, NDArray] = self.env.observe(cur_agent)
        reward = float(self.env.rewards[cur_agent])
        terminated = bool(self.env.terminations[cur_agent])
        truncated = bool(self.env.truncations[cur_agent])
        info: dict[str, Any] = self.env.infos[cur_agent]

        return obs, reward, terminated, truncated, info

    def export_spawned_config(self) -> PreyPredScenarioConfig:
        """Export current scenario configuration."""
        return self.scenario.export_spawned_config(self.world)

    def save_spawned_config(
        self,
        output_path: str | Path,
        *,
        format: Literal["yaml", "json"] = "yaml",
        indent: int = 2,
    ) -> PreyPredScenarioConfig:
        """Save current environment configuration to file."""
        config = self.export_spawned_config()
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        json_dump = config.model_dump_json(indent=indent)
        if format == "json":
            output.write_text(json_dump, encoding="utf-8")
        else:
            json_dict: dict[str, Any] = yaml.safe_load(json_dump)
            yaml_text = yaml.dump(json_dict, sort_keys=False, indent=indent)
            output.write_text(yaml_text, encoding="utf-8")
        return config
