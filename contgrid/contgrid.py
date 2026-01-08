from enum import StrEnum
from typing import Any, Generic, Literal, SupportsFloat

import gymnasium
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from gymnasium import Env, spaces
from gymnasium.core import ActType, ObsType
from gymnasium.utils import seeding
from matplotlib.backends.backend_agg import FigureCanvasAgg
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict

from contgrid.core.action import (
    DEFAULT_ACTION_CONFIG,
    ActionMode,
    ActionModeConfig,
    ContinuousFullVelocity,
    ContinuousMinimalVelocity,
    DiscreteAngDirectional,
    DiscreteDirectionVelocity,
    DiscreteMinimalVelocity,
)
from contgrid.core.const import ALPHABET
from contgrid.core.entities import Entity, EntityShape
from contgrid.core.grid import Grid
from contgrid.core.render import (
    DEFAULT_RENDER_CONFIG,
    EnvRenderer,
    RenderConfig,
    Renderer,
)
from contgrid.core.scenario import BaseScenario, ScenarioConfigT
from contgrid.core.typing import Position
from contgrid.core.utils import AgentSelector
from contgrid.core.world import Agent, World


class ActionOption(StrEnum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    CONTINUOUS_MINIMAL = "continuous-minimal"


class EnvConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario: BaseScenario
    max_cycles: int = 100
    render_config: RenderConfig = DEFAULT_RENDER_CONFIG
    action_opt: str = ActionOption.CONTINUOUS_MINIMAL.value
    local_ratio: float | None = None
    verbose: bool = False


class BaseEnv(Generic[ObsType, ActType, ScenarioConfigT]):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 10,
    }

    native_action_modes: dict[str, type[ActionMode]] = {
        ContinuousMinimalVelocity.name: ContinuousMinimalVelocity,
        ContinuousFullVelocity.name: ContinuousFullVelocity,
        DiscreteMinimalVelocity.name: DiscreteMinimalVelocity,
        DiscreteDirectionVelocity.name: DiscreteDirectionVelocity,
        DiscreteAngDirectional.name: DiscreteAngDirectional,
    }

    def __init__(
        self,
        scenario: BaseScenario[ScenarioConfigT, ObsType],
        max_cycles: int | None = None,
        render_config: RenderConfig = DEFAULT_RENDER_CONFIG,
        custom_renderers: list[Renderer] | None = None,
        action_config: ActionModeConfig = DEFAULT_ACTION_CONFIG,
        local_ratio: float | None = None,
        verbose: bool = False,
    ):
        super().__init__()

        action_mode: str | type[ActionMode] = action_config.action_mode
        action_mode_kwargs: dict[str, Any] = action_config.action_mode_kwargs
        match action_mode:
            case str():
                if action_mode not in self.native_action_modes:
                    raise ValueError(f"Unknown action mode {action_mode}")
                self.action_mode: ActionMode = self.native_action_modes[action_mode](
                    **action_mode_kwargs
                )
            case _:
                self.action_mode = action_mode(**action_mode_kwargs)

        self.render_config: RenderConfig = render_config
        self.render_mode: Literal["human", "rgb_array"] = render_config.render_mode
        self.viewer = None
        self.world: World = scenario.make_world(verbose)
        self.grid: Grid = self.world.grid
        self.width = render_config.width_px
        self.height = render_config.height_px
        self.dpi = render_config.dpi
        self.fig = None
        self.ax = None
        self.max_size = 1
        self.verbose: bool = verbose
        self.renderers: list[Renderer] = [EnvRenderer(render_config)] + (
            custom_renderers or []
        )

        # Set up the drawing window

        self.render_on = False
        self._seed()

        self.max_cycles: int | None = max_cycles
        self.scenario = scenario
        self.local_ratio = local_ratio

        self.scenario.reset_world(self.world, self.np_random)

        self.agents: list[str] = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map: dict[str, int] = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }

        self._agent_selector: AgentSelector[str] = AgentSelector(self.agents)

        # set spaces
        self.action_spaces: dict[str, spaces.Space[ActType]] = dict()
        self.observation_spaces: dict[str, spaces.Space[ObsType]] = dict()
        for agent in self.world.agents:
            self.action_spaces[agent.name] = self.action_mode.define_action_space(agent)
            self.observation_spaces[agent.name] = scenario.observation_space(
                agent, self.world
            )

        # self.state_space = spaces.Box(
        #     low=-np.float32(np.inf),
        #     high=+np.float32(np.inf),
        #     shape=(state_dim,),
        #     dtype=np.float32,
        # )

        # Get the original cam_range
        # This will be used to scale the rendering
        all_poses = [entity.state.pos for entity in self.world.entities]
        self.original_cam_range = np.max(np.abs(np.array(all_poses)))

        self.steps = 0

        self.current_actions: list[ActType | None] = [None] * self.num_agents

        self.agent_selection: str = self._agent_selector.reset()

    def movable_agent_action_dim(self, action_opt: ActionOption) -> int:
        if action_opt == ActionOption.CONTINUOUS_MINIMAL:
            return self.world.dim_p
        else:
            return self.world.dim_p * 2 + 1

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    def observation_space(self, agent: str):
        return self.observation_spaces[agent]

    def action_space(self, agent: str):
        return self.action_spaces[agent]

    def _seed(self, seed: int | None = None):
        self.np_random, seed = seeding.np_random(seed)

    def observe(self, agent: str) -> ObsType:
        return self.scenario.observation(
            self.world.agents[self._index_map[agent]], self.world
        )

    def state(self) -> tuple[ObsType, ...]:
        states: tuple[ObsType, ...] = tuple(
            self.scenario.observation(
                self.world.agents[self._index_map[agent]], self.world
            )
            for agent in self.possible_agents
        )
        return states

    def reset(self, seed: int | None = None, options: dict[str, Any] | None = None):
        if seed is not None:
            self._seed(seed=seed)
        self.scenario.reset_world(self.world, self.np_random)

        self.agents = self.possible_agents[:]
        self.rewards = {name: 0.0 for name in self.agents}
        self._cumulative_rewards = {name: 0.0 for name in self.agents}
        self.terminations = {name: False for name in self.agents}
        self.truncations = {name: False for name in self.agents}
        self.infos = {name: {} for name in self.agents}

        self.agent_selection = self._agent_selector.reset()
        self.steps = 0

        self.current_actions: list[ActType | None] = [None] * self.num_agents

        cur_agent = self.agent_selection
        self.infos[cur_agent] = self.scenario.info(
            self.world.agents[self._index_map[cur_agent]], self.world
        )

    def step(self, action: ActType) -> None:
        if (
            self.terminations[self.agent_selection]
            or self.truncations[self.agent_selection]
        ):
            self._was_dead_step(action)
            return
        cur_agent = self.agent_selection
        current_idx = self._index_map[self.agent_selection]
        next_idx = (current_idx + 1) % self.num_agents
        self.agent_selection = self._agent_selector.next()

        self.current_actions[current_idx] = action

        if next_idx == 0:
            self._execute_world_step()
            self.steps += 1
            if self.max_cycles is not None and self.steps >= self.max_cycles:
                for a in self.agents:
                    self.truncations[a] = True
        else:
            self._clear_rewards()

        self._cumulative_rewards[cur_agent] = 0
        self._accumulate_rewards()
        self.terminations[cur_agent] = (
            self.terminations[cur_agent]
            or self.world.agents[self._index_map[cur_agent]].terminated
        )
        self.infos[cur_agent] = self.scenario.info(
            self.world.agents[self._index_map[cur_agent]], self.world
        )

        if self.render_mode == "human":
            self.render()

    def _execute_world_step(self):
        # set action for each agent
        for i, agent in enumerate(self.world.agents):
            self.action_mode.update_agent_action(
                agent, self.current_actions[i], self.world
            )

        self.world.step()

        global_reward = 0.0
        if self.local_ratio is not None:
            global_reward = float(self.scenario.global_reward(self.world))

        for agent in self.world.agents:
            agent_reward = float(self.scenario.reward(agent, self.world))
            if self.local_ratio is not None:
                reward = (
                    global_reward * (1 - self.local_ratio)
                    + agent_reward * self.local_ratio
                )
            else:
                reward = agent_reward

            self.rewards[agent.name] = reward

    def _was_dead_step(self, action: ActType) -> None:
        """Helper function that performs step() for dead agents.

        Does the following:

        1. Removes dead agent from .agents, .terminations, .truncations, .rewards, ._cumulative_rewards, and .infos
        2. Loads next agent into .agent_selection: if another agent is dead, loads that one, otherwise load next live agent
        3. Clear the rewards dict

        Examples:
            Highly recommended to use at the beginning of step as follows:

        def step(self, action):
            if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
                self._was_dead_step()
                return
            # main contents of step
        """
        if action is not None:
            raise ValueError("when an agent is dead, the only valid action is None")

        # removes dead agent
        agent = self.agent_selection
        assert self.terminations[agent] or self.truncations[agent], (
            "an agent that was not dead as attempted to be removed"
        )
        del self.terminations[agent]
        del self.truncations[agent]
        del self.rewards[agent]
        del self._cumulative_rewards[agent]
        del self.infos[agent]
        self.agents.remove(agent)

        # finds next dead agent or loads next live agent (Stored in _skip_agent_selection)
        _deads_order = [
            agent
            for agent in self.agents
            if (self.terminations[agent] or self.truncations[agent])
        ]
        if _deads_order:
            if getattr(self, "_skip_agent_selection", None) is None:
                self._skip_agent_selection = self.agent_selection
            self.agent_selection = _deads_order[0]
        else:
            if getattr(self, "_skip_agent_selection", None) is not None:
                assert self._skip_agent_selection is not None
                self.agent_selection = self._skip_agent_selection
            self._skip_agent_selection = None
        self._clear_rewards()

    def _clear_rewards(self) -> None:
        """Clears all items in .rewards."""
        for agent in self.rewards:
            self.rewards[agent] = 0

    def _accumulate_rewards(self) -> None:
        """Adds .rewards dictionary to ._cumulative_rewards dictionary.

        Typically called near the end of a step() method
        """
        for agent, reward in self.rewards.items():
            self._cumulative_rewards[agent] += reward

    def enable_render(self, mode: str = "human") -> None:
        if not self.render_on or self.fig is None or self.ax is None:
            if self.fig is None or self.ax is None:
                self.fig, self.ax = plt.subplots(
                    figsize=(self.width / 100, self.height / 100), dpi=self.dpi
                )
                self.ax.set_aspect("equal")
                self.ax.axis("off")  # Remove axes for clean rendering
            if mode == "human":
                plt.ion()  # Turn on interactive mode
            self.render_on = True

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)
        assert self.fig is not None and self.ax is not None

        for renderer in self.renderers:
            renderer.render(self.fig, self.ax, self.world, self.grid)

        # Tight layout often produces better results
        self.fig.tight_layout(pad=0)
        # Save the plot for debug
        if self.render_mode == "rgb_array":
            # Convert matplotlib figure to numpy array
            canvas = FigureCanvasAgg(self.fig)
            canvas.draw()
            buf = canvas.buffer_rgba()
            rgb_array = np.asarray(buf)[:, :, :3]  # Remove alpha channel
            return rgb_array
        elif self.render_mode == "human":
            plt.draw()
            plt.pause(1.0 / self.metadata["render_fps"])
            return

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None

    def all_possible_states(self) -> dict[str, dict[Position, ObsType]]:
        """
        Get all possible states for each agent in the environment.

        Returns
        -------
        possible_states : dict[str, dict[Position, ObsType]]
            A dictionary where each key is an agent's name and the value is another
            dictionary mapping that agent's positions to observations for that agent.
        """
        possible_states: dict[str, dict[Position, ObsType]] = {}

        # Get all free cells in the grid (non-wall cells)
        layout = self.grid.layout
        n_rows = len(layout)
        n_cols = len(layout[0]) if layout else 0

        free_positions: list[Position] = []
        for r in range(n_rows):
            for c in range(n_cols):
                if layout[r][c] != "#":
                    # Convert grid cell to continuous position (center of cell)
                    x = c * self.grid.cell_size
                    y = (n_rows - 1 - r) * self.grid.cell_size
                    free_positions.append((x, y))

        # For each agent, compute observations for all possible positions
        for agent in self.world.agents:
            agent_states: dict[Position, ObsType] = {}
            # Store original position
            original_pos = agent.state.pos.copy()

            for pos in free_positions:
                # Temporarily set agent to this position
                agent.state.pos = np.array(pos, dtype=np.float64)
                # Get observation for this position
                obs = self.scenario.observation(agent, self.world)
                agent_states[pos] = obs

            # Restore original position
            agent.state.pos = original_pos
            possible_states[agent.name] = agent_states

        return possible_states


class BaseGymEnv(Env[ObsType, ActType], Generic[ObsType, ActType, ScenarioConfigT]):
    """
    A continuous grid environment with rooms and corridors.

    This environment is built upon the `BaseEnv` class and utilizes a custom
    scenario to create a world with multiple rooms connected by corridors.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 10}

    def __init__(
        self,
        scenario: BaseScenario[ScenarioConfigT, ObsType],
        render_config: RenderConfig = DEFAULT_RENDER_CONFIG,
        render_mode: str | None = None,
        action_config: ActionModeConfig = DEFAULT_ACTION_CONFIG,
        local_ratio: float | None = None,
        verbose: bool = False,
    ):
        if render_mode is not None:
            render_config = render_config.model_copy(
                update={"render_mode": render_mode}
            )
        if isinstance(action_config, dict):
            action_config = ActionModeConfig.model_validate(action_config)

        self.scenario: BaseScenario[ScenarioConfigT, ObsType] = scenario
        self.world: World = self.scenario.make_world(verbose=verbose)
        self.env = BaseEnv(
            self.scenario,
            max_cycles=None,
            render_config=render_config,
            action_config=action_config,
            local_ratio=local_ratio,
        )
        self.action_spaces = self.env.action_spaces
        self.observation_spaces = self.env.observation_spaces
        self.possible_agents = self.env.possible_agents
        self._index_map = self.env._index_map

        self.observation_space = self.env.observation_space(self.agent_selection)
        self.action_space = self.env.action_space(self.agent_selection)

    @property
    def agent_selection(self) -> str:
        """Get the currently selected agent."""
        return self.env.agent_selection

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        self.env.step(action)
        obs = self.env.observe(self.env.agent_selection)
        reward = self.env.rewards[self.env.agent_selection]
        terminated = self.env.terminations[self.env.agent_selection]
        truncated = self.env.truncations[self.env.agent_selection]
        info = self.env.infos[self.env.agent_selection]

        return obs, reward, terminated, truncated, info

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        self.env.reset(seed=seed, options=options)
        obs = self.env.observe(self.env.agent_selection)
        info = self.env.infos[self.env.agent_selection]

        return obs, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()
