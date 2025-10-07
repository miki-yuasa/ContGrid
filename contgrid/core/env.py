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

from .const import ALPHABET, Color
from .entities import Entity, EntityShape
from .grid import Grid
from .scenario import BaseScenario, ScenarioConfigT
from .utils import AgentSelector
from .world import DEFAULT_WORLD_CONFIG, Agent, World, WorldConfig


class RenderConfig(BaseModel):
    render_mode: Literal["human", "rgb_array"] = "rgb_array"
    draw_grid: bool = True
    width_px: int = 700
    height_px: int = 700
    dpi: int = 100


DEFAULT_RENDER_CONFIG = RenderConfig()


class ActionMode(StrEnum):
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"
    CONTINUOUS_MINIMAL = "continuous-minimal"


class EnvConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    scenario: BaseScenario
    max_cycles: int = 100
    render_config: RenderConfig = DEFAULT_RENDER_CONFIG
    action_mode: str = ActionMode.CONTINUOUS_MINIMAL.value
    local_ratio: float | None = None
    verbose: bool = False


class BaseEnv(Generic[ObsType, ActType, ScenarioConfigT]):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "is_parallelizable": True,
        "render_fps": 10,
    }

    continuous_modes: list[ActionMode] = [
        ActionMode.CONTINUOUS,
        ActionMode.CONTINUOUS_MINIMAL,
    ]

    def __init__(
        self,
        scenario: BaseScenario[ScenarioConfigT, ObsType],
        max_cycles: int = 100,
        render_config: RenderConfig = DEFAULT_RENDER_CONFIG,
        action_mode: str = ActionMode.CONTINUOUS_MINIMAL.value,
        local_ratio: float | None = None,
        verbose: bool = False,
    ):
        super().__init__()

        self.render_config: RenderConfig = render_config
        self.render_mode = render_config.render_mode
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

        # Set up the drawing window

        self.renderOn = False
        self._seed()

        self.max_cycles = max_cycles
        self.scenario = scenario
        self.action_mode: ActionMode = ActionMode(action_mode)
        self.local_ratio = local_ratio

        self.scenario.reset_world(self.world, self.np_random)

        self.agents: list[str] = [agent.name for agent in self.world.agents]
        self.possible_agents = self.agents[:]
        self._index_map: dict[str, int] = {
            agent.name: idx for idx, agent in enumerate(self.world.agents)
        }

        self._agent_selector: AgentSelector[str] = AgentSelector(self.agents)

        # set spaces
        self.action_spaces: dict[str, spaces.Space] = dict()
        self.observation_spaces: dict[str, spaces.Space] = dict()
        state_dim: int = 0
        for agent in self.world.agents:
            if agent.movable:
                space_dim = self.movable_agent_action_dim(self.action_mode)
            elif self.action_mode in self.continuous_modes:
                space_dim = 0
            else:
                space_dim = 1

            if not agent.silent:
                if self.action_mode in self.continuous_modes:
                    space_dim += self.world.dim_c
                else:
                    space_dim *= self.world.dim_c

            # obs_dim = len(self.scenario.observation(agent, self.world))
            # state_dim += obs_dim
            match self.action_mode:
                case "continuous":
                    self.action_spaces[agent.name] = spaces.Box(
                        low=0, high=agent.u_range, shape=(space_dim,)
                    )
                case "continuous-minimal":
                    self.action_spaces[agent.name] = spaces.Box(
                        low=-agent.u_range, high=agent.u_range, shape=(space_dim,)
                    )
                case "discrete":
                    self.action_spaces[agent.name] = spaces.Discrete(space_dim)
                case _:
                    raise ValueError(f"Unknown action mode {self.action_mode}")

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

    def movable_agent_action_dim(self, action_mode: ActionMode) -> int:
        if action_mode == ActionMode.CONTINUOUS_MINIMAL:
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
            if self.steps >= self.max_cycles:
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
            action = self.current_actions[i]
            scenario_action: list[NDArray | int | np.integer | ActType] = []
            if agent.movable:
                mdim = self.movable_agent_action_dim(self.action_mode)
                if self.action_mode in self.continuous_modes:
                    assert isinstance(action, np.ndarray)
                    scenario_action.append(action[0:mdim])
                    action = action[mdim:]
                else:
                    assert isinstance(action, int) or isinstance(action, np.integer)
                    scenario_action.append(action % mdim)
                    action //= mdim
            if not agent.silent:
                assert action
                scenario_action.append(action)
            self._set_action(scenario_action, agent, self.action_spaces[agent.name])

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

    # set env action for a particular agent
    def _set_action(
        self,
        action: list[NDArray | int | np.integer | ActType],
        agent: Agent,
        action_space,
        time=None,
    ):
        agent.action.u = np.zeros(self.world.dim_p)
        agent.action.c = np.zeros(self.world.dim_c)

        if agent.movable:
            # physical action
            agent.action.u = np.zeros(self.world.dim_p)
            match self.action_mode:
                case "continuous":
                    assert isinstance(action[0], np.ndarray)
                    agent.action.u[0] += action[0][2] - action[0][1]
                    agent.action.u[1] += action[0][4] - action[0][3]
                case "continuous-minimal":
                    assert isinstance(action[0], np.ndarray)
                    agent.action.u[0] += action[0][0]
                    agent.action.u[1] += action[0][1]
                case "discrete":
                    if action[0] == 1:
                        agent.action.u[0] = -1.0
                    if action[0] == 2:
                        agent.action.u[0] = +1.0
                    if action[0] == 3:
                        agent.action.u[1] = -1.0
                    if action[0] == 4:
                        agent.action.u[1] = +1.0

            sensitivity = 5.0
            if agent.accel is not None:
                sensitivity = agent.accel
            agent.action.u *= sensitivity
            action = action[1:]
        if not agent.silent:
            # communication action
            if self.action_mode in self.continuous_modes:
                assert isinstance(action[0], np.ndarray)
                agent.action.c = action[0]
            else:
                agent.action.c = np.zeros(self.world.dim_c)
                agent.action.c[action[0]] = 1.0
            action = action[1:]
        # make sure we used all elements of action
        assert len(action) == 0

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
        if not self.renderOn:
            if self.fig is None:
                self.fig, self.ax = plt.subplots(
                    figsize=(self.width / 100, self.height / 100), dpi=self.dpi
                )
                self.ax.set_aspect("equal")
                self.ax.axis("off")  # Remove axes for clean rendering
            if mode == "human":
                plt.ion()  # Turn on interactive mode
            self.renderOn = True

    def render(self):
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        self.enable_render(self.render_mode)
        assert self.fig is not None and self.ax is not None

        self.draw()
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

    def draw(self, alphabet: str = ALPHABET):
        # clear axes
        assert self.ax is not None, (
            "Call enable_render() or set render_mode to human or rgb_array"
        )
        # update bounds to center around agent
        all_poses = [entity.state.pos for entity in self.world.all_entities]

        # Find the limits of the environment
        all_poses_np = np.array(all_poses)
        x_min, y_min = np.min(all_poses_np, axis=0)

        self.ax.clear()
        self.ax.set_xlim(x_min, x_min + self.grid.width)
        self.ax.set_ylim(y_min, y_min + self.grid.height)
        self.ax.set_aspect("equal")
        self.ax.axis("off")
        self.ax.set_facecolor("white")

        # If draw_grid is enabled, draw the grid lines
        if self.render_config.draw_grid:
            # Draw grid lines
            for x in range(self.grid.width_cells + 1):
                self.ax.axvline(x=x_min + x, color="black", linewidth=0.5, zorder=1)
            for y in range(self.grid.height_cells + 1):
                self.ax.axhline(y=y_min + y, color="black", linewidth=0.5, zorder=1)

        # The scaling factor is used for dynamic rescaling of the rendering - a.k.a Zoom In/Zoom Out effect
        # The 0.9 is a factor to keep the entities from appearing "too" out-of-bounds

        # update geometry and text positions
        text_line = 0
        for entity in self.world.all_entities:
            # geometry
            x: float
            y: float
            x, y = entity.state.pos

            assert entity.color
            self._draw_shape(entity)

            # text
            if isinstance(entity, Agent):
                if entity.silent:
                    continue
                if np.all(entity.state.c == 0):
                    word = "_"
                elif self.action_mode in self.continuous_modes:
                    word = (
                        "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                    )
                else:
                    word = alphabet[np.argmax(entity.state.c)]

                message = entity.name + " sends " + word + "   "
                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                self.ax.text(
                    message_x_pos, message_y_pos, message, fontsize=12, color="black"
                )
                text_line += 1

    def _draw_shape(self, entity: Entity):
        # Convert color tuple to matplotlib-compatible format (0-1 range)
        x: float = entity.state.pos[0]
        y: float = entity.state.pos[1]
        size: float = entity.size
        shape: EntityShape = entity.shape
        color_normalized = tuple(c / 255.0 for c in entity.color)
        assert self.ax
        if shape == EntityShape.CIRCLE:
            # Draw filled circle
            circle = patches.Circle(
                (x, y),
                size,
                facecolor=color_normalized,
                edgecolor="black",
                linewidth=0.5,
                zorder=2,
                hatch=entity.hatch,
                hatch_linewidth=0.3,
            )
            self.ax.add_patch(circle)
        elif shape == EntityShape.SQUARE:
            # Draw filled rectangle (square)
            line_width: float = 0.0 if self.render_config.draw_grid else 0.5
            rect = patches.Rectangle(
                (x, y),
                size,
                size,
                facecolor=color_normalized,
                edgecolor="black",
                linewidth=line_width,
                zorder=0,
                hatch=entity.hatch,
                hatch_linewidth=0.3,
            )
            self.ax.add_patch(rect)
        else:
            raise ValueError(f"Unknown shape: {shape}")

    def close(self):
        if self.fig is not None:
            plt.close(self.fig)
            self.fig = None
            self.ax = None


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
        world_config: WorldConfig = DEFAULT_WORLD_CONFIG,
        max_cycles: int = 100,
        render_config: RenderConfig = DEFAULT_RENDER_CONFIG,
        action_mode: str = ActionMode.CONTINUOUS_MINIMAL.value,
        local_ratio: float | None = None,
        verbose: bool = False,
    ):
        self.scenario: BaseScenario[ScenarioConfigT, ObsType] = scenario
        self.world: World = self.scenario.make_world(verbose=verbose)
        self.env = BaseEnv(
            self.scenario, max_cycles, render_config, action_mode, local_ratio
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
