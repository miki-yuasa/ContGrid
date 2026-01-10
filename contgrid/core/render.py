from abc import ABC, abstractmethod
from typing import Any, Literal

import matplotlib.patches as patches
import numpy as np
from gymnasium.core import Env
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from pydantic import BaseModel

from .agent import Agent
from .const import ALPHABET
from .entities import Entity, EntityShape
from .grid import Grid
from .world import World


class RenderConfig(BaseModel):
    render_mode: Literal["human", "rgb_array"] = "rgb_array"
    draw_grid: bool = False
    width_px: int = 700
    height_px: int = 700
    dpi: int = 100


DEFAULT_RENDER_CONFIG = RenderConfig()


class Renderer(ABC):
    """
    Abstract base class for environment renderers.
    """

    alphabet: str = ALPHABET

    @abstractmethod
    def __init__(self, render_config: RenderConfig) -> None:
        """
        Initialize the renderer with the given configuration.

        Parameters
        ----------
        render_config : RenderConfig
            Configuration for rendering.
        """
        self.render_config = render_config

    @abstractmethod
    def render(self, fig: Figure, ax: Axes, world: World, grid: Grid) -> None:
        """
        Render the environment state onto the given matplotlib figure and axes.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure to render on.
        ax : Axes
            Matplotlib axes to render on.
        world : World
            The world state to render.
        grid : Grid
            The grid for spatial queries.
        options : dict[str, Any]
            Additional rendering options.
        """
        pass


class EnvRenderer(Renderer):
    """
    Renderer for the continuous grid world environment.
    Custom renders are rendered on top of this.
    """

    def __init__(self, render_config: RenderConfig) -> None:
        self.render_config = render_config

        self.width = render_config.width_px
        self.height = render_config.height_px

    def render(
        self,
        fig: Figure,
        ax: Axes,
        world: World,
        grid: Grid,
    ) -> None:
        # update bounds to center around agent
        all_poses = [entity.draw_pos for entity in world.all_entities]

        # Find the limits of the environment
        all_poses_np = np.array(all_poses)
        x_min, y_min = np.min(all_poses_np, axis=0)

        ax.clear()
        ax.set_xlim(x_min, x_min + grid.width)
        ax.set_ylim(y_min, y_min + grid.height)
        ax.set_aspect("equal")
        ax.axis("off")
        ax.set_facecolor("white")

        # If draw_grid is enabled, draw the grid lines
        if self.render_config.draw_grid:
            # Draw grid lines
            for x in range(grid.width_cells + 1):
                ax.axvline(x=x_min + x, color="black", linewidth=0.5, zorder=1)
            for y in range(grid.height_cells + 1):
                ax.axhline(y=y_min + y, color="black", linewidth=0.5, zorder=1)

        # The scaling factor is used for dynamic rescaling of the rendering - a.k.a Zoom In/Zoom Out effect
        # The 0.9 is a factor to keep the entities from appearing "too" out-of-bounds

        # update geometry and text positions
        text_line = 0
        for entity in world.all_entities:
            # geometry
            x: float
            y: float
            x, y = entity.state.pos

            assert entity.color
            self._draw_shape(entity, ax)

            # text
            if isinstance(entity, Agent):
                if entity.silent:
                    continue
                if np.all(entity.state.c == 0):
                    word = "_"
                # elif self.action_opt in self.continuous_modes:
                #     word = (
                #         "[" + ",".join([f"{comm:.2f}" for comm in entity.state.c]) + "]"
                #     )
                else:
                    word = self.alphabet[np.argmax(entity.state.c)]

                message = entity.name + " sends " + word + "   "
                message_x_pos = self.width * 0.05
                message_y_pos = self.height * 0.95 - (self.height * 0.05 * text_line)
                ax.text(
                    message_x_pos, message_y_pos, message, fontsize=12, color="black"
                )
                text_line += 1

    def _draw_shape(self, entity: Entity, ax: Axes) -> None:
        # Convert color tuple to matplotlib-compatible format (0-1 range)
        x: float = entity.draw_pos[0]
        y: float = entity.draw_pos[1]
        size: float = entity.size
        shape: EntityShape = entity.shape
        color_normalized = tuple(c / 255.0 for c in entity.color)
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
            ax.add_patch(circle)
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
            ax.add_patch(rect)
        else:
            raise ValueError(f"Unknown shape: {shape}")


class PostRenderer(ABC):
    """
    Abstract base class for post-renderers that add additional elements to the rendering after the main environment's `env.render()` is called.
    """

    @abstractmethod
    def render(self, fig: Figure, ax: Axes, env: Env, **kwargs: Any) -> None:
        """
        Render additional elements onto the given matplotlib figure and axes.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure to render on.
        ax : Axes
            Matplotlib axes to render on.
        env : Env
            The gymnasium environment being rendered.
        **kwargs : Any
            Additional rendering options specific to the renderer implementation.
        """
        pass

    def get_image(self, fig: Figure) -> NDArray[np.uint8]:
        """
        Extract RGB image array from the figure canvas.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure to extract image from.

        Returns
        -------
        NDArray[np.uint8]
            RGB array of the rendered image.
        """
        fig.canvas.draw()
        buf = fig.canvas.buffer_rgba()  # type: ignore[attr-defined]
        img_array = np.asarray(buf)
        # Convert RGBA to RGB
        return img_array[:, :, :3].astype(np.uint8)


class DiscreteActionVectorRenderer(PostRenderer):
    """
    Renderer for environments with discrete action vectors.
    Draws arrows representing action probabilities, centered at the agent's position,
    with arrow lengths proportional to probabilities.

    Examples
    --------
    >>> import numpy as np
    >>> # Define 4 cardinal directions
    >>> vectors = np.array([[1, 0], [0, 1], [-1, 0], [0, -1]], dtype=np.float64)
    >>> renderer = DiscreteActionVectorRenderer(vectors, agent_idx=0)
    >>>
    >>> # Update probabilities and render
    >>> probs = np.array([0.5, 0.3, 0.1, 0.1])
    >>> renderer.set_probabilities(probs)
    >>> renderer.render(fig, ax, env)
    """

    def __init__(
        self,
        action_vectors: NDArray[np.float64],
        agent_idx: int = 0,
        max_arrow_length: float = 0.5,
        arrow_color: str = "blue",
        arrow_alpha: float = 0.7,
        arrow_width: float = 0.02,
        head_width: float = 0.08,
        head_length: float = 0.05,
        min_prob_threshold: float = 0.01,
    ) -> None:
        """
        Initialize the discrete action vector renderer.

        Parameters
        ----------
        action_vectors : NDArray[np.float64]
            Array of shape (num_actions, 2) where each row is a direction vector (x, y).
            These vectors define the possible discrete actions.
        agent_idx : int
            Index of the agent to render arrows for.
        max_arrow_length : float
            Maximum length of arrows when probability is 1.0.
        arrow_color : str
            Color of the arrows.
        arrow_alpha : float
            Base alpha (transparency) of the arrows.
        arrow_width : float
            Width of the arrow shaft.
        head_width : float
            Width of the arrow head.
        head_length : float
            Length of the arrow head.
        min_prob_threshold : float
            Minimum probability threshold below which arrows are not drawn.
        """
        self.action_vectors = action_vectors
        self.agent_idx = agent_idx
        self.max_arrow_length = max_arrow_length
        self.arrow_color = arrow_color
        self.arrow_alpha = arrow_alpha
        self.arrow_width = arrow_width
        self.head_width = head_width
        self.head_length = head_length
        self.min_prob_threshold = min_prob_threshold

        # Normalize action vectors to unit vectors for consistent scaling
        norms = np.linalg.norm(action_vectors, axis=1, keepdims=True)
        # Avoid division by zero for zero vectors
        norms = np.where(norms == 0, 1, norms)
        self.unit_vectors = action_vectors / norms

        # Initialize with uniform probabilities
        self.probabilities: NDArray[np.float64] = np.ones(len(action_vectors)) / len(
            action_vectors
        )

    def set_probabilities(self, probabilities: NDArray[np.float64]) -> None:
        """
        Update the action probabilities.

        Parameters
        ----------
        probabilities : NDArray[np.float64]
            Array of shape (num_actions,) containing probabilities for each action.

        Raises
        ------
        ValueError
            If probabilities length doesn't match the number of action vectors.
        """
        if len(probabilities) != len(self.action_vectors):
            raise ValueError(
                f"Probabilities length ({len(probabilities)}) must match "
                f"action_vectors length ({len(self.action_vectors)})"
            )
        self.probabilities = probabilities

    def render(self, fig: Figure, ax: Axes, env: Env, **kwargs: Any) -> None:
        """
        Render action probability arrows onto the given matplotlib axes.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure to render on.
        ax : Axes
            Matplotlib axes to render on.
        env : Env
            The gymnasium environment, expected to have a `world` attribute with agents.
        **kwargs : Any
            Optional keyword arguments:
            - probabilities: NDArray[np.float64] - Override stored probabilities for this render
            - agent_idx: int - Override stored agent index for this render
        """
        # Allow overriding probabilities and agent_idx per render call
        probabilities = kwargs.get("probabilities", self.probabilities)
        agent_idx = kwargs.get("agent_idx", self.agent_idx)

        # Get agent position from environment
        if not hasattr(env, "world"):
            return

        world: World = env.world  # type: ignore
        if agent_idx >= len(world.agents):
            return

        agent = world.agents[agent_idx]
        agent_pos = agent.state.pos

        # Draw arrows for each action
        for unit_vec, prob in zip(self.unit_vectors, probabilities):
            if prob < self.min_prob_threshold:
                continue

            # Scale arrow length by probability
            arrow_length = prob * self.max_arrow_length
            dx = unit_vec[0] * arrow_length
            dy = unit_vec[1] * arrow_length

            # Draw arrow from agent position
            ax.arrow(
                agent_pos[0],
                agent_pos[1],
                dx,
                dy,
                color=self.arrow_color,
                alpha=self.arrow_alpha
                * prob,  # More transparent for lower probabilities
                width=self.arrow_width,
                head_width=self.head_width,
                head_length=self.head_length,
                length_includes_head=True,
                zorder=10,
            )
