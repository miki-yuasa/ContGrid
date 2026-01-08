from abc import ABC, abstractmethod
from typing import Any, Literal

import matplotlib.patches as patches
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
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
