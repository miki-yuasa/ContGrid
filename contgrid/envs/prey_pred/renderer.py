"""Custom renderer for the PreyPred environment."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from matplotlib import patches
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from contgrid.core import Grid, World
from contgrid.core.render import RenderConfig, Renderer

if TYPE_CHECKING:
    from .scenario import PreyPredScenario


class PreyPredRenderer(Renderer):
    """Custom renderer adding region boundaries, trajectory paths, and HUD."""

    def __init__(
        self,
        scenario: PreyPredScenario,
        render_config: RenderConfig,
    ) -> None:
        super().__init__(render_config)
        self.scenario = scenario
        self.render_config = render_config

    def render(
        self,
        fig: Figure,
        ax: Axes,
        world: World,
        grid: Grid,
        **kwargs: Any,
    ) -> None:
        """Render PreyPred-specific overlays onto the axes.

        Args:
            fig: Matplotlib Figure.
            ax: Matplotlib Axes.
            world: ContGrid World state.
            grid: ContGrid Grid state.
            **kwargs: Additional keyword arguments.
        """
        opts = self.scenario.config.render_options

        # 1. Draw 4x4 Region boxes
        if opts.show_region_boxes:
            for region in self.scenario.config.regions:
                min_x, max_x, min_y, max_y = region.bounds
                width = max_x - min_x
                height = max_y - min_y
                rect = patches.Rectangle(
                    (min_x, min_y),
                    width,
                    height,
                    facecolor="#F3F4F6",
                    edgecolor="#D1D5DB",
                    linestyle="--",
                    linewidth=1.0,
                    zorder=0.5,
                )
                ax.add_patch(rect)

                # Subdued Region watermark / label in corner
                ax.text(
                    min_x + 0.2,
                    max_y - 0.35,
                    f"R{region.region_id}",
                    fontsize=8,
                    color="#9CA3AF",
                    weight="bold",
                    zorder=0.6,
                )

        # 2. Draw deterministic trajectory paths
        if opts.show_trajectories:
            for item in self.scenario.trajectory_items:
                waypoints = item.trajectory.get_path_waypoints(num_points=100)
                if len(waypoints) > 1:
                    xs = [p[0] for p in waypoints]
                    ys = [p[1] for p in waypoints]
                    ax.plot(
                        xs,
                        ys,
                        color=item.color.value,
                        linestyle=":",
                        linewidth=1.5,
                        alpha=0.6,
                        zorder=1.5,
                    )

        # 3. Draw HUD if enabled
        if opts.show_hud:
            counts = self.scenario.prey_capture_counts
            hud_text = (
                f"t={self.scenario.sim_time:.1f}s | "
                f"Captures: [{counts[0]},{counts[1]},{counts[2]},{counts[3]}]"
            )
            ax.text(
                0.5,
                1.02,
                hud_text,
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="medium",
                bbox={
                    "boxstyle": "round,pad=0.3",
                    "facecolor": "#FFFFFF",
                    "edgecolor": "#E5E7EB",
                    "alpha": 0.85,
                },
                zorder=10,
            )
