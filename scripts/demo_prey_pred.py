"""Executable demonstration script for PreyPredEnv.

Runs a sample episode, prints telemetry, and saves an animated GIF and
trajectory plot into the configured output directory.
"""

from __future__ import annotations

import shutil
from collections.abc import Sequence
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags
from PIL import Image

import contgrid  # noqa: F401

_DEFAULT_OUTPUT_DIR: Path = (
    Path(__file__).resolve().parent.parent / "assets" / "envs"
)

_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    default=str(_DEFAULT_OUTPUT_DIR),
    help="Directory to save generated demo media files.",
)
_ARTIFACT_DIR = flags.DEFINE_string(
    "artifact_dir",
    default=None,
    help="Optional secondary artifact directory to copy generated media.",
)
_NUM_STEPS = flags.DEFINE_integer(
    "num_steps",
    default=120,
    help="Number of simulation steps to run.",
    lower_bound=1,
)
_SEED = flags.DEFINE_integer(
    "seed",
    default=42,
    help="Random seed for reproducibility.",
)


def run_demo(
    num_steps: int = 120,
    seed: int = 42,
    output_dir: Path | None = None,
    artifact_dir: Path | None = None,
) -> None:
    """Run a demonstration episode and record visual media.

    Args:
        num_steps: Number of simulation steps to run.
        seed: Random seed for reproducibility.
        output_dir: Output directory for media assets.
        artifact_dir: Optional secondary directory to copy generated media.
    """
    if output_dir is None:
        output_dir = _DEFAULT_OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    env = gym.make("contgrid/PreyPred-v0", render_mode="rgb_array")
    obs, info = env.reset(seed=seed)

    frames: list[Image.Image] = []
    agent_trail: list[tuple[float, float]] = []

    init_frame = env.render()
    if init_frame is not None:
        frames.append(Image.fromarray(np.asarray(init_frame)))

    agent_pos = obs["agent_pos"]
    agent_trail.append((float(agent_pos[0]), float(agent_pos[1])))

    print("=" * 80)
    print("ContGrid PreyPred-v0 Demonstration Rollout")
    print(f"Initial Agent Position: ({agent_pos[0]:.2f}, {agent_pos[1]:.2f})")
    print("=" * 80)

    total_reward = 0.0
    for step in range(num_steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        pos = obs["agent_pos"]
        agent_trail.append((float(pos[0]), float(pos[1])))

        counts = obs["prey_capture_counts"]
        print(
            f"Step {step + 1:03d}: Action=[{action[0]}, {action[1]}] | "
            f"Agent=({pos[0]:.2f}, {pos[1]:.2f}) | "
            f"Counts=[{counts[0]}, {counts[1]}, {counts[2]}, {counts[3]}] | "
            f"Reward={reward:+.3f} | Terminated={terminated}"
        )

        frame = env.render()
        if frame is not None:
            frames.append(Image.fromarray(np.asarray(frame)))

        if terminated or truncated:
            print(
                f"Episode ended at step {step + 1} "
                f"(terminated={terminated}, truncated={truncated}, "
                f"success={info['is_success']})"
            )
            break

    env.close()

    # Save animated GIF
    gif_path = output_dir / "prey_pred_sample.gif"
    if frames:
        frames[0].save(
            gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=100,
            loop=0,
        )
        print(f"\nSaved animated demo to: {gif_path}")

        if artifact_dir is not None and artifact_dir.exists():
            shutil.copy2(gif_path, artifact_dir / "prey_pred_sample.gif")
            print(f"Copied demo GIF to artifact directory: {artifact_dir}")

    # Save trajectory summary plot
    fig, ax = plt.subplots(figsize=(6, 6), dpi=120)
    ax.set_facecolor("#FAFAFA")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 15)
    ax.set_aspect("equal")
    ax.set_title("PreyPred Sample Agent Trajectory Trace", fontsize=11)

    regions = [
        (1.5, 5.5, 1.5, 5.5, "Region 0"),
        (8.5, 12.5, 1.5, 5.5, "Region 1"),
        (1.5, 5.5, 8.5, 12.5, "Region 2"),
        (8.5, 12.5, 8.5, 12.5, "Region 3"),
    ]
    for min_x, max_x, min_y, max_y, label in regions:
        rect = plt.Rectangle(
            (min_x, min_y),
            max_x - min_x,
            max_y - min_y,
            facecolor="#EEEEEE",
            edgecolor="#CCCCCC",
            linestyle="--",
        )
        ax.add_patch(rect)
        ax.text(
            (min_x + max_x) / 2,
            (min_y + max_y) / 2,
            label,
            ha="center",
            va="center",
            color="#AAAAAA",
            fontsize=8,
        )

    xs = [p[0] for p in agent_trail]
    ys = [p[1] for p in agent_trail]
    ax.plot(
        xs,
        ys,
        color="#2563EB",
        linewidth=1.8,
        alpha=0.8,
        label="Agent Path",
    )
    ax.scatter(
        [xs[0]],
        [ys[0]],
        color="green",
        s=60,
        zorder=5,
        label="Start",
    )
    ax.scatter(
        [xs[-1]],
        [ys[-1]],
        color="red",
        s=60,
        zorder=5,
        label="End",
    )
    ax.legend(loc="upper right", fontsize=8)
    ax.set_xlabel("X (world units)")
    ax.set_ylabel("Y (world units)")
    plt.tight_layout()

    traj_path = output_dir / "prey_pred_trajectory.png"
    plt.savefig(traj_path)
    plt.close(fig)
    print(f"Saved trajectory trace plot to: {traj_path}")

    if (
        artifact_dir is not None
        and artifact_dir.exists()
        and traj_path.exists()
    ):
        shutil.copy2(traj_path, artifact_dir / "prey_pred_trajectory.png")
        print(f"Copied trajectory trace to artifact directory: {artifact_dir}")

    print("=" * 80)
    print(
        f"Demonstration completed successfully. Total reward: {total_reward:.2f}"
    )
    print("=" * 80)


def main(argv: Sequence[str]) -> None:
    """Entry point parsing flags and executing the demonstration."""
    if len(argv) > 1:
        raise app.UsageError("Too many command-line arguments.")

    out_dir = Path(_OUTPUT_DIR.value)
    art_dir = Path(_ARTIFACT_DIR.value) if _ARTIFACT_DIR.value else None
    run_demo(
        num_steps=_NUM_STEPS.value,
        seed=_SEED.value,
        output_dir=out_dir,
        artifact_dir=art_dir,
    )


if __name__ == "__main__":
    app.run(main)
