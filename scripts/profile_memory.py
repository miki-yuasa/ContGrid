"""
Memory profiling script for ContGrid environment.

This script profiles memory usage to identify memory-intensive operations
that might benefit from Rust optimization.

Usage:
    python scripts/profile_memory.py

Note: Requires memory_profiler package
    pip install memory-profiler
"""

import gc
import sys
from typing import Any

import gymnasium as gym
import numpy as np

import contgrid

try:
    from memory_profiler import memory_usage, profile

    HAS_MEMORY_PROFILER = True
except ImportError:
    HAS_MEMORY_PROFILER = False
    print(
        "Warning: memory_profiler not installed. Install with: pip install memory-profiler"
    )
    print("Falling back to basic memory tracking...")


def get_size(obj: Any, seen: set | None = None) -> int:
    """Recursively find the size of an object and its contents."""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()

    obj_id = id(obj)
    if obj_id in seen:
        return 0

    seen.add(obj_id)

    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, "__dict__"):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes, bytearray)):
        try:
            size += sum([get_size(i, seen) for i in obj])
        except TypeError:
            pass

    return size


def format_bytes(bytes_value: int) -> str:
    """Format bytes into human-readable format."""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_value < 1024.0:
            return f"{bytes_value:.2f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.2f} TB"


def profile_memory_basic():
    """Basic memory profiling without memory_profiler package."""
    print("=" * 80)
    print("BASIC MEMORY PROFILING")
    print("=" * 80)

    # Profile environment creation
    print("\n1. Environment Creation Memory:")
    gc.collect()
    env = gym.make("contgrid/Rooms-v0", max_episode_steps=100)
    env_size = get_size(env)
    print(f"   Environment object size: {format_bytes(env_size)}")

    # Profile world
    world_size = get_size(env.unwrapped.world)
    print(f"   World object size: {format_bytes(world_size)}")

    # Profile agents
    agents_size = sum(get_size(agent) for agent in env.unwrapped.world.agents)
    print(f"   Agents total size: {format_bytes(agents_size)}")

    # Profile landmarks
    landmarks_size = sum(
        get_size(landmark) for landmark in env.unwrapped.world.landmarks
    )
    print(f"   Landmarks total size: {format_bytes(landmarks_size)}")

    # Profile walls
    walls_size = get_size(env.unwrapped.world.wall_collision_checker)
    print(f"   WallCollisionChecker size: {format_bytes(walls_size)}")
    print(
        f"   - Wall bounds array size: {format_bytes(env.unwrapped.world.wall_collision_checker.wall_bounds.nbytes)}"
    )

    # Profile grid
    grid_size = get_size(env.unwrapped.world.grid)
    print(f"   Grid object size: {format_bytes(grid_size)}")

    # Profile scenario
    scenario_size = get_size(env.unwrapped.scenario)
    print(f"   Scenario object size: {format_bytes(scenario_size)}")

    # Reset and profile observation
    print("\n2. Observation Memory:")
    observation, info = env.reset(seed=42)
    obs_size = get_size(observation)
    print(f"   Observation size: {format_bytes(obs_size)}")

    # Break down observation components
    if isinstance(observation, dict):
        for key, value in observation.items():
            if isinstance(value, np.ndarray):
                print(
                    f"   - {key}: {format_bytes(value.nbytes)} ({value.shape}, {value.dtype})"
                )

    # Profile action space
    print("\n3. Action Space Memory:")
    action = env.action_space.sample()
    action_size = get_size(action)
    print(f"   Action size: {format_bytes(action_size)}")
    if isinstance(action, np.ndarray):
        print(
            f"   Action array: {format_bytes(action.nbytes)} ({action.shape}, {action.dtype})"
        )

    # Profile rendering
    print("\n4. Rendering Memory:")
    env_render = gym.make(
        "contgrid/Rooms-v0", max_episode_steps=100, render_mode="rgb_array"
    )
    env_render.reset(seed=42)
    frame = env_render.render()
    if frame is not None:
        frame_size = frame.nbytes if isinstance(frame, np.ndarray) else get_size(frame)
        print(f"   Frame size: {format_bytes(frame_size)}")
        if isinstance(frame, np.ndarray):
            print(f"   Frame dimensions: {frame.shape}, dtype: {frame.dtype}")
    env_render.close()

    # Memory during episode
    print("\n5. Memory During Episode Execution:")
    gc.collect()
    frames = []
    observations = []

    for step in range(50):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        observations.append(observation)

        if terminated or truncated:
            break

    total_obs_size = sum(get_size(obs) for obs in observations)
    print(
        f"   Total observations memory ({len(observations)} steps): {format_bytes(total_obs_size)}"
    )
    print(
        f"   Average per observation: {format_bytes(total_obs_size / len(observations))}"
    )

    env.close()

    print("\n" + "=" * 80)
    print("MEMORY RECOMMENDATIONS")
    print("=" * 80)
    print("\nLarge memory consumers:")

    components = [
        ("Environment", env_size),
        ("World", world_size),
        ("WallCollisionChecker", walls_size),
        ("Scenario", scenario_size),
        ("Agents", agents_size),
        ("Landmarks", landmarks_size),
    ]

    components.sort(key=lambda x: x[1], reverse=True)

    for name, size in components:
        percentage = (size / env_size) * 100
        print(f"  - {name}: {format_bytes(size)} ({percentage:.1f}% of environment)")

        if name == "WallCollisionChecker" and size > 1024 * 1024:  # > 1MB
            print(
                "    * RECOMMENDATION: Large collision checker could benefit from Rust"
            )
            print("      - Use more memory-efficient data structures")
            print("      - Consider spatial indexing (BVH, octree)")


if HAS_MEMORY_PROFILER:

    @profile
    def profile_episode_with_memory_profiler():
        """Profile an episode with line-by-line memory tracking."""
        env = gym.make("contgrid/Rooms-v0", max_episode_steps=100)
        observation, info = env.reset(seed=42)

        for _ in range(100):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break

        env.close()
        return observation

    def profile_memory_advanced():
        """Advanced memory profiling with memory_profiler."""
        print("\n" + "=" * 80)
        print("ADVANCED MEMORY PROFILING (with memory_profiler)")
        print("=" * 80)

        print("\nMemory usage during episode execution:")
        mem_usage = memory_usage((profile_episode_with_memory_profiler,), interval=0.01)

        print(f"\nPeak memory usage: {max(mem_usage):.2f} MB")
        print(f"Baseline memory: {min(mem_usage):.2f} MB")
        print(f"Memory delta: {max(mem_usage) - min(mem_usage):.2f} MB")

        print("\nRunning detailed line-by-line profiling...")
        print("(Check the output above for line-by-line memory usage)\n")


def main():
    """Main memory profiling routine."""
    print("Starting ContGrid Memory Profiling...\n")

    # Always run basic profiling
    profile_memory_basic()

    # Run advanced profiling if available
    if HAS_MEMORY_PROFILER:
        profile_memory_advanced()
    else:
        print("\n" + "=" * 80)
        print("For more detailed profiling, install memory_profiler:")
        print("  pip install memory-profiler")
        print("=" * 80)


if __name__ == "__main__":
    main()
