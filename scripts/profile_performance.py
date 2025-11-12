"""
Performance profiling script for ContGrid environment.

This script profiles the computational performance of key functions and classes
to identify bottlenecks that would benefit from Rust optimization.

Usage:
    python scripts/profile_performance.py
"""

import cProfile
import io
import pstats
import time
from dataclasses import dataclass
from typing import Callable

import gymnasium as gym
import numpy as np

from contgrid.core.grid import WallCollisionChecker
from contgrid.envs.rooms import (
    DEFAULT_ROOMS_SCENARIO_CONFIG,
    DEFAULT_WORLD_CONFIG,
    RoomsScenario,
)


@dataclass
class ProfilingResult:
    """Store profiling results for a specific operation."""

    name: str
    total_time: float
    calls: int
    time_per_call: float
    percentage: float = 0.0


class PerformanceProfiler:
    """Profile performance of ContGrid operations."""

    def __init__(self, num_episodes: int = 100, steps_per_episode: int = 100):
        self.num_episodes = num_episodes
        self.steps_per_episode = steps_per_episode
        self.results: list[ProfilingResult] = []

    def profile_function(
        self, func: Callable, name: str, *args, **kwargs
    ) -> ProfilingResult:
        """Profile a single function."""
        print(f"\nProfiling: {name}")

        start_time = time.perf_counter()
        _ = func(*args, **kwargs)
        end_time = time.perf_counter()

        elapsed = end_time - start_time

        profiling_result = ProfilingResult(
            name=name, total_time=elapsed, calls=1, time_per_call=elapsed
        )

        print(f"  Time: {elapsed * 1000:.2f} ms")

        return profiling_result

    def profile_environment_creation(self) -> ProfilingResult:
        """Profile environment creation."""

        def create_env():
            env = gym.make(
                "contgrid/Rooms-v0", max_episode_steps=self.steps_per_episode
            )
            return env

        return self.profile_function(create_env, "Environment Creation")

    def profile_reset(self, env) -> ProfilingResult:
        """Profile environment reset."""

        def reset():
            return env.reset(seed=42)

        return self.profile_function(reset, "Environment Reset")

    def profile_step(self, env) -> ProfilingResult:
        """Profile single environment step."""
        env.reset(seed=42)
        action = env.action_space.sample()

        def step():
            return env.step(action)

        return self.profile_function(step, "Single Step")

    def profile_full_episode(self) -> ProfilingResult:
        """Profile a complete episode."""

        def run_episode():
            env = gym.make(
                "contgrid/Rooms-v0", max_episode_steps=self.steps_per_episode
            )
            observation, info = env.reset(seed=42)

            for _ in range(self.steps_per_episode):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

            env.close()
            return observation

        return self.profile_function(
            run_episode, f"Full Episode ({self.steps_per_episode} steps)"
        )

    def profile_wall_collision_checker(self) -> dict[str, ProfilingResult]:
        """Profile WallCollisionChecker operations."""
        print("\n=== Profiling WallCollisionChecker ===")

        grid = DEFAULT_WORLD_CONFIG.grid
        checker = WallCollisionChecker(grid.layout, grid.cell_size, verbose=False)

        R = 0.25  # Robot radius
        C = 0.1  # Contact margin
        num_checks = 10000

        results = {}

        # Profile is_collision
        def test_is_collision():
            for i in range(num_checks):
                pos = (5.0 + i * 0.001, 5.0 + i * 0.001)
                checker.is_collision(R, C, pos, 100.0, 0.1)

        result = self.profile_function(
            test_is_collision, f"is_collision ({num_checks} calls)"
        )
        result.calls = num_checks
        result.time_per_call = result.total_time / num_checks
        results["is_collision"] = result

        # Profile clip_new_position
        def test_clip_position():
            for i in range(num_checks):
                curr_pos = (5.0, 5.0)
                new_pos = (5.0 + i * 0.0001, 5.0 + i * 0.0001)
                checker.clip_new_position(R, C, curr_pos, new_pos)

        result = self.profile_function(
            test_clip_position, f"clip_new_position ({num_checks} calls)"
        )
        result.calls = num_checks
        result.time_per_call = result.total_time / num_checks
        results["clip_new_position"] = result

        # Profile is_position_valid
        def test_is_valid():
            for i in range(num_checks):
                pos = (5.0 + i * 0.001, 5.0 + i * 0.001)
                checker.is_position_valid(R, C, pos)

        result = self.profile_function(
            test_is_valid, f"is_position_valid ({num_checks} calls)"
        )
        result.calls = num_checks
        result.time_per_call = result.total_time / num_checks
        results["is_position_valid"] = result

        return results

    def profile_world_step(self) -> ProfilingResult:
        """Profile World.step() in isolation."""
        print("\n=== Profiling World.step ===")

        # Create world with scenario
        scenario = RoomsScenario(DEFAULT_ROOMS_SCENARIO_CONFIG, DEFAULT_WORLD_CONFIG)
        world = scenario.make_world(verbose=False)
        scenario.reset_world(world, np.random.default_rng(42))

        # Set random actions
        for agent in world.agents:
            agent.action.u = np.random.randn(2) * agent.u_range

        num_steps = 1000

        def test_world_step():
            for _ in range(num_steps):
                world.step()

        result = self.profile_function(
            test_world_step, f"World.step ({num_steps} calls)"
        )
        result.calls = num_steps
        result.time_per_call = result.total_time / num_steps

        return result

    def profile_scenario_operations(self) -> dict[str, ProfilingResult]:
        """Profile RoomsScenario operations."""
        print("\n=== Profiling RoomsScenario Operations ===")

        scenario = RoomsScenario(DEFAULT_ROOMS_SCENARIO_CONFIG, DEFAULT_WORLD_CONFIG)
        world = scenario.make_world(verbose=False)
        np_random = np.random.default_rng(42)

        results = {}

        # Profile reset_world
        def test_reset():
            scenario.reset_world(world, np_random)

        results["reset_world"] = self.profile_function(test_reset, "reset_world")

        # Profile observation
        scenario.reset_world(world, np_random)
        agent = world.agents[0]

        num_obs = 1000

        def test_observation():
            for _ in range(num_obs):
                _ = scenario.observation(agent, world)

        result = self.profile_function(
            test_observation, f"observation ({num_obs} calls)"
        )
        result.calls = num_obs
        result.time_per_call = result.total_time / num_obs
        results["observation"] = result

        # Profile reward
        def test_reward():
            for _ in range(num_obs):
                _ = scenario.reward(agent, world)

        result = self.profile_function(test_reward, f"reward ({num_obs} calls)")
        result.calls = num_obs
        result.time_per_call = result.total_time / num_obs
        results["reward"] = result

        # Profile info
        def test_info():
            for _ in range(num_obs):
                _ = scenario.info(agent, world)

        result = self.profile_function(test_info, f"info ({num_obs} calls)")
        result.calls = num_obs
        result.time_per_call = result.total_time / num_obs
        results["info"] = result

        return results

    def profile_rendering(self) -> ProfilingResult:
        """Profile rendering operations."""
        print("\n=== Profiling Rendering ===")

        env = gym.make(
            "contgrid/Rooms-v0",
            max_episode_steps=self.steps_per_episode,
            render_mode="rgb_array",
        )
        env.reset(seed=42)

        num_renders = 100

        def test_render():
            for _ in range(num_renders):
                _ = env.render()

        result = self.profile_function(test_render, f"render ({num_renders} calls)")
        result.calls = num_renders
        result.time_per_call = result.total_time / num_renders

        env.close()

        return result

    def run_detailed_cprofile(self) -> str:
        """Run cProfile on a full episode to get detailed stats."""
        print("\n=== Running Detailed cProfile ===")

        profiler = cProfile.Profile()

        profiler.enable()

        # Run multiple episodes
        for episode in range(self.num_episodes):
            env = gym.make(
                "contgrid/Rooms-v0", max_episode_steps=self.steps_per_episode
            )
            observation, info = env.reset(seed=42 + episode)

            for _ in range(self.steps_per_episode):
                action = env.action_space.sample()
                observation, reward, terminated, truncated, info = env.step(action)

                if terminated or truncated:
                    break

            env.close()

        profiler.disable()

        # Get stats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats("cumulative")
        ps.print_stats(50)  # Top 50 functions

        return s.getvalue()

    def generate_report(self, all_results: dict) -> str:
        """Generate a comprehensive report."""
        report = []
        report.append("=" * 80)
        report.append("CONTGRID PERFORMANCE PROFILING REPORT")
        report.append("=" * 80)
        report.append("\nConfiguration:")
        report.append(f"  Episodes: {self.num_episodes}")
        report.append(f"  Steps per episode: {self.steps_per_episode}")
        report.append("\n")

        # Calculate total time across all operations
        total_time = 0.0
        flat_results = []

        for category, results in all_results.items():
            if isinstance(results, dict):
                for result in results.values():
                    total_time += result.total_time
                    flat_results.append(result)
            else:
                total_time += results.total_time
                flat_results.append(results)

        # Calculate percentages
        for result in flat_results:
            result.percentage = (result.total_time / total_time) * 100

        # Sort by total time
        flat_results.sort(key=lambda x: x.total_time, reverse=True)

        report.append("\n" + "=" * 80)
        report.append("PERFORMANCE HOTSPOTS (Sorted by Total Time)")
        report.append("=" * 80)
        report.append(
            f"\n{'Operation':<40} {'Total (ms)':<12} {'Per Call (μs)':<15} {'Calls':<10} {'%':<8}"
        )
        report.append("-" * 80)

        for result in flat_results:
            report.append(
                f"{result.name:<40} "
                f"{result.total_time * 1000:<12.2f} "
                f"{result.time_per_call * 1000000:<15.2f} "
                f"{result.calls:<10} "
                f"{result.percentage:<8.2f}"
            )

        report.append("\n" + "=" * 80)
        report.append("RECOMMENDATIONS FOR RUST OPTIMIZATION")
        report.append("=" * 80)

        # Identify high-impact operations
        high_impact = [r for r in flat_results if r.percentage > 5.0 or r.calls > 1000]

        report.append("\nHigh-impact operations (>5% of total time or >1000 calls):")
        for i, result in enumerate(high_impact, 1):
            report.append(f"\n{i}. {result.name}")
            report.append(
                f"   - Total time: {result.total_time * 1000:.2f} ms ({result.percentage:.1f}%)"
            )
            report.append(f"   - Calls: {result.calls}")
            report.append(
                f"   - Time per call: {result.time_per_call * 1000000:.2f} μs"
            )

            # Provide recommendations
            if "collision" in result.name.lower():
                report.append(
                    "   - RECOMMENDATION: Excellent candidate for Rust optimization"
                )
                report.append("     * Involves heavy numerical computation")
                report.append("     * Called frequently in inner loops")
                report.append("     * Pure computational logic, easy to port")
            elif "World.step" in result.name:
                report.append("   - RECOMMENDATION: Consider Rust optimization")
                report.append("     * Core simulation loop")
                report.append("     * Contains physics calculations")
                report.append("     * May benefit from SIMD operations")
            elif (
                "observation" in result.name.lower() or "reward" in result.name.lower()
            ):
                report.append("   - RECOMMENDATION: Moderate priority for Rust")
                report.append("     * Involves distance calculations")
                report.append("     * Could benefit from vectorization")

        report.append("\n" + "=" * 80)
        report.append("SUMMARY")
        report.append("=" * 80)
        report.append(f"\nTotal profiled time: {total_time * 1000:.2f} ms")
        report.append(f"Number of operations profiled: {len(flat_results)}")
        report.append("\nTop 3 bottlenecks:")
        for i, result in enumerate(flat_results[:3], 1):
            report.append(f"  {i}. {result.name}: {result.percentage:.1f}%")

        return "\n".join(report)


def main():
    """Main profiling routine."""
    print("Starting ContGrid Performance Profiling...")
    print("This may take a few minutes...\n")

    profiler = PerformanceProfiler(num_episodes=10, steps_per_episode=100)

    all_results = {}

    # Profile environment operations
    env = gym.make("contgrid/Rooms-v0", max_episode_steps=profiler.steps_per_episode)
    all_results["env_creation"] = profiler.profile_environment_creation()
    all_results["reset"] = profiler.profile_reset(env)
    all_results["step"] = profiler.profile_step(env)
    env.close()

    all_results["full_episode"] = profiler.profile_full_episode()

    # Profile core components
    all_results["wall_collision"] = profiler.profile_wall_collision_checker()
    all_results["world_step"] = profiler.profile_world_step()
    all_results["scenario_ops"] = profiler.profile_scenario_operations()
    all_results["rendering"] = profiler.profile_rendering()

    # Generate report
    report = profiler.generate_report(all_results)
    print("\n" + report)

    # Save report to file
    output_file = "tests/out/performance_profile.txt"
    import os

    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        f.write(report)
    print(f"\n\nReport saved to: {output_file}")

    # Run detailed cProfile
    print("\n\nRunning detailed cProfile analysis...")
    detailed_stats = profiler.run_detailed_cprofile()

    detailed_file = "tests/out/cprofile_detailed.txt"
    with open(detailed_file, "w") as f:
        f.write(detailed_stats)
    print(f"Detailed cProfile report saved to: {detailed_file}")

    # Print top functions from cProfile
    print("\n" + "=" * 80)
    print("TOP 20 FUNCTIONS BY CUMULATIVE TIME (from cProfile)")
    print("=" * 80)
    print(detailed_stats.split("\n")[6:26])  # Print first 20 lines of stats


if __name__ == "__main__":
    main()
