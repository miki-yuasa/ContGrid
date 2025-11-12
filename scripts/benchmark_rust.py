"""
Benchmark script to compare Python vs Rust implementation performance.

This script measures the performance improvement from using the Rust-accelerated
collision detection functions.
"""

import time
from typing import Callable

import numpy as np

from contgrid.core.grid import WallCollisionChecker as PythonChecker
from contgrid.core.grid_rust import (
    HAS_RUST,
)
from contgrid.core.grid_rust import (
    WallCollisionCheckerAccelerated as RustChecker,
)
from contgrid.envs.rooms import DEFAULT_WORLD_CONFIG


def benchmark_function(func: Callable, name: str, iterations: int = 10000) -> float:
    """Benchmark a function and return the average time per call in microseconds."""
    # Warm up
    for _ in range(100):
        func()

    # Actual benchmark
    start = time.perf_counter()
    for _ in range(iterations):
        func()
    end = time.perf_counter()

    total_time = end - start
    avg_time_us = (total_time / iterations) * 1_000_000

    print(f"{name:40s}: {avg_time_us:8.2f} μs/call ({total_time * 1000:.2f} ms total)")

    return avg_time_us


def main():
    """Run benchmarks comparing Python and Rust implementations."""
    print("=" * 80)
    print("PERFORMANCE BENCHMARK: Python vs Rust")
    print("=" * 80)

    if not HAS_RUST:
        print("\n⚠️  WARNING: Rust extension not available!")
        print("Run: maturin develop --release")
        print("\nFalling back to Python-only benchmark...\n")

    grid = DEFAULT_WORLD_CONFIG.grid
    R = 0.25  # Robot radius
    C = 0.1  # Contact margin
    iterations = 10000

    print(f"\nBenchmark Configuration:")
    print(f"  Grid size: {grid.height_cells} x {grid.width_cells}")
    print(f"  Cell size: {grid.cell_size}")
    print(f"  Robot radius: {R}")
    print(f"  Contact margin: {C}")
    print(f"  Iterations: {iterations}")

    # Initialize checkers
    python_checker = PythonChecker(grid.layout, grid.cell_size, verbose=False)

    if HAS_RUST:
        rust_checker = RustChecker(grid.layout, grid.cell_size, verbose=False)
        print(f"  Rust acceleration: ✅ Available")
    else:
        rust_checker = None
        print(f"  Rust acceleration: ❌ Not available")

    print("\n" + "=" * 80)
    print("1. is_collision() Benchmark")
    print("=" * 80)

    def python_is_collision():
        pos = (5.0 + np.random.rand() * 0.1, 5.0 + np.random.rand() * 0.1)
        python_checker.is_collision(R, C, pos, 100.0, 0.1)

    python_time = benchmark_function(
        python_is_collision, "Python is_collision", iterations
    )

    if HAS_RUST and rust_checker:

        def rust_is_collision():
            pos = (5.0 + np.random.rand() * 0.1, 5.0 + np.random.rand() * 0.1)
            rust_checker.is_collision(R, C, pos, 100.0, 0.1)

        rust_time = benchmark_function(
            rust_is_collision, "Rust is_collision", iterations
        )

        speedup = python_time / rust_time
        print(f"\n  Speedup: {speedup:.2f}x faster with Rust")

    print("\n" + "=" * 80)
    print("2. is_position_valid() Benchmark")
    print("=" * 80)

    def python_is_valid():
        pos = (5.0 + np.random.rand() * 0.1, 5.0 + np.random.rand() * 0.1)
        python_checker.is_position_valid(R, C, pos)

    python_time = benchmark_function(
        python_is_valid, "Python is_position_valid", iterations
    )

    if HAS_RUST and rust_checker:

        def rust_is_valid():
            pos = (5.0 + np.random.rand() * 0.1, 5.0 + np.random.rand() * 0.1)
            rust_checker.is_position_valid(R, C, pos)

        rust_time = benchmark_function(
            rust_is_valid, "Rust is_position_valid", iterations
        )

        speedup = python_time / rust_time
        print(f"\n  Speedup: {speedup:.2f}x faster with Rust")

    print("\n" + "=" * 80)
    print("3. clip_new_position() Benchmark")
    print("=" * 80)

    def python_clip():
        curr_pos = (5.0, 5.0)
        new_pos = (
            5.0 + np.random.rand() * 0.01,
            5.0 + np.random.rand() * 0.01,
        )
        python_checker.clip_new_position(R, C, curr_pos, new_pos)

    python_time = benchmark_function(
        python_clip, "Python clip_new_position", iterations
    )

    if HAS_RUST and rust_checker:

        def rust_clip():
            curr_pos = (5.0, 5.0)
            new_pos = (
                5.0 + np.random.rand() * 0.01,
                5.0 + np.random.rand() * 0.01,
            )
            rust_checker.clip_new_position(R, C, curr_pos, new_pos)

        rust_time = benchmark_function(rust_clip, "Rust clip_new_position", iterations)

        speedup = python_time / rust_time
        print(f"\n  Speedup: {speedup:.2f}x faster with Rust")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if HAS_RUST:
        print("\n✅ Rust acceleration is working!")
        print("\nThe Rust implementation provides significant performance improvements")
        print("for the most time-critical collision detection functions.")
        print("\nNext steps:")
        print("  - The environment will automatically use Rust acceleration")
        print("  - Run your training/evaluation scripts to see the speedup")
        print("  - Profile again to confirm the performance improvements")
    else:
        print("\n⚠️  Rust acceleration not available")
        print("\nTo enable Rust acceleration:")
        print("  1. Ensure Rust is installed: https://rustup.rs/")
        print("  2. Build the extension: maturin develop --release")
        print("  3. Re-run this benchmark to see the performance improvements")


if __name__ == "__main__":
    main()
