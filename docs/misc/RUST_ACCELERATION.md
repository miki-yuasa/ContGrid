# Rust Performance Acceleration

ContGrid now includes optional Rust acceleration for performance-critical collision detection operations.

## Performance Impact

With Rust acceleration enabled:
- **~650x faster** collision checking
- **~500x faster** position clipping
- **~3.4x overall speedup** in simulation (when rendering is enabled)
- **~875x faster** for training workloads (rendering disabled)

## Installation

### Prerequisites
- Rust toolchain: https://rustup.rs/
- Maturin (already in dev dependencies)

### Building the Extension

```bash
# Activate your virtual environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Build and install the Rust extension (release mode for max performance)
maturin develop --release
```

That's it! The environment will automatically use Rust acceleration when available.

## Verification

Check if Rust acceleration is active:

```python
from contgrid.core.grid_rust import HAS_RUST
print(f"Rust acceleration: {'✅ Enabled' if HAS_RUST else '❌ Not available'}")
```

Run the integration test:

```bash
python tests/test_rust_integration.py
```

Run the benchmark to see performance gains:

```bash
python scripts/benchmark_rust.py
```

## Fallback Behavior

If the Rust extension is not available (e.g., Rust not installed, build failed), ContGrid automatically falls back to the pure Python implementation without any code changes required.

## Development

### Rebuilding After Changes

```bash
# Quick rebuild (faster compilation, less optimization)
maturin develop

# Release rebuild (slower compilation, maximum optimization)
maturin develop --release
```

### Running Tests

```bash
# Run all tests (including Rust integration)
pytest

# Run only Rust-specific tests
python tests/test_rust_integration.py
```

## Technical Details

See [RUST_OPTIMIZATION.md](RUST_OPTIMIZATION.md) for detailed performance analysis and implementation details.

## Files

- `src/lib.rs` - Rust implementation (PyO3 bindings)
- `contgrid/core/grid_rust.py` - Python wrapper with fallback
- `scripts/benchmark_rust.py` - Performance benchmarks
- `Cargo.toml` - Rust project configuration
- `RUST_OPTIMIZATION.md` - Detailed performance report
