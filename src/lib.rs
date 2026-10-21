use pyo3::prelude::*;

pub mod collision;
pub mod geom;
pub mod sampler;

pub use collision::WallCollisionCheckerRust;

/// Python module definition for contgrid_rust
#[pymodule]
fn contgrid_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WallCollisionCheckerRust>()?;
    Ok(())
}
