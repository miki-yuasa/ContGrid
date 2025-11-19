use pyo3::prelude::*;

/// WallCollisionChecker implemented in Rust for performance
#[pyclass]
pub struct WallCollisionCheckerRust {
    wall_bounds: Vec<[f64; 4]>, // [min_x, max_x, min_y, max_y]
    l: f64,
    verbose: bool,
}

#[pymethods]
impl WallCollisionCheckerRust {
    #[new]
    fn new(layout: Vec<Vec<String>>, l: f64, verbose: bool) -> Self {
        let n_rows = layout.len();
        let mut walls = Vec::new();

        for (r, row) in layout.iter().enumerate() {
            for (c, char_str) in row.iter().enumerate() {
                if char_str == "#" {
                    // Bottom-left corner (anchor)
                    let x = c as f64 * l;
                    let y = (n_rows - 1 - r) as f64 * l;

                    // Wall boundaries
                    let min_x = x - l / 2.0;
                    let max_x = x + l / 2.0;
                    let min_y = y - l / 2.0;
                    let max_y = y + l / 2.0;

                    walls.push([min_x, max_x, min_y, max_y]);
                }
            }
        }

        if verbose {
            println!("Memoized CollisionChecker (Rust): Pre-processed {} wall segments.", walls.len());
        }

        WallCollisionCheckerRust {
            wall_bounds: walls,
            l,
            verbose,
        }
    }

    /// Check if there is a collision at the given robot position
    fn is_collision(
        &self,
        r: f64,
        c: f64,
        robot_pos: (f64, f64),
        collision_force: f64,
        contact_margin: f64,
    ) -> PyResult<(bool, [f64; 2])> {
        if self.wall_bounds.is_empty() {
            return Ok((false, [0.0, 0.0]));
        }

        let (x, y) = robot_pos;
        let r_eff = r + c;
        let r_eff_sq = r_eff * r_eff;

        let candidates = self.get_collision_candidates(r_eff, robot_pos);

        let mut force = [0.0, 0.0];
        let mut is_collided = false;

        for [wall_min_x, wall_max_x, wall_min_y, wall_max_y] in candidates {
            let closest_x = x.clamp(wall_min_x, wall_max_x);
            let closest_y = y.clamp(wall_min_y, wall_max_y);

            let delta_x = x - closest_x;
            let delta_y = y - closest_y;
            let dist_sq = delta_x * delta_x + delta_y * delta_y;

            if dist_sq < r_eff_sq {
                let dist = dist_sq.sqrt();
                let penetration = r_eff - dist;

                if dist > 1e-10 {
                    let force_magnitude = penetration / dist;
                    force[0] += delta_x * force_magnitude;
                    force[1] += delta_y * force_magnitude;
                }

                is_collided = true;
            }
        }

        if is_collided {
            force[0] *= collision_force;
            force[1] *= collision_force;
        }

        Ok((is_collided, force))
    }

    /// Check if a position is valid (no collision with walls)
    fn is_position_valid(&self, r: f64, allowed_overlap: f64, pos: (f64, f64)) -> bool {
        let (test_x, test_y) = pos;
        let min_distance = r - allowed_overlap;

        for [wall_min_x, wall_max_x, wall_min_y, wall_max_y] in &self.wall_bounds {
            let closest_x = test_x.clamp(*wall_min_x, *wall_max_x);
            let closest_y = test_y.clamp(*wall_min_y, *wall_max_y);
            let dx = test_x - closest_x;
            let dy = test_y - closest_y;
            let dist = (dx * dx + dy * dy).sqrt();

            if dist < min_distance {
                if self.verbose {
                    println!("Distance to wall: {} < min_distance: {}", dist, min_distance);
                    println!("- Position ({}, {}) is invalid due to wall at ({}, {}, {}, {})",
                             pos.0, pos.1, wall_min_x, wall_min_y, wall_max_x, wall_max_y);
                }
                return false;
            }
        }
        true
    }

    /// Support for deepcopy in Python
    fn __deepcopy__(&self, _memo: &Bound<'_, pyo3::types::PyDict>) -> Self {
        WallCollisionCheckerRust {
            wall_bounds: self.wall_bounds.clone(),
            l: self.l,
            verbose: self.verbose,
        }
    }

    /// Clip the new position to ensure no collision with walls
    fn clip_new_position(
        &self,
        r: f64,
        allowed_overlap: f64,
        curr_pos: (f64, f64),
        new_pos: (f64, f64),
    ) -> (f64, f64) {
        // If the new position is valid, return it as is
        if self.is_position_valid(r, allowed_overlap, new_pos) {
            if self.verbose {
                println!("New position is valid: ({}, {})", new_pos.0, new_pos.1);
            }
            return new_pos;
        }

        // If current position is invalid, return it (don't make things worse)
        if !self.is_position_valid(r, allowed_overlap, curr_pos) {
            if self.verbose {
                println!("Current position is already invalid, returning it");
            }
            return curr_pos;
        }

        // Analytically find the maximum valid scaling factor
        let (curr_x, curr_y) = curr_pos;
        let (new_x, new_y) = new_pos;

        // Movement vector
        let dx = new_x - curr_x;
        let dy = new_y - curr_y;

        // If no movement, return current position
        if dx.abs() < 1e-10 && dy.abs() < 1e-10 {
            return curr_pos;
        }

        // Find the minimum scaling factor that would cause collision
        let mut min_collision_t = 1.0;
        let min_distance = r - allowed_overlap;

        // Check collision with each wall
        for [wall_min_x, wall_max_x, wall_min_y, wall_max_y] in &self.wall_bounds {
            if let Some(t) = self.find_wall_collision_parameter(
                curr_x, curr_y, dx, dy,
                *wall_min_x, *wall_max_x, *wall_min_y, *wall_max_y,
                min_distance,
            ) {
                if t >= 0.0 && t < min_collision_t {
                    min_collision_t = t;
                }
            }
        }

        // Use a small safety margin to ensure we don't quite reach the collision point
        let safety_margin = 1e-6;
        let safe_t = (min_collision_t - safety_margin).max(0.0);

        let mut final_x = curr_x + safe_t * dx;
        let mut final_y = curr_y + safe_t * dy;

        // If safe_t is very small, we might end up not moving at all
        // Check if we can move along one axis (slide along the wall)
        if safe_t < 1e-6 {
            // Try moving only in x direction
            let test_pos_x = (new_x, curr_y);
            if self.is_position_valid(r, allowed_overlap, test_pos_x) {
                final_x = new_x;
                final_y = curr_y;
            } else {
                // Try moving only in y direction
                let test_pos_y = (curr_x, new_y);
                if self.is_position_valid(r, allowed_overlap, test_pos_y) {
                    final_x = curr_x;
                    final_y = new_y;
                }
            }
        }

        (final_x, final_y)
    }
}

impl WallCollisionCheckerRust {
    /// Get wall bounds that are candidates for collision
    fn get_collision_candidates(&self, r_eff: f64, robot_pos: (f64, f64)) -> Vec<[f64; 4]> {
        if self.wall_bounds.is_empty() {
            return Vec::new();
        }

        let (x, y) = robot_pos;

        let min_check_x = x - r_eff;
        let max_check_x = x + r_eff;
        let min_check_y = y - r_eff;
        let max_check_y = y + r_eff;

        self.wall_bounds
            .iter()
            .filter(|[wall_min_x, wall_max_x, wall_min_y, wall_max_y]| {
                *wall_min_x < max_check_x
                    && *wall_max_x > min_check_x
                    && *wall_min_y < max_check_y
                    && *wall_max_y > min_check_y
            })
            .copied()
            .collect()
    }

    /// Find the parameter t where a moving circle would first collide with a wall
    fn find_wall_collision_parameter(
        &self,
        start_x: f64,
        start_y: f64,
        dx: f64,
        dy: f64,
        wall_min_x: f64,
        wall_max_x: f64,
        wall_min_y: f64,
        wall_max_y: f64,
        min_distance: f64,
    ) -> Option<f64> {
        // Expand the wall by min_distance in all directions (Minkowski sum)
        let expanded_min_x = wall_min_x - min_distance;
        let expanded_max_x = wall_max_x + min_distance;
        let expanded_min_y = wall_min_y - min_distance;
        let expanded_max_y = wall_max_y + min_distance;

        let mut t_min: f64 = 0.0;
        let mut t_max: f64 = f64::INFINITY;

        // Check intersection with vertical walls (x-direction)
        if dx.abs() > 1e-10 {
            let t1 = (expanded_min_x - start_x) / dx;
            let t2 = (expanded_max_x - start_x) / dx;

            let (t1, t2) = if t1 > t2 { (t2, t1) } else { (t1, t2) };

            t_min = t_min.max(t1);
            t_max = t_max.min(t2);
        } else if start_x < expanded_min_x || start_x > expanded_max_x {
            return None;
        }

        // Check intersection with horizontal walls (y-direction)
        if dy.abs() > 1e-10 {
            let t1 = (expanded_min_y - start_y) / dy;
            let t2 = (expanded_max_y - start_y) / dy;

            let (t1, t2) = if t1 > t2 { (t2, t1) } else { (t1, t2) };

            t_min = t_min.max(t1);
            t_max = t_max.min(t2);
        } else if start_y < expanded_min_y || start_y > expanded_max_y {
            return None;
        }

        // Check if there's a valid intersection
        if t_min <= t_max && t_max >= 0.0 {
            Some(t_min)
        } else {
            None
        }
    }
}

/// Python module definition
#[pymodule]
fn contgrid_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<WallCollisionCheckerRust>()?;
    Ok(())
}
