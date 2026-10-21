use pyo3::prelude::*;
use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

use crate::collision::WallCollisionCheckerRust;
use crate::geom::{clip_segment_to_box, InternalLineSeg, MIN_SEGMENT_LENGTH};

/// Maximum number of random Gaussian perturbation attempts per obstacle.
const MAX_RANDOM_ATTEMPTS: usize = 100;

/// Number of deterministic equidistant candidate samples per segment during fallback.
const FALLBACK_NUM_SAMPLES_PER_SEGMENT: usize = 20;

/// Normalized segment range start for fallback grid evaluation.
const FALLBACK_T_START: f64 = 0.05;

/// Normalized segment range end for fallback grid evaluation.
const FALLBACK_T_END: f64 = 0.95;

/// Maximum random perturbation added to deterministic parameter t in fallback.
const FALLBACK_T_JITTER: f64 = 0.02;

/// Lower clamp limit for parameter t along a segment.
const FALLBACK_T_MIN: f64 = 0.01;

/// Upper clamp limit for parameter t along a segment.
const FALLBACK_T_MAX: f64 = 0.99;

#[inline]
pub fn check_candidate_valid(
    checker: &WallCollisionCheckerRust,
    cx: f64,
    cy: f64,
    obstacle_radius: f64,
    min_spacing_sq: f64,
    goal_pos: (f64, f64),
    goal_clearance_sq: f64,
    agent_pos: Option<(f64, f64)>,
    min_agent_dist_sq: f64,
    all_obstacles: &[(f64, f64)],
) -> bool {
    // Wall collision check
    if !checker.is_position_valid(obstacle_radius, 0.0, (cx, cy)) {
        return false;
    }

    // Distance to existing & already-placed obstacles
    for &(ox, oy) in all_obstacles {
        let dx = cx - ox;
        let dy = cy - oy;
        if dx * dx + dy * dy < min_spacing_sq {
            return false;
        }
    }

    // Distance to goal
    let gdx = cx - goal_pos.0;
    let gdy = cy - goal_pos.1;
    if gdx * gdx + gdy * gdy < goal_clearance_sq {
        return false;
    }

    // Distance to agent
    if let Some((ax, ay)) = agent_pos {
        let adx = cx - ax;
        let ady = cy - ay;
        if adx * adx + ady * ady < min_agent_dist_sq {
            return false;
        }
    }

    true
}

pub fn sample_path_gaussian(
    checker: &WallCollisionCheckerRust,
    segments: Vec<[f64; 4]>,
    num_obstacles: usize,
    obstacle_radius: f64,
    gaussian_std: f64,
    min_spacing: f64,
    edge_buffer: f64,
    grid_bounds: [f64; 4],
    room_bounds: Vec<Option<[f64; 4]>>,
    existing_obstacles: Vec<(f64, f64)>,
    goal_pos: (f64, f64),
    goal_clearance: f64,
    agent_pos: Option<(f64, f64)>,
    min_agent_dist: f64,
    seed: u64,
) -> PyResult<Vec<(f64, f64)>> {
    if num_obstacles == 0 || segments.is_empty() {
        return Ok(Vec::new());
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let normal_dist = if gaussian_std > 0.0 {
        Normal::new(0.0, gaussian_std).ok()
    } else {
        None
    };

    let mut all_obstacles = existing_obstacles;
    let mut positions = Vec::with_capacity(num_obstacles);

    let min_spacing_sq = min_spacing * min_spacing;
    let goal_clearance_sq = goal_clearance * goal_clearance;
    let min_agent_dist_sq = min_agent_dist * min_agent_dist;

    let base_segs: Vec<InternalLineSeg> = segments
        .iter()
        .map(|s| InternalLineSeg::new(s[0], s[1], s[2], s[3]))
        .filter(|s| s.length > MIN_SEGMENT_LENGTH)
        .collect();

    if base_segs.is_empty() {
        return Ok(Vec::new());
    }

    for i in 0..num_obstacles {
        let req_room_bound = room_bounds.get(i).copied().flatten();

        let (target_segs, clip_min_x, clip_max_x, clip_min_y, clip_max_y) = if let Some(rb) = req_room_bound {
            let mut clipped = Vec::new();
            for s in &segments {
                if let Some(cs) = clip_segment_to_box(s[0], s[1], s[2], s[3], rb, edge_buffer) {
                    clipped.push(cs);
                }
            }
            let segs = if clipped.is_empty() {
                base_segs.clone()
            } else {
                clipped
            };
            (
                segs,
                rb[0] + edge_buffer,
                rb[1] - edge_buffer,
                rb[2] + edge_buffer,
                rb[3] - edge_buffer,
            )
        } else {
            (
                base_segs.clone(),
                grid_bounds[0] + edge_buffer,
                grid_bounds[1] - edge_buffer,
                grid_bounds[2] + edge_buffer,
                grid_bounds[3] - edge_buffer,
            )
        };

        if target_segs.is_empty() {
            continue;
        }

        let mut cum_lens = Vec::with_capacity(target_segs.len());
        let mut total_len = 0.0;
        for s in &target_segs {
            total_len += s.length;
            cum_lens.push(total_len);
        }

        let mut found_pos = None;

        for _ in 0..MAX_RANDOM_ATTEMPTS {
            let r_val: f64 = rng.gen::<f64>() * total_len;
            let seg_idx = match cum_lens.binary_search_by(|v| v.partial_cmp(&r_val).unwrap()) {
                Ok(idx) => idx,
                Err(idx) => idx.min(target_segs.len() - 1),
            };
            let seg = &target_segs[seg_idx];
            let t: f64 = rng.gen::<f64>();
            let (n1, n2) = if let Some(dist) = normal_dist {
                (dist.sample(&mut rng), dist.sample(&mut rng))
            } else {
                (0.0, 0.0)
            };

            let (raw_cx, raw_cy) = seg.sample_with_noise(t, n1, n2, gaussian_std);

            let (cx, cy) = if req_room_bound.is_some() {
                (
                    raw_cx.clamp(clip_min_x, clip_max_x),
                    raw_cy.clamp(clip_min_y, clip_max_y),
                )
            } else {
                if raw_cx < clip_min_x || raw_cx > clip_max_x || raw_cy < clip_min_y || raw_cy > clip_max_y {
                    continue;
                }
                (raw_cx, raw_cy)
            };

            if check_candidate_valid(
                checker,
                cx,
                cy,
                obstacle_radius,
                min_spacing_sq,
                goal_pos,
                goal_clearance_sq,
                agent_pos,
                min_agent_dist_sq,
                &all_obstacles,
            ) {
                found_pos = Some((cx, cy));
                break;
            }
        }

        if found_pos.is_none() {
            let mut seg_indices: Vec<usize> = (0..target_segs.len()).collect();
            seg_indices.shuffle(&mut rng);

            let t_span = FALLBACK_T_END - FALLBACK_T_START;
            let num_intervals = (FALLBACK_NUM_SAMPLES_PER_SEGMENT - 1) as f64;
            let mut t_values: Vec<f64> = (0..FALLBACK_NUM_SAMPLES_PER_SEGMENT)
                .map(|k| FALLBACK_T_START + t_span * (k as f64) / num_intervals)
                .collect();

            'outer: for seg_idx in seg_indices {
                let seg = &target_segs[seg_idx];
                t_values.shuffle(&mut rng);

                for &t in &t_values {
                    let t_jitter = rng.gen_range(-FALLBACK_T_JITTER..FALLBACK_T_JITTER);
                    let t_perturbed = (t + t_jitter).clamp(FALLBACK_T_MIN, FALLBACK_T_MAX);
                    let (n1, n2) = if let Some(dist) = normal_dist {
                        (dist.sample(&mut rng), dist.sample(&mut rng))
                    } else {
                        (0.0, 0.0)
                    };
                    let (raw_cx, raw_cy) = seg.sample_with_noise(t_perturbed, n1, n2, gaussian_std);

                    let (cx, cy) = if req_room_bound.is_some() {
                        (
                            raw_cx.clamp(clip_min_x, clip_max_x),
                            raw_cy.clamp(clip_min_y, clip_max_y),
                        )
                    } else {
                        if raw_cx < clip_min_x || raw_cx > clip_max_x || raw_cy < clip_min_y || raw_cy > clip_max_y {
                            continue;
                        }
                        (raw_cx, raw_cy)
                    };

                    if check_candidate_valid(
                        checker,
                        cx,
                        cy,
                        obstacle_radius,
                        min_spacing_sq,
                        goal_pos,
                        goal_clearance_sq,
                        agent_pos,
                        min_agent_dist_sq,
                        &all_obstacles,
                    ) {
                        found_pos = Some((cx, cy));
                        break 'outer;
                    }
                }
            }
        }

        if let Some(pos) = found_pos {
            all_obstacles.push(pos);
            positions.push(pos);
        }
    }

    Ok(positions)
}
