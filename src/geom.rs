/// Minimum length below which a segment is treated as degenerate.
pub const MIN_SEGMENT_LENGTH: f64 = 1e-6;

/// Threshold for treating line directions as parallel to boundary axes.
pub const PARALLEL_LINE_EPSILON: f64 = 1e-10;

/// Internal 2D line segment with precomputed directional unit vectors.
#[derive(Clone, Copy)]
pub struct InternalLineSeg {
    pub x1: f64,
    pub y1: f64,
    pub x2: f64,
    pub y2: f64,
    pub dx: f64,
    pub dy: f64,
    pub length: f64,
    pub ux: f64,
    pub uy: f64,
    pub px: f64,
    pub py: f64,
}

impl InternalLineSeg {
    pub fn new(x1: f64, y1: f64, x2: f64, y2: f64) -> Self {
        let dx = x2 - x1;
        let dy = y2 - y1;
        let length = (dx * dx + dy * dy).sqrt();
        let (ux, uy, px, py) = if length > MIN_SEGMENT_LENGTH {
            let inv_len = 1.0 / length;
            let u_x = dx * inv_len;
            let u_y = dy * inv_len;
            (u_x, u_y, -u_y, u_x)
        } else {
            (1.0, 0.0, 0.0, 1.0)
        };
        Self {
            x1,
            y1,
            x2,
            y2,
            dx,
            dy,
            length,
            ux,
            uy,
            px,
            py,
        }
    }

    #[inline]
    pub fn sample_with_noise(&self, t: f64, n1: f64, n2: f64, std: f64) -> (f64, f64) {
        let bx = self.x1 + t * self.dx;
        let by = self.y1 + t * self.dy;
        if std > 0.0 {
            (
                bx + (n1 * self.ux + n2 * self.px) * std,
                by + (n1 * self.uy + n2 * self.py) * std,
            )
        } else {
            (bx, by)
        }
    }
}

/// Clip a line segment to an axis-aligned bounding box with an edge buffer using Liang-Barsky.
pub fn clip_segment_to_box(
    x1: f64,
    y1: f64,
    x2: f64,
    y2: f64,
    bounds: [f64; 4],
    edge_buffer: f64,
) -> Option<InternalLineSeg> {
    let min_x = bounds[0] + edge_buffer;
    let max_x = bounds[1] - edge_buffer;
    let min_y = bounds[2] + edge_buffer;
    let max_y = bounds[3] - edge_buffer;

    let dx = x2 - x1;
    let dy = y2 - y1;

    let mut t0: f64 = 0.0;
    let mut t1: f64 = 1.0;

    let p = [-dx, dx, -dy, dy];
    let q = [x1 - min_x, max_x - x1, y1 - min_y, max_y - y1];

    for i in 0..4 {
        if p[i].abs() < PARALLEL_LINE_EPSILON {
            if q[i] < 0.0 {
                return None;
            }
        } else {
            let t = q[i] / p[i];
            if p[i] < 0.0 {
                t0 = t0.max(t);
            } else {
                t1 = t1.min(t);
            }
        }
    }

    if t0 > t1 {
        return None;
    }

    let cx1 = x1 + t0 * dx;
    let cy1 = y1 + t0 * dy;
    let cx2 = x1 + t1 * dx;
    let cy2 = y1 + t1 * dy;

    let seg = InternalLineSeg::new(cx1, cy1, cx2, cy2);
    if seg.length > MIN_SEGMENT_LENGTH {
        Some(seg)
    } else {
        None
    }
}
