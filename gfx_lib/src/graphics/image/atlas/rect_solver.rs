// ##########
// Fit multiple rectangles into one

use crate::utils::NextPot;

/// Rectangle struct with key identifier
pub struct Rect<K> {
    pub key: K, // Rect identifier
    pub pos: [u32; 2],
    pub size: [u32; 2],
    pub rotated: bool, // Rotated CW 90 degrees
}
impl <K> std::fmt::Debug for Rect<K> {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(f, "[{:?}, {:?}, {}]", self.pos, self.size, self.rotated)
    }
}
impl <K> Rect<K> {
    pub fn new(key: K, w: u32, h: u32) -> Self { Self {
        key,
        pos: [0, 0],
        size: [w, h],
        rotated: false,
    } }
}

/// Enum of errors that may occur during solving, none is critical and ::solve should never panic
/// just try again with different parameters
pub enum SolverError {
    ImageIsTooBig, // Then one of images dimension cannot fit inside max bounds
    AreaIsTooBig, // Then area of all images is too big for max bounds
    CannotFitOnOnePage(usize), // Currently only used to generate single page atlases
}
impl std::error::Error for SolverError {}
impl std::fmt::Debug for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        match self {
            SolverError::ImageIsTooBig => write!(f, "One of rectangles is too big to fit inside max bounds"),
            SolverError::AreaIsTooBig => write!(f, "Area of given rectangles is bigger then max bounds"),
            SolverError::CannotFitOnOnePage(idx) => write!(f, "Cannot fit all rectangles on one page. Failed at {}", idx),
        }
    }
}
impl std::fmt::Display for SolverError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        (self as &dyn std::fmt::Debug).fmt(f)
    }
}

/// Empty space that can be used to put rect into
#[derive(Debug, Copy, Clone)]
struct Space { pos: (u32, u32), size: (u32, u32) }

/// Solver for collection of rectangles
pub struct Solver {
    max_dims: u32,
    padding: [u32; 2],
    can_rotate: bool,
}
impl Solver {
    pub fn with_params(max_dims: u32, padding: [u32; 2], can_rotate: bool) -> Self { Self {
        max_dims,
        padding,
        can_rotate,
    } }

    /// Solve for given params, return true is succeeded, else false
    fn solve_for<K>(dims: u32, pads: [u32; 2], can_rotate: bool, rects: &mut Vec<Rect<K>>) -> bool {

        let mut spaces = vec![Space { pos: (0, 0), size: (dims, dims) }];

        let px = pads[0] * 2;
        let py = pads[1] * 2;

        'outer: for (idx, rect) in rects.iter_mut().enumerate() {

            for space_idx in 0 .. spaces.len() {
                let space = spaces[space_idx];

                // Search for space to fit rect
                let mut fits = true;
                let mut rotated = false;
                if rect.size[0] + px > space.size.0 || rect.size[1] + py > space.size.1 { fits = false; }
                // Try rotated
                if !fits && can_rotate && (rect.size[1] + px > space.size.0 || rect.size[0] + py > space.size.1) {
                    fits = false;
                    rotated = true;
                }
                // Cant fit in this space? => Continue to the next
                if !fits { continue }

                rect.rotated = rotated;
                rect.pos = [space.pos.0 + pads[0], space.pos.1 + pads[1]];
                if rotated { rect.size = [rect.size[1], rect.size[0]]; }

                // If space exits, put image in said space, then reduce and process remaining space

                let rw = rect.size[0] + px;
                let rh = rect.size[1] + py;

                // if fully fits, remove space
                if space.size.0 == rw && space.size.1 == rh {
                    spaces.remove(space_idx);
                }
                else if space.size.1 == rh {
                    // Full height, change pos x and width
                    spaces[space_idx] = Space {
                        pos: (space.pos.0 + rw, space.pos.1),
                        size: (space.size.0 - rw, space.size.1)
                    }
                }
                else if space.size.0 == rw {
                    // Full width, change pos y and height
                    spaces[space_idx] = Space {
                        pos: (space.pos.0, space.pos.1 + rh),
                        size: (space.size.0, space.size.1 - rh)
                    }
                }
                else {
                    // Use lower space as updated and add space to the right as new

                    // Bottom
                    spaces[space_idx] = Space {
                        pos: (space.pos.0, space.pos.1 + rh),
                        size: (rw, space.size.1 - rh)
                    };

                    // Right
                    spaces.push(Space {
                        pos: (space.pos.0 + rw, space.pos.1),
                        size: (space.size.0 - rw, space.size.1)
                    })
                }

                continue 'outer;
            }
            // if None available, return error
//            panic!("{:?} :: fit: {:?}", spaces, rect);
            return false;
        }

        true
    }

    /// Return minimum possible POT rects was packed into
    pub fn solve<K>(&self, rects: &mut Vec<Rect<K>>) -> Result<[u32; 2], SolverError> {
        let mut total_rects_area = 0;
        for r in rects.iter() {
            if r.size[0] + self.padding[0]*2 > self.max_dims || r.size[1] + self.padding[1]*2 > self.max_dims {
                return Err(SolverError::ImageIsTooBig);
            }
            total_rects_area += (r.size[0] + self.padding[0]*2) * (r.size[1] + self.padding[1]*2);
        }

        let given_area = self.max_dims * self.max_dims;

        if total_rects_area > given_area { return Err(SolverError::AreaIsTooBig) }

        let max_pot = self.max_dims.next_pot();
        let mut pot = ((total_rects_area as f32).sqrt() as u32).next_pot();

        for p in pot .. max_pot+1 {
            let dim = 2.0f32.powf(p as f32) as u32;
            if Self::solve_for(dim, self.padding, self.can_rotate, rects) {
                return Ok([dim; 2])
            }
        }

        Err(SolverError::CannotFitOnOnePage(0))
    }
    /*
//    https://observablehq.com/@mourner/simple-rectangle-packing
    result = {
  // calculate total box area and maximum box width
  let area = 0;
  let maxWidth = 0;
  for (const box of boxes) {
    area += box.w * box.h;
    maxWidth = Math.max(maxWidth, box.w);
  }

  // sort the boxes for insertion by height, descending
  boxes.sort((a, b) => b.h - a.h);

  // aim for a squarish resulting container,
  // slightly adjusted for sub-100% space utilization
  const startWidth = Math.max(Math.ceil(Math.sqrt(area / 0.95)), maxWidth);

  // start with a single empty space, unbounded at the bottom
  const spaces = [{x: 0, y: 0, w: startWidth, h: Infinity}];
  const packed = [];

  for (const box of boxes) {
    // look through spaces backwards so that we check smaller spaces first
    for (let i = spaces.length - 1; i >= 0; i--) {
      const space = spaces[i];

      // look for empty spaces that can accommodate the current box
      if (box.w > space.w || box.h > space.h) continue;

      // found the space; add the box to its top-left corner
      // |-------|-------|
      // |  box  |       |
      // |_______|       |
      // |         space |
      // |_______________|
      packed.push(Object.assign({}, box, {x: space.x, y: space.y}));

      if (box.w === space.w && box.h === space.h) {
        // space matches the box exactly; remove it
        const last = spaces.pop();
        if (i < spaces.length) spaces[i] = last;

      } else if (box.h === space.h) {
        // space matches the box height; update it accordingly
        // |-------|---------------|
        // |  box  | updated space |
        // |_______|_______________|
        space.x += box.w;
        space.w -= box.w;

      } else if (box.w === space.w) {
        // space matches the box width; update it accordingly
        // |---------------|
        // |      box      |
        // |_______________|
        // | updated space |
        // |_______________|
        space.y += box.h;
        space.h -= box.h;

      } else {
        // otherwise the box splits the space into two spaces
        // |-------|-----------|
        // |  box  | new space |
        // |_______|___________|
        // | updated space     |
        // |___________________|
        spaces.push({
          x: space.x + box.w,
          y: space.y,
          w: space.w - box.w,
          h: box.h
        });
        space.y += box.h;
        space.h -= box.h;
      }
      break;
    }
    yield {packed, spaces};
  }
}
    */
}