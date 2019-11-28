
pub mod old_renderer;
pub mod object;
pub mod renderer;
pub mod light;

use cgmath::{Matrix4, SquareMatrix, Vector3, Deg, Vector4, Matrix3};

pub struct Camera {
    pub pos: [f32; 3],
    pub angle: [f32; 3], // Degrees

    // Current direction (read only)
    pub right: [f32; 3],
    pub forward: [f32; 3],
    pub up: [f32; 3],

    pub dirty: bool,
    pub view: Matrix4<f32>,
    pub projection: Matrix4<f32>
}
impl Camera {
    pub fn new(projection: Matrix4<f32>) -> Self {
        Self {
            pos: [0.0, 0.0, 0.0],
            angle: [0.0, 0.0, 0.0],

            right: [1.0, 0.0, 0.0],
            forward: [0.0, 1.0, 0.0],
            up: [0.0, 0.0, 1.0],

            dirty: true,
            view: Matrix4::identity(),
            projection
        }
    }

    pub fn set_projection(&mut self, projection: Matrix4<f32>) {
        self.projection = projection;
    }

    fn update(&mut self) {
        if self.dirty {
            self.dirty = false;

            for a in self.angle.iter_mut() {
                if *a > 360.0 { *a %= 360.0; }
                if *a < -360.0 { *a = -((-*a) % 360.0); }
            }

            self.view = {
                let tr = Matrix4::from_translation(Vector3 {
                    x: self.pos[0], y: self.pos[1], z: self.pos[2]
                });
                let ax = Matrix4::from_angle_x(Deg(self.angle[0]));
                let ay = Matrix4::from_angle_y(Deg(self.angle[1]));
                let az = Matrix4::from_angle_z(Deg(self.angle[2]));
                ax * ay * az * tr
            };

            let normal = Matrix4::invert(&self.view).unwrap();

            self.right   = [normal[0][0], normal[0][1], normal[0][2]];
            self.up      = [normal[1][0], normal[1][1], normal[1][2]];
            self.forward = [normal[2][0], normal[2][1], normal[2][2]];
        }
    }

    pub fn get_view_projection(&mut self) -> Matrix4<f32> {
        self.update();
        self.projection * self.view
    }

    pub fn set_pos_arr(&mut self, pos: [f32; 3]) {
        if self.pos[0] != pos[0] || self.pos[1] != pos[1] || self.pos[2] != pos[2] {
            self.pos = pos;
            self.dirty = true;
        }
    }

    pub fn set_angle(&mut self, angle: [f32; 3]) {
        if self.angle[0] != angle[0] || self.angle[1] != angle[1] || self.angle[2] != angle[2] {
            self.angle = angle;
            self.dirty = true;
        }
    }

    pub fn move_by(&mut self, forward: f32, right: f32, up: f32) {
        if forward != 0.0 || right != 0.0 || up != 0.0 {
            self.dirty = true;
            let mut ddx = 0.0f32;
            let mut ddy = 0.0f32;
            let mut ddz = 0.0f32;
            if forward != 0.0 {
                ddx += self.forward[0] * forward;
                ddy += self.forward[1] * forward;
                ddz += self.forward[2] * forward;
            }
            if right != 0.0 {
                ddx += self.right[0] * right;
                ddy += self.right[1] * right;
                ddz += self.right[2] * right;
            }
            if up != 0.0 {
                ddx += self.up[0] * up;
                ddy += self.up[1] * up;
                ddz += self.up[2] * up;
            }
            self.pos[0] -= ddx;
            self.pos[1] -= ddy;
            self.pos[2] -= ddz;
        }
    }

}