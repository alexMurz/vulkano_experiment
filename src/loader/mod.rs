
pub mod blender;

#[derive(Debug, Copy, Clone)]
pub struct Face<'f> {
    pub vert_count: usize,
    pub vert: &'f [[f32; 3]],
    pub norm: &'f [[f32; 3]],
    pub uv: &'f [[f32; 2]],
}