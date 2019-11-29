
use vulkano::buffer::BufferAccess;
use vulkano::descriptor::DescriptorSet;

use cgmath::{Matrix4, SquareMatrix, Vector3, Deg, Vector4, Matrix3};
use std::sync::Arc;

#[derive(Default, Copy, Clone)]
pub struct Vertex3D {
    position: [f32; 3], // Vertex position in Model Space
    color: [f32; 4], // Vertex tint
    normal: [f32; 3], // Vertex Normal
    // if 0 use flat shading
    // flat shading is calculated internally and does not use provided normal
    flat_shading: u32,
    uv: [f32; 2], // Texture UV
    material_id: u32, // Index of material
}
impl Vertex3D {
    #[inline]
    pub fn empty() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            normal: [0.0, 0.0, 0.0],
            flat_shading: 0,
            uv: [0.0, 0.0],
            material_id: 0
        }
    }
    #[inline]
    pub fn from_position(x: f32, y: f32, z: f32) -> Self {
        Self {
            position: [x, y, z],
            .. Vertex3D::empty()
        }
    }

    #[inline]
    pub fn position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.position = [x, y, z];
        self
    }
    #[inline]
    pub fn normal(mut self, x: f32, y: f32, z: f32) -> Self {
        self.normal = [x, y, z];
        self
    }
    #[inline]
    pub fn color(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.color = [r, g, b, a];
        self
    }
    #[inline]
    pub fn flat_shading(mut self, flag: bool) -> Self {
        self.flat_shading = if flag { 1 } else { 0 };
        self
    }
    #[inline]
    pub fn uv(mut self, u: f32, v: f32) -> Self {
        self.uv = [u, v];
        self
    }
    #[inline]
    pub fn material(mut self, id: u32) -> Self {
        self.material_id = id;
        self
    }

}
vulkano::impl_vertex!(Vertex3D, position, color, normal, flat_shading, uv, material_id);

// Simplified vertex for working with screening
#[derive(Default, Copy, Clone)]
pub struct ScreenVertex {
    position: [f32; 2]
}
impl ScreenVertex {
    pub fn new(x: f32, y: f32) -> Self { Self { position: [x, y] } }
}
vulkano::impl_vertex!(ScreenVertex, position);

pub trait MeshAccess {
    fn get_vbo(&self) -> Arc<dyn BufferAccess + Send + Sync>;

    fn has_ibo(&self) -> bool;
    fn get_ibo(&self) -> Arc<dyn BufferAccess + Send + Sync>;
}

/// Describes mesh geometry with optional
/// Should be reused if possible
pub struct MeshData {
    pub vbo: Arc<dyn BufferAccess + Send + Sync + 'static>,
    pub ibo: Option<Arc<dyn BufferAccess + Send + Sync + 'static>>,
}
impl MeshAccess for MeshData {
    fn get_vbo(&self) -> Arc<dyn BufferAccess + Send + Sync> { self.vbo.clone() }

    fn has_ibo(&self) -> bool { self.ibo.is_some() }
    fn get_ibo(&self) -> Arc<dyn BufferAccess + Send + Sync> { self.ibo.as_ref().unwrap().clone() }
}

/// Describes material information
pub struct Material {
    pub diffuse: [f32; 3],
}

pub struct ObjectInstance {
    // Vertex info
    pub mesh_data: Arc<dyn MeshAccess + Send + Sync>,

    // Model matrix info
    pub pos: [f32; 3],
    pub scl: [f32; 3],
    pub rot: [f32; 3], // Radians

    // Contains buffers for material info
//    descriptor_set: Box<dyn DescriptorSet + Send + Sync>,
}
/// Clone instance and reuse mesh data
impl Clone for ObjectInstance {
    fn clone(&self) -> Self {
        Self {
            mesh_data: self.mesh_data.clone(),
            pos: self.pos,
            scl: self.scl,
            rot: self.rot,
        }
    }
}
impl MeshAccess for ObjectInstance {
    fn get_vbo(&self) -> Arc<dyn BufferAccess + Send + Sync> { self.mesh_data.get_vbo() }

    fn has_ibo(&self) -> bool { self.mesh_data.has_ibo() }
    fn get_ibo(&self) -> Arc<dyn BufferAccess + Send + Sync> { self.mesh_data.get_ibo() }
}
impl ObjectInstance {

    pub fn model_matrix(&self) -> Matrix4<f32> {
        Matrix4::from_nonuniform_scale(self.scl[0], self.scl[1], self.scl[2])
            * Matrix4::from_angle_x(cgmath::Rad(self.rot[0]))
            * Matrix4::from_angle_y(cgmath::Rad(self.rot[1]))
            * Matrix4::from_angle_z(cgmath::Rad(self.rot[2]))
            * Matrix4::from_translation(cgmath::vec3(self.pos[0], self.pos[1], self.pos[2]))
    }

    pub fn set_pos(&mut self, x: f32, y: f32, z: f32) {
        self.pos = [x, y, z];
    }

}