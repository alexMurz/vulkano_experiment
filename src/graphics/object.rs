
use vulkano::{
    device::Queue,
    buffer::{ BufferAccess, TypedBufferAccess, BufferUsage, ImmutableBuffer },
    descriptor::{
        PipelineLayoutAbstract,
        descriptor_set::{ DescriptorSet, PersistentDescriptorSet },
    },
    pipeline::{
        GraphicsPipelineAbstract,
    },
    sampler::Sampler,
    image::{ ImageViewAccess, ImageAccess },
    sync::GpuFuture,
};

use cgmath::{Matrix4, SquareMatrix, Vector3, Deg, Vector4, Matrix3, Matrix, BaseFloat, vec3};
use std::sync::Arc;
use cgmath_culling::{FrustumCuller, BoundingBox, Intersection};
use crate::graphics::renderer_3d::mesh::{ Vertex3D };


// Simplified vertex for working with screening
#[derive(Default, Copy, Clone)]
pub struct ScreenVertex {
    pub pos: [f32; 2],
    pub col: [f32; 4],
    pub uv: [f32; 2],
}
impl ScreenVertex {
    pub fn empty() -> Self { ScreenVertex::default() }

    pub fn with_pos(x: f32, y: f32) -> Self { Self {
        pos: [x, y], .. ScreenVertex::empty()
    } }
    #[inline] pub fn pos(mut self, x: f32, y: f32) -> Self { self.pos = [x, y]; self }
    #[inline] pub fn get_pos(&self) -> [f32; 2] { self.pos }
    #[inline] pub fn get_pos_x(&self) -> f32 { self.pos[0] }
    #[inline] pub fn get_pos_y(&self) -> f32 { self.pos[1] }

    pub fn with_col(r: f32, g: f32, b: f32, a: f32) -> Self { Self {
        col: [r, g, b, a], .. ScreenVertex::empty()
    } }
    #[inline] pub fn uni_color(mut self, i: f32, a: f32) -> Self { self.col = [i, i, i, a]; self }
    #[inline] pub fn col(mut self, r: f32, g: f32, b: f32, a: f32) -> Self { self.col = [r, g, b, a]; self }
    #[inline] pub fn get_col(&self) -> [f32; 4] { self.col }
    #[inline] pub fn get_col_r(&self) -> f32 { self.col[0] }
    #[inline] pub fn get_col_g(&self) -> f32 { self.col[1] }
    #[inline] pub fn get_col_b(&self) -> f32 { self.col[2] }
    #[inline] pub fn get_col_a(&self) -> f32 { self.col[3] }

    pub fn with_uv(u: f32, v: f32) -> Self { Self {
        uv: [u, v], .. ScreenVertex::empty()
    } }
    #[inline] pub fn uv(mut self, u: f32, v: f32) -> Self { self.uv = [u, v]; self }
    #[inline] pub fn get_uv(&self) -> [f32; 2] { self.uv }
    #[inline] pub fn get_uv_x(&self) -> f32 { self.uv[0] }
    #[inline] pub fn get_uv_y(&self) -> f32 { self.uv[1] }

}
vulkano::impl_vertex!(ScreenVertex, pos, col, uv);

#[derive(Copy, Clone)]
pub struct ScreenInstance {
    pub inst_transform: [[f32; 4]; 4],
    pub inst_color: [f32; 4],
    pub inst_uv_a: [f32; 2],
    pub inst_uv_b: [f32; 2],
}
impl Default for ScreenInstance {
    fn default() -> Self { Self {
        inst_transform: Matrix4::identity().into(),
        inst_color: [1.0; 4],
        inst_uv_a: [0.0; 2],
        inst_uv_b: [1.0; 2],
    } }
}
impl ScreenInstance {
    pub fn new() -> Self { Self::default() }

    pub fn set_transform<R: Into<cgmath::Rad<f32>>>(&mut self, x: f32, y: f32, w: f32, h: f32, rot: R) {
        self.inst_transform = (Matrix4::from_translation(vec3(x, y, 0.0))
            * Matrix4::from_angle_z(rot)
            * Matrix4::from_nonuniform_scale(w, h, 1.0)).into();
    }

    pub fn set_color(&mut self, r: f32, g: f32, b: f32, a: f32) { self.inst_color = [r,g,b,a] }

    pub fn set_uv(&mut self, u0: f32, v0: f32, u1: f32, v1: f32) {
        self.inst_uv_a = [u0, v0];
        self.inst_uv_b = [u1, v1];
    }

}
vulkano::impl_vertex!(ScreenInstance, inst_transform, inst_color, inst_uv_a, inst_uv_b);










