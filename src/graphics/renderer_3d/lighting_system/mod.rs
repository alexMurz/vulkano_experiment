
use std::sync::Arc;
use vulkano::{
    device::Queue,
    framebuffer::{ Subpass, RenderPassAbstract },
    image::{ AttachmentImage, ImageAccess, ImageViewAccess },
    command_buffer::{ AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState },
    pipeline::{
        GraphicsPipelineAbstract, GraphicsPipeline,
        blend::{ AttachmentBlend, BlendOp, BlendFactor },
    },
    descriptor::{
        DescriptorSet,
        descriptor_set::PersistentDescriptorSet
    },
    buffer::{ BufferAccess, ImmutableBuffer, BufferUsage, CpuAccessibleBuffer },

    sync::GpuFuture,
};

use crate::graphics::object::{ ScreenVertex };
use cgmath::{Matrix4, SquareMatrix, vec3, Vector3, InnerSpace};
use std::cell::RefCell;

mod light_draw_systems;
pub mod shadow_mapper;
pub mod lighting_pass;

pub enum LightKind {
    // Ambient light
    Ambient,

    // Point light without shadows
    PointLight,

    // Cone shadow with light
    ConeWithShadow(ShadowKind::Cone)
}
impl LightKind {
    fn has_shadow(&self) -> bool {
        match self {
            LightKind::ConeWithShadow(_) => true,
            _ => false
        }
    }
}

#[allow(non_snake_case)]
pub mod ShadowKind {
    use cgmath::{Matrix4, vec3, Point3, SquareMatrix};
    use vulkano::image::{AttachmentImage};
    use std::sync::Arc;
    use vulkano::framebuffer::FramebufferAbstract;
    use vulkano::buffer::{CpuAccessibleBuffer};
    use vulkano::descriptor::DescriptorSet;

    // Data type for data_buffer
    use crate::graphics::renderer_3d::lighting_system::light_draw_systems::shadow_cone_light;

    pub struct Cone {
        proj_deg: f32,
        proj: Matrix4<f32>, // Projection matrix
        pub vp: Matrix4<f32>,

        // Just retranslation from LightSource
        pub dist: f32,
        pub pos: [f32; 3],
        pub col: [f32; 3],

        // Mapper Data
        pub resolution: [u32; 2], // Shadow resolution, managed by shadow mapper
        pub image: Option<Arc<AttachmentImage>>, // depth map of shadow, managed by shadow mapper
        pub framebuffer: Option<Arc<dyn FramebufferAbstract + Send + Sync>>, // Framebuffer used for shadow mapping
        // Lighting Data
        pub data_changed: bool, // Then need to update data
        pub data_buffer: Option<Arc<CpuAccessibleBuffer<shadow_cone_light::fs::ty::LightData>>>, // Data passed to rendering pipeline, managed by renderer, data is specified for renderer type
        pub data_set: Option<Arc<dyn DescriptorSet + Send + Sync>>
    }
    impl Default for Cone {
        fn default() -> Self { Self {
            proj_deg: 90.0,
            proj: Matrix4::identity(),
            vp: Matrix4::identity(),
            resolution: [256, 256],
            dist: 20.0,
            pos: [0.0, 0.0, 0.0],
            col: [1.0, 1.0, 1.0],
            image: None,
            framebuffer: None,
            data_changed: true,
            data_buffer: None,
            data_set: None,
        }}
    }
    impl Cone {
        pub fn with_projection(deg: f32, resolution: [u32; 2]) -> Self {
            let mut c = Self::default();
            c.resolution = resolution;
            c.set_projection(deg, 20.0);
            c
        }
        pub fn set_projection(&mut self, deg: f32, distance: f32) {
            self.proj_deg = deg;
            self.proj = cgmath::perspective(cgmath::Deg(deg), 1.0, 1.0,  distance.max(1.1));
            self.data_changed = true;
        }
        pub fn update(&mut self, pos: &[f32; 3], dir: &[f32; 3], col: &[f32; 3], dist: f32) {
            self.pos = *pos;
            self.col = *col;
            self.dist = dist;
            self.set_projection(self.proj_deg, dist);
            self.vp = self.proj *
                Matrix4::look_at(
                Point3::new(pos[0], pos[1], pos[2]),
                Point3::new(pos[0] + dir[0], pos[1] + dir[1], pos[2] + dir[2]),
                vec3(0.0, 1.0, 0.0)
            );

            println!("Update VP to {:?}", pos);
            self.data_changed = true;
        }
    }
}

pub struct LightSource {
    pub active: bool,
    dirty: bool, // Then transform is modified
    kind: LightKind,
    pos: [f32; 3],
    dir: [f32; 3],
    col: [f32; 3],
    dist: f32
}
/// Compare LightSources by address
impl PartialEq for LightSource {
    fn eq(&self, other: &LightSource) -> bool { self as *const _ == other as *const _ }
}
impl LightSource {
    pub fn new(kind: LightKind) -> Self {
        Self {
            active: true,
            dirty: false,
            kind,
            pos: [0.0, 0.0, 0.0],
            dir: [0.0, 1.0, 0.0],
            col: [1.0, 1.0, 1.0],
            dist: 1.0
        }
    }

    pub fn get_pos(&self) -> [f32; 3] { self.pos }
    pub fn pos(&mut self, x: f32, y: f32, z: f32) {
        self.pos = [x, y, z];
        self.dirty = true;
    }
    pub fn pos_vec(&mut self, v: [f32; 3]) {
        self.pos = v;
        self.dirty = true;
    }

    pub fn get_dir(&self) -> [f32; 3] { self.dir }
    pub fn dir(&mut self, x: f32, y: f32, z: f32) {
        self.dir = [x, y, z];
        self.dirty = true;
    }
    pub fn dir_vec(&mut self, v: [f32; 3]) {
        self.dir = v;
        self.dirty = true;
    }

    /// Modifies dir for current pos,
    pub fn look_at(&mut self, x: f32, y: f32, z: f32) {
        let v = Vector3::normalize(vec3(self.pos[0] - x, self.pos[1] - y, self.pos[2] - z));
        self.dir(-v.x, -v.y, -v.z)
    }

    pub fn get_col(&self) -> [f32; 3] { self.col }
    pub fn col(&mut self, r: f32, g: f32, b: f32) {
        self.col = [r, g, b];
    }
    pub fn col_vec(&mut self, v: [f32; 3]) { self.col = v; }

    /// Modify light intensity [0.0 .. 1.0]
    pub fn int(&mut self, int: f32) {
        let m = int / self.col[0].max(self.col[1].max(self.col[2]));
        self.col[0] *= m;
        self.col[1] *= m;
        self.col[2] *= m;
    }

    pub fn get_dist(&self) -> f32 { self.dist }
    pub fn dist(&mut self, dist: f32) {
        self.dist = dist;
        self.dirty = true;
    }

    pub fn set_dirty(&mut self) { self.dirty = true; }
    pub fn update(&mut self) {
        if self.dirty {
            match &mut self.kind {
                LightKind::ConeWithShadow(cone) => cone.update(&self.pos, &self.dir, &self.col, self.dist),
                _ => ()
            }
            self.dirty = false;
        }
    }
}









