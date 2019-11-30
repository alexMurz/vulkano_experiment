
use std::sync::Arc;
use vulkano::device::Queue;
use vulkano::framebuffer::{Subpass, RenderPassAbstract};
use vulkano::pipeline::{GraphicsPipelineAbstract, GraphicsPipeline};
use vulkano::buffer::{BufferAccess, ImmutableBuffer, BufferUsage, CpuAccessibleBuffer};
use crate::graphics::object::{Vertex3D, ObjectInstance, MeshAccess, ScreenVertex};
use vulkano::sync::GpuFuture;
use vulkano::pipeline::blend::{AttachmentBlend, BlendOp, BlendFactor};
use vulkano::descriptor::DescriptorSet;
use cgmath::{Matrix4, SquareMatrix, vec3, Vector3, InnerSpace};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState};
use vulkano::image::{AttachmentImage, ImageAccess, ImageViewAccess};

mod light_draw_systems;
pub mod shadow_mapper;
pub mod lighting_pass;


pub enum LightKind {

    // Ambient light
    Ambient,

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
    use cgmath::{Matrix4, Rad, vec3, Point3, SquareMatrix};
    use vulkano::image::{ImageViewAccess, ImageAccess, AttachmentImage};
    use std::sync::Arc;
    use vulkano::framebuffer::FramebufferAbstract;
    use vulkano::buffer::{BufferAccess, CpuAccessibleBuffer};
    use vulkano::descriptor::DescriptorSet;
    use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
    use std::any::Any;

    // Data type for data_buffer
    use crate::graphics::renderer::lighting_system::light_draw_systems::shadow_cone_light;

    pub struct Cone {
        proj_rad: Rad<f32>,
        proj: Matrix4<f32>, // Projection matrix
        pub vp: Matrix4<f32>,

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
            proj_rad: Rad(0.0),
            proj: Matrix4::identity(),
            vp: Matrix4::identity(),
            resolution: [1, 1],
            image: None,
            framebuffer: None,
            data_changed: true,
            data_buffer: None,
            data_set: None,
        }}
    }
    impl Cone {
        pub fn with_projection<A: Into<Rad<f32>>>(angle: A, distance: f32) -> Self {
            let mut c = Self::default();
            c.set_projection(angle, distance);
            c
        }
        pub fn set_projection<A: Into<Rad<f32>>>(&mut self, angle: A, distance: f32) {
            self.proj_rad = angle.into();
            self.proj = cgmath::perspective(self.proj_rad, 1.0, 1.0, distance.max(1.1));
            self.data_changed = true;
        }
        pub fn update(&mut self, pos: &[f32; 3], dir: &[f32; 3], pow: f32) {
            // TODO: Angle from direction (0.0) for now
            self.set_projection(self.proj_rad, pow);
            self.vp = self.proj * Matrix4::look_at(
                Point3::new(pos[0], pos[1], pos[2]),
                Point3::new(pos[0] - dir[0], pos[1] - dir[1], pos[2] - dir[2]),
                vec3(0.0, 1.0, 0.0)
            );
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
    pow: f32
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
            pow: 1.0
        }
    }

    pub fn pos(&mut self, x: f32, y: f32, z: f32) {
        self.pos = [x, y, z];
        self.dirty = true;
    }
    pub fn pos_vec(&mut self, v: [f32; 3]) {
        self.pos = v;
        self.dirty = true;
    }

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
        let v = Vector3::normalize(vec3(x - self.pos[0], y - self.pos[1], z - self.pos[2]));
        self.dir(v.x, v.y, v.z)
    }

    pub fn col(&mut self, r: f32, g: f32, b: f32) {
        self.col = [r, g, b];
    }
    pub fn col_vec(&mut self, v: [f32; 3]) {
        self.col = v;
    }

    pub fn get_pow(&self) -> f32 { self.pow }
    pub fn pow(&mut self, pow: f32) {
        self.pow = pow;
        self.dirty = true;
    }

    pub fn set_dirty(&mut self) { self.dirty = true; }
    pub fn update(&mut self) {
        if self.dirty {
            match &mut self.kind {
                LightKind::ConeWithShadow(cone) => cone.update(&self.pos, &self.dir, self.pow),
                _ => ()
            }
            self.dirty = false;
        }
    }
}










