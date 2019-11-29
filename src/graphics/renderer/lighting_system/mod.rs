
use std::sync::Arc;
use vulkano::device::Queue;
use vulkano::framebuffer::{Subpass, RenderPassAbstract};
use vulkano::pipeline::{GraphicsPipelineAbstract, GraphicsPipeline};
use vulkano::buffer::{BufferAccess, ImmutableBuffer, BufferUsage, CpuAccessibleBuffer};
use crate::graphics::object::{Vertex3D, ObjectInstance, MeshAccess, ScreenVertex};
use vulkano::sync::GpuFuture;
use vulkano::pipeline::blend::{AttachmentBlend, BlendOp, BlendFactor};
use vulkano::descriptor::DescriptorSet;
use cgmath::{ Matrix4, SquareMatrix };
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState};
use vulkano::image::{AttachmentImage, ImageAccess, ImageViewAccess};

pub mod shadow_cone_light;
use shadow_cone_light::ShadedConeLight;
use std::cell::RefCell;
use crate::graphics::renderer::shadow_mapping::ShadowSource;

// Apply diffirent lighting methods

pub struct LightingPass {
    // Full screen VBO square
    vbo: Arc<dyn BufferAccess + Send + Sync>,

    shadow_cone_light: ShadedConeLight,

    view_projection: Matrix4<f32>,
}
impl LightingPass {
    pub fn new<R>(queue: Arc<Queue>, subpass: Subpass<R>) -> Self
        where R: RenderPassAbstract + Send + Sync + Clone + 'static
    {

        let vbo = {
            let (a, b) = ImmutableBuffer::from_iter(vec![
                ScreenVertex::new(-1.0, -1.0),
                ScreenVertex::new(-1.0,  1.0),
                ScreenVertex::new( 1.0, -1.0),
                ScreenVertex::new( 1.0,  1.0),
            ].iter().cloned(), BufferUsage::vertex_buffer(), queue.clone()).unwrap();
            b.flush().unwrap();
            a
        };

        let shadow_cone_light = shadow_cone_light::ShadedConeLight::new(
            queue.clone(),
            Subpass::clone(&subpass)
        );

        Self {
            vbo,
            shadow_cone_light,
            view_projection: Matrix4::identity(),
        }
    }

    pub fn set_view_projection(&mut self, vp: Matrix4<f32>) {
        if !self.view_projection.eq(&vp) {
            self.view_projection = vp;
            self.shadow_cone_light.to_world = Matrix4::invert(&vp).unwrap();
        }
    }

    pub fn set_attachments(&mut self,
                           diffuse_buffer: Arc<AttachmentImage>,
                           normal_buffer: Arc<AttachmentImage>,
                           depth_buffer: Arc<AttachmentImage>,
    )
    {
        self.shadow_cone_light.set_attachments(
            diffuse_buffer.clone(),
            normal_buffer.clone(),
            depth_buffer.clone(),
        );
    }

    pub fn render_source(&mut self, dyn_state: &DynamicState, light: Arc<RefCell<ShadowSource>>) -> Option<AutoCommandBuffer>
    {
        if !light.borrow().active { None }
        else { Some(self.shadow_cone_light.render(light, self.vbo.clone(), dyn_state)) }
    }
}
