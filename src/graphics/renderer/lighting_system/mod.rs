
use std::sync::Arc;
use vulkano::device::Queue;
use vulkano::framebuffer::{Subpass, RenderPassAbstract};
use vulkano::pipeline::{GraphicsPipelineAbstract, GraphicsPipeline};
use vulkano::buffer::{BufferAccess, ImmutableBuffer, BufferUsage, CpuAccessibleBuffer};
use crate::graphics::object::{Vertex3D, ObjectInstance, MeshAccess};
use vulkano::sync::GpuFuture;
use vulkano::pipeline::blend::{AttachmentBlend, BlendOp, BlendFactor};
use vulkano::descriptor::DescriptorSet;
use cgmath::{ Matrix4, SquareMatrix };
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState};
use vulkano::image::AttachmentImage;
use crate::graphics::renderer::lighting_system::shadeless::Shadeless;


pub mod shadeless;


// Apply diffirent lighting methods

pub struct LightingPass {
    // Full screen VBO square
    vbo: Arc<dyn BufferAccess + Send + Sync>,

    // Only use is then debugging
    shadeless: Shadeless,
}
impl LightingPass {
    pub fn new<R>(queue: Arc<Queue>, subpass: Subpass<R>) -> Self
        where R: RenderPassAbstract + Send + Sync + Clone + 'static
    {

        let vbo = {
            let (a, b) = ImmutableBuffer::from_iter(vec![
                Vertex3D::from_position(-1.0, -1.0, 0.0).uv(0.0, 0.0),
                Vertex3D::from_position(-1.0,  1.0, 0.0).uv(0.0, 1.0),
                Vertex3D::from_position( 1.0, -1.0, 0.0).uv(1.0, 0.0),
                Vertex3D::from_position( 1.0,  1.0, 0.0).uv(1.0, 1.0),
            ].iter().cloned(), BufferUsage::vertex_buffer(), queue.clone()).unwrap();
            b.flush().unwrap();
            a
        };

        let shadeless_pass = shadeless::Shadeless::new(queue.clone(), Subpass::clone(&subpass));

        Self {
            vbo,
            shadeless: shadeless_pass
        }
    }

    pub fn set_view_projection(&mut self, vp: Matrix4<f32>) {}

    pub fn set_attachments(&mut self,
                           diffuse_buffer: Arc<AttachmentImage>,
                           normal_buffer: Arc<AttachmentImage>,
                           depth_buffer: Arc<AttachmentImage>,
    )
    {
        self.shadeless.set_attachments(
            diffuse_buffer.clone(),
            normal_buffer.clone(),
            depth_buffer.clone(),
        );
    }

    pub fn render(&mut self, dyn_state: &DynamicState) -> AutoCommandBuffer {
        self.shadeless.render(self.vbo.clone(), dyn_state)
    }
}
