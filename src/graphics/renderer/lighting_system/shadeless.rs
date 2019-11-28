
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

// Use given geometry and other stuff to render it all into final image
// Apply light sources in additive manner

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "\
#version 450

layout(location = 0) in vec3 position;
layout(location = 0) out vec2 v_screen_coords;

void main() {
    v_screen_coords = position.xy;
    gl_Position = vec4(position.xy, 0.0, 1.0);
}"
    }
}
mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
#version 450


layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_diffuse;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput u_depth;

layout(location = 0) out vec4 s_color;

layout(location = 0) in vec2 v_screen_coords;

void main() {
    float depth = subpassLoad(u_depth).x;
    if (depth >= 1.0) { discard; }
    s_color = subpassLoad(u_diffuse);
}"
    }
}


pub struct Shadeless {
    queue: Arc<Queue>,
    // No lighting or shadows, only material color
    shadeless_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    // Desc set with attachments
    attachment_set: Option<Arc<dyn DescriptorSet + Send + Sync>>,
}
impl Shadeless {
    pub fn new<R>(queue: Arc<Queue>, subpass: Subpass<R>) -> Self
        where R: RenderPassAbstract + Send + Sync + 'static
    {
        let shadeless_pipeline = {
            let vs = vs::Shader::load(queue.device().clone())
                .expect("failed to create shader module");
            let fs = fs::Shader::load(queue.device().clone())
                .expect("failed to create shader module");

            Arc::new(GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex3D>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_strip()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(fs.main_entry_point(), ())
                .blend_collective(AttachmentBlend {
                    enabled: true,
                    color_op: BlendOp::Add,
                    color_source: BlendFactor::One,
                    color_destination: BlendFactor::One,
                    alpha_op: BlendOp::Max,
                    alpha_source: BlendFactor::One,
                    alpha_destination: BlendFactor::One,
                    mask_red: true,
                    mask_green: true,
                    mask_blue: true,
                    mask_alpha: true,
                })
                .render_pass(subpass)
                .build(queue.device().clone())
                .unwrap()) as Arc<dyn GraphicsPipelineAbstract + Send + Sync>
        };

        Self {
            queue,
            shadeless_pipeline,
            attachment_set: None,
        }
    }

    pub fn set_attachments(&mut self,
                           diffuse_buffer: Arc<AttachmentImage>,
                           normal_buffer: Arc<AttachmentImage>,
                           depth_buffer: Arc<AttachmentImage>,
    )
    {
        self.attachment_set = Some(Arc::new(
            PersistentDescriptorSet::start(self.shadeless_pipeline.clone(), 0)
                .add_image(diffuse_buffer).unwrap()
                .add_image(normal_buffer).unwrap()
                .add_image(depth_buffer).unwrap()
                .build().unwrap()
        ));
    }


    pub fn render<'f>(&mut self, vbo: Arc<dyn BufferAccess + Send + Sync>, dyn_state: &DynamicState) -> AutoCommandBuffer
    {
        if self.attachment_set.is_none() {
            panic!("Attachments not specified, use set_attachments");
        }

        let attachment_set = self.attachment_set.as_ref().unwrap().clone();

        let mut cbb = AutoCommandBufferBuilder::secondary_graphics(
            self.queue.device().clone(),
            self.queue.family(),
            self.shadeless_pipeline.clone().subpass()
        ).unwrap()
            .draw(self.shadeless_pipeline.clone(), dyn_state, vec![vbo.clone()],
                  (attachment_set), ()
            ).unwrap();

        cbb.build().unwrap()
    }

}