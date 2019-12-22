
// Bake Image, flip Y coordinate, sense Vulkan considers TopLeft corner being 0 and
// everything else BottomLeft

use std::{
    sync::Arc,
};

use vulkano::{
    device::Queue,
    pipeline::{ GraphicsPipeline, GraphicsPipelineAbstract },
    descriptor::DescriptorSet,
    sampler::Sampler,
    sync::{ self, GpuFuture },
};
use vulkano::framebuffer::{Subpass, RenderPassAbstract};
use crate::graphics::object::ScreenVertex;
use vulkano::buffer::{BufferUsage, BufferAccess, ImmutableBuffer};
use vulkano::image::AttachmentImage;
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "\
#version 450

layout(location = 0) in vec2 pos;

void main() {
    gl_Position = vec4(pos, 0.0, 1.0);
}"
    }
}
mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
#version 450

layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_color;
layout(location = 0) out vec4 s_color;

void main() {
    vec3 col = subpassLoad(u_color).rgb;
    s_color = vec4(col, 1.0);
}"
    }
}


/// Bake image, with some extra corrections
pub struct PostBakeImage {
    queue: Arc<Queue>,
    vbo: Arc<dyn BufferAccess + Send + Sync>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    attachment_set: Option<Arc<dyn DescriptorSet + Send + Sync>>,
}
impl PostBakeImage {
    pub fn new<R>(queue: Arc<Queue>, subpass: Subpass<R>) -> Self
        where R: RenderPassAbstract + Send + Sync + 'static
    {

        let pipeline = {
            let vs = vs::Shader::load(queue.device().clone())
                .expect("failed to create shader module");
            let fs = fs::Shader::load(queue.device().clone())
                .expect("failed to create shader module");

            Arc::new(GraphicsPipeline::start()
                .vertex_input_single_buffer::<ScreenVertex>()
//                .vertex_input_single_buffer::<ScreenVertex>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(fs.main_entry_point(), ())
                .render_pass(subpass)
                .build(queue.device().clone())
                .unwrap()) as Arc<dyn GraphicsPipelineAbstract + Send + Sync>
        };

        let vbo = {
            let s = 1.0;
            let (a, b) = ImmutableBuffer::from_iter(vec![
                ScreenVertex::with_pos(-s, -s).uv(0.0, 0.0),
                ScreenVertex::with_pos(-s,  s).uv(0.0, 1.0),
                ScreenVertex::with_pos( s, -s).uv(1.0, 0.0),
                ScreenVertex::with_pos(-s,  s).uv(0.0, 1.0),
                ScreenVertex::with_pos( s,  s).uv(1.0, 1.0),
                ScreenVertex::with_pos( s, -s).uv(1.0, 0.0),
            ].iter().cloned(), BufferUsage::vertex_buffer(), queue.clone()).unwrap();
            b.flush().unwrap();
            a
        };

        Self {
            queue,
            vbo,
            pipeline,
            attachment_set: None,
        }
    }

    pub fn set_attachment(&mut self, transient: Arc<AttachmentImage>) {
        self.attachment_set = Some(Arc::new(
            PersistentDescriptorSet::start(self.pipeline.clone(), 0)
                .add_image(transient).unwrap()
                .build().unwrap()
        ));
    }

    pub fn render(&self, cbb: AutoCommandBufferBuilder, dyn_state: &DynamicState) -> AutoCommandBufferBuilder {

        assert!(self.attachment_set.is_some()); // Check for color, normal, depth attachments

        let attachment_set = self.attachment_set.clone().unwrap();

        cbb.draw(self.pipeline.clone(), dyn_state, vec![self.vbo.clone()],
                 (attachment_set), ()
        ).unwrap()
    }
}


