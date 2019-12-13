
use std::sync::Arc;
use vulkano::device::Queue;
use vulkano::framebuffer::{Subpass, RenderPassAbstract};
use vulkano::pipeline::{GraphicsPipelineAbstract, GraphicsPipeline};
use vulkano::buffer::{BufferAccess, ImmutableBuffer, BufferUsage, CpuAccessibleBuffer};
use crate::graphics::object::{ ScreenVertex };
use vulkano::pipeline::blend::{AttachmentBlend, BlendOp, BlendFactor};
use vulkano::descriptor::DescriptorSet;
use cgmath::{ Matrix4, SquareMatrix };
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState};
use vulkano::image::{AttachmentImage, ImageAccess, ImageViewAccess};
use vulkano::sampler::{Sampler, Filter, MipmapMode, SamplerAddressMode, BorderColor};

use std::cell::RefCell;
use crate::graphics::renderer_3d::lighting_system::{
    LightSource, LightKind, ShadowKind
};

// Use given geometry and other stuff to render it all into final image
// Apply cone light source and its shadow in additive manner
// Same rules for all kinds of lighting

pub mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "\
#version 450

layout(location = 0) in vec2 pos;
layout(location = 0) out vec2 v_screen_coords;

void main() {
    v_screen_coords = pos;
    gl_Position = vec4(pos, 0.0, 1.0);
}"
    }
}
pub mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
#version 450


layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_diffuse;
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normal;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput u_depth;

layout(location = 0) out vec4 s_color;

layout(location = 0) in vec2 v_screen_coords;

layout(push_constant) uniform PushData {
    mat4 to_world;
    vec3 light_pos;
    vec3 light_col;
    float light_pow;
} push;


void main() {
    float depth = subpassLoad(u_depth).x;
    if (depth >= 1.0) { discard; }

    vec4 world = push.to_world * vec4(v_screen_coords, depth, 1.0);
    world /= world.w;
    vec3 normal = normalize(-subpassLoad(u_normal).xyz);
    vec3 col = subpassLoad(u_diffuse).rgb;

    vec3 light_pos = push.light_pos;
    float light_pow = push.light_pow;

    vec3 L = normalize(light_pos - world.xyz);
    float cosTheta = dot(L, normal);

    float light_distance = length(light_pos - world.xyz);
    float light_percent = max(cosTheta * sqrt(1.0 - light_distance/light_pow), 0.0);

    s_color = vec4(col * light_percent * push.light_col, 1.0);
}"
    }
}


pub struct PointLight {
    queue: Arc<Queue>,
    // No lighting or shadows, only material color
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    // Desc set with attachments
    attachment_set: Option<Arc<dyn DescriptorSet + Send + Sync>>,

    // Sampler
    sampler: Arc<Sampler>,

    // Push constants
    pub to_world: Matrix4<f32>,
}
impl PointLight {
    pub fn new<R>(queue: Arc<Queue>,
                  subpass: Subpass<R>) -> Self
        where R: RenderPassAbstract + Send + Sync + 'static
    {
        let pipeline = {
            let vs = vs::Shader::load(queue.device().clone())
                .expect("failed to create shader module");
            let fs = fs::Shader::load(queue.device().clone())
                .expect("failed to create shader module");

            Arc::new(GraphicsPipeline::start()
                .vertex_input_single_buffer::<ScreenVertex>()
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


        let sampler = Sampler::new(queue.device().clone(),
            Filter::Linear, Filter::Linear, MipmapMode::Linear,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            SamplerAddressMode::ClampToEdge,
            0.0, 1.0, 0.0, 0.0,
        ).unwrap();

        Self {
            queue,
            pipeline,
            attachment_set: None,

            sampler,

            to_world: Matrix4::identity(),
        }
    }

    pub fn set_attachments(&mut self,
                           diffuse_buffer: Arc<AttachmentImage>,
                           normal_buffer: Arc<AttachmentImage>,
                           depth_buffer: Arc<AttachmentImage>,
    )
    {
        self.attachment_set = Some(Arc::new(
            PersistentDescriptorSet::start(self.pipeline.clone(), 0)
                .add_image(diffuse_buffer).unwrap()
                .add_image(normal_buffer).unwrap()
                .add_image(depth_buffer).unwrap()
                .build().unwrap()
        ));
    }


    pub fn render<'f>(&mut self,
                      source: &mut LightSource,
                      vbo: Arc<dyn BufferAccess + Send + Sync>,
                      dyn_state: &DynamicState) -> AutoCommandBuffer
    {
        assert!(self.attachment_set.is_some()); // Check for color, normal, depth attachments

        let attachment_set = self.attachment_set.clone().unwrap();

        let mut cbb = AutoCommandBufferBuilder::secondary_graphics(
            self.queue.device().clone(),
            self.queue.family(),
            self.pipeline.clone().subpass()
        ).unwrap()
            .draw(self.pipeline.clone(), dyn_state, vec![vbo.clone()],
                  (attachment_set), (fs::ty::PushData {
                    _dummy0: [0; 4].into(),
                    to_world: self.to_world.into(),
                    light_pow: source.get_dist().into(),
                    light_pos: source.get_pos().into(),
                    light_col: source.get_col().into(),
                })
            ).unwrap();

        cbb.build().unwrap()
    }

}