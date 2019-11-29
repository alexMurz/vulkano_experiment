
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
use vulkano::image::{AttachmentImage, ImageAccess, ImageViewAccess};
use vulkano::sampler::{Sampler, Filter, MipmapMode, SamplerAddressMode, BorderColor};

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
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normal;
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput u_depth;

layout(location = 0) out vec4 s_color;

layout(set = 0, binding = 3) uniform sampler2D u_shadow;

layout(location = 0) in vec2 v_screen_coords;

layout(push_constant) uniform PushData {
    mat4 to_world;
    mat4 shadow_biased;
} push;


vec2 poissonDisk[4] = vec2[](
  vec2( -0.94201624, -0.39906216 ),
  vec2( 0.94558609, -0.76890725 ),
  vec2( -0.094184101, -0.92938870 ),
  vec2( 0.34495938, 0.29387760 )
);
float spread = 700.0;

// #define SHADOW_POISSON
float sampleShadow(vec4 shadow_coord, float bias) {
    float shade = 1.0;
#ifdef SHADOW_POISSON
    for (int i=0;i<4;i++){
        if (texture(u_shadow, shadow_coord.xy + poissonDisk[i]/spread).r < shadow_coord.z - bias) shade -= 0.2;
    }
#else
    if (texture(u_shadow, shadow_coord.xy).r < shadow_coord.z - bias) shade = 0.0;
#endif
    return shade;
}

void main() {
    float depth = subpassLoad(u_depth).x;
    if (depth >= 1.0) { discard; }

    vec4 world = push.to_world * vec4(v_screen_coords, depth, 1.0);
    world /= world.w;
    vec2 uv = v_screen_coords.xy * vec2(0.5, 0.5) + vec2(0.5, 0.5);
    vec3 normal = normalize(subpassLoad(u_normal).xyz);
    vec3 col = subpassLoad(u_diffuse).rgb;

    vec3 light_pos = vec3(0.0, -3.0, 0.0);
    vec3 L = normalize(light_pos - world.xyz);
    float cosTheta = dot(L, normal);

    float light_percent = max(abs(cosTheta), 0.0);
    float light_distance = length(L) / 20.0;
    light_percent *= 1.0 / exp(light_distance);

    vec4 shadow_coord = push.shadow_biased * world;
    shadow_coord /= shadow_coord.w;
    float shade = sampleShadow(shadow_coord, 0.05*tan(acos(-cosTheta)));

    s_color = vec4(col * shade * light_percent, 1.0);
}"
    }
}


pub struct Shadeless {
    queue: Arc<Queue>,
    // No lighting or shadows, only material color
    shadeless_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    // Desc set with attachments
    attachment_set: Option<Arc<dyn DescriptorSet + Send + Sync>>,

    // Sampler
    sampler: Arc<Sampler>,
    shadow_image: Arc<dyn ImageViewAccess + Send + Sync>,

    // Push constants
    pub to_world: Matrix4<f32>,
    pub shadow_biased: Matrix4<f32>,
}
impl Shadeless {
    pub fn new<R>(queue: Arc<Queue>, shadow_image: Arc<dyn ImageViewAccess + Send + Sync>, subpass: Subpass<R>) -> Self
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

        let sampler = Sampler::new(queue.device().clone(),
            Filter::Linear, Filter::Linear, MipmapMode::Linear,
            SamplerAddressMode::ClampToBorder(BorderColor::FloatOpaqueWhite),
            SamplerAddressMode::ClampToBorder(BorderColor::FloatOpaqueWhite),
            SamplerAddressMode::ClampToBorder(BorderColor::FloatOpaqueWhite),
            0.0, 1.0, 0.0, 0.0,
        ).unwrap();

        Self {
            queue,
            shadeless_pipeline,
            attachment_set: None,

            sampler,
            shadow_image,
            to_world: Matrix4::identity(),
            shadow_biased: Matrix4::identity(),
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
                .add_sampled_image(self.shadow_image.clone(), self.sampler.clone()).unwrap()
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
                  (attachment_set), (fs::ty::PushData {
                    to_world: self.to_world.into(),
                    shadow_biased: self.shadow_biased.into()
                })
            ).unwrap();

        cbb.build().unwrap()
    }

}