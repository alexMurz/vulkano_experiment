
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
use vulkano::sampler::{Sampler, Filter, MipmapMode, SamplerAddressMode, BorderColor};

use std::cell::RefCell;
use crate::graphics::renderer::lighting_system::{
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

layout(location = 0) in vec2 position;
layout(location = 0) out vec2 v_screen_coords;

void main() {
    v_screen_coords = position;
    gl_Position = vec4(position, 0.0, 1.0);
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
    float pow;
} push;

void main() {
    float depth = subpassLoad(u_depth).x;
    if (depth >= 1.0) { discard; }

    vec3 col = subpassLoad(u_diffuse).rgb;

    s_color = vec4(col * push.pow, 1.0);
}"
    }
}

const SHADOW_BIAS: Matrix4<f32> = Matrix4::new(
    0.5, 0.0, 0.0, 0.0,
    0.0, 0.5, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.5, 0.5, 0.0, 1.0,
);

pub struct AmbientLight {
    queue: Arc<Queue>,
    // No lighting or shadows, only material color
    pub pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    // Desc set with attachments
    attachment_set: Option<Arc<dyn DescriptorSet + Send + Sync>>,
    // Push constants
    pub to_world: Matrix4<f32>,
}
impl AmbientLight {
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



        Self {
            queue,
            pipeline,
            attachment_set: None,
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
                    pow: source.get_pow().into(),
                })
            ).unwrap();

        cbb.build().unwrap()

        
//        // Write matrix data to buffer
//
//        let light_data_buffer = CpuAccessibleBuffer::from_data(
//            self.queue.device().clone(), BufferUsage::uniform_buffer(),
//            fs::ty::LightData {
//                shadow_biased: (SHADOW_BIAS * source.view_projection).into(),
//                light_pow: 20.0.into(),
//                light_pos: source.light_pos.into()
//            }
//        ).unwrap();
//
////        {
////            let mut writer = self.light_data_buffer.write().unwrap();
////            writer.shadow_biased = (SHADOW_BIAS * source.view_projection).into();
////            writer.light_pos = source.light_pos.into();
////        }
//
//        // Create desc set TODO: Make source be owner of set
//        let light_data_set = Arc::new(
//            PersistentDescriptorSet::start(self.pipeline.clone(), 1)
////                .add_image(source.image.clone()).unwrap()
////                .add_sampler(self.sampler.clone()).unwrap()
//                .add_sampled_image(source.image.clone(), self.sampler.clone()).unwrap()
//                .add_buffer(light_data_buffer.clone()).unwrap()
//                .build().unwrap()
//        );
//
//        let attachment_set = self.attachment_set.as_ref().unwrap().clone();
//
//        let mut cbb = AutoCommandBufferBuilder::secondary_graphics(
//            self.queue.device().clone(),
//            self.queue.family(),
//            self.pipeline.clone().subpass()
//        ).unwrap()
//            .draw(self.pipeline.clone(), dyn_state, vec![vbo.clone()],
//                  (attachment_set, light_data_set), (fs::ty::PushData {
//                    to_world: self.to_world.into(),
//                })
//            ).unwrap();
//
//        cbb.build().unwrap()
    }

}