
use std::sync::Arc;
use vulkano::device::Queue;
use vulkano::framebuffer::{Subpass, RenderPassAbstract};
use vulkano::pipeline::{GraphicsPipelineAbstract, GraphicsPipeline};
use vulkano::buffer::{BufferAccess, ImmutableBuffer, BufferUsage, CpuAccessibleBuffer};
use crate::graphics::object::{Vertex3D, ObjectInstance, MeshAccess};
use vulkano::sync::GpuFuture;
use vulkano::pipeline::blend::{AttachmentBlend, BlendOp, BlendFactor};
use vulkano::descriptor::{DescriptorSet, PipelineLayoutAbstract};
use cgmath::{ Matrix4, SquareMatrix };
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState};
use vulkano::image::AttachmentImage;
use crate::graphics::renderer::lighting_system::shadeless::Shadeless;

// Generate shadow map from viewport of camera

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec3 position;

layout(location = 0) out vec4 v_world;

layout(push_constant) uniform PushData {
    mat4 model;
} push;

layout(set = 0, binding = 0) uniform Matrixes {
    mat4 vp; // Camera ViewProjection
} mat;

void main() {
    v_world = (push.model * vec4(position, 1.0));
    gl_Position = mat.vp * v_world;
}"
    }
}
mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
#version 450

layout(location = 0) in vec4 v_world;

layout(location = 0) out vec4 f_shadow;

layout(push_constant) uniform PushData {
    mat4 vp; // Shadow viewport VP
} push;

void main() {
    vec4 shadow = push.vp * v_world;
    shadow /= shadow.w;
    f_shadow = vec4(shadow.z);
}"
    }
}

const BIAS_MATRIX: Matrix4<f32> = Matrix4::new(
    0.5, 0.0, 0.0, 0.0,
    0.0, 0.5, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.5, 0.5, 0.0, 1.0
);


pub struct ShadowMappingPass {
    queue: Arc<Queue>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    // Shadow VP transfered using push
    uniform_dirty: bool,
    uniform_set: Arc<dyn DescriptorSet + Send + Sync>, // set to transfer Camera VP
    uniform_buff: Arc<CpuAccessibleBuffer<vs::ty::Matrixes>>, // Matrixes { Camera VP }
    view_projection: Matrix4<f32>, // Camera VP

}
impl ShadowMappingPass {

    pub fn new<R>(queue: Arc<Queue>, subpass: Subpass<R>) -> Self
        where R: RenderPassAbstract + Send + Sync + 'static
    {
        let pipeline = {
            let vs = vs::Shader::load(queue.device().clone())
                .expect("failed to create shader module");
            let fs = fs::Shader::load(queue.device().clone())
                .expect("failed to create shader module");

            Arc::new(GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex3D>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(fs.main_entry_point(), ())
                .depth_stencil_simple_depth()
//                .blend_collective(AttachmentBlend {
//                    enabled: true,
//                    color_op: BlendOp::Min,
//                    color_source: BlendFactor::One,
//                    color_destination: BlendFactor::One,
//                    alpha_op: BlendOp::Max,
//                    alpha_source: BlendFactor::One,
//                    alpha_destination: BlendFactor::One,
//                    mask_red: true,
//                    mask_green: true,
//                    mask_blue: true,
//                    mask_alpha: true,
//                })
                .render_pass(subpass)
                .build(queue.device().clone())
                .unwrap()) as Arc<dyn GraphicsPipelineAbstract + Send + Sync>
        };

        let uniform_buff = CpuAccessibleBuffer::from_data(
            queue.device().clone(), BufferUsage::uniform_buffer(), vs::ty::Matrixes {
                vp: Matrix4::identity().into()
            }
        ).unwrap();

        let uniform_set = Arc::new(PersistentDescriptorSet::start(pipeline.clone(), 0)
            .add_buffer(uniform_buff.clone()).unwrap()
            .build().unwrap()
        );

        Self {
            queue,
            pipeline,

            uniform_dirty: false,
            uniform_set,
            uniform_buff,
            view_projection: Matrix4::identity(),
        }
    }

    pub fn set_view_projection(&mut self, vp: Matrix4<f32>) {
        if !self.view_projection.eq(&vp) {
            self.uniform_dirty = true;
            self.view_projection = vp;
        }
    }

    /// Use shadow_mat: Matrix4<f32> - ViewProjection of shadow source, unbiased
    pub fn render<'f>(&mut self, shadow_mat: Matrix4<f32>,
                      dyn_state: &DynamicState,
                      geometry: &Vec<&'f ObjectInstance>) -> AutoCommandBuffer {
        if self.uniform_dirty {
            self.uniform_dirty = false;
            let mut writer = self.uniform_buff.write().unwrap();
            writer.vp = self.view_projection.into();
        }

        let mut cbb = AutoCommandBufferBuilder::secondary_graphics(
            self.queue.device().clone(),
            self.queue.family(),
            self.pipeline.clone().subpass()
        ).unwrap();

        for i in geometry {
            if i.has_ibo() {
                unimplemented!()
            } else {
                let vs_push = vs::ty::PushData {
                    model: i.model_matrix().into()
                };
                let fs_push = fs::ty::PushData {
                    vp: {
                        shadow_mat
                    }.into()
                };
                cbb = cbb.draw(self.pipeline.clone(), dyn_state,
                               vec![i.get_vbo()],
                               (self.uniform_set.clone()), (vs_push, fs_push))
                    .unwrap();
            }
        }

        cbb.build().unwrap()
    }
}