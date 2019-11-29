
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

// Pass for baking geometry and material data
// First in line


mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec4 color;
layout(location = 2) in vec3 normal;
layout(location = 3) in uint flat_shading;
layout(location = 4) in vec2 uv;
layout(location = 5) in uint material_id;

layout(location = 0) out vec3 v_color;
layout(location = 1) out vec3 v_norm;
layout(location = 2) out vec3 v_pos;
layout(location = 3) out uint v_flat_shading;

layout(push_constant) uniform PushData {
    mat4 model;
} push;

layout(set = 0, binding = 0) uniform Matrixes {
    mat4 vp;
} mat;

void main() {
    v_norm = normal;
    v_color = color.rgb;
    v_flat_shading = flat_shading;

    v_pos = (push.model * vec4(position, 1.0)).xyz;
    gl_Position = mat.vp * vec4(v_pos, 1.0);
}"
    }
}
mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
#version 450

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec3 f_normal;

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec3 v_norm;
layout(location = 2) in vec3 v_pos;
layout(location = 3) in flat uint v_flat_shading;

void main() {
    vec3 faceNormal;
    if (v_flat_shading > 0) {
        vec3 tangent = dFdx( v_pos );
        vec3 bitangent = dFdy( v_pos );
        faceNormal = normalize( -cross( tangent, bitangent ) );
    } else faceNormal = v_norm;

    f_color = vec4(v_color, 1.0);
    f_normal = faceNormal;
}"
    }
}


pub struct GeometryPass {
    queue: Arc<Queue>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    // Matrix info (used to update)
    pub view_projection: Matrix4<f32>,

    // Matrixes Uniform
    uniform_dirty: bool, // True then uniform requires update
    uniform_set: Arc<dyn DescriptorSet + Send + Sync>, // desc set
    uniform_matrixes_buffer: Arc<CpuAccessibleBuffer<vs::ty::Matrixes>>, // Buffer with `matrixes`
}
impl GeometryPass {
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
//                    color_op: BlendOp::Add,
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

        let uniform_matrixes_buffer = {
            CpuAccessibleBuffer::from_data(
                queue.device().clone(), BufferUsage::uniform_buffer(),
                vs::ty::Matrixes {
                    vp: Matrix4::identity().into()
                }
            ).unwrap()
        };

        let uniform_set = Arc::new(
            PersistentDescriptorSet::start(pipeline.clone(), 0)
                .add_buffer(uniform_matrixes_buffer.clone()).unwrap()
                .build().unwrap()
        );

        Self {
            queue,
            pipeline,

            view_projection: Matrix4::identity(),

            uniform_dirty: true,
            uniform_set,
            uniform_matrixes_buffer
        }
    }

    pub fn set_view_projection(&mut self, vp: Matrix4<f32>) {
        if !self.view_projection.eq(&vp) {
            self.uniform_dirty = true;
            self.view_projection = vp;
        }
    }

    pub fn render<'f>(&mut self, dyn_state: &DynamicState, geometry: &Vec<&'f ObjectInstance>) -> AutoCommandBuffer {
        if self.uniform_dirty {
            self.uniform_dirty = false;
            let mut writer = self.uniform_matrixes_buffer.write().unwrap();
            writer.vp = self.view_projection.into();
        }

        let mut cbb = AutoCommandBufferBuilder::secondary_graphics(
            self.queue.device().clone(),
            self.queue.family(),
            self.pipeline.clone().subpass()
        ).unwrap();

        for i in geometry {
            if i.has_ibo() { unimplemented!() } else {
                let push = vs::ty::PushData {
                    model: i.model_matrix().into()
                };
                cbb = cbb.draw(self.pipeline.clone(), dyn_state,
                               vec![i.get_vbo()],
                               (self.uniform_set.clone()), (push))
                    .unwrap();
            }
        }

        cbb.build().unwrap()
    }

}