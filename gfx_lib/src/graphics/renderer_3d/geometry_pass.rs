
use std::sync::Arc;
use vulkano::{
    device::Queue,
    framebuffer::{ Subpass, RenderPassAbstract },
    buffer::{ BufferAccess, ImmutableBuffer, BufferUsage, CpuAccessibleBuffer },
    pipeline::{
        GraphicsPipeline, GraphicsPipelineAbstract,
        blend::{ AttachmentBlend, BlendOp, BlendFactor },
        vertex::{ SingleBufferDefinition, OneVertexOneInstanceDefinition },
    },
    descriptor::{
        DescriptorSet,
        descriptor_set::PersistentDescriptorSet,
    },
    command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState},
    sampler::{Sampler, Filter, MipmapMode, SamplerAddressMode}
};
use cgmath::{ Matrix4, SquareMatrix };
use crate::graphics::renderer_3d::mesh::{
    Vertex3D, MeshAccess, MaterialMeshSlice, MeshData, MaterialData, ObjectInstance, MaterialDrawMode
};


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
layout(location = 3) in vec2 uv;

layout(location = 0) out vec3 v_color;
layout(location = 1) out vec3 v_norm;
layout(location = 2) out vec3 v_pos;
layout(location = 3) out vec2 v_uv;

layout(push_constant) uniform PushData {
    mat4 model;
    mat4 normal;
} push;

layout(set = 0, binding = 0) uniform Matrixes {
    mat4 vp;
} mat;

void main() {
    v_uv = uv;
    v_norm = mat3(push.normal) * normal;
    v_color = color.rgb;

    v_pos = (push.model * vec4(position, 1.0)).xyz;
    gl_Position = mat.vp * vec4(v_pos, 1.0);
}"
    }
}
pub mod flat_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
#version 450

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec3 f_normal;

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec3 v_norm;
layout(location = 2) in vec3 v_pos;
layout(location = 3) in vec2 v_uv;

layout(set = 1, binding = 0) uniform Material {
    vec3 diffuse;
    vec2 uv_remap_a;
    vec2 uv_remap_b;
    int flat_shading;
} material;

void main() {
    vec3 faceNormal;
    if (material.flat_shading > 0) {
        vec3 tangent = dFdx( v_pos );
        vec3 bitangent = dFdy( v_pos );
        faceNormal = normalize( -cross( tangent, bitangent ) );
    } else faceNormal = v_norm;

    f_color = vec4(v_color * material.diffuse, 1.0);
    f_normal = faceNormal;
}"
    }
}
mod tex_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
#version 450

layout(location = 0) out vec4 f_color;
layout(location = 1) out vec3 f_normal;

layout(location = 0) in vec3 v_color;
layout(location = 1) in vec3 v_norm;
layout(location = 2) in vec3 v_pos;
layout(location = 3) in vec2 v_uv;

layout(set = 1, binding = 0) uniform Material {
    vec3 diffuse;
    vec2 uv_remap_a;
    vec2 uv_remap_b;
    int flat_shading;
} material;
layout(set = 1, binding = 1) uniform sampler2D u_texture;

void main() {
    vec3 faceNormal;
    if (material.flat_shading > 0) {
        vec3 tangent = dFdx( v_pos );
        vec3 bitangent = dFdy( v_pos );
        faceNormal = normalize( -cross( tangent, bitangent ) );
    } else faceNormal = v_norm;

    vec2 uv = material.uv_remap_a + v_uv * (material.uv_remap_b - material.uv_remap_a);
    vec3 col = texture(u_texture, uv).rgb * material.diffuse * v_color;
    f_color = vec4(col, 1.0);
    f_normal = faceNormal;
}"
    }
}

pub struct FlatPass {
    queue: Arc<Queue>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    // Matrix info (used to update)
    pub view_projection: Matrix4<f32>,

    // Matrixes Uniform
    uniform_dirty: bool, // True then uniform requires update
    uniform_mat_set: Arc<dyn DescriptorSet + Send + Sync>, // desc set
    uniform_matrixes_buffer: Arc<CpuAccessibleBuffer<vs::ty::Matrixes>>, // Buffer with `matrixes`
}
impl FlatPass {
    pub fn new<R>(queue: Arc<Queue>, subpass: Subpass<R>) -> Self
        where R: RenderPassAbstract + Send + Sync + 'static
    {
        let pipeline = {
            let vs = vs::Shader::load(queue.device().clone())
                .expect("failed to create shader module");
            let fs = flat_fs::Shader::load(queue.device().clone())
                .expect("failed to create shader module");

            Arc::new(GraphicsPipeline::start()
                .vertex_input(SingleBufferDefinition::<Vertex3D>::new())
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(fs.main_entry_point(), ())
                .depth_stencil_simple_depth()
                .front_face_counter_clockwise() // Due to flipped Y coordinate, also change vertex order to CW
                .cull_mode_back()
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
            uniform_mat_set: uniform_set,
            uniform_matrixes_buffer
        }
    }

    pub fn set_view_projection(&mut self, vp: Matrix4<f32>) {
        if !self.view_projection.eq(&vp) {
            self.uniform_dirty = true;
            self.view_projection = vp;
        }
    }

    pub fn render<'f>(&mut self, dyn_state: &DynamicState, mut cbb: AutoCommandBufferBuilder, matrices: (Matrix4<f32>, Matrix4<f32>), mat: &mut MaterialMeshSlice) -> AutoCommandBufferBuilder {
        if self.uniform_dirty {
            self.uniform_dirty = false;
            let mut writer = self.uniform_matrixes_buffer.write().unwrap();
            writer.vp = self.view_projection.into();
        }

        let push = vs::ty::PushData {
            model: matrices.0.into(),
            normal: matrices.1.into(),
        };

        if mat.ibo_slice.is_some() {
            cbb.draw_indexed(
                self.pipeline.clone(), dyn_state,
                vec![mat.vbo_slice.clone()],
                mat.ibo_slice.clone().unwrap(),
                (self.uniform_mat_set.clone(), mat.material.get_uniform(&self.pipeline, 1)),
                (push)
            ).unwrap()
        } else {
            cbb.draw(
                self.pipeline.clone(), dyn_state,
                vec![mat.vbo_slice.clone()],
                (self.uniform_mat_set.clone(), mat.material.get_uniform(&self.pipeline, 1)),
                (push)
            ).unwrap()
        }
    }

}

pub struct TexPass {
    queue: Arc<Queue>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    // Matrix info (used to update)
    pub view_projection: Matrix4<f32>,

    // Image
    sampler: Arc<Sampler>,

    // Matrixes Uniform
    uniform_dirty: bool, // True then uniform requires update
    uniform_mat_set: Arc<dyn DescriptorSet + Send + Sync>, // desc set
    uniform_matrixes_buffer: Arc<CpuAccessibleBuffer<vs::ty::Matrixes>>, // Buffer with `matrixes`
}
impl TexPass {
    pub fn new<R>(queue: Arc<Queue>, subpass: Subpass<R>) -> Self
        where R: RenderPassAbstract + Send + Sync + 'static
    {
        let pipeline = {
            let vs = vs::Shader::load(queue.device().clone())
                .expect("failed to create shader module");
            let fs = tex_fs::Shader::load(queue.device().clone())
                .expect("failed to create shader module");

            Arc::new(GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex3D>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(fs.main_entry_point(), ())
                .depth_stencil_simple_depth()
                .front_face_counter_clockwise() // Due to flipped Y coordinate, also change vertex order to CW
                .cull_mode_back()
                .render_pass(subpass)
                .build(queue.device().clone())
                .unwrap()) as Arc<dyn GraphicsPipelineAbstract + Send + Sync>
        };

        let sampler = Sampler::new(
            queue.device().clone(), Filter::Linear, Filter::Linear,
            MipmapMode::Linear,
            SamplerAddressMode::ClampToEdge, SamplerAddressMode::ClampToEdge, SamplerAddressMode::ClampToEdge,
            0.0, 1.0, 0.0, 0.0
        ).unwrap();

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

            sampler,

            view_projection: Matrix4::identity(),

            uniform_dirty: true,
            uniform_mat_set: uniform_set,
            uniform_matrixes_buffer
        }
    }

    pub fn set_view_projection(&mut self, vp: Matrix4<f32>) {
        if !self.view_projection.eq(&vp) {
            self.uniform_dirty = true;
            self.view_projection = vp;
        }
    }

    pub fn render<'f>(&mut self, dyn_state: &DynamicState, mut cbb: AutoCommandBufferBuilder, matrices: (Matrix4<f32>, Matrix4<f32>), mat: &mut MaterialMeshSlice) -> AutoCommandBufferBuilder {
        if self.uniform_dirty {
            self.uniform_dirty = false;
            let mut writer = self.uniform_matrixes_buffer.write().unwrap();
            writer.vp = self.view_projection.into();
        }

        let push = vs::ty::PushData {
            model: matrices.0.into(),
            normal: matrices.1.into(),
        };

        if mat.ibo_slice.is_some() {
            cbb.draw_indexed(
                self.pipeline.clone(), dyn_state,
                vec![mat.vbo_slice.clone()],
                mat.ibo_slice.clone().unwrap(),
                (self.uniform_mat_set.clone(), mat.material.get_uniform(&self.pipeline, 1)),
                (push)
            ).unwrap()
        } else {
            cbb.draw(
                self.pipeline.clone(), dyn_state,
                vec![mat.vbo_slice.clone()],
                (self.uniform_mat_set.clone(), mat.material.get_uniform(&self.pipeline, 1)),
                (push)
            ).unwrap()
        }
//        cbb.draw(self.pipeline.clone(), dyn_state,
//                 vec![mat.vbo_slice.clone()],
//                 (self.uniform_mat_set.clone(), mat.material.get_uniform(&self.pipeline, 1)),
//                 (push))
//            .unwrap()
    }

}

pub struct GeometryPass {
    queue: Arc<Queue>,
    view_projection: Matrix4<f32>,
    flat_pass: FlatPass,
    tex_pass: TexPass,
}
impl GeometryPass {
    pub fn new<R>(queue: Arc<Queue>, subpass: Subpass<R>) -> Self
        where R: RenderPassAbstract + Send + Sync + Clone + 'static
    {
        let flat_pass = FlatPass::new(queue.clone(), subpass.clone());
        let tex_pass = TexPass::new(queue.clone(), subpass.clone());

        Self {
            queue,
            view_projection: Matrix4::identity(),

            flat_pass,
            tex_pass
        }
    }

    pub fn set_view_projection(&mut self, vp: Matrix4<f32>) {
        self.view_projection = vp;
        self.flat_pass.set_view_projection(vp);
        self.tex_pass.set_view_projection(vp);
    }

    //noinspection RsMatchCheck
    pub fn render<'f>(&mut self, dyn_state: &DynamicState, geometry: &mut Vec<ObjectInstance>) -> AutoCommandBuffer {

        let mut cbb = AutoCommandBufferBuilder::secondary_graphics(
            self.queue.device().clone(),
            self.queue.family(),
            self.flat_pass.pipeline.clone().subpass()
        ).unwrap();

        let vp = self.view_projection;
        for i in geometry.iter_mut().filter(|x| x.mesh_data.ready_for_use() && x.mesh_data.visible_in(vp * x.model_matrix())) {
            let matrices = (i.model_matrix(), i.normal_matrix());
            for m in i.materials.iter_mut() {

                match m.material.mode() {
                    MaterialDrawMode::NoTexture => cbb = self.flat_pass.render(dyn_state, cbb, matrices, m),
                    MaterialDrawMode::WithDiffuse => cbb = self.tex_pass.render(dyn_state, cbb, matrices, m),
                    _ => panic!("Unsupported mode"),
                }
            }
        }

        cbb.build().unwrap()
    }

}