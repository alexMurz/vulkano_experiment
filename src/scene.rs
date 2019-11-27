
use vulkano::buffer::BufferUsage;
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::command_buffer::AutoCommandBuffer;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::DynamicState;
use vulkano::device::Queue;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::framebuffer::Subpass;
use vulkano::pipeline::GraphicsPipeline;
use vulkano::pipeline::GraphicsPipelineAbstract;
use vulkano::pipeline::viewport::Viewport;

use std::sync::Arc;
use crate::graphics::renderer::{RendererDrawable, RenderExecutor, ShadowExecutor};

use cgmath::{Matrix4, SquareMatrix, Vector3, vec3};
use blend::Blend;
use crate::loader;


#[derive(Default, Debug, Clone)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
    normal: [f32; 3],
}
vulkano::impl_vertex!(Vertex, position, color, normal);

mod vs {
    vulkano_shaders::shader!{
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;

layout(location = 0) out vec3 v_color;
layout(location = 1) out vec3 v_norm;
layout(location = 2) out vec3 v_pos;

layout(push_constant) uniform PushData {
    mat4 model;
    mat4 mvp;
} push;

void main() {
    v_norm = normal;
    v_color = color;
    v_pos = (push.model * vec4(position, 1.0)).xyz;
    gl_Position = push.mvp * vec4(position, 1.0);
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

void main() {
    vec3 tangent = dFdx( v_pos );
    vec3 bitangent = dFdy( v_pos );
    vec3 faceNormal = normalize( -cross( tangent, bitangent ) );

    f_color = vec4(v_color, 0.7);
    f_normal = faceNormal; // v_norm; // vec3(0.0, 1.0, 0.0);
}"
    }
}

mod shadow_map {
    pub mod vs {
        vulkano_shaders::shader!{
            ty: "vertex",
            src: "
#version 450

layout(location = 0) in vec3 position;
layout(location = 1) in vec3 color;
layout(location = 2) in vec3 normal;

layout(push_constant) uniform PushData {
    mat4 mvp;
} push;

void main() {
    vec4 pos = push.mvp * vec4(position, 1.0);
    gl_Position = pos;
}"
        }
    }

    pub mod fs {
        vulkano_shaders::shader!{
            ty: "fragment",
            src: "
#version 450

void main() {
}"
        }
    }
}

pub struct Scene {
    queue: Arc<Queue>,
    floor_vbo: Arc<CpuAccessibleBuffer<[Vertex]>>,
    object_vbo: Arc<CpuAccessibleBuffer<[Vertex]>>,

    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    shadow_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    dynamic_state: DynamicState,
    t: usize,
}
impl Scene {
    pub fn new<R>(queue: Arc<Queue>, subpass: Subpass<R>, shadow_subpass: Subpass<R>) -> Self
        where R: RenderPassAbstract + Send + Sync + 'static
    {

        let floor_vbo = {
            let z = 2f32;
            let s = 10f32;
            CpuAccessibleBuffer::from_iter(queue.device().clone(), BufferUsage::all(), [
                Vertex { position: [-s, z, -s], color: [1.0, 1.0, 1.0], normal: [0.0, 1.0, 0.0] },
                Vertex { position: [-s, z,  s], color: [1.0, 1.0, 1.0], normal: [0.0, 1.0, 0.0] },
                Vertex { position: [ s, z,  s], color: [1.0, 1.0, 1.0], normal: [0.0, 1.0, 0.0] },
                Vertex { position: [-s, z, -s], color: [1.0, 1.0, 1.0], normal: [0.0, 1.0, 0.0] },
                Vertex { position: [ s, z,  s], color: [1.0, 1.0, 1.0], normal: [0.0, 1.0, 0.0] },
                Vertex { position: [ s, z, -s], color: [1.0, 1.0, 1.0], normal: [0.0, 1.0, 0.0] },
            ].iter().cloned()).expect("Failed to create VBO for `Floor`")
        };

        let object_vbo = {
            let blend = Blend::from_path("src/data/test.blend");
            let mut vertices = Vec::new();
            loader::blender::load_model_faces(&blend, "myCubeMesh", |mut face| {
                for i in 0..3 {
                    vertices.push(Vertex {
                        position: face.vert[i],
                        normal: face.norm[i],
                        color: [1.0, 1.0, 1.0]
                    });
                }
            });

            let s = 0.95f32;
            let z = 0.0;
            CpuAccessibleBuffer::from_iter(queue.device().clone(), BufferUsage::all(), vertices.iter().cloned()
//                                           [
//                Vertex { position: [-s, -s, z], color: [1.0, 0.0, 1.0], normal: [0.0, 0.0,-1.0] },
//                Vertex { position: [-s,  s, z], color: [1.0, 0.0, 1.0], normal: [0.0, 0.0,-1.0] },
//                Vertex { position: [ s,  s, z], color: [1.0, 0.0, 1.0], normal: [0.0, 0.0,-1.0] },
//                Vertex { position: [-s, -s, z], color: [1.0, 0.0, 1.0], normal: [0.0, 0.0,-1.0] },
//                Vertex { position: [ s,  s, z], color: [1.0, 0.0, 1.0], normal: [0.0, 0.0,-1.0] },
//                Vertex { position: [ s, -s, z], color: [1.0, 0.0, 1.0], normal: [0.0, 0.0,-1.0] },
//            ].iter().cloned()
            ).expect("Failed to create VBO for `Object`")
        };


        let pipeline = {
            let vs = vs::Shader::load(queue.device().clone())
                .expect("failed to create shader module");
            let fs = fs::Shader::load(queue.device().clone())
                .expect("failed to create shader module");

            Arc::new(GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(fs.main_entry_point(), ())
                .depth_stencil_simple_depth()
                .render_pass(subpass)
                .build(queue.device().clone())
                .unwrap()) as Arc<_>
        };


        let shadow_pipeline = {
            let vs = shadow_map::vs::Shader::load(queue.device().clone())
                .expect("failed to create shader module");
            let fs = shadow_map::fs::Shader::load(queue.device().clone())
                .expect("failed to create shader module");

            Arc::new(GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(fs.main_entry_point(), ())
                .depth_stencil_simple_depth()
                .render_pass(shadow_subpass)
                .build(queue.device().clone())
                .unwrap()) as Arc<_>
        };


        let dyn_state = DynamicState::none();

        Self {
            queue,
            floor_vbo,
            object_vbo,
            pipeline,
            shadow_pipeline,

            dynamic_state: dyn_state,
            t: 0
        }

    }


    pub fn act(&mut self, delta: f32) {
        self.t += 1;
    }

    pub fn render(&mut self, executor: &mut RenderExecutor) {
        let x = (self.t as f32 / 60.0f32).sin() * 1.0f32;

        let dim = executor.viewport();

        let push_constants1 = vs::ty::PushData {
            model: Matrix4::identity().into(),
            mvp: (executor.view_projection).into(),
        };
        let push_constants2 = vs::ty::PushData {
            model: Matrix4::from_angle_z(cgmath::Rad(x)).into(),
            mvp: (executor.view_projection *
                Matrix4::from_angle_z(cgmath::Rad(x))
            ).into(),
        };

        self.dynamic_state.viewports = Some(vec![Viewport {
            origin: [0.0, 0.0],
            dimensions: [dim[0] as f32, dim[1] as f32],
            depth_range: 0.0 .. 1.0,
        }]);

        let cb =
            AutoCommandBufferBuilder::secondary_graphics(
                self.queue.device().clone(),
                self.queue.family(),
                self.pipeline.clone().subpass()
            ).unwrap()
                .draw(self.pipeline.clone(), &self.dynamic_state, vec![self.floor_vbo.clone()], (), push_constants1).unwrap()
                .draw(self.pipeline.clone(), &self.dynamic_state, vec![self.object_vbo.clone()], (), push_constants2).unwrap()
                .build().unwrap();
        executor.execute(cb);
    }

    pub fn render_shadows(&mut self, executor: &mut ShadowExecutor) {
        let x = (self.t as f32 / 60.0f32).sin() * 1.0f32;

        let push_constants1 = shadow_map::vs::ty::PushData {
            mvp: (executor.view_projection).into(),
        };
        let push_constants2 = shadow_map::vs::ty::PushData {
            mvp: (executor.view_projection *
                Matrix4::from_angle_z(cgmath::Rad(x))
            ).into(),
        };

        let dim = executor.viewport();
        self.dynamic_state.viewports = Some(vec![Viewport {
            origin: [0.0, 0.0],
            dimensions: [dim[0] as f32, dim[1] as f32],
            depth_range: 0.0 .. 1.0,
        }]);

        let cb =
            AutoCommandBufferBuilder::secondary_graphics(
                self.queue.device().clone(),
                self.queue.family(),
                self.shadow_pipeline.clone().subpass()
            ).unwrap()
                .draw(self.shadow_pipeline.clone(), &self.dynamic_state, vec![self.floor_vbo.clone()], (), push_constants1).unwrap()
                .draw(self.shadow_pipeline.clone(), &self.dynamic_state, vec![self.object_vbo.clone()], (), push_constants2).unwrap()
                .build().unwrap();
        executor.execute(cb);
    }
}

