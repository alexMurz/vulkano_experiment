use std::sync::Arc;
use vulkano::pipeline::{GraphicsPipelineAbstract, GraphicsPipeline};
use vulkano::command_buffer::{DynamicState, AutoCommandBufferBuilder, AutoCommandBuffer, CommandBuffer};
use vulkano::framebuffer::{RenderPassAbstract, Subpass, FramebufferAbstract, Framebuffer};
use std::cell::RefCell;
use vulkano::image::{AttachmentImage, ImageUsage, ImageAccess};
use vulkano::device::Queue;
use vulkano::pipeline::viewport::Viewport;
use vulkano::format::Format;
use crate::graphics::renderer_3d::{
    mesh::{ Vertex3D, ObjectInstance },
    lighting_system::{ LightSource, LightKind }
};
use cgmath::{Matrix4, Point3, vec3};
use vulkano::buffer::BufferAccess;

mod depth_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "\
#version 450

layout(location = 0) in vec3 position;

layout(push_constant) uniform PushData {
    mat4 mvp;
} push;

void main() {
    gl_Position = push.mvp * vec4(position, 1.0);
}"
    }
}
mod depth_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "\
#version 450
void main() {}"
    }
}


/// Generate depth map using given view_projection
/// Has its own render pass
/// Sampled attachment does not change unless resolution changes
pub struct ShadowMapping {
    queue: Arc<Queue>,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    // Generate Depth Buffer
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,
    dyn_state: DynamicState,
}
impl ShadowMapping {

    pub fn new(queue: Arc<Queue>) -> Self {
        let render_pass = Arc::new(vulkano::single_pass_renderpass!(
            queue.device().clone(),
            attachments: {
                depth: {
                    load: Clear,
                    store: Store,
                    format: Format::D16Unorm,
                    samples: 1,
                }
            },
            pass: {
                 color: [],
                 depth_stencil: {depth}
            }
        ).unwrap()) as Arc<dyn RenderPassAbstract + Send + Sync>;

        let pipeline = {
            let vs = depth_vs::Shader::load(queue.device().clone())
                .expect("failed to create shader module");
            let fs = depth_fs::Shader::load(queue.device().clone())
                .expect("failed to create shader module");

            Arc::new(GraphicsPipeline::start()
                .vertex_input_single_buffer::<Vertex3D>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(fs.main_entry_point(), ())
                .depth_stencil_simple_depth()
//                .blend_alpha_blending()
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .front_face_counter_clockwise() // Due to flipped Y coordinate, also change vertex order to CW
                .cull_mode_front()
                .build(queue.device().clone())
                .unwrap()
            ) as Arc<dyn GraphicsPipelineAbstract + Send + Sync>
        };


        Self {
            queue,
            render_pass,
            pipeline,
            dyn_state: DynamicState::none(),
        }
    }


    fn create_source_info(&self, resolution: [u32; 2]) -> (Arc<dyn FramebufferAbstract + Send + Sync>, Arc<AttachmentImage>) {
        let img = AttachmentImage::sampled(
            self.queue.device().clone(),
            resolution,
            Format::D16Unorm
        ).unwrap();

        let framebuffer = Arc::new(Framebuffer::start(self.render_pass.clone())
            .add(img.clone()).unwrap()
            .build().unwrap()
        );

        (framebuffer, img)
    }


    pub fn render_image(&mut self,
                        info: &mut LightKind,
                        geometry: &Vec<ObjectInstance>)
        -> AutoCommandBuffer
    {

        let mut cbb = AutoCommandBufferBuilder::primary_one_time_submit(
            self.queue.device().clone(), self.queue.family()).unwrap();

        match info {
            LightKind::ConeWithShadow(cone) => {

                if cone.framebuffer.is_none() || ImageAccess::dimensions(cone.image.as_ref().unwrap()).width_height() != cone.resolution {
                    let (fb, img) = self.create_source_info(cone.resolution);
                    cone.framebuffer = Some(fb);
                    cone.image = Some(img);
                    cone.data_set = None; // Invalidate data set
                }

                cbb = cbb.begin_render_pass(
                    cone.framebuffer.as_ref().unwrap().clone(),
                    false,
                    vec![1.0f32.into()]
                ).unwrap();

                self.dyn_state.viewports = Some(vec![Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [cone.resolution[0] as f32, cone.resolution[1] as f32],
                    depth_range: 0.0 .. 1.0
                }]);

                for i in geometry.iter(){
                    let mvp = cone.vp * i.model_matrix();
                    if i.mesh_data.visible_in(mvp) {
                        let vs_push = depth_vs::ty::PushData {
                            mvp: mvp.into()
                        };
                        for mat in i.materials.iter().filter(|x| x.material.is_cast_shadow()) {
                            if mat.ibo_slice.is_some() {
                                cbb = cbb.draw_indexed(self.pipeline.clone(), &self.dyn_state,
                                                       vec![mat.vbo_slice.clone()],
                                                       mat.ibo_slice.clone().unwrap(),
                                                       (), (vs_push)).unwrap();
                            } else {
                                cbb = cbb.draw(self.pipeline.clone(), &self.dyn_state,
                                               vec![mat.vbo_slice.clone()],
                                               (), (vs_push)).unwrap();
                            }
                        }

                    }
                }

                cbb = cbb.end_render_pass().unwrap()
            },
            _ => ()
        }

        cbb.build().unwrap()
    }

}