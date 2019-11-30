use std::sync::Arc;
use vulkano::pipeline::{GraphicsPipelineAbstract, GraphicsPipeline};
use vulkano::command_buffer::{DynamicState, AutoCommandBufferBuilder, AutoCommandBuffer};
use crate::graphics::object::{Vertex3D, ObjectInstance, MeshAccess};
use vulkano::framebuffer::{RenderPassAbstract, Subpass, FramebufferAbstract, Framebuffer};
use std::cell::RefCell;
use vulkano::image::{AttachmentImage, ImageUsage, ImageAccess};
use vulkano::device::Queue;
use vulkano::pipeline::viewport::Viewport;
use vulkano::format::Format;
use crate::graphics::renderer::lighting_system::{LightSource, LightKind};

mod depth_vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec3 position;

layout(push_constant) uniform PushData {
    mat4 mvp;
} push;

void main() {
    vec4 pos = push.mvp * vec4(position, 1.0);
    gl_Position = pos;
}"
    }
}
mod depth_fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
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
        let render_pass = Arc::new(vulkano::ordered_passes_renderpass!(queue.device().clone(),
            attachments: {
                depth: {
                    load: Clear,
                    store: Store,
                    format: Format::D16Unorm,
                    samples: 1,
                }
            },
            passes: [
                {
                    color: [],
                    depth_stencil: {depth},
                    input: []
                }
            ]
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
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .cull_mode_front()
                .build(queue.device().clone())
                .unwrap()) as Arc<dyn GraphicsPipelineAbstract + Send + Sync>
        };


        Self {
            queue,
            render_pass,
            pipeline,
            dyn_state: DynamicState::none(),
        }
    }


    fn create_source_info(&self, resolution: [u32; 2]) -> (Arc<dyn FramebufferAbstract + Send + Sync>, Arc<AttachmentImage>) {
        let img = AttachmentImage::multisampled_with_usage(
            self.queue.device().clone(), resolution, 1, Format::D16Unorm,
            ImageUsage {
                sampled: true,
                depth_stencil_attachment: true,
                .. ImageUsage::none()
            }
        ).unwrap();

        let framebuffer = Arc::new(Framebuffer::start(self.render_pass.clone())
            .add(img.clone()).unwrap()
            .build().unwrap()
        );

        (framebuffer, img)
    }


    pub fn render_image<'f>(&mut self,
                        info: &mut LightKind,
                        geometry: &Vec<&'f ObjectInstance>) -> AutoCommandBuffer
    {

        let mut cbb = AutoCommandBufferBuilder::primary_one_time_submit(
            self.queue.device().clone(), self.queue.family()).unwrap();

        match info {
            LightKind::ConeWithShadow(cone) => {

                if cone.framebuffer.is_none() || ImageAccess::dimensions(cone.image.as_ref().unwrap()).width_height() != cone.resolution {
                    let (fb, img) = self.create_source_info(cone.resolution);
                    cone.framebuffer = Some(fb);
                    cone.image = Some(img);
                }

                let mut cbb = AutoCommandBufferBuilder::primary_one_time_submit(
                    self.queue.device().clone(), self.queue.family()).unwrap()
                    .begin_render_pass(cone.framebuffer.clone().unwrap(), false, vec![1.0.into()]).unwrap();

                self.dyn_state.viewports = Some(vec![Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [cone.resolution[0] as _, cone.resolution[1] as _],
                    depth_range: 0.0 .. 1.0
                }]);

                for i in geometry {
                    if i.has_ibo() {
                        unimplemented!()
                    } else {
                        let vs_push = depth_vs::ty::PushData {
                            mvp: (cone.vp * i.model_matrix()).into()
                        };
                        cbb = cbb.draw(self.pipeline.clone(), &self.dyn_state,
                                       vec![i.get_vbo()],
                                       (), (vs_push))
                            .unwrap();
                    }
                }

                cbb = cbb.end_render_pass().unwrap()
            },
            _ => ()
        }

        cbb.build().unwrap()
    }

}