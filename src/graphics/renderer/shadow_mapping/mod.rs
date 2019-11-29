
use std::sync::Arc;
use vulkano::device::{Queue, DeviceOwned};
use vulkano::framebuffer::{Subpass, RenderPassAbstract, FramebufferBuilder, Framebuffer, FramebufferAbstract};
use vulkano::pipeline::{GraphicsPipelineAbstract, GraphicsPipeline};
use vulkano::buffer::{BufferAccess, ImmutableBuffer, BufferUsage, CpuAccessibleBuffer, DeviceLocalBuffer};
use crate::graphics::object::{Vertex3D, ObjectInstance, MeshAccess, ScreenVertex};
use vulkano::sync::GpuFuture;
use vulkano::pipeline::blend::{AttachmentBlend, BlendOp, BlendFactor};
use vulkano::descriptor::{DescriptorSet, PipelineLayoutAbstract};
use cgmath::{ Matrix4, SquareMatrix };
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState};
use vulkano::image::{ImageAccess, AttachmentImage, ImageUsage, StorageImage, Dimensions, ImageViewAccess};
use vulkano::format::{Format, FormatDesc};
use crate::graphics::renderer::lighting_system::shadeless::Shadeless;
use vulkano::pipeline::viewport::Viewport;
use std::cell::RefCell;


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
    gl_Position = push.mvp * vec4(position, 1.0);
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


const BIAS_MATRIX: Matrix4<f32> = Matrix4::new(
    0.5, 0.0, 0.0, 0.0,
    0.0, 0.5, 0.0, 0.0,
    0.0, 0.0, 1.0, 0.0,
    0.5, 0.5, 0.0, 1.0
);


/// Holds reference to matrix and output image attachment
/// Output doesn't change unless resolution changes
pub struct ShadowSource {
    pub active: bool, // if active, will be updated
    pub view_projection: Matrix4<f32>,
    pub image: Arc<AttachmentImage>
}

/// Generate shadow map from viewport of camera
/// Has its own render pass
/// Sampled attachment does not change unless resolution changes
pub struct ShadowMapping {
    queue: Arc<Queue>,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,

    // Generate Depth Buffer
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

    // Shadow Resolution
    dyn_state: DynamicState,

    sources: Vec<Arc<RefCell<ShadowSource>>>,
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
//                .cull_mode_disabled()
                .cull_mode_front()
                .build(queue.device().clone())
                .unwrap()) as Arc<dyn GraphicsPipelineAbstract + Send + Sync>
        };


        Self {
            queue,
            render_pass,
            pipeline,
            dyn_state: DynamicState::none(),
            sources: Vec::new()
        }
    }

    pub fn new_source(&mut self, resolution: [u32; 2]) -> Arc<RefCell<ShadowSource>> {
        let img = AttachmentImage::with_usage(
            self.queue.device().clone(), resolution, Format::D16Unorm,
            ImageUsage {
                sampled: true,
                depth_stencil_attachment: true,
                .. ImageUsage::none()
            }
        ).unwrap();
        let source = Arc::new(RefCell::new(ShadowSource {
            active: true,
            view_projection: Matrix4::identity(),
            image: img
        }));
        self.sources.push(source.clone());
        source
    }


    fn render_image<'f>(&mut self,
                        source_ref: Arc<RefCell<ShadowSource>>,
                        geometry: &Vec<&'f ObjectInstance>) -> AutoCommandBuffer
    {
        let source = source_ref.borrow();

        let framebuffer = Arc::new(Framebuffer::start(self.render_pass.clone())
            .add(source.image.clone()).unwrap()
            .build().unwrap()
        );

        let mut cbb = AutoCommandBufferBuilder::primary_one_time_submit(
            self.queue.device().clone(), self.queue.family()).unwrap()
            .begin_render_pass(framebuffer.clone(), false, vec![1.0.into()]).unwrap();

        let img_dim = ImageAccess::dimensions(&source.image).width_height();
        self.dyn_state.viewports = Some(vec![Viewport {
            origin: [0.0, 0.0],
            dimensions: [img_dim[0] as _, img_dim[1] as _],
            depth_range: 0.0 .. 1.0
        }]);

        for i in geometry {
            if i.has_ibo() {
                unimplemented!()
            } else {
                let vs_push = depth_vs::ty::PushData {
                    mvp: (source.view_projection * i.model_matrix()).into()
                };
                cbb = cbb.draw(self.pipeline.clone(), &self.dyn_state,
                               vec![i.get_vbo()],
                               (), (vs_push))
                    .unwrap();
            }
        }

        cbb.end_render_pass().unwrap().build().unwrap()
    }

    /// Use shadow_mat: Matrix4<f32> - ViewProjection of shadow source, unbiased
    pub fn render<'f>(&mut self, geometry: &Vec<&'f ObjectInstance>) -> AutoCommandBuffer {
        let mut cbb = AutoCommandBufferBuilder::primary_one_time_submit(
            self.queue.device().clone(),
            self.queue.family()
        ).unwrap();
        unsafe {
            for i in 0..self.sources.len() {
                cbb = cbb.execute_commands(self.render_image(self.sources[i].clone(), geometry)).unwrap();
            }
        };
        cbb.build().unwrap()
    }
}