use std::sync::Arc;
use vulkano::device::Queue;
use vulkano::framebuffer::{RenderPassAbstract, Subpass, Framebuffer};
use vulkano::format::Format;
use crate::graphics::renderer::geometry_pass::GeometryPass;
use vulkano::image::{AttachmentImage, ImageUsage, ImageAccess, ImageViewAccess};
use cgmath::{Matrix4, Point3, vec3};
use vulkano::sync::GpuFuture;
use crate::graphics::object::{ObjectInstance, Vertex3D, MeshData};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState, CommandBuffer};
use vulkano::pipeline::viewport::Viewport;
use vulkano::buffer::{BufferUsage, ImmutableBuffer};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;

mod geometry_pass;
mod lighting_system;

use lighting_system::lighting_pass::LightingPass;
use lighting_system::{ LightSource, LightKind, ShadowKind };

pub struct Renderer {
    // Basics
    queue: Arc<Queue>,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    dyn_state: DynamicState,

    // FB Attachments,
    diffuse_buffer: Arc<AttachmentImage>,
    normal_buffer: Arc<AttachmentImage>,
    depth_buffer: Arc<AttachmentImage>,

    // Passes
    geom_pass: GeometryPass,
    lighting_pass: LightingPass,
}
impl Renderer {
    pub fn new(queue: Arc<Queue>, output_format: Format) -> Self {
        let render_pass = Arc::new(vulkano::ordered_passes_renderpass!(queue.device().clone(),
            attachments: {
                final_color: {
                    load: Clear,
                    store: Store,
                    format: output_format,
                    samples: 1,
                },
                // Will be bound to `self.diffuse_buffer`.
                diffuse: {
                    load: Clear,
                    store: DontCare,
                    format: Format::A2B10G10R10UnormPack32,
                    samples: 1,
                },
                // Will be bound to `self.normals_buffer`.
                normals: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R16G16B16A16Sfloat,
                    samples: 1,
                },
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16Unorm,
                    samples: 1,
                }
            },
            passes: [
                // Write to the diffuse, normals and depth attachments.
                {
                    color: [diffuse, normals],
                    depth_stencil: {depth},
                    input: []
                },
                // Apply lighting by reading these three attachments and writing to `final_color`.
                {
                    color: [final_color],
                    depth_stencil: {},
                    input: [diffuse, normals, depth]
                }
            ]
        ).unwrap()) as Arc<dyn RenderPassAbstract + Send + Sync>;

        let atch_usage = ImageUsage {
            transient_attachment: true,
            input_attachment: true,
            ..ImageUsage::none()
        };
        let diffuse_buffer = AttachmentImage::with_usage(
            queue.device().clone(), [1, 1], Format::A2B10G10R10UnormPack32, atch_usage
        ).unwrap();
        let normal_buffer = AttachmentImage::with_usage(
            queue.device().clone(), [1, 1], Format::R16G16B16A16Sfloat, atch_usage
        ).unwrap();
        let depth_buffer = AttachmentImage::with_usage(
            queue.device().clone(), [1, 1], Format::D16Unorm, atch_usage
        ).unwrap();

        let geom_pass = GeometryPass::new(
            queue.clone(),
            Subpass::from(render_pass.clone(), 0).unwrap()
        );
        let mut lighting_pass = LightingPass::new(
            queue.clone(),
            Subpass::from(render_pass.clone(), 1).unwrap()
        );

        let light_count = 100;
        let light_res = [64, 64];

        /* Test ambient */ if true {
            let source = lighting_pass.create_source(LightKind::Ambient);
            source.borrow_mut().active = true;
            source.borrow_mut().pow(0.05);
        }


        for i in 0..light_count {
            let x = (i as f32 / light_count as f32 * 3.1415 * 2.0).sin() * 5.0;
            let y = (i as f32 / light_count as f32 * 3.1415 * 2.0).cos() * 5.0;
            let source = lighting_pass.create_source(LightKind::ConeWithShadow(
                ShadowKind::Cone::with_projection(45.0, light_res)
            ));
            source.borrow_mut().pos(x, -2.0, y);
            source.borrow_mut().look_at(0.0, 0.0, 0.0);
            source.borrow_mut().pow(10.0);
        }

//        for i in 0..light_count {
//            let x = (i as f32 / light_count as f32 * 3.1415 * 2.0).sin() * 5.0;
//            let y = (i as f32 / light_count as f32 * 3.1415 * 2.0).cos() * 5.0;
//            let mut ss = shadow_mapping.new_source(light_res);
//            ss.borrow_mut().light_pos = [x, -5.0, y];
//            ss.borrow_mut().view_projection = {
//                cgmath::perspective(cgmath::Deg(45.0), 1.0, 1.0, 20.0)
//                    * Matrix4::look_at(Point3::new(x,-5.0, y), Point3::new(0.0, 0.0, 0.0), vec3(0.0, 1.0, 0.0))
//
//            };
//        }

        Self {
            queue,
            render_pass,
            dyn_state: DynamicState::none(),

            diffuse_buffer,
            normal_buffer,
            depth_buffer,

            geom_pass,
            lighting_pass
        }
    }

    pub fn generate_mesh_from_data(&self, data: Vec<Vertex3D>) -> MeshData {
        let (vbo, future) = ImmutableBuffer::from_iter(
            data.iter().cloned(),
            BufferUsage::vertex_buffer(),
            self.queue.clone(),
        ).unwrap();

        // Wait on future
        future.flush().unwrap();

        MeshData {
            vbo,
            ibo: None
        }
    }

    pub fn set_view_projection(&mut self, view_projection: Matrix4<f32>) {
        self.geom_pass.set_view_projection(view_projection);
        self.lighting_pass.set_view_projection(view_projection);
    }

    pub fn render<'f, F, I>(&mut self, prev_future: F, final_image: I, geometry: &Vec<&'f ObjectInstance>) -> Box<dyn GpuFuture>
        where
            F: GpuFuture + 'static,
            I: ImageAccess + ImageViewAccess + Send + Sync + Clone + 'static,
    {

        let img_dims = ImageAccess::dimensions(&final_image).width_height();
        if ImageAccess::dimensions(&self.depth_buffer).width_height() != img_dims {

            let atch_usage = ImageUsage { transient_attachment: true, input_attachment: true, ..ImageUsage::none() };

            self.diffuse_buffer = AttachmentImage::with_usage(self.queue.device().clone(),
                                                              img_dims,
                                                              Format::A2B10G10R10UnormPack32,
                                                              atch_usage
            ).unwrap();

            self.normal_buffer = AttachmentImage::with_usage(self.queue.device().clone(),
                                                              img_dims,
                                                              Format::R16G16B16A16Sfloat,
                                                              atch_usage
            ).unwrap();

            self.depth_buffer = AttachmentImage::with_usage(self.queue.device().clone(),
                                                            img_dims,
                                                            Format::D16Unorm,
                                                            atch_usage
            ).unwrap();

            self.dyn_state.viewports = Some(vec![Viewport {
                origin: [0.0, 0.0],
                dimensions: [img_dims[0] as f32, img_dims[1] as f32],
                depth_range: 0.0 .. 1.0
            }]);

            self.lighting_pass.set_attachments(
                self.diffuse_buffer.clone(),
                self.normal_buffer.clone(),
                self.depth_buffer.clone(),
            );
        }

        // Prepare shadow map
        // Perform updating of lighting and wait on it
        let shadow_cb = self.lighting_pass.update(geometry);

        let framebuffer = Arc::new(
            Framebuffer::start(self.render_pass.clone())
                .add(final_image.clone()).unwrap()
                .add(self.diffuse_buffer.clone()).unwrap()
                .add(self.normal_buffer.clone()).unwrap()
                .add(self.depth_buffer.clone()).unwrap()
                .build().unwrap()
        );

        let mut main_cbb = AutoCommandBufferBuilder::primary_one_time_submit(
            self.queue.device().clone(), self.queue.family()
        ).unwrap()
            .begin_render_pass(framebuffer.clone(), true, vec![
                [0.0, 0.0, 0.0, 1.0].into(),
                [0.0, 0.0, 0.0, 0.0].into(),
                [0.0, 0.0, 0.0, 0.0].into(),
                1.0.into(),
            ]).unwrap();

        // Do geometry pass
        main_cbb = unsafe {
            main_cbb.execute_commands(self.geom_pass.render(&self.dyn_state, geometry)).unwrap()
        };

        // Do Finalization
        main_cbb = unsafe {
            main_cbb = main_cbb.next_subpass(true).unwrap();
            main_cbb = self.lighting_pass.render(main_cbb, &self.dyn_state);
            main_cbb
        };

        let main_cb = main_cbb.end_render_pass().unwrap().build().unwrap();

        Box::new(prev_future
                     .then_execute(self.queue.clone(), shadow_cb).unwrap()
                     .then_execute(self.queue.clone(), main_cb).unwrap()
        )
    }

}