use std::sync::Arc;
use vulkano::device::Queue;
use vulkano::framebuffer::{RenderPassAbstract, Subpass, Framebuffer};
use vulkano::format::Format;
use crate::graphics::renderer_3d::geometry_pass::GeometryPass;
use vulkano::image::{AttachmentImage, ImageUsage, ImageAccess, ImageViewAccess};
use cgmath::{Matrix4, Point3, vec3};
use vulkano::sync::GpuFuture;
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState, CommandBuffer};
use vulkano::pipeline::viewport::Viewport;
use vulkano::buffer::{BufferUsage, ImmutableBuffer};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;

use lighting_system::lighting_pass::LightingPass;
use lighting_system::{ LightSource, LightKind, ShadowKind };
use std::cell::RefCell;

mod geometry_pass;
pub mod lighting_system;
pub mod mesh;
use mesh::{
    Vertex3D, MeshAccess,
    MaterialMeshSlice,
    MeshData, MeshDataAsync,
    MaterialData, ObjectInstance,
};

pub struct Renderer3D {
    // Geometry to draw
    pub render_geometry: Vec<ObjectInstance>,

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
/// Comms with game_listener
impl Renderer3D {

    pub fn generate_mesh_from_data(&self, data: Vec<Vertex3D>) -> Arc<dyn MeshAccess + Send + Sync> {
        MeshData::from_data(self.queue.clone(), data)
    }

    pub fn generate_mesh_from_data_later(&self, data: Vec<Vertex3D>) -> Arc<dyn MeshAccess + Send + Sync> {
        MeshDataAsync::from_data(self.queue.clone(), data)
    }

    pub fn create_light_source(&mut self, kind: LightKind) -> Arc<RefCell<LightSource>> {
        self.lighting_pass.create_source(kind)
    }
    pub fn remove_light_source(&mut self, source: Arc<RefCell<LightSource>>) -> bool {
        self.lighting_pass.remove_source(source)
    }

}
/// Comms with main loop
impl Renderer3D {
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


        Self {
            render_geometry: Vec::new(),

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

    pub fn set_view_projection(&mut self, view_projection: Matrix4<f32>) {
        self.geom_pass.set_view_projection(view_projection);
        self.lighting_pass.set_view_projection(view_projection);
    }

    pub fn render<'f, F, I>(&mut self, prev_future: F, final_image: I) -> Box<dyn GpuFuture>
        where
            F: GpuFuture + 'static,
            I: ImageAccess + ImageViewAccess + Send + Sync + Clone + 'static,
    {
        for r in self.render_geometry.iter_mut() { r.update(); }

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
        let shadow_cb = self.lighting_pass.update(&self.render_geometry);

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
            main_cbb.execute_commands(self.geom_pass.render(&self.dyn_state, &mut self.render_geometry)).unwrap()
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