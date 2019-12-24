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
use vulkano::buffer::{BufferUsage, ImmutableBuffer, BufferAccess, BufferSlice};
use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;

use lighting_system::lighting_pass::LightingPass;
use lighting_system::{ LightSource, LightKind, ShadowKind };
use std::cell::RefCell;

mod geometry_pass;

pub mod lighting_system;
pub mod mesh;
pub mod post_processing;

use crate::loader::{
    ObjectInfo, MaterialSlice, MaterialInfo, MaterialImageUsage
};
use crate::sync::{ Loader, LoaderError };
use mesh::{
    Vertex3D, MeshAccess,
    MaterialMeshSlice,
    MaterialData, ObjectInstance,
};
use crate::graphics::image::atlas::{TextureRegion, ImageResolver};
use crate::graphics::renderer_3d::post_processing::bake_image::PostBakeImage;
use crate::graphics::renderer_3d::mesh::ImmutableMeshData;

const DIFFUSE_FORMAT: Format = Format::A2B10G10R10UnormPack32;
const DEPTH_FORMAT: Format = Format::D16Unorm;

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
    transient_depth_buffer: Arc<AttachmentImage>,
    depth_buffer: Arc<AttachmentImage>,

    // Passes
    geom_pass: GeometryPass,
    lighting_pass: LightingPass,
}
/// Comms with game_listener
impl Renderer3D {

    pub fn generate_mesh_from_data(&self, data: Vec<Vertex3D>, indices: Option<Vec<u32>>) -> Loader<Arc<dyn MeshAccess + Send + Sync + 'static>> {
        ImmutableMeshData::from_data(self.queue.clone(), data, indices).into()
    }

    pub fn generate_object(&self, mut object: ObjectInfo, mut image_resolver: Box<dyn ImageResolver + Send + 'static>) -> Loader<ObjectInstance> {

        fn generate_material(material: &MaterialInfo, resolver: &mut Box<dyn ImageResolver + Send + 'static>) -> MaterialData {
            let mut data = MaterialData::new();
            data.set_alpha(material.dissolve);
            data.set_cast_shadow(material.cast_shadow);
            if material.diffuse_tex.is_some() {
                data.set_diffuse_region(resolver.get(
                    MaterialImageUsage::Diffuse,
                    material.diffuse_tex.as_ref().unwrap()
                ).unwrap());
            }
            data
        }

        let mut mesh_loader = self.generate_mesh_from_data(
            object.vertices.iter()
                .map(|x| x.clone().into())
                .collect(),
            if object.indices.is_empty() { None } else { Some(object.indices.iter().cloned().collect()) }
        );
        let queue = self.queue.clone();
        Loader::with_closure(move || {
            let mesh = mesh_loader.take();
            // Object instance that we are building
            let mut inst = ObjectInstance::new(mesh.clone());
            // Futures to wait on before loading is complete
            let mut future = Box::new(vulkano::sync::now(queue.device().clone())) as Box<dyn GpuFuture + Send + Sync>;

            // Build Gpu-friendly materials from `MaterialInfo`
            for m in object.materials.iter() { match m {
                MaterialSlice::WithIndices { material, indices } => {
                    let (ibo_buffer, ibo_future) = ImmutableBuffer::from_iter(
                        indices.iter().cloned(),
                        BufferUsage::index_buffer(),
                        queue.clone()
                    ).unwrap();
                    future = Box::new(future.join(ibo_future));

                    inst.materials.push(MaterialMeshSlice {
                        vbo_slice: mesh.get_vbo_slice(),
                        ibo_slice: Some(ibo_buffer),
                        material: generate_material(material, &mut image_resolver)
                    })
                },
                MaterialSlice::WithVertexSlice { material, vertex_slice } => {
                    inst.materials.push(MaterialMeshSlice {
                        vbo_slice: {
                            Arc::new(BufferSlice::from_typed_buffer_access(mesh.get_vbo())
                                .slice(vertex_slice.clone()
                            ).unwrap())
                        },
                        ibo_slice: None,
                        material: generate_material(material, &mut image_resolver)
                    })
                }
            } }

            // Wait on GpuFutures
            future.flush().unwrap();
            image_resolver.flush();

            inst
        })
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
                    format: DIFFUSE_FORMAT,
                    samples: 1,
                },
                // Will be bound to `self.normals_buffer`.
                normals: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R16G16B16A16Sfloat,
                    samples: 1,
                },
                // Depth used for geometry pass
                transient_depth: {
                    load: Clear,
                    store: DontCare,
                    format: DEPTH_FORMAT,
                    samples: 1,
                },
                // Buffered depth
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: DEPTH_FORMAT,
                    samples: 1,
                }
            },
            passes: [
                // Write to the diffuse, normals and depth attachments.
                {
                    color: [ diffuse ],
                    depth_stencil: { transient_depth },
                    input: []
                },
                // Write depth again, for shadow mapping
                {
                    color: [ normals ],
                    depth_stencil: { depth },
                    input: []
                },

                // Apply lighting by reading these three attachments and writing to `final_color`.
                {
                    color: [ final_color ],
                    depth_stencil: {},
                    input: [ diffuse, normals, depth ]
                }
            ]
        ).unwrap()) as Arc<dyn RenderPassAbstract + Send + Sync>;

        let atch_usage = ImageUsage {
            transient_attachment: true,
            input_attachment: true,
            ..ImageUsage::none()
        };

        let diffuse_buffer = AttachmentImage::with_usage(
            queue.device().clone(), [1, 1], DIFFUSE_FORMAT, atch_usage
        ).unwrap();
        let normal_buffer = AttachmentImage::with_usage(
            queue.device().clone(), [1, 1], Format::R16G16B16A16Sfloat, atch_usage
        ).unwrap();
        let transient_depth_buffer = AttachmentImage::with_usage(
            queue.device().clone(), [1, 1], DEPTH_FORMAT, atch_usage
        ).unwrap();
        let depth_buffer = AttachmentImage::with_usage(
            queue.device().clone(), [1, 1], DEPTH_FORMAT, atch_usage
        ).unwrap();


        let geom_pass = GeometryPass::new(
            queue.clone(),
            Subpass::from(render_pass.clone(), 1).unwrap(),
            Subpass::from(render_pass.clone(), 0).unwrap()
        );
        let mut lighting_pass = LightingPass::new(
            queue.clone(),
            Subpass::from(render_pass.clone(), 2).unwrap()
        );


        Self {
            render_geometry: Vec::new(),

            queue,
            render_pass,
            dyn_state: DynamicState::none(),

            diffuse_buffer,
            normal_buffer,
            transient_depth_buffer,
            depth_buffer,

            geom_pass,
            lighting_pass,
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

            let atch_usage = ImageUsage {
                transient_attachment: true,
                input_attachment: true,
                ..ImageUsage::none()
            };

            self.diffuse_buffer = AttachmentImage::with_usage(
                self.queue.device().clone(),
                img_dims,
                DIFFUSE_FORMAT,
                atch_usage
            ).unwrap();

            self.normal_buffer = AttachmentImage::with_usage(
                self.queue.device().clone(),
                img_dims,
                Format::R16G16B16A16Sfloat,
                atch_usage
            ).unwrap();

            self.transient_depth_buffer = AttachmentImage::with_usage(
                self.queue.device().clone(),
                img_dims,
                DEPTH_FORMAT,
                atch_usage
            ).unwrap();

            self.depth_buffer = AttachmentImage::with_usage(
                self.queue.device().clone(),
                img_dims,
                DEPTH_FORMAT,
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
                .add(self.transient_depth_buffer.clone()).unwrap()
                .add(self.depth_buffer.clone()).unwrap()
                .build().unwrap()
        );

        let mut main_cbb = AutoCommandBufferBuilder::primary_one_time_submit(
            self.queue.device().clone(), self.queue.family()
        ).unwrap()
            .begin_render_pass(framebuffer.clone(), false, vec![
                [0.0, 0.0, 0.0, 1.0].into(),
                [0.0, 0.0, 0.0, 0.0].into(),
                [0.0, 0.0, 0.0, 0.0].into(),
                1.0.into(),
                1.0.into(),
            ]).unwrap();

        // Do geometry depth only
        main_cbb = self.geom_pass.bake_materials(main_cbb, &self.dyn_state, &mut self.render_geometry);

        // Do geometry pass
        main_cbb = main_cbb.next_subpass(false).unwrap();
        main_cbb = self.geom_pass.bake_depth_normal(main_cbb, &self.dyn_state, &mut self.render_geometry);

        // Do Lighting
        main_cbb = main_cbb.next_subpass(true).unwrap();
        main_cbb = self.lighting_pass.render(main_cbb, &self.dyn_state);

        let main_cb = main_cbb.end_render_pass().unwrap().build().unwrap();

        Box::new(prev_future
                     .then_execute(self.queue.clone(), shadow_cb).unwrap()
                     .then_execute(self.queue.clone(), main_cb).unwrap()
        )
    }

}