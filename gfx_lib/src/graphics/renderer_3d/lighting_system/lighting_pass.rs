
use std::sync::Arc;
use vulkano::device::Queue;
use vulkano::framebuffer::{Subpass, RenderPassAbstract};
use vulkano::pipeline::{GraphicsPipelineAbstract, GraphicsPipeline};
use vulkano::buffer::{BufferAccess, ImmutableBuffer, BufferUsage, CpuAccessibleBuffer};
use crate::graphics::object::{ScreenVertex};
use vulkano::sync::GpuFuture;
use cgmath::{ Matrix4, SquareMatrix };
use vulkano::command_buffer::{AutoCommandBuffer, AutoCommandBufferBuilder, DynamicState};
use vulkano::image::{AttachmentImage};
use std::cell::RefCell;

use crate::graphics::renderer_3d::{
    mesh::{ Vertex3D, ObjectInstance },
    lighting_system::light_draw_systems::{
        shadow_cone_light::ShadedConeLight,
        ambient_light::AmbientLight,
        point_light::PointLight,
    },
};

use crate::graphics::renderer_3d::lighting_system::shadow_mapper::ShadowMapping;
use crate::graphics::renderer_3d::lighting_system::{LightSource, LightKind };

// Apply diffirent lighting methods

pub struct LightingPass {
    queue: Arc<Queue>,
    // Full screen VBO square
    vbo: Arc<dyn BufferAccess + Send + Sync>,

    // Sources
    sources: Vec<Arc<RefCell<LightSource>>>,

    // Shadow mapper
    shadow_mapper: ShadowMapping,

    // Pass for one kind of light source
    // Flat lights
    ambient_light: AmbientLight,
    point_light: PointLight,

    // Lights with shadows
    shadow_cone_light: ShadedConeLight,


    // camera view_projection
    view_projection: Matrix4<f32>,
}
impl LightingPass {
    pub fn new<R>(queue: Arc<Queue>, subpass: Subpass<R>) -> Self
        where R: RenderPassAbstract + Send + Sync + Clone + 'static
    {

        let vbo = {
            let (a, b) = ImmutableBuffer::from_iter(vec![
                ScreenVertex::with_pos(-1.0, -1.0),
                ScreenVertex::with_pos(-1.0,  1.0),
                ScreenVertex::with_pos( 1.0, -1.0),
                ScreenVertex::with_pos( 1.0,  1.0),
            ].iter().cloned(), BufferUsage::vertex_buffer(), queue.clone()).unwrap();
            b.flush().unwrap();
            a
        };

        let shadow_mapper = ShadowMapping::new(queue.clone());

        let shadow_cone_light = ShadedConeLight::new(
            queue.clone(),
            Subpass::clone(&subpass)
        );
        let ambient_light = AmbientLight::new(
            queue.clone(),
            Subpass::clone(&subpass)
        );
        let point_light = PointLight::new(
            queue.clone(),
            Subpass::clone(&subpass)
        );

        Self {
            queue,
            vbo,
            sources: Vec::new(),

            shadow_mapper,

            ambient_light,
            point_light,
            shadow_cone_light,

            view_projection: Matrix4::identity(),
        }
    }

    /// Create and enable new light source
    pub fn create_source(&mut self, kind: LightKind) -> Arc<RefCell<LightSource>>{
        let source = Arc::new(RefCell::new(LightSource::new(kind)));
        self.sources.push(source.clone());
        source
    }

    /// Remove source
    /// Return true if source found and removed or else false
    pub fn remove_source(&mut self, arc: Arc<RefCell<LightSource>>) -> bool {
        for i in 0 .. self.sources.len() {
            if self.sources[i] == arc {
                self.sources.remove(i);
                return true;
            }
        }
        false
    }

    pub fn set_view_projection(&mut self, vp: Matrix4<f32>) {
        if !self.view_projection.eq(&vp) {
            self.view_projection = vp;
            let to_world = Matrix4::invert(&vp).unwrap();

            self.ambient_light.to_world = to_world;
            self.point_light.to_world = to_world;
            self.shadow_cone_light.to_world = to_world;
        }
    }

    pub fn set_attachments(&mut self,
                           diffuse_buffer: Arc<AttachmentImage>,
                           normal_buffer: Arc<AttachmentImage>,
                           depth_buffer: Arc<AttachmentImage>,
    )
    {

        self.ambient_light.set_attachments(
            diffuse_buffer.clone(),
            normal_buffer.clone(),
            depth_buffer.clone(),
        );

        self.point_light.set_attachments(
            diffuse_buffer.clone(),
            normal_buffer.clone(),
            depth_buffer.clone(),
        );

        self.shadow_cone_light.set_attachments(
            diffuse_buffer.clone(),
            normal_buffer.clone(),
            depth_buffer.clone(),
        );
    }

    pub fn update<'f>(&mut self, geometry: &Vec<ObjectInstance>)  -> AutoCommandBuffer {
        let mut cbb = AutoCommandBufferBuilder::primary_one_time_submit(
            self.queue.device().clone(),
            self.queue.family()
        ).unwrap();
        for s in self.sources.iter_mut() {
            if s.borrow().active {
                s.borrow_mut().update();
                if s.borrow().kind.has_shadow() {
                    unsafe {
                        cbb = cbb.execute_commands(self.shadow_mapper.render_image(
                            &mut s.borrow_mut().kind, geometry
                        )).unwrap();
                    };
                }
            }
        }
        cbb.build().unwrap()
    }

    pub fn render(&mut self, mut cbb: AutoCommandBufferBuilder, dyn_state: &DynamicState) -> AutoCommandBufferBuilder {
        for s in self.sources.iter_mut() {
            let mut source = &mut s.borrow_mut();
            if source.active {
                match &mut source.kind {
                    LightKind::Ambient => unsafe {
                        cbb = cbb.execute_commands(self.ambient_light.render(
                            source, self.vbo.clone(), dyn_state
                        )).unwrap();
                    },
                    LightKind::PointLight => unsafe {
                        cbb = cbb.execute_commands(self.point_light.render(
                            source, self.vbo.clone(), dyn_state
                        )).unwrap();
                    },

                    // Shaded
                    LightKind::ConeWithShadow(cone) => unsafe {
                        cbb = cbb.execute_commands(self.shadow_cone_light.render(
                            cone, self.vbo.clone(), dyn_state
                        )).unwrap();
                    },
                }
            }
        }
        cbb
    }

}