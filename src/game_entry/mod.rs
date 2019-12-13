
use crate::main_processor::{
    GameListener, Frame, FrameRequest
};
use winit::VirtualKeyCode as Keys;
use cgmath::{Matrix4, SquareMatrix, vec3};
use std::{
    io::Cursor,
    iter::Iterator,
    sync::Arc,
};
use crate::graphics::{
    Camera,
    image::{
        ImageContentAbstract, ImageContent,
        sampler_pool::{ SamplerParams, SamplerPool },
    },
    renderer_3d::{
        lighting_system::{ LightSource, LightKind, ShadowKind },
        mesh::{ Vertex3D, MeshAccess, MaterialMeshSlice, MeshData, MaterialData, ObjectInstance },
        Renderer3D,
    },
    renderer_2d::Renderer2D,
    object::{
        ScreenVertex, ScreenInstance
    }
};
use vulkano::{
    image::{ ImageAccess, AttachmentImage, ImageUsage },
    sampler::{ Sampler, Filter, MipmapMode, SamplerAddressMode },
    buffer::BufferAccess,
    format::Format,
    sync::GpuFuture
};

mod ui_2d_pass;

/// Main Game Entry
pub struct GameEntry {
    camera: Camera,
    floor_object: ObjectInstance,
    test_object: ObjectInstance,

    pass_2d: ui_2d_pass::UI2DPass,

    renderer_2d: Renderer2D,
    renderer_3d: Renderer3D,

    time: f32, // Time sence beginning
    speed_mod: f32, // Cam Speed
    holding_mouse: bool, // Is currently holding mouse
}
impl GameEntry {
    pub fn new(init_frame: &mut Frame) -> Self {
        let image = ImageContent::from_bytes(
            init_frame.queue.clone(),
            init_frame.sampler_pool.with_params(SamplerParams::simple_repeat()),
            Cursor::new(include_bytes!("../data/icon512.png").to_vec()),
            Format::R8G8B8A8Srgb,
        );

        // 2D UI Pass
        let mut pass_2d = ui_2d_pass::UI2DPass::new(init_frame);

        // Renderer2D renders into attachment to use in renderer3D
        let mut renderer_2d = Renderer2D::new(init_frame.queue.clone(), Format::R8G8B8A8Snorm, 1000*1000);

        let mut renderer_3d = Renderer3D::new(init_frame.queue.clone(), init_frame.image.format());

        /* Setup lighting */ {
            /* Ambient */ {
                let mut source = renderer_3d.create_light_source(LightKind::Ambient);
                source.borrow_mut().active = true;
                source.borrow_mut().col(0.1, 0.1, 0.3);
                source.borrow_mut().dist(0.1);
            }

            /* Spot Light */{
                let mut source = renderer_3d.create_light_source(LightKind::PointLight);
                source.borrow_mut().active = false;
                source.borrow_mut().pos(0.0, -1.0, 3.0);
                source.borrow_mut().col(0.8, 0.4, 0.2);
                source.borrow_mut().dist(10.0);
            }

            let light_count = 4;
            let light_intensity = 0.5; // 1.0 / light_count as f32;
            let res_sq = 1024;
            let light_res = [res_sq, res_sq];
            for i in 0..light_count {
                let x = (i as f32 / light_count as f32 * 3.1415 * 2.0).sin() * 5.0;
                let y = (i as f32 / light_count as f32 * 3.1415 * 2.0).cos() * 5.0;
                let mut source = renderer_3d.create_light_source(LightKind::ConeWithShadow(
                    ShadowKind::Cone::with_projection(45.0, light_res)
                ));
                source.borrow_mut().pos(x, -2.0, y);
                source.borrow_mut().look_at(0.0, 0.0, 0.0);
                source.borrow_mut().int(light_intensity);
                source.borrow_mut().dist(10.0);
            }
        }

        // Create
        let (mut floor_object, mut test_object) = {

            let floor_size = 10.0;
            let floor_mesh = renderer_3d.generate_mesh_from_data(vec![
                Vertex3D::from_position(-floor_size, 0.0,-floor_size).uv(0.0, 0.0).color(1.0, 1.0, 1.0, 1.0),
                Vertex3D::from_position(-floor_size, 0.0, floor_size).uv(0.0, 1.0).color(1.0, 1.0, 1.0, 1.0),
                Vertex3D::from_position( floor_size, 0.0,-floor_size).uv(1.0, 0.0).color(1.0, 1.0, 1.0, 1.0),

                Vertex3D::from_position(-floor_size, 0.0, floor_size).uv(0.0, 1.0).color(1.0, 1.0, 1.0, 1.0),
                Vertex3D::from_position( floor_size, 0.0, floor_size).uv(1.0, 1.0).color(1.0, 1.0, 1.0, 1.0),
                Vertex3D::from_position( floor_size, 0.0,-floor_size).uv(1.0, 0.0).color(1.0, 1.0, 1.0, 1.0),
            ]);

            let mut vertices = Vec::new();
            let blend = blend::Blend::from_path("src/data/test.blend");
            crate::loader::blender::load_model_faces(&blend, "Sphere", |face| {
                for i in 0..face.vert_count {
                    vertices.push(
                        Vertex3D::from_position(face.vert[i][0], face.vert[i][1], face.vert[i][2])
                            .normal(face.norm[i][0], face.norm[i][1], face.norm[i][2])
                            .uv(face.uv[i][0], face.uv[i][1])
                            .color(1.0, 1.0, 1.0, 1.0)
                    );
                }
            });
            let obj_mesh = renderer_3d.generate_mesh_from_data_later(vertices);

            let mut floor_obj = ObjectInstance::new(floor_mesh);
            floor_obj.materials.push(MaterialMeshSlice {
                slice: floor_obj.mesh_data.get_vbo_slice(),
                material: {
                    let mut md = MaterialData::new();
//                    md.set_diffuse_texture_with_sampler(renderer_2d_att.clone(), sampler_pool.with_params(SamplerParams::simple_repeat()));
                    md
                }
            });
            floor_obj.set_pos(0.0, 2.0, 0.0);

            let mut test_obj = ObjectInstance::new(obj_mesh);
            test_obj.materials.push(MaterialMeshSlice {
                slice: test_obj.mesh_data.get_vbo_slice(),
                material: {
                    let mut md = MaterialData::new();
                    md.set_flat_shading(false);
                    md.set_diffuse_texture_with_sampler(pass_2d.output.clone(), init_frame.sampler_pool.with_params(SamplerParams::simple_repeat()));
                    md
                }
            });
            (floor_obj, test_obj)
        };

        renderer_3d.render_geometry.push(floor_object.clone());
        renderer_3d.render_geometry.push(test_object.clone());

        Self {
            camera: Camera::new(Matrix4::identity()),

            pass_2d,

            renderer_2d,
            renderer_3d,

            floor_object,
            test_object,

            time: 0.0,
            speed_mod: 0.0,
            holding_mouse: false,
        }
    }

    fn pass_2d(&mut self, delta: f32, frame: &mut Frame, future: Box<dyn GpuFuture>) -> Box<dyn GpuFuture> {
        self.pass_2d.render(&mut self.renderer_2d, future)
    }
}

impl GameListener for GameEntry {

    fn dimensions_changed(&mut self, frame: &mut Frame, width: u32, height: u32) {
        self.camera.set_projection(cgmath::perspective(
            cgmath::Deg(60.0), width as f32 / height as f32, 0.1, 100.0
        ));

//        let aspect = width as f32 / height as f32;
        self.renderer_2d.set_viewport_window(1.0, 1.0);
    }


    fn update(&mut self, delta: f32, frame: &mut Frame, future: Box<dyn GpuFuture>) -> Box<dyn GpuFuture> {
        self.time += delta;

        /* Process Camera Movement */ {
            let forward = if frame.key_state(Keys::W) { 1.0 } else if frame.key_state(Keys::S) { -1.0 } else { 0.0 };
            let right = if frame.key_state(Keys::A) { 1.0 } else if frame.key_state(Keys::D) { -1.0 } else { 0.0 };
            let up = if frame.key_state(Keys::R) { 1.0 } else if frame.key_state(Keys::F) { -1.0 } else { 0.0 };

            let sprint = if frame.key_state(Keys::LShift) { 2.0 } else { 1.0 };
            let speed = 1.05f32.powf(self.speed_mod) * delta * sprint * 2.0;
            self.camera.move_by(-forward * speed,  -right * speed, -up * speed);

            if self.holding_mouse {
                let spd = frame.cursor_spd();
                self.camera.rotate_by(-spd[1] * 40.0, spd[0] * 40.0, 0.0);
            }
        }

        let f = self.pass_2d(delta, frame, future);

        self.renderer_3d.set_view_projection(self.camera.get_view_projection());
        self.renderer_3d.render(f, frame.image.clone())
    }

    fn key_pressed(&mut self, frame: &mut Frame, keycode: Keys) {
        use Keys::*;
        match keycode {
            Escape => frame.request(FrameRequest::ExitApplication),
//            F2 => self.async_2d = !self.async_2d,
            F1 => {
                self.holding_mouse = !self.holding_mouse;
                frame.request(FrameRequest::HoldCursor(Some(self.holding_mouse)))
            },
            _ => (),
        }
    }

    fn mouse_wheel(&mut self, frame: &mut Frame, x: f32, y: f32) {
        self.speed_mod += y;
    }
}
