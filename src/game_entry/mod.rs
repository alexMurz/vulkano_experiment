
use winit::VirtualKeyCode as Keys;
use cgmath::{Matrix4, SquareMatrix, vec3};
use std::{
    io::Cursor,
    iter::Iterator,
    sync::Arc,
    path::Path,
};
use gfx_lib::{
    main_processor::{
        GameListener, Frame, FrameRequest
    },
    graphics::{
        Camera,
        image::{
            ImageContentAbstract, ImageContent,
            atlas::{
                AtlasBuilder,
                TextureAtlas,
                AtlasImageResolver,
                DirectoryImageResolver,
            },
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
    },
    sync::{ Loader, LoaderError },
};
use std::ops::Deref;

use vulkano::{
    format::Format,
    sync::GpuFuture,
};
use vulkano::image::ImageAccess;

mod ui_2d_pass;

/// Main Game Entry
pub struct GameEntry {
    camera: Camera,
//    atlas: TextureAtlas,

    pass_2d: ui_2d_pass::UI2DPass,

    renderer_2d: Renderer2D,
    renderer_3d: Renderer3D,

    time: f32, // Time sence beginning
    speed_mod: f32, // Cam Speed
    holding_mouse: bool, // Is currently holding mouse
}
impl GameEntry {
    pub fn new(init_frame: &mut Frame) -> Self {
//        let image2 = ImageContent::new_with_bytes(
//            init_frame.queue.clone(),
//            init_frame.sampler_pool.with_params(SamplerParams::simple_repeat()),
//            Cursor::new(include_bytes!("../data/icon128.png").to_vec()),
//            Format::R8G8B8A8Srgb,
//        );
//        let image1 = ImageContent::new_with_bytes(
//            init_frame.queue.clone(),
//            init_frame.sampler_pool.with_params(SamplerParams::simple_repeat()),
//            Cursor::new(include_bytes!("../data/icon512.png").to_vec()),
//            Format::R8G8B8A8Srgb,
//        );

        // 2D UI Pass
        let mut pass_2d = ui_2d_pass::UI2DPass::new(init_frame);

        // Wait on image before use
//        image1.flush();
//        image2.flush();

        // Atlas Image
//        let am = ImageContent::load_image(
//            init_frame.queue.clone(),
//            Cursor::new(include_bytes!("../data/icon512.png").to_vec()),
//            Format::R8G8B8A8Srgb,
//        );

        // Atlas Test
        let img_bytes = include_bytes!("../data/icon512.png").to_vec();
        let atlas = TextureAtlas::start()
            .set_max_dims(1024)
            .set_padding(1, 1)
            .set_background_color(1.0, 0.0, 1.0, 1.0)
            .set_format(Format::R8G8B8A8Snorm)
            .add_loader("icon512.png", ImageContent::load_image(
                init_frame.queue.clone(),
                Cursor::new(img_bytes),
                Format::R8G8B8A8Srgb)
            ).unwrap().set_scl([0.7, 0.7]).next()
//            .add_data("icon512.png", Cursor::new(img_bytes)).unwrap().set_scl([0.5, 0.5]).next()
            .build(init_frame).unwrap()
            .unwrap();

        // Transient image between renders and bake
//        ImageContent::load_image()

        let mut renderer_2d = Renderer2D::new(init_frame.queue.clone(), Format::R8G8B8A8Snorm, 1000);
        let mut renderer_3d = Renderer3D::new(init_frame.queue.clone(), init_frame.image.format());
        // Bake output onto output renderer and flip Y

        /* Setup lighting */ {
            /* Ambient */ {
                let mut source = renderer_3d.create_light_source(LightKind::Ambient);
                source.borrow_mut().active = true;
                source.borrow_mut().col(0.2, 0.2, 0.2);
//                source.borrow_mut().dist(0.1);
            }

            /* Spot Light */{
                let mut source = renderer_3d.create_light_source(LightKind::PointLight);
                source.borrow_mut().active = false;
                source.borrow_mut().pos(0.0, 5.0, 0.0);
                source.borrow_mut().col(1.0, 0.5, 0.5);
                source.borrow_mut().int(0.2);
                source.borrow_mut().dist(20.0);
            }

            let light_count = 5;
            let light_intensity = 0.2; // 1.0 / light_count as f32;
            let res_sq = 1024;
            let light_res = [res_sq, res_sq];

            for i in 0..light_count {
                let x = (i as f32 / light_count as f32 * 3.1415 * 2.0).sin() * 5.0;
                let y = (i as f32 / light_count as f32 * 3.1415 * 2.0).cos() * 5.0;
                let mut source = renderer_3d.create_light_source(LightKind::ConeWithShadow(
                    ShadowKind::Cone::with_projection(90.0, light_res)
                ));
                source.borrow_mut().pos(x, 5.0, y);
                source.borrow_mut().look_at(0.0, 0.0, 0.0);
                source.borrow_mut().int(light_intensity);
                source.borrow_mut().dist(20.0);
            }
        }

        // Create objects
        let mut geom = {

            let floor_size = 5.0;
            let floor_mesh = renderer_3d.generate_mesh_from_data(vec![
                Vertex3D::from_position(-floor_size, 0.0,-floor_size).uv(0.0, 0.0).normal(0.0, 1.0, 0.0),
                Vertex3D::from_position(-floor_size, 0.0, floor_size).uv(0.0, 1.0).normal(0.0, 1.0, 0.0),
                Vertex3D::from_position( floor_size, 0.0,-floor_size).uv(1.0, 0.0).normal(0.0, 1.0, 0.0),
                Vertex3D::from_position( floor_size, 0.0, floor_size).uv(1.0, 1.0).normal(0.0, 1.0, 0.0),
            ], Some(vec![0, 1, 2, 1, 3, 2]));

            let mut floor_obj = ObjectInstance::new(floor_mesh.unwrap());
            floor_obj.materials.push(MaterialMeshSlice {
                vbo_slice: floor_obj.mesh_data.get_vbo_slice(),
                ibo_slice: Some(floor_obj.mesh_data.get_ibo()),
                material: {
                    let mut md = MaterialData::new();
                    md.set_diffuse_texture_with_sampler(atlas.get_image(), init_frame.sampler_pool.with_params(SamplerParams::simple_repeat()));
                    md
                }
            });
            floor_obj.set_pos(0.0, -2.0, 0.0);

            let mut object_data = gfx_lib::loader::obj::load_objects(
                &Path::new("src/data/test.obj"),
                vec!["Plane"]
            ).unwrap();

            let mut obj1 = renderer_3d.generate_object(
                object_data.remove("Plane").unwrap(),
//                DirectoryImageResolver::new(
//                    Path::new("src/data"),
//                    init_frame.queue.clone(),
//                    init_frame.sampler_pool.with_params(SamplerParams::simple_repeat())
//                ).unwrap()
                AtlasImageResolver::new(&atlas)
            ).unwrap();

            vec![floor_obj, obj1]
        };

        for v in geom.drain(..) { renderer_3d.render_geometry.push(v) }

        Self {
            camera: {
                let mut c = Camera::new(Matrix4::identity());
                c.pos[2] = -5.0;
                c
            },
//            atlas,

            pass_2d,

            renderer_2d,
            renderer_3d,

            time: 0.0,
            speed_mod: 0.0,
            holding_mouse: false,
        }
    }

    fn pass_2d(&mut self, delta: f32, frame: &mut Frame, future: Box<dyn GpuFuture + Send + Sync>) -> Box<dyn GpuFuture + Send + Sync> {
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


    fn update(&mut self, delta: f32, frame: &mut Frame, mut future: Box<dyn GpuFuture>) -> Box<dyn GpuFuture> {
        self.time += delta;
//        println!("FPS: {}", 1.0 / delta);

        /* Process Camera Movement */ {
            let forward = if frame.key_state(Keys::W) { 1.0 } else if frame.key_state(Keys::S) { -1.0 } else { 0.0 };
            let right = if frame.key_state(Keys::A) { 1.0 } else if frame.key_state(Keys::D) { -1.0 } else { 0.0 };
            let up = if frame.key_state(Keys::F) { 1.0 } else if frame.key_state(Keys::R) { -1.0 } else { 0.0 };

            let sprint = if frame.key_state(Keys::LShift) { 2.0 } else { 1.0 };
            let speed = 1.05f32.powf(self.speed_mod) * delta * sprint * 2.0;
            self.camera.move_by(-forward * speed,  -right * speed, -up * speed);

            if self.holding_mouse {
                let spd = frame.cursor_spd();
                self.camera.rotate_by( spd[1] * 40.0, spd[0] * 40.0, 0.0);
            }
        }

//        self.pass_2d.output = frame.image.clone();
//        let mut future = self.pass_2d.render(&mut self.renderer_2d, future);

        self.renderer_3d.set_view_projection(self.camera.get_view_projection());
        future = self.renderer_3d.render(future, frame.image.clone());

//        future = self.renderer_3d.render(future, self.transient_image.clone());
//        future = {
//            self.bake_renderer.begin(frame.image.clone());
//            self.bake_renderer.sta
//        };

        future
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
