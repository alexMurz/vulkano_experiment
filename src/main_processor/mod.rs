// Point of this class is to process, sort, categorize and send event to listeners and renderer
// Creates window from config


use vulkano_win::{ VkSurfaceBuild };
use vulkano::{
    instance::{ Instance, QueueFamily, PhysicalDevice, MemoryType, ApplicationInfo },
    device::{ Queue, QueuesIter, Device, DeviceExtensions },
    swapchain::{ self, Surface, Swapchain, SurfaceTransform, PresentMode, AcquireError },
    image::{ SwapchainImage },
    sync::{ self, GpuFuture, FlushError },
};
use winit::{EventsLoop, dpi::{LogicalPosition, LogicalSize}, VirtualKeyCode, ElementState, Window};

use std::sync::Arc;
use std::cell::RefCell;

use crate::loader;
use crate::graphics::{
    renderer,
    Camera,
    object::{ Vertex3D, ObjectInstance }
};

pub mod settings;
use settings::{
    GameSettings,
    BackendInfo
};
use vulkano::swapchain::SwapchainAcquireFuture;
use vulkano::device::DeviceOwned;


/// Main Processor listener
pub trait GameListener {
    fn update(&mut self, delta: f32);

    fn key_pressed(&mut self, keycode: VirtualKeyCode) { }
    fn key_released(&mut self, keycode: VirtualKeyCode) { }
}

/// Resource Provider for GameListener
pub trait ResourceProvider {
    fn key_state(&self, keycode: VirtualKeyCode) -> bool;
}


/// Main Processor
/// Sends events and backend attachments to its GameListener
pub struct MainProcessor {
    used_settings: GameSettings,

    backend: BackendInfo,
    swapchain: SwapchainConfig,

    keyboard_state: Arc<RefCell<KeyboardState>>,
    listener: Option<Box<dyn GameListener>>,
}
impl MainProcessor {
    pub fn new(settings: GameSettings) -> Option<Self> {
        let backend = settings.generate_backend()?;
        let swapchain = SwapchainConfig::create(&backend)?;
        Some(Self {
            used_settings: settings,

            backend,
            swapchain,

            keyboard_state: Arc::new(RefCell::new(KeyboardState::new())),
            listener: None
        })
    }

    pub fn set_game_listener(&mut self, game_listener: Box<dyn GameListener>) {
        self.listener = Some(game_listener);
    }

    pub fn start_loop(&mut self) {
        let mut renderer = renderer::Renderer::new(self.swapchain.main_queue.clone(), self.swapchain.swapchain.format());

        // Gpu Future
        let mut last_sync = Box::new(sync::now(self.swapchain.device())) as Box<dyn GpuFuture>;

        // Running flag
        let mut is_running = true;

        let mut prev_time = time::precise_time_ns();
        while is_running {
            let time = time::precise_time_ns();
            let delta = (time - prev_time) as f32 / 1e9;
            prev_time = time;

            // Do update
            if self.listener.is_some() {
                self.listener.as_mut().unwrap().update(delta);
            }

            // Do swapchain maintenance
            self.swapchain.update_if_required(&self.backend);

            // Do acquire swapchain image
            let (image_num, acquire_future) = match self.swapchain.acquire() {
                Ok(r) => r,
                Err(AcquireError::OutOfDate) => {
                    self.swapchain.recreate();
                    continue;
                },
                Err(err) => panic!("{:?}", err),
            };

            // Do drawing

            // ...

            let future = renderer.render(
                last_sync.join(acquire_future),
                self.swapchain.images[image_num].clone(),
                &vec![]
            );


            // Update GpuFuture
            match future.then_swapchain_present(
                self.swapchain.main_queue.clone(),
                self.swapchain.swapchain.clone(),
                image_num
            ).then_signal_fence_and_flush() {
                Ok(future) => {
                    // This wait is required when using NVIDIA or running on macOS. See https://github.com/vulkano-rs/vulkano/issues/1247
                    future.wait(None).unwrap();
                    last_sync = Box::new(future) as Box<_>;
                }
                Err(FlushError::OutOfDate) => {
                    self.swapchain.recreate();
                    last_sync = Box::new(sync::now(self.swapchain.device().clone())) as Box<_>;
                }
                Err(e) => {
                    println!("{:?}", e);
                    return;
                }
            }

            // Process Events
            let mut kb = &mut self.keyboard_state;
            let mut sc = &mut self.swapchain;
            self.backend.event_loop.poll_events(|e| {
                use winit::Event;
                match e {
                    Event::WindowEvent { event, ..} => {
                        use winit::WindowEvent;
                        match event {
                            WindowEvent::Resized(_) => sc.recreate(),
                            WindowEvent::CloseRequested => is_running = false,
                            WindowEvent::KeyboardInput { input: winit::KeyboardInput{ virtual_keycode: Some(keycode), state, .. }, .. } => {
                                use winit::VirtualKeyCode::*;
                                kb.borrow_mut().key_event(keycode, state);
                                match keycode {
                                    Escape => is_running = false,
                                    _ => (),
                                }
                            }
                            _ => (),
                        }


                    },
                    _ => (),
                }
            });
        }
    }
}

/// Contain all associated information about swapchain
pub struct SwapchainConfig {
    swapchain: Arc<Swapchain<Window>>,
    images: Vec<Arc<SwapchainImage<Window>>>,

    main_queue: Arc<Queue>,
    queues: QueuesIter,

    recreate: bool
}
impl SwapchainConfig {

    pub fn create(backend: &BackendInfo) -> Option<Self> {
        let window = backend.window();

        // Dims
        let mut dimensions = if let Some(dimensions) = window.get_inner_size() {
            let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
            [dimensions.0, dimensions.1]
        } else {
            return None;
        };

//        println!("Window with dimensions {:?}", dimensions);

        // Select physical device
        let physical = {
            let dev = PhysicalDevice::enumerate(&backend.instance).next().unwrap();
//            println!("Using device: {} (type: {:?})", dev.name(), dev.ty());
            dev
        };

        let queue_family = physical.queue_families().find(|&q|
            q.supports_graphics() && backend.surface.is_supported(q).unwrap_or(false)
        ).unwrap();

        let device_ext = DeviceExtensions { khr_swapchain: true, .. DeviceExtensions::none() };

        let (device, mut queues) = Device::new(
            physical, physical.supported_features(), &device_ext,
            [(queue_family, 0.5)].iter().cloned()
        ).unwrap();

        let main_queue = queues.next().unwrap();

        let (mut swapchain, mut images) = {
            let caps = backend.surface.capabilities(physical).unwrap();
//            println!("Device Caps: {:?}", caps);

            let usage = caps.supported_usage_flags;
            let format = caps.supported_formats[0].0;
            let alpha = caps.supported_composite_alpha.iter().next().unwrap();

//            println!("Formats: {:?}", caps.present_modes);

            let image_count = caps.min_image_count;
            Swapchain::new(device.clone(), backend.surface.clone(),
                           image_count, format, dimensions, 1,
                           usage, &main_queue, SurfaceTransform::Identity,
                           alpha, PresentMode::Immediate,
                           true, None).unwrap()
        };

        Some(Self {
            swapchain,
            images,

            main_queue,
            queues,

            recreate: false })
    }

    pub fn device(&self) -> Arc<Device> {
        self.swapchain.device().clone()
    }

    pub fn recreate(&mut self) { self.recreate = true; }

    pub fn update_if_required(&mut self, backend: &BackendInfo) {
        if self.recreate {
            let window = backend.window();
            let dimensions = if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
                [dimensions.0, dimensions.1]
            } else {
                return;
            };
            let (new_swapchain, new_images) = self.swapchain.recreate_with_dimension(dimensions).unwrap();
            self.swapchain = new_swapchain;
            self.images = new_images;
            self.recreate = false;
        }
    }

    pub fn acquire(&self) -> Result<(usize, SwapchainAcquireFuture<Window>), AcquireError> {
        swapchain::acquire_next_image(self.swapchain.clone(), None)
    }

}

pub fn start() {
    let mut event_loop = winit::EventsLoop::new();

    // Create instance
    let instance = {
        let extensions = vulkano_win::required_extensions();
        Instance::new(None, &extensions, None).unwrap()
    };

    // Select physical device
    let physical = {
        let dev = PhysicalDevice::enumerate(&instance).next().unwrap();
        println!("Using device: {} (type: {:?})", dev.name(), dev.ty());
        dev
    };

    let default_width = 800.0;
    let default_height = 600.0;
    let default_aspect = (default_width / default_height) as f32;

    let surface = {
        let wb = winit::WindowBuilder::new()
            .with_title("Title")
            .with_dimensions((default_width, default_height).into())
            .with_min_dimensions((640.0, 480.0).into())
            .with_max_dimensions((1920.0, 1080.0).into())
            .with_resizable(true);
        wb.build_vk_surface(&event_loop, instance.clone()).unwrap()
    };
    let window = surface.window();
    let mut mouse_hold_state = false;

    // Dims
    let mut dimensions = if let Some(dimensions) = window.get_inner_size() {
        let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
        [dimensions.0, dimensions.1]
    } else {
        return;
    };

    let queue_family = physical.queue_families().find(|&q|
        q.supports_graphics() && surface.is_supported(q).unwrap_or(false)
    ).unwrap();

    let device_ext = DeviceExtensions { khr_swapchain: true, .. DeviceExtensions::none() };

    let (device, mut queues) = Device::new(
        physical, physical.supported_features(), &device_ext,
        [(queue_family, 0.5)].iter().cloned()
    ).unwrap();

    let main_queue = queues.next().unwrap();

    let (mut swapchain, mut images) = {
        let caps = surface.capabilities(physical).unwrap();
        println!("Device Caps: {:?}", caps);

        let usage = caps.supported_usage_flags;
        let format = caps.supported_formats[0].0;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        println!("Formats: {:?}", caps.present_modes);

        let image_count = caps.min_image_count;

        Swapchain::new(device.clone(), surface.clone(),
                       image_count, format, dimensions, 1,
                       usage, &main_queue, SurfaceTransform::Identity,
                       alpha, PresentMode::Immediate,
                       true, None).unwrap()
    };

//    let mut renderer = renderer::Renderer::new(main_queue.clone(), &mut queues, swapchain.format());

    let mut renderer = renderer::Renderer::new(main_queue.clone(), swapchain.format());

    let mut camera = {
        let projection = cgmath::perspective(cgmath::Deg(60.0), default_aspect, 0.01, 100.0);
        Camera::new(projection)
    };

    let (mut floor_object, mut test_object) = {
//        use graphics::object::{ Vertex3D, ObjectInstance };

        let floor_size = 150.0;
        let floor_mesh = renderer.generate_mesh_from_data(vec![
            Vertex3D::from_position(-floor_size, 0.0,-floor_size).flat_shading(true),
            Vertex3D::from_position(-floor_size, 0.0, floor_size).flat_shading(true),
            Vertex3D::from_position( floor_size, 0.0,-floor_size).flat_shading(true),

            Vertex3D::from_position(-floor_size, 0.0, floor_size).flat_shading(true),
            Vertex3D::from_position( floor_size, 0.0, floor_size).flat_shading(true),
            Vertex3D::from_position( floor_size, 0.0,-floor_size).flat_shading(true),
        ]);

        let mut vertices = Vec::new();
        let blend = blend::Blend::from_path("src/data/test.blend");
        loader::blender::load_model_faces(&blend, "Sphere", |face| {
            for i in 0..face.vert_count {
                vertices.push(
                    Vertex3D::from_position(face.vert[i][0], face.vert[i][1], face.vert[i][2])
                        .normal(face.norm[i][0], face.norm[i][1], face.norm[i][2])
                        .uv(face.uv[i][0], face.uv[i][1])
                        .color(1.0, 1.0, 0.0, 1.0)
//                        .flat_shading(true)
                );
            }
        });
        let obj_mesh = renderer.generate_mesh_from_data(vertices);

        let mut floor_obj = ObjectInstance::new(Arc::new(floor_mesh));
        floor_obj.set_pos(0.0, 2.0, 0.0);

        let mut test_obj = ObjectInstance::new(Arc::new(obj_mesh));
        (floor_obj, test_obj)
    };

    let mut prev_sync = Box::new(sync::now(device.clone())) as Box<dyn GpuFuture>;
    let mut running = true;
    let mut recreate_swapchain = false;

    let mut prev_time = time::precise_time_ns();
    let mut t = 0.0f32;

    let mut rot = [0.0f32, 0.0f32, 0.0f32];
    let move_speed = 2.032;
    // Q, W, E, A, S, D
    mod Button {
        pub const Q: usize = 0;
        pub const W: usize = 1;
        pub const E: usize = 2;
        pub const A: usize = 3;
        pub const S: usize = 4;
        pub const D: usize = 5;
    }
    let mut button_states = [false; 6];

    let mut cursor_center = LogicalPosition::new((dimensions[0] / 2) as f64, (dimensions[1] / 2) as f64); // (dimensions[0] / 2, dimensions[1] / 2).into();
    window.set_cursor_position(cursor_center).unwrap();

    let mut t_fps = 0.0;
    let mut frames = 0;
    while running {
        let time = time::precise_time_ns();
        let delta_ns = time - prev_time;
        prev_time = time;
        let delta = (delta_ns as f64 / 1e9f64) as f32;

        frames += 1;
        t_fps += delta;
        if t_fps >= 1.0 {
            println!("FPS: {}", frames);
            t_fps -= 1.0;
            frames = 0;
        }


        /* Update */{
            t += delta;
            test_object.set_pos(t.sin()*3.0, 0.0, 0.0);

            let m = move_speed * delta;

            let right = if button_states[Button::A] { -m }
            else if button_states[Button::D] { m }
            else { 0.0 };

            let forward = if button_states[Button::W] { -m }
            else if button_states[Button::S] { m }
            else { 0.0 };

            if button_states[Button::Q] { rot[1] -= delta * 90.0; }
            if button_states[Button::E] { rot[1] += delta * 90.0; }

            camera.move_by(forward, right, 0.0);
            camera.set_angle(rot);
        }

        floor_object.update();
        test_object.update();

        prev_sync.cleanup_finished();

        if recreate_swapchain {
            let dimensions = if let Some(dimensions) = window.get_inner_size() {
                let dimensions: (u32, u32) = dimensions.to_physical(window.get_hidpi_factor()).into();
                [dimensions.0, dimensions.1]
            } else {
                return;
            };
            let (new_swapchain, new_images) = swapchain.recreate_with_dimension(dimensions).unwrap();
            swapchain = new_swapchain;
            images = new_images;
            recreate_swapchain = false;
        }

        let (image_num, acquire_future) = match swapchain::acquire_next_image(swapchain.clone(), None) {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                recreate_swapchain = true;
                return;
            }
            Err(err) => panic!("{:?}", err)
        };

        renderer.set_view_projection(camera.get_view_projection());

        let future = renderer.render(
            prev_sync.join(acquire_future),
            images[image_num].clone(),
            &vec![&floor_object, &test_object]
        );

        let future = future
            .then_swapchain_present(main_queue.clone(), swapchain.clone(), image_num)
            .then_signal_fence_and_flush();

        match future {
            Ok(future) => {
                // This wait is required when using NVIDIA or running on macOS. See https://github.com/vulkano-rs/vulkano/issues/1247
                future.wait(None).unwrap();
                prev_sync = Box::new(future) as Box<_>;
            }
            Err(FlushError::OutOfDate) => {
                recreate_swapchain = true;
                prev_sync = Box::new(sync::now(device.clone())) as Box<_>;
            }
            Err(e) => {
                println!("{:?}", e);
                return;
            }
        }


        event_loop.poll_events(|event| {
            match event {
                winit::Event::WindowEvent { event, .. } => {
                    match event {
                        winit::WindowEvent::CursorMoved { position, .. } => {
                            if mouse_hold_state {
                                let dx = position.x - cursor_center.x;
                                let dy = position.y - cursor_center.y;
                                window.set_cursor_position(cursor_center).unwrap();
                                rot[1] += dx as f32 * 0.2;
                                rot[0] -= dy as f32 * 0.2;
                            }
                        },

                        winit::WindowEvent::KeyboardInput { input: winit::KeyboardInput { virtual_keycode: Some(key), state, .. }, .. } => {
                            use winit::VirtualKeyCode::*;
                            use winit::ElementState;
                            match key {
                                Q => button_states[Button::Q] = state == ElementState::Pressed,
                                W => button_states[Button::W] = state == ElementState::Pressed,
                                E => button_states[Button::E] = state == ElementState::Pressed,
                                A => button_states[Button::A] = state == ElementState::Pressed,
                                S => button_states[Button::S] = state == ElementState::Pressed,
                                D => button_states[Button::D] = state == ElementState::Pressed,

                                F1 => if state == ElementState::Pressed {
                                    mouse_hold_state = !mouse_hold_state;
                                    window.grab_cursor(mouse_hold_state);
                                    window.hide_cursor(mouse_hold_state);
                                    if mouse_hold_state {
                                        window.set_cursor_position(cursor_center).unwrap();
                                    }
                                },

                                Escape => running = false,
                                _ => ()
                            }
                        }
                        winit::WindowEvent::CloseRequested => {
                            running = false;
                        },
                        winit::WindowEvent::Resized(size) => {
                            recreate_swapchain = true;
                            let aspect = size.width as f32 / size.height as f32;
                            let projection = cgmath::perspective(cgmath::Deg(60.0), aspect, 0.01, 50.0);
                            camera.set_projection(projection);
                        },
                        _ => ()
                    }
                },
                _ => ()
            }
        })

    }

}

/// Remap keys for ease of use with my way of using them

pub struct KeyboardState {
    keys: [bool; 255],
}
impl KeyboardState {

    pub fn new() -> Self { Self {
        keys: [false; 255]
    } }

    /// Set from ElementState
    pub fn key_event(&mut self, keycode: VirtualKeyCode, state: ElementState) {
        if state == ElementState::Pressed {
            self.key_down(keycode);
        } else {
            self.key_up(keycode);
        }
    }

    /// Then Key Pressed
    pub fn key_down(&mut self, keycode: VirtualKeyCode) {
        self.keys[keycode as usize] = true;
    }

    /// Then Key Released
    pub fn key_up(&mut self, keycode: VirtualKeyCode) {
        self.keys[keycode as usize] = false;
    }

    pub fn state_of(&self, keycode: VirtualKeyCode) -> bool { self.keys[keycode as usize] }

}

mod test {
    use winit::{VirtualKeyCode, ElementState};
    use crate::main_processor::KeyboardState;

    #[test] fn test_keyboard_state() {
        let mut state = KeyboardState::new();

        state.key_event(VirtualKeyCode::Q, ElementState::Pressed);
        assert_eq!(state.state_of(VirtualKeyCode::Q), true);
        assert_eq!(state.state_of(VirtualKeyCode::W), false);

        state.key_event(VirtualKeyCode::W, ElementState::Pressed);
        assert_eq!(state.state_of(VirtualKeyCode::Q), true);
        assert_eq!(state.state_of(VirtualKeyCode::W), true);

        state.key_event(VirtualKeyCode::Q, ElementState::Released);
        assert_eq!(state.state_of(VirtualKeyCode::Q), false);
        assert_eq!(state.state_of(VirtualKeyCode::W), true);

        state.key_event(VirtualKeyCode::W, ElementState::Released);
        assert_eq!(state.state_of(VirtualKeyCode::Q), false);
        assert_eq!(state.state_of(VirtualKeyCode::W), false);
    }

}