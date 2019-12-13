// Point of this class is to process, sort, categorize and send event to listeners and renderer
// Creates window from config


use vulkano_win::{ VkSurfaceBuild };
use vulkano::{
    instance::{ Instance, QueueFamily, PhysicalDevice, MemoryType, ApplicationInfo },
    device::{ Queue, QueuesIter, Device, DeviceExtensions, DeviceOwned },
    swapchain::{ self, Surface, Swapchain, SurfaceTransform, PresentMode, AcquireError, SwapchainAcquireFuture},
    image::{ SwapchainImage },
    sync::{ self, GpuFuture, FlushError },
};
use winit::{EventsLoop, dpi::{LogicalPosition, LogicalSize}, VirtualKeyCode, ElementState, Window, MouseButton};

use std::sync::Arc;
use std::cell::{RefCell, Ref};

use crate::loader;
use crate::graphics::{
    renderer_3d::{
        self,
        mesh::{ Vertex3D, MeshData, ObjectInstance },
    },
    Camera,
};

pub mod settings;
use settings::{
    GameSettings,
    WindowInfo
};
use cgmath::Matrix4;
use std::error::Error;
use crate::graphics::image::sampler_pool::SamplerPool;


/// Main Processor listener
pub trait GameListener {
    fn dimensions_changed(&mut self, frame: &mut Frame, width: u32, height: u32) {}

    fn update(&mut self, delta: f32, frame: &mut Frame, future: Box<dyn GpuFuture>) -> Box<dyn GpuFuture>;

    fn key_pressed(&mut self, frame: &mut Frame, keycode: VirtualKeyCode) { }
    fn key_released(&mut self, frame: &mut Frame, keycode: VirtualKeyCode) { }

    fn mouse_wheel(&mut self, frame: &mut Frame, x: f32, y: f32) { }
}

/// Requests on to do to some parts of backend from window user
/// Ex: Window settings, ...
pub enum FrameRequest {
    ExitApplication,
    HoldCursor(Option<bool>), // If none => switch state
}

/// Holds state of application
/// Holds and/or executes requests from FrameRequest
/// Like is mouse holding and things like that
#[derive(Debug)]
struct ApplicationState {
    running: bool,
    hold_cursor: bool,
}
impl Default for ApplicationState {
    fn default() -> Self { Self {
        running: true,
        hold_cursor: false,
    } }
}
impl ApplicationState {
    fn accept(&mut self, frame: Frame) {
        for r in frame.requests.iter() {
            use FrameRequest::*;
            match r {
                ExitApplication => self.running = false,
                HoldCursor(flag) => self.hold_cursor = flag.unwrap_or(!self.hold_cursor),
            }
        }
    }
}

/// Frame info (draw geometry, clicked buttons)
pub struct Frame<'v> {
    pub queue: Arc<Queue>, // Main Queue
    pub image: Arc<SwapchainImage<Window>>, // Output image, None in init frame
    pub sampler_pool: &'v mut SamplerPool, // Samplet pool

    requests: Vec<FrameRequest>,

    keyboard: &'v mut KeyboardState,
    mouse: &'v mut MouseState,
}
/// Init and interaction with IO
impl <'v> Frame<'v> {
    pub fn cursor_pos(&self) -> [f32; 2] { self.mouse.position }
    pub fn cursor_spd(&self) -> [f32; 2] { self.mouse.speed }
    pub fn cursor_btn(&self, button: winit::MouseButton) -> bool { self.mouse.state_of(button) }

    pub fn key_state(&self, keycode: VirtualKeyCode) -> bool { self.keyboard.state_of(keycode) }
}
/// Frame requests
impl <'v> Frame<'v> {
    pub fn request(&mut self, request: FrameRequest) { self.requests.push(request) }
}

/// Start Listener in one function
pub fn start_with_settings_and_listener<F>(
    settings: GameSettings,
    mut init_listener: F) -> Result<(), String>
    where F: FnMut(&mut Frame) -> Box<dyn GameListener>
{
    let mut application_state = ApplicationState::default();
    let mut window = settings.generate_window()?;
    let mut swapchain = SwapchainConfig::create(&window)?;
    let mut keyboard = KeyboardState::new();
    let mut mouse = MouseState::new();
    let mut sampler_pool = SamplerPool::new(swapchain.device());

    macro_rules! new_frame {
        ($img_idx:expr) => {
            Frame {
                queue: swapchain.main_queue.clone(),
                image: swapchain.images[$img_idx].clone(),
                sampler_pool: &mut sampler_pool,
                keyboard: &mut keyboard,
                mouse: &mut mouse,
                requests: vec![],
            }
        };
        () => { new_frame!(0) };
    }

    let mut listener = {
        let mut init_frame = new_frame!();
        let mut l = init_listener(&mut init_frame);
        let dims = swapchain.swapchain.dimensions();
        l.dimensions_changed(&mut init_frame, dims[0], dims[1]);
        application_state.accept(init_frame);
        l
    };

    let mut last_sync = Box::new(sync::now(swapchain.device())) as Box<dyn GpuFuture>;
    let mut prev_time = time::precise_time_ns();
    while application_state.running {
        let time = time::precise_time_ns();
        let delta = (time - prev_time) as f32 / 1e9;
        prev_time = time;
        println!("FPS: {}", 1.0 / delta);

        // Do swapchain maintenance
        swapchain.update_if_required(&window);

        // Do acquire swapchain image
        let (image_num, acquire_future) = match swapchain.acquire() {
            Ok(r) => r,
            Err(AcquireError::OutOfDate) => {
                swapchain.recreate();
                continue;
            },
            Err(e) => return Err(format!("{:?}", e)),
        };


        let mut frame = new_frame!(image_num);

        // Do update and drawing using future to receive next GpuFuture
        let future = listener.update(delta, &mut frame, Box::new(last_sync.join(acquire_future)));

        // Present future to swapchain
        match future.then_swapchain_present(
            swapchain.main_queue.clone(),
            swapchain.swapchain.clone(),
            image_num
        ).then_signal_fence_and_flush() {
            Ok(future) => {
                // This wait is required when using NVIDIA or running on macOS. See https://github.com/vulkano-rs/vulkano/issues/1247
                future.wait(None).unwrap();
                last_sync = Box::new(future) as Box<_>;
            }
            Err(FlushError::OutOfDate) => {
                swapchain.recreate();
                last_sync = Box::new(sync::now(swapchain.device().clone())) as Box<_>;
            }
            Err(e) => return Err(format!("{:?}", e)),
        }

        // Process Events
        window.event_loop.poll_events(|e| {
            use winit::Event;
            match e {
                Event::DeviceEvent { event, .. } => {
                    use winit::DeviceEvent;
                    match event {
                        DeviceEvent::MouseMotion { delta } => { frame.mouse.move_event(delta) },
                        _ => (),
                    }
                },
                Event::WindowEvent { event, .. } => {
                    use winit::WindowEvent;
                    match event {
                        WindowEvent::Resized(s) => {
                            listener.dimensions_changed(&mut frame, s.width as u32, s.height as u32);
                            swapchain.recreate();
                        },
                        WindowEvent::CloseRequested => application_state.running = false,
                        WindowEvent::CursorMoved { position, .. } => frame.mouse.pos_event(position),
                        WindowEvent::MouseWheel { delta, .. } => {
                            match delta {
                                winit::MouseScrollDelta::LineDelta(x, y) => {
                                    if x != 0.0 || y != 0.0 { listener.mouse_wheel(&mut frame, x, y) }
                                },
                                winit::MouseScrollDelta::PixelDelta(p) => {
                                    panic!("Currently not supported");
                                },
                            }
                        },
                        WindowEvent::MouseInput { button, state, .. } => frame.mouse.key_event(button, state),
                        WindowEvent::KeyboardInput { input: winit::KeyboardInput { virtual_keycode: Some(keycode), state, .. }, .. } => {
                            frame.keyboard.key_event(keycode, state);
                            match state {
                                ElementState::Pressed => listener.key_pressed(&mut frame, keycode),
                                ElementState::Released => listener.key_released(&mut frame, keycode),
                            }
                        }
                        _ => (),
                    }
                },
                _ => (),
            }
        });

        // Also move frame to release mouse and keyboard fields for modding
        application_state.accept(frame);

        // Update mouse position after events but before setting forced, centred position
        mouse.update(delta);

        // Apply application state
        {
            let window = window.window();
            window.hide_cursor(application_state.hold_cursor);
            if application_state.hold_cursor {
                let size = window.get_outer_size().unwrap();
                let pos = LogicalPosition::new(size.width/2.0, size.height/2.0);
                window.set_cursor_position(pos).unwrap();
            }
        }
    }

    Ok(())
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

    pub fn create(backend: &WindowInfo) -> Result<Self, String> {

        // Dims
        let mut dimensions = if let Some(dimensions) = backend.window().get_inner_size() {
            let dimensions: (u32, u32) = dimensions.to_physical(backend.window().get_hidpi_factor()).into();
            [dimensions.0, dimensions.1]
        } else {
            return Err(String::from("Window already closed (swapchain)"));
        };

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

            let usage = caps.supported_usage_flags;
            let format = caps.supported_formats[0].0;
            let alpha = caps.supported_composite_alpha.iter().next().unwrap();

            let image_count = caps.min_image_count;
            match Swapchain::new(device.clone(), backend.surface.clone(),
                                 image_count, format, dimensions, 1,
                                 usage, &main_queue, SurfaceTransform::Identity,
                                 alpha, PresentMode::Immediate,
                                 true, None) {
                Ok(s) => s,
                Err(e) => return Err(format!("{:?}", e)),
            }
        };

        Ok(Self {
            swapchain,
            images,

            main_queue,
            queues,

            recreate: false
        })
    }

    pub fn device(&self) -> Arc<Device> {
        self.swapchain.device().clone()
    }

    pub fn recreate(&mut self) { self.recreate = true; }

    pub fn update_if_required(&mut self, backend: &WindowInfo) {
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

/// Current state of keyboard keys being pressed
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

/// Save mouse state, position, button states and move delta
#[derive(Debug, PartialEq)]
pub struct MouseState {
    // Buttons
    buttons: [bool; 3],
    other_buttons: Vec<u8>, // Other currently down buttons with u8 ID

    // Cursor Position and Move Speed
    position: [f32; 2], // Current LogicalPosition
    speed_updated: u8, // Then speed is updated current frame, 2 just now, 1 last frame, 0 old news
    speed: [f32; 2], // Cursor move speed per sec (adjusted to delta)

    // Scroll Wheel is not buffered and reported as it comes
}
impl Default for MouseState {
    fn default() -> Self { Self {
        buttons: [false; 3],
        other_buttons: Vec::new(),

        position: [0.0; 2],
        speed_updated: 2,
        speed: [0.0; 2],
    }}
}
impl MouseState {

    pub fn new() -> Self { MouseState::default() }

    /// Update cursor speed
    pub fn update(&mut self, delta: f32) {
        if self.speed_updated <= 2 {
            // 0, just set and will not be used until 1
            // 1 use
            // 2 old news
            if self.speed_updated == 0 {
                self.speed = [self.speed[0] * delta, self.speed[1] * delta];
            } else if self.speed_updated == 2 {
                self.speed = [0.0; 2];
            }
            self.speed_updated += 1;
        }
    }

    /// Sets new current position
    pub fn pos_event(&mut self, pos: LogicalPosition) {
        self.position = [pos.x as f32, pos.y as f32];
    }

    pub fn move_event(&mut self, mouse_delta: (f64, f64)) {
        self.speed_updated = 0;
        self.speed = [mouse_delta.0 as f32, mouse_delta.1 as f32];
    }

    /// New Mouse Button Pressed
    pub fn key_event(&mut self, button: MouseButton, state: ElementState) {
        match button {
            MouseButton::Left => self.buttons[0] = state == ElementState::Pressed,
            MouseButton::Right => self.buttons[1] = state == ElementState::Pressed,
            MouseButton::Middle => self.buttons[2] = state == ElementState::Pressed,
            MouseButton::Other(id) => {
                if state == ElementState::Pressed {
                    if !self.other_buttons.contains(&id) { self.other_buttons.push(id) }
                } else {
                    self.other_buttons.retain(|f| *f != id);
                }
            },
        };
    }

    pub fn state_of(&self, button: MouseButton) -> bool {
        match button {
            MouseButton::Left => self.buttons[0],
            MouseButton::Right => self.buttons[1],
            MouseButton::Middle => self.buttons[2],
            MouseButton::Other(id) => self.other_buttons.contains(&id),
        }
    }

}

mod test {
    use winit::{ElementState};

    #[test] fn test_keyboard_state() {
        use crate::main_processor::KeyboardState;
        use winit::{VirtualKeyCode};

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

    #[test] fn test_mouse_state() {
        use crate::main_processor::MouseState;
        use winit::{ MouseButton, dpi::LogicalPosition };

        let mut state = MouseState::new();

        // Test Default Buttons
        {
            fn with_state(buttons: [bool; 3]) -> MouseState { MouseState {
                buttons, ..MouseState::default()
            } }

            state.key_event(MouseButton::Right, ElementState::Pressed);
            assert_eq!(state, with_state([false, true, false]));

            state.key_event(MouseButton::Left, ElementState::Pressed);
            assert_eq!(state, with_state([true, true, false]));

            state.key_event(MouseButton::Right, ElementState::Released);
            assert_eq!(state, with_state([true, false, false]));

            state.key_event(MouseButton::Right, ElementState::Released);
            assert_eq!(state, with_state([true, false, false]));

            state.key_event(MouseButton::Left, ElementState::Released);
            assert_eq!(state, with_state([false, false, false]));
        }

        // Test Extra Buttons
        {
            fn with_others(other_buttons: Vec<u8>) -> MouseState { MouseState {
                other_buttons, ..MouseState::default()
            } }

            state.key_event(MouseButton::Other(0), ElementState::Pressed);
            assert_eq!(state, with_others(vec![0]));

            state.key_event(MouseButton::Other(1), ElementState::Pressed);
            assert_eq!(state, with_others(vec![0, 1]));

            state.key_event(MouseButton::Other(1), ElementState::Pressed);
            assert_eq!(state, with_others(vec![0, 1]));

            state.key_event(MouseButton::Other(0), ElementState::Released);
            assert_eq!(state, with_others(vec![1]));

            state.key_event(MouseButton::Other(0), ElementState::Released);
            assert_eq!(state, with_others(vec![1]));

            state.key_event(MouseButton::Other(1), ElementState::Released);
            assert_eq!(state, with_others(vec![]));
        }

        // Test position update
        {

//            fn with_cursor(x: f32, y: f32, px: f32, py: f32, dx: f32, dy: f32, position_dirty: bool) -> MouseState { MouseState {
//                position: [x, y], prev_position: [px, py], speed: [dx, dy], position_dirty, .. MouseState::default()
//            }}
//
//            // Quick cycle
//            state.pos_event(LogicalPosition::new(10.0, 10.0));
//            assert_eq!(state, with_cursor(10.0, 10.0, 0.0, 0.0, 0.0, 0.0, true));
//
//            state.update(0.1);
//            assert_eq!(state, with_cursor(10.0, 10.0, 10.0, 10.0, 1.0, 1.0, true));
//
//            state.update(0.1);
//            assert_eq!(state, with_cursor(10.0, 10.0, 10.0, 10.0, 0.0, 0.0, false));
//
//            // Fast Cycle
//            state.pos_event(LogicalPosition::new(0.0, 0.0));
//            assert_eq!(state, with_cursor(0.0, 0.0, 10.0, 10.0, 0.0, 0.0, true));
//
//            state.update(0.1);
//            state.pos_event(LogicalPosition::new(10.0, 0.0));
//            assert_eq!(state, with_cursor(10.0, 0.0, 0.0, 0.0, -1.0, -1.0, true));
//
//            state.update(0.1);
//            state.pos_event(LogicalPosition::new(20.0, 0.0));
//            assert_eq!(state, with_cursor(20.0, 0.0, 10.0, 0.0, 1.0, 0.0, true));
//
//            state.update(0.1);
//            assert_eq!(state, with_cursor(20.0, 0.0, 20.0, 0.0, 1.0, 0.0, true));
//
//            state.update(0.1);
//            assert_eq!(state, with_cursor(20.0, 0.0, 20.0, 0.0, 0.0, 0.0, false));
        }


    }

}