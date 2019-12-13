
// Settings of window / game and others

use vulkano_win::{ self, VkSurfaceBuild };
use vulkano::{
    instance::{ Instance, PhysicalDevice, ApplicationInfo, Version },
    swapchain::{ PresentMode, Surface },
};

use winit::{EventsLoop, WindowBuilder, Window, dpi::{ LogicalPosition, LogicalSize }};
use std::sync::Arc;
use std::borrow::Cow;
use std::cell::RefCell;

pub const WINDOW_TITLE: &str = "API";
const ENGINE_NAME: &str = "Insomnia";
const ENGINE_VER: Version = Version {
    major: 0,
    minor: 1,
    patch: 0
};

#[derive(Debug, Copy, Clone)]
pub enum WindowMode {
    // Windowed with optional window position
    Windowed(Option<(u32, u32)>),
    // Borderless full screen
    Borderless,
    // Exclusive full screen
    Fullscreen
}

#[derive(Copy, Clone)]
pub struct GameSettings {
    // Window settings
    pub window_size: (u32, u32),
    pub window_mode: WindowMode,

    // Vulkan settings
    pub present_mode: PresentMode,
}

/// Default settings
impl Default for GameSettings {
    fn default() -> Self { Self {
        window_size: (800, 600),
        window_mode: WindowMode::Windowed(None),
        present_mode: PresentMode::Fifo,
    }}
}

/// Editor for settings
impl GameSettings {
}

/// Usage of selected settings
impl GameSettings {
    /// Returns None if window no longer exist and application should close
    pub fn generate_window(&self) -> Result<WindowInfo, String> {
        let mut event_loop = EventsLoop::new();

        // Create instance
        let instance = {

            let app_info = ApplicationInfo {
                application_name: Some(Cow::Borrowed(WINDOW_TITLE)),
                application_version: Some(Version{
                    major: 0,
                    minor: 1,
                    patch: 0
                }),
                engine_name: Some(Cow::Borrowed(ENGINE_NAME)),
                engine_version: Some(ENGINE_VER)
            };

            let extensions = vulkano_win::required_extensions();
            Instance::new(Some(&app_info), &extensions, None).unwrap()
        };


        let surface = {
            let mut wb = winit::WindowBuilder::new().with_title(WINDOW_TITLE);

            match self.window_mode {
                WindowMode::Windowed(pos) => {
//                    if let Some(p) = pos { window.set_position(LogicalPosition::new(p.0 as _, p.1 as _)); }
//                    let size = LogicalSize::from_physical(self.window_size.into(), window.get_hidpi_factor());
//                    window.set_inner_size(size);
                    wb = wb.with_dimensions(self.window_size.into());
                },
                WindowMode::Borderless => {
//                    let dims = window.get_current_monitor().get_dimensions();
                    wb = wb.with_dimensions(LogicalSize::from_physical(event_loop.get_primary_monitor().get_dimensions(), event_loop.get_primary_monitor().get_hidpi_factor()))
                        .with_decorations(false);
                }
                WindowMode::Fullscreen => return Err(String::from("Fullscreen currently not supported")),
            }

            wb.build_vk_surface(&event_loop, instance.clone()).unwrap()
        };
        let window = surface.window();

        // Extra settings for window
        match self.window_mode {
            WindowMode::Borderless => window.set_position((0, 0).into()),
            _ => (),
        }

        // Check if window still exists
        if window.get_inner_size().is_none() { Err(String::from("Window already closed")) }
        else { Ok(WindowInfo {
            event_loop,
            instance,
            surface
        }) }
    }
}

/// Contains backend information
pub struct WindowInfo {
    pub event_loop: EventsLoop,
    pub instance: Arc<Instance>,
    pub surface: Arc<Surface<Window>>,
}
impl WindowInfo {
    pub fn window(&self) -> &Window { self.surface.window() }
}
