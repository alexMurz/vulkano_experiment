

use vulkano::buffer::{BufferUsage, CpuAccessibleBuffer, BufferAccess};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::{PersistentDescriptorSet, FixedSizeDescriptorSet};
use vulkano::device::{Device, DeviceExtensions, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, FramebufferAbstract, Subpass, RenderPassAbstract};
use vulkano::image::{SwapchainImage, ImageUsage};
use vulkano::image::attachment::AttachmentImage;
use vulkano::instance::Instance;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::swapchain::{AcquireError, PresentMode, SurfaceTransform, Swapchain, SwapchainCreationError, ColorSpace, Surface};
use vulkano::swapchain;
use vulkano::sync::GpuFuture;
use vulkano::sync;

use vulkano_win::VkSurfaceBuild;

use winit::{Window, ElementState, VirtualKeyCode};

use cgmath::{Matrix3, Matrix4, Point3, Vector3, Rad, SquareMatrix};

use std::iter;
use std::sync::Arc;
use std::time::Instant;
use vulkano::descriptor::DescriptorSet;

mod graphics;
use graphics::{ old_renderer, renderer };
use crate::graphics::Camera;
use crate::graphics::old_renderer::RendererState;
use graphics::renderer::Renderer;

extern crate blend;
use blend::{Blend};
use std::any::Any;

mod scene;
mod loader;

fn main() {
    start();
//    test_blend();
}

fn test_blend() {
    let blend = Blend::from_path("src/data/test.blend");
    loader::blender::load_model_faces(&blend, "myCubeMesh", |face| {
        println!("Face");
        println!(" -- {:?}", face);
    });
//    for obj in blend.get_by_code(*b"OB") {
//        if obj.is_valid("data") && obj.get("data").code() == *b"ME\0\0" {
//            let name = &obj.get("id").get_string("name")[2..];
//            let loc = obj.get_f32_vec("loc");
//            println!("\"{}\" at {:?}, {:?}", name, loc,  obj.get("data").code());
//        }
//    }
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

    let surface = {
        let mut wb = winit::WindowBuilder::new()
            .with_title("Title")
            .with_dimensions((800.0, 600.0).into());
        wb.build_vk_surface(&event_loop, instance.clone()).unwrap()
    };
    let window = surface.window();

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
        physical, physical.supported_features(), &device_ext, [(queue_family, 0.5)].iter().cloned()
    ).unwrap();

    let main_queue = queues.next().unwrap();

    let (mut swapchain, mut images) = {
        let caps = surface.capabilities(physical).unwrap();
        let usage = caps.supported_usage_flags;
        let format = caps.supported_formats[0].0;
        let alpha = caps.supported_composite_alpha.iter().next().unwrap();

        println!("Formats: {:?}", caps.present_modes);

        let image_count = caps.min_image_count;

        Swapchain::new(device.clone(), surface.clone(), image_count, format, dimensions, 1,
                       usage, &main_queue, SurfaceTransform::Identity, alpha, PresentMode::Fifo, true, None).unwrap()
    };

//    let mut renderer = renderer::Renderer::new(main_queue.clone(), &mut queues, swapchain.format());

    let mut renderer = renderer::Renderer::new(main_queue.clone(), swapchain.format());

    let mut camera = {
        let projection = cgmath::perspective(cgmath::Deg(60.0), 1.0, 0.01, 100.0);
        Camera::new(projection)
    };
    let (mut floor_object, mut test_object) = {
        use graphics::object::{ MeshData, Vertex3D, ObjectInstance };

        let floor_size = 5.0;
        let floor_mesh = renderer.generate_mesh_from_data(vec![
            Vertex3D::from_position(-floor_size, 0.0, -floor_size),
            Vertex3D::from_position(-floor_size, 0.0,  floor_size),
            Vertex3D::from_position( floor_size, 0.0, -floor_size),

            Vertex3D::from_position( floor_size, 0.0, -floor_size),
            Vertex3D::from_position(-floor_size, 0.0,  floor_size),
            Vertex3D::from_position( floor_size, 0.0,  floor_size),
        ]);
        let obj_mesh = renderer.generate_mesh_from_data(vec![
            Vertex3D::from_position(-0.5, -0.5, 0.0),
            Vertex3D::from_position(-0.5,  0.5, 0.0).color(0.0, 1.0, 0.0, 1.0),
            Vertex3D::from_position( 0.5, -0.5, 0.0),
        ]);
        (
            ObjectInstance {
                mesh_data: Arc::new(floor_mesh),
                pos: [0.0, 2.0, 0.0],
                scl: [1.0, 1.0, 1.0],
                rot: [0.0, 0.0, 0.0]
            },
            ObjectInstance {
                mesh_data: Arc::new(obj_mesh),
                pos: [0.0, 0.0, 0.0],
                scl: [1.0, 1.0, 1.0],
                rot: [0.0, 0.0, 0.0]
            }
        )
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

    while running {
        let time = time::precise_time_ns();
        let delta_ns = time - prev_time;
        prev_time = time;
        let delta = (delta_ns as f64 / 1e9f64) as f32;

        /* Update */{
            t += delta;
            test_object.set_pos(t.sin(), 0.0, 0.0);

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

//        scene.act(delta);


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

//        let future = renderer.render(
//            prev_sync.join(acquire_future),
//            images[image_num].clone(),
//            |state| {
//                match state {
//                    RendererState::Drawing(mut executor) => scene.render(&mut executor),
//                    RendererState::Shadow(mut executor) => scene.render_shadows(&mut executor),
//                    _ => ()
//                };
//            },
//        );

        renderer.set_view_projection(camera.get_view_projection());
        let future = renderer.render(
            prev_sync.join(acquire_future),
            images[image_num].clone(),
            vec![&floor_object, &test_object].as_ref()
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
            Err(sync::FlushError::OutOfDate) => {
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
                        winit::WindowEvent::KeyboardInput { input: winit::KeyboardInput { virtual_keycode: Some(key), state, .. }, .. } => {
                            use winit::VirtualKeyCode::*;
                            match key {
                                Q => button_states[Button::Q] = state == ElementState::Pressed,
                                W => button_states[Button::W] = state == ElementState::Pressed,
                                E => button_states[Button::E] = state == ElementState::Pressed,
                                A => button_states[Button::A] = state == ElementState::Pressed,
                                S => button_states[Button::S] = state == ElementState::Pressed,
                                D => button_states[Button::D] = state == ElementState::Pressed,

                                Escape => running = false,
                                _ => ()
                            }
                        }
                        winit::WindowEvent::CloseRequested => running = false,
                        winit::WindowEvent::Resized(size) => {
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



















