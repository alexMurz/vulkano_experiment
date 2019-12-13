
#![feature(float_to_from_bytes)]

extern crate blend;
extern crate rayon;

use std::cell::{RefCell, Cell};
use std::iter;
use std::sync::Arc;
use std::time::Instant;

use cgmath::{Matrix3, Matrix4, Point3, Rad, SquareMatrix, vec3, Vector3};
use vulkano::buffer::{BufferAccess, BufferUsage, CpuAccessibleBuffer};
use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState};
use vulkano::descriptor::descriptor_set::{FixedSizeDescriptorSet, PersistentDescriptorSet};
use vulkano::descriptor::DescriptorSet;
use vulkano::device::{Device, DeviceExtensions, Queue};
use vulkano::format::Format;
use vulkano::framebuffer::{Framebuffer, RenderPassAbstract, Subpass};
use vulkano::image::{ImageUsage, SwapchainImage};
use vulkano::image::attachment::AttachmentImage;
use vulkano::instance::Instance;
use vulkano::instance::PhysicalDevice;
use vulkano::pipeline::{GraphicsPipeline, GraphicsPipelineAbstract};
use vulkano::pipeline::vertex::TwoBuffersDefinition;
use vulkano::pipeline::viewport::Viewport;
use vulkano::swapchain::{AcquireError, ColorSpace, PresentMode, Surface, SurfaceTransform, Swapchain, SwapchainCreationError};
use vulkano::swapchain;
use vulkano::sync;
use vulkano::sync::GpuFuture;
use vulkano_win::{create_vk_surface, VkSurfaceBuild};
use winit::{ElementState, MouseCursor, VirtualKeyCode, Window};
use winit::dpi::LogicalPosition;

use graphics::renderer_3d;

use crate::graphics::Camera;
use std::sync::atomic::AtomicBool;

mod utils;

mod graphics;

mod game_entry;
mod loader;
mod main_processor;

fn main() {
    start();

//use std::sync::{Arc, Mutex};
//    use std::thread;
//    use std::time::Duration;
//
//    let num = Arc::new(Mutex::new(5));
//    // allow `num` to be shared across threads (Arc) and modified
//    // (Mutex) safely without a data race.
//
//    let num_clone = num.clone();
//    // create a cloned reference before moving `num` into the thread.
//
//    thread::spawn(move || {
//        loop {
//            *num.lock().unwrap() += 1;
//            // modify the number.
//            thread::sleep(Duration::from_millis(1_500));
//        }
//    });
//
//    output(num_clone);
//
//
//    fn output(num: Arc<Mutex<i32>>) {
//        loop {
//            println!("{:?}", *num.lock().unwrap());
//            // read the number.
//            //  - lock(): obtains a mutable reference; may fail,
//            //    thus return a Result
//            //  - unwrap(): ignore the error and get the real
//            //    reference / cause panic on error.
//            thread::sleep(Duration::from_secs(1));
//        }
//    }
}

fn start() {
    use main_processor::{
        settings::{ GameSettings, WindowMode }
    };

    let settings = GameSettings {
//        window_mode: WindowMode::Borderless,
        .. GameSettings::default()
    };

    match main_processor::start_with_settings_and_listener(
        settings,
        |frame| { Box::new(game_entry::GameEntry::new(frame)) }
    ) {
        Ok(_) => (),
        Err(err) => println!("Finished with error: {}", err),
    }
}


mod test {

    #[test] fn test_arc() {
        use std::sync::Arc;

        let arc1 = Arc::new(5.0);
        let arc2 = arc1.clone();
        assert_eq!(arc1, arc2)
    }

    #[test] fn test_arc_vec() {
        use std::sync::Arc;

        let a = Arc::new(1.0);
        let b = Arc::new(2.0);
        let c = Arc::new(3.0);

        let mut vec = vec![a.clone(), b.clone(), c.clone()];
        vec.retain(|x| *x != b);
        assert_eq!(vec, vec![a, c]);
    }

    #[test] fn test_address_cmp() {
        use std::sync::Arc;
        use std::cell::RefCell;

        #[derive(Debug)]
        struct Data {}
        // Compare by address
        impl PartialEq for Data {
            fn eq(&self, other: &Data) -> bool { self as *const _ == other as *const _ }
        }

        let a = Arc::new(RefCell::new(Data{}));
        let b = a.clone();
        let c = Arc::new(RefCell::new(Data{}));

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    #[test] fn test_obj_holder() {
        struct Holder<T> {
            pub obj: T
        }
        impl <T> Holder<T> {
            fn new(obj: T) -> Self { Self { obj } }
        }
        impl <T> std::ops::Deref for Holder<T> {
            type Target = T;
            fn deref(&self) -> &Self::Target { &self.obj }
        }
        impl <T> std::ops::DerefMut for Holder<T> {
            fn deref_mut(&mut self) -> &mut Self::Target { &mut self.obj }
        }

        let mut obj = Holder::new(5.0);
        *obj *= 3.0;
        assert_eq!(15.0, *obj);
    }
}

















