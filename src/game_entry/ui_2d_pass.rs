
use rayon::prelude::*;
use std::{
    io::Cursor,
    iter::Iterator,
    sync::Arc,
};
use gfx_lib::{
    main_processor::Frame,
    graphics::{
        image::{
            ImageContent,
            sampler_pool::SamplerParams,
        },
        renderer_2d::{
            Renderer2D,
            cache::{ Render2DCache, Render2DCacheError },
        },
        object::ScreenInstance,
    }
};

use vulkano::{
    format::Format,
    image::{ ImageUsage, AttachmentImage },
    sync::GpuFuture
};
use vulkano::image::{ImageViewAccess, ImageAccess};

// Game specific 2D Render for UI into image

pub struct UI2DPass {
    image: ImageContent,
    cache: Render2DCache,
    pub output: Arc<AttachmentImage>,
}
impl UI2DPass {

    pub fn new(frame: &mut Frame) -> Self {
        let image = ImageContent::new_with_bytes(
            frame.queue.clone(),
            frame.sampler_pool.with_params(SamplerParams::simple_repeat()),
            Cursor::new(include_bytes!("../data/icon512.png").to_vec()),
            Format::R8G8B8A8Srgb,
        );

        let res = 1024 * 1;
        let renderer_2d_att = AttachmentImage::with_usage(
            frame.queue.device().clone(), [res, res], Format::R8G8B8A8Snorm,
            ImageUsage {
                color_attachment: true,
                sampled: true,
                .. ImageUsage::none()
            }
        ).unwrap();

        let count = 2;
        let s = 1.0 / count as f32;

        let mut cache = Render2DCache::new(frame, count * count);
        cache.set_image((image.get_image(), image.get_sampler()));

        cache.append({
            let s = 0.8;
            let s2 = s / 2.0;
            let mut instance = ScreenInstance::new();
            instance.set_transform(s2, s2, s, s, cgmath::Rad(0.0));
            instance
        }).unwrap();

//        for x in 0 .. count { for y in 0 .. count {
//            let mut instance = ScreenInstance::new();
//            instance.set_transform(
//                (x as f32 + 0.5) * s, (y as f32 + 0.5) * s,
//                s, s, cgmath::Rad(0.0)
//            );
//            let xp = x as f32 / count as f32;
//            let yp = y as f32 / count as f32;
//            instance.set_color(1.0 - x as f32, 1.0 - y as f32, 1.0, 1.0);
//            cache.append(instance).unwrap();
//        } }

        Self {
            image,
            cache,
            output: renderer_2d_att,
        }
    }


    // Test just pushing instances in renderpass
    fn render_req(&mut self, renderer: &mut Renderer2D, future: Box<dyn GpuFuture + Send + Sync>) -> Box<dyn GpuFuture + Send + Sync> {
        renderer.begin(self.output.clone());
        let mut pass = renderer.start_image_content(&mut self.image);

        let count = 4;
        let w_step = 1.0 / count as f32;
        let h_step = 1.0 / count as f32;
        let w_size = w_step * 0.7;
        let h_size = w_step * 0.7;

//        let time = self.time;
        let mut prep_instance = |idx: usize| {
            let x = idx % count;
            let y = idx / count;
            let angle = 0.0; // time * idx as f32 / (count * count) as f32 * 360.0;
            let mut inst = ScreenInstance::new();
            inst.set_transform(
                w_step * (x as f32 + 0.5),
                h_step * (y as f32 + 0.5),
                w_size, h_size, cgmath::Deg(angle));
            inst.set_color(1.0, 1.0, 1.0, 1.0);
            inst
        };

        let time1 = time::precise_time_ns();

        let async_2d = false;
        let arr: Vec<_> = if async_2d {
            (0..(count * count) as usize).into_par_iter().map(prep_instance).collect()
        } else {
            (0..(count * count) as usize).into_iter().map(prep_instance).collect()
        };
        pass.render_instances_vec(arr);

        pass.end_call();
        renderer.end(future)
    }

    // Test drawing already prepared cache
    fn render_cache<I>(&mut self, renderer: &mut Renderer2D, output: I, future: Box<dyn GpuFuture + Send + Sync>) -> Box<dyn GpuFuture + Send + Sync>
        where I: ImageAccess + ImageViewAccess + Clone + Send + Sync + 'static
    {
        renderer.begin(output);

        renderer.render_cache(&mut self.cache);

        renderer.end(future)
    }

    pub fn render_into<I>(&mut self, renderer: &mut Renderer2D, output: I, future: Box<dyn GpuFuture + Send + Sync>) -> Box<dyn GpuFuture + Send + Sync>
        where I: ImageAccess + ImageViewAccess + Clone + Send + Sync + 'static
    {
        if !self.image.is_ready() { return future; }
        self.render_cache(renderer, output, future)
    }

    pub fn render(&mut self, renderer: &mut Renderer2D, future: Box<dyn GpuFuture + Send + Sync>) -> Box<dyn GpuFuture + Send + Sync> {
        if !self.image.is_ready() { return future; }
        self.render_cache(renderer, self.output.clone(), future)
    }
}