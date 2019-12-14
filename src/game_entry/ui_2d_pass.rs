
use rayon::prelude::*;
use std::{
    io::Cursor,
    iter::Iterator,
    sync::Arc,
};
use crate::{
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

// Game specific 2D Render for UI into image

pub struct UI2DPass {
    image: ImageContent,
    cache: Render2DCache,
    pub output: Arc<AttachmentImage>,
}
impl UI2DPass {
    pub fn new(frame: &mut Frame) -> Self {
        let image = ImageContent::from_bytes(
            frame.queue.clone(),
            frame.sampler_pool.with_params(SamplerParams::simple_repeat()),
            Cursor::new(include_bytes!("../data/icon512.png").to_vec()),
            Format::R8G8B8A8Srgb,
        );

        let res = 1024 * 8;
        let renderer_2d_att = AttachmentImage::with_usage(
            frame.queue.device().clone(), [res, res], Format::R8G8B8A8Snorm,
            ImageUsage {
                color_attachment: true,
                sampled: true,
                .. ImageUsage::none()
            }
        ).unwrap();

        let count = 1000;
        let s = 1.0 / count as f32;

        let mut cache = Render2DCache::new(frame, count * count);
        cache.set_image((image.get_image(), image.get_sampler()));

        for x in 0 .. count { for y in 0 .. count {
            let mut instance = ScreenInstance::new();
            instance.set_transform(
                x as f32 * s, y as f32 * s,
                s, s, cgmath::Rad(0.0)
            );
            let xp = x as f32 / count as f32;
            let yp = y as f32 / count as f32;
            instance.set_color(xp, yp, 1.0-yp, 1.0);
            cache.append(instance).unwrap();
        } }

        Self {
            image,
            cache,
            output: renderer_2d_att,
        }
    }


    // Test just pushing instances in renderpass
    fn render_req(&mut self, renderer: &mut Renderer2D, future: Box<dyn GpuFuture>) -> Box<dyn GpuFuture> {
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

        pass.end_pass();
        renderer.end(future)
    }

    // Test drawing already prepared cache
    fn render_cache(&mut self, renderer: &mut Renderer2D, future: Box<dyn GpuFuture>) -> Box<dyn GpuFuture> {
        renderer.begin(self.output.clone());

        renderer.render_cache(&mut self.cache);

        renderer.end(future)
    }

    pub fn render(&mut self, renderer: &mut Renderer2D, future: Box<dyn GpuFuture>) -> Box<dyn GpuFuture> {
        if !self.image.is_ready() { return future; }

//        self.render_req(renderer, future)
        self.render_cache(renderer, future)
    }
}