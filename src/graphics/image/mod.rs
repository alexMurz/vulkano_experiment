
use vulkano::{
    device::Queue,
    format::Format,
    descriptor::{
        DescriptorSet, PipelineLayoutAbstract,
        descriptor_set::PersistentDescriptorSet
    },
    sampler::{ Sampler, SamplerAddressMode, MipmapMode, Filter },
    image::{ ImmutableImage, ImageDimensions, ImageViewAccess, ImageAccess },
};
use std::{
    ops::Deref,
    io::Cursor,
    sync::{ Arc, Mutex },
    error, fmt,
};
use vulkano::pipeline::GraphicsPipelineAbstract;

mod loader;
pub mod sampler_pool;

#[derive(Debug)]
pub enum AccessError {
    NotReadyError,
    Panic
}
impl error::Error for AccessError {
    fn description(&self) -> &str {
        match self {
            AccessError::NotReadyError => "ImageContent not yet loaded. Use image.is_ready()",
            AccessError::Panic => "Panic!",
        }
    }
}
impl fmt::Display for AccessError {
    fn fmt(&self, fmt: &mut fmt::Formatter) -> Result<(), fmt::Error> {
        write!(fmt, "{}", (self as &dyn error::Error).description())
    }
}

pub trait ImageContentAbstract {
    fn get_uniform(&mut self, pipeline: &Arc<dyn GraphicsPipelineAbstract + Send + Sync>, set_id: usize) -> Arc<dyn DescriptorSet + Send + Sync>;
}

/// Contains asynchronously loaded Immutable image and uniform associated for it
pub struct ImageContent {
    sampler: Arc<Sampler>,
    image: Arc<ImmutableImage<Format>>,
    ready: Arc<Mutex<bool>>,

    // If None => Recreate uniform
    uniform: Option<Arc<dyn DescriptorSet + Send + Sync>>,
}
/// Clone image but leave uniform, used to reuse image in multiple DescriptorSets
impl Clone for ImageContent {
    fn clone(&self) -> Self { Self {
        sampler: self.sampler.clone(),
        image: self.image.clone(),
        ready: self.ready.clone(),
        uniform: None,
    }}
}
/// Create new instance, check and access image
impl ImageContent {
    pub fn from_bytes(queue: Arc<Queue>, sampler: Arc<Sampler>, bytes: Cursor<Vec<u8>>, format: Format) -> Self {
        let (image, future) = loader::load_png_image_from_bytes(queue, bytes, format);

        let ready = Arc::new(Mutex::new(false));
        let t = ready.clone();
        rayon::spawn(move || {
            future.flush().unwrap();
            *t.lock().unwrap() = true;
        });

        Self {
            sampler,
            image,
            ready,
            uniform: None,
        }
    }
    pub fn is_ready(&self) -> bool { *self.ready.lock().unwrap() }
    pub fn recreate_uniform(&mut self) { self.uniform = None; }

    /// Return sampler
    pub fn get_sampler(&self) -> Arc<Sampler> { self.sampler.clone() }
    /// Return image with no check if it is ready to use
    pub fn get_image(&self) -> Arc<dyn ImageViewAccess + Send + Sync> { self.image.clone() }

    fn access(&self) -> Result<Arc<dyn ImageViewAccess + Send + Sync>, AccessError> {
        if !self.is_ready() {
            Err(AccessError::NotReadyError)
        } else {
            Ok(self.image.clone())
        }
    }
}
/// Content Access Interface for ImageContent
/// Access to this images uniform
impl ImageContentAbstract for ImageContent {

    fn get_uniform(&mut self, pipeline: &Arc<dyn GraphicsPipelineAbstract + Send + Sync>, set_id: usize) -> Arc<dyn DescriptorSet + Send + Sync> {
//        assert_eq!(*self.ready.lock().unwrap(), true, "Texture not yet ready to be used");

        if self.uniform.is_none() {
            self.uniform = Some(Arc::new(PersistentDescriptorSet::start(pipeline.clone(), set_id)
                .add_sampled_image(self.access().unwrap(), self.sampler.clone()).unwrap()
                .build().unwrap()
            ));
        }

        self.uniform.clone().unwrap()
    }
}


