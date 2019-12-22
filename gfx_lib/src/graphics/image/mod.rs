
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
use vulkano::sync::GpuFuture;
use crate::sync::Loader;

mod loader;
pub mod sampler_pool;
pub mod atlas;

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
    image: Loader<Arc<dyn ImageViewAccess + Send + Sync>>,
    // If None => Recreate uniform
    uniform: Option<Arc<dyn DescriptorSet + Send + Sync>>,
}
/// Clone image but leave uniform, used to reuse image in multiple DescriptorSets
impl Clone for ImageContent {
    fn clone(&self) -> Self { Self {
        sampler: self.sampler.clone(),
        image: self.image.clone(),
        uniform: None,
    }}
}
/// Create new instance, check and access image
impl ImageContent {

    /// Load image info
    pub fn load_image_data(bytes: Cursor<Vec<u8>>) -> loader::PNGData {
        loader::load_png_data_from_bytes(bytes)
    }

    /// Load image, from file bytes
    pub fn load_image(queue: Arc<Queue>, bytes: Cursor<Vec<u8>>, format: Format) -> Loader<Arc<dyn ImageViewAccess + Send + Sync>> {
        let (a, b) = loader::load_png_image_from_bytes(queue, bytes, format);
        Loader::with_gpu_future(a, b)
    }

    pub fn new_with_bytes(queue: Arc<Queue>, sampler: Arc<Sampler>, bytes: Cursor<Vec<u8>>, format: Format) -> Self {
        let image_loader = ImageContent::load_image(queue, bytes, format);

        Self {
            sampler,
            image: image_loader,
            uniform: None,
        }
    }
    pub fn is_ready(&self) -> bool { self.image.is_ready() }
    pub fn recreate_uniform(&mut self) { self.uniform = None; }

    /// Return sampler
    pub fn get_sampler(&self) -> Arc<Sampler> { self.sampler.clone() }
    /// Return image with no check if it is ready to use
    pub fn get_image(&self) -> Arc<dyn ImageViewAccess + Send + Sync> { self.image.get_ref().clone() }

    /// Wait for image to load
    pub fn flush(&self) { while !self.is_ready() {} }

    pub fn access(&self) -> Result<Arc<dyn ImageViewAccess + Send + Sync>, AccessError> {
//        if !self.is_ready() {
//            Err(AccessError::NotReadyError)
//        } else {
            Ok(self.get_image())
//        }
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


