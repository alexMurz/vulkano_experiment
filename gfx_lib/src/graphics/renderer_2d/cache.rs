

use std::{
    sync::Arc
};

use vulkano::{
    sampler::Sampler,
    image::ImageViewAccess,
    command_buffer::{AutoCommandBufferBuilder, CommandBuffer},
    descriptor::{
        DescriptorSet, PipelineLayoutAbstract,
        descriptor_set::PersistentDescriptorSet,
    },
    buffer::{ CpuAccessibleBuffer, BufferUsage, BufferAccess },
    sync::GpuFuture,
};

use crate::{
    main_processor::Frame,
    graphics::image::ImageContentAbstract,
    graphics::object::ScreenInstance
};
use vulkano::buffer::BufferSlice;
use crate::utils::with;

/// Errors
pub enum Render2DCacheError {
    /// Then appending to buffer will overflow it
    /// Param: (capacity)
    CapacityOverflow(usize),
    /// Basically array out of bounds exception, then accessing
    /// Param: (given_pos, array_size)
    OutOfBounds(usize, usize),
}
impl std::error::Error for Render2DCacheError {
    fn description(&self) -> &str {
        let err_str = match self {
            Render2DCacheError::CapacityOverflow(capacity) => format!("Appending over current capacity {}", capacity),
            Render2DCacheError::OutOfBounds(pos, size) => format!("Index ({}) out of bounds [0; {})", pos, size),
            _ => format!("This Render2DCacheError currently not described. ty: {:?}", self),
        };
        "123"
    }
}
impl std::fmt::Debug for Render2DCacheError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(fmt, "{}", (self as &dyn std::error::Error).description())
    }
}
impl std::fmt::Display for Render2DCacheError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        write!(fmt, "{}", (self as &dyn std::error::Error).description())
    }
}

/// Instance buffer, witch can be created and modified only when necessary. Simple instance caching
pub struct Render2DCache {
    // Associated Instance buffer
    buffer: Arc<CpuAccessibleBuffer<[ScreenInstance]>>,
    // Currently used buffer slice, slice is non if no data in buffer
    slice: Option<Arc<dyn BufferAccess + Send + Sync>>,
    // Associated Image uniform
    image: Option<(Arc<dyn ImageViewAccess + Send + Sync>, Arc<Sampler>)>,
    // Uniform with texture
    uniform_set: Option<Arc<dyn DescriptorSet + Send + Sync>>,
    // Current position in buffer
    pos: usize,
    // Buffer Capacity
    capacity: usize,
}
impl Render2DCache {
    pub fn new(frame: &mut Frame, capacity: usize) -> Self {
        let buffer = unsafe {
            CpuAccessibleBuffer::uninitialized_array(
                frame.queue.device().clone(),
                capacity,
                BufferUsage::all(),
            ).unwrap()
        };

        Self {
            buffer,
            slice: None,
            image: None,
            uniform_set: None,
            pos: 0,
            capacity,
        }
    }

    // ##############
    // Locals

    fn update_slice(&mut self) {
        let slice = BufferSlice::from_typed_buffer_access(self.buffer.clone()).slice(0 .. self.pos).unwrap();
        self.slice = Some(Arc::new(slice));
    }

    // ##############
    // Globals

    /// Return current position in buffer (length of used buffer), less or equal to capacity
    #[inline] pub fn position(&self) -> usize { self.pos }
    #[inline] pub fn capacity(&self) -> usize { self.capacity }

    /// Set new image descriptor, ezpz
    pub fn set_image(&mut self, image: (Arc<dyn ImageViewAccess + Send + Sync>, Arc<Sampler>)) {
        self.image = Some(image);
        self.uniform_set = None; // Reset uniform to update it with new texture
    }

    /// Reallocate buffer with different capacity. Slow, blocking.
    /// If not all data can fit in new capacity, excess will be trimmed
    /// All errors currently just panic
    pub fn realloc(&mut self, frame: &mut Frame, new_capacity: usize) {
        // noice
        if new_capacity == self.capacity { return }

        let new_buffer = unsafe {
            CpuAccessibleBuffer::uninitialized_array(
                frame.queue.device().clone(),
                new_capacity,
                BufferUsage::all()
            ).unwrap()
        };

        let mut cbb = AutoCommandBufferBuilder::primary_one_time_submit(
            frame.queue.device().clone(), frame.queue.family()
        ).unwrap()
            .copy_buffer(self.buffer.clone(), new_buffer.clone()).unwrap()
            .build().unwrap();
        cbb.execute(frame.queue.clone()).unwrap().flush().unwrap();

        // New buffer is ready, drop old arc
        self.buffer = new_buffer;
        self.capacity = new_capacity;
        self.pos = self.pos.min(self.capacity);
        self.update_slice();
    }

    /// Appends new instance to end of the buffer and increase position
    pub fn append(&mut self, instance: ScreenInstance) -> Result<(), Render2DCacheError> {
        if self.pos >= self.capacity { return Err(Render2DCacheError::CapacityOverflow(self.capacity)) }

        // Writer dropped right after setting new value, releasing self
        self.buffer.write().unwrap()[self.pos] = instance;

        self.pos += 1;
        self.update_slice();
        Ok(())
    }

    /// Replace instance at position with different
    pub fn set(&mut self, pos: usize, instance: ScreenInstance) -> Result<(), Render2DCacheError> {
        if pos >= self.pos { return Err(Render2DCacheError::OutOfBounds(pos, self.pos)) }
        let mut writer = self.buffer.write().unwrap();
        writer[pos] = instance;
        Ok(())
    }

    /// Grants access to (Buffer, TextureSet)
    /// Create Set for Pipeline
    /// Panics then texture not set
    pub fn access<Pl>(&mut self, pipeline: &Pl, set_id: usize) -> (Option<Arc<dyn BufferAccess + Send + Sync>>, Arc<dyn DescriptorSet + Send + Sync>)
        where
            Pl: PipelineLayoutAbstract + Send + Sync + Clone + 'static,
    {

        assert!(self.image.is_some(), "Texture not set!");

        if self.uniform_set.is_none() {
            let img= self.image.as_ref().unwrap();
            self.uniform_set = Some(Arc::new(
                PersistentDescriptorSet::start(pipeline.clone(), set_id)
                    .add_sampled_image(img.0.clone(), img.1.clone()).unwrap()
                    .build().unwrap()
            ));
        }

        (self.slice.clone(), self.uniform_set.clone().unwrap())
    }
}


