
use std::sync::Arc;

use vulkano::{
    device::Queue
};

pub mod bake_image;

/// Trait describing post processor base functionality
pub trait PostProcessor {

}

/// Contains post processors to be applied to output attachment of framebuffer
///
pub struct PostProcessingBatch {
    queue: Arc<Queue>,
//    processors: Vec<dyn PostProcessor>,
}
impl PostProcessingBatch {
//    pub fn new(queue: Arc<Queue>) -> Self {
//
//    }
}

