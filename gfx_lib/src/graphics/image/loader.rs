

use vulkano::{
    device::Queue,
    format::Format,
    image::{ ImmutableImage, Dimensions },
    buffer::{ ImmutableBuffer, BufferAccess },
    sync::GpuFuture,
};

use std::{
    io::Cursor,
    sync::{ Arc, Mutex },
};

/// Raw PNG Loader
pub fn load_png_image_from_bytes(queue: Arc<Queue>, bytes: Cursor<Vec<u8>>, format: Format)
    -> (Arc<ImmutableImage<Format>>, Box<dyn GpuFuture + Send + Sync>)
{
    load_png_data_from_bytes(bytes).load_image(queue, format)
}

/// Raw Image Data
pub struct PNGData {
    pub dimensions: (u32, u32),
    pub data: Vec<u8>,
}
impl PNGData {
    pub fn load_image(self, queue: Arc<Queue>, format: Format) -> (Arc<ImmutableImage<Format>>, Box<dyn GpuFuture + Send + Sync>) {
        let (image, future) = ImmutableImage::from_iter(
            self.data.iter().cloned(),
            Dimensions::Dim2d {
                width: self.dimensions.0,
                height: self.dimensions.1,
            },
            format,
            queue.clone()
        ).unwrap();
        (image, Box::new(future))
    }
}

/// Prepare data for raw PNG loading
pub fn load_png_data_from_bytes(bytes: Cursor<Vec<u8>>) -> PNGData {
    let decoder = png::Decoder::new(bytes);
    let (info, mut reader) = decoder.read_info().unwrap();
    let mut image_data = Vec::new();
    image_data.resize((info.width * info.height * 4) as usize, 0);
    reader.next_frame(&mut image_data).unwrap();
    PNGData {
        dimensions: (info.width, info.height),
        data: image_data,
    }
}