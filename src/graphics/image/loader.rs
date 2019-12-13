

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
    let decoder = png::Decoder::new(bytes);
    let (info, mut reader) = decoder.read_info().unwrap();
//    println!("color_type: {}", info.color_type);
    let dimensions = Dimensions::Dim2d { width: info.width, height: info.height };
    let mut image_data = Vec::new();

    // Add format size based size
//    let size: usize = format.size().unwrap();
    image_data.resize((info.width * info.height * 4) as usize, 0);
    reader.next_frame(&mut image_data).unwrap();

    let (image, future) = ImmutableImage::from_iter(
        image_data.iter().cloned(),
        dimensions,
        format,
        queue.clone()
    ).unwrap();

    (image, Box::new(future))
}