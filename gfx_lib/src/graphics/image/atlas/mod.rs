// ##########
// Texture Atlas Generator
// Atlas is a single texture contain multiple, smaller textures

use std::{
    sync::Arc,
    ops::Index,
    io::Cursor,
    path::{ PathBuf, Path },
    collections::{ BTreeMap, HashMap }
};
use crate::{
    main_processor::Frame,
    graphics::{
        object::ScreenVertex,
        renderer_2d::Renderer2D,
        image::{
            ImageContentAbstract,
            sampler_pool::SamplerParams
        },
    },
};
use vulkano::{
    format::Format,
    image::{
        ImageAccess, ImageViewAccess, AttachmentImage, ImageUsage,
        Dimensions,
        StorageImage, ImmutableImage, MipmapsCount, ImageLayout
    },
    buffer::{ ImmutableBuffer, BufferUsage },
    command_buffer::{ AutoCommandBufferBuilder, AutoCommandBuffer, DynamicState, CommandBuffer },
    pipeline::{
        viewport::Viewport,
        GraphicsPipeline, GraphicsPipelineAbstract,
    },
    sampler::Sampler,
    descriptor::{
        descriptor_set::{ PersistentDescriptorSet, DescriptorSet },
    },
    framebuffer::{ Framebuffer, RenderPassAbstract, Subpass },
    sync::GpuFuture
};
use cgmath::{ Matrix4, SquareMatrix };
use crate::graphics::object::ScreenInstance;
use crate::loader::MaterialImageUsage;
use crate::sync::Loader;
use crate::graphics::image::ImageContent;
use vulkano::device::Queue;
use std::ops::Range;
use crate::graphics::image::loader::PNGData;


pub mod rect_solver;


pub enum AtlasError {
    // Cant fill all images in selected bounds
    SolverError(rect_solver::SolverError),
    NameAlreadyInUse(String), // Then image with the same name already added
}
impl std::error::Error for AtlasError {}
impl std::fmt::Debug for AtlasError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        match self {
            AtlasError::SolverError(e) => write!(f, "Solver Error: {:?}", e),
            AtlasError::NameAlreadyInUse(name) => write!(f, "Image with name \"{}\" already registered", name),
            _ => write!(f, "Error not described"),
        }
    }
}
impl std::fmt::Display for AtlasError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        (self as &dyn std::fmt::Debug).fmt(fmt)
    }
}
impl From<rect_solver::SolverError> for AtlasError {
    fn from(e: rect_solver::SolverError) -> Self { AtlasError::SolverError(e) }
}

/// Atlas Builder Entry (image holder)
/// Container in different structure just in case if some other info will be required
struct BuilderEntry {
    name: String,
    dims: [u32; 2], // Target image dims, may not be same but image will be squished
    image: EntryImage,
}
/// Contains image access types
enum EntryImage {
    Image(Arc<dyn ImageViewAccess + Send + Sync>), // Image already loaded, can be used here and now
    ImageLoader(Loader<Arc<dyn ImageViewAccess + Send + Sync>>), // Same as `Image` but it is future for Image
    Request(Option<PNGData>), // Request to load image
}
impl EntryImage {
    /// Loads `EntryImage::Request`, changing it to be `EntryImage::ImageLoader`
    fn load_requests(&mut self, queue: &Arc<Queue>, format: &Format) {
        match self {
            EntryImage::Request(data) => {
                println!("Make loader from request");
                let (i, f) = data.take().unwrap().load_image(queue.clone(), format.clone());
                *self = EntryImage::ImageLoader(
                    Loader::with_gpu_future(i as Arc<dyn ImageViewAccess + Send + Sync>, f)
                )
            },
            _ => (),
        }
    }
}

/// into `EntryImage::Image`
impl From<Arc<dyn ImageViewAccess + Send + Sync>> for EntryImage {
    fn from(o: Arc<dyn ImageViewAccess + Send + Sync>) -> Self { EntryImage::Image(o) }
}
/// into `EntryImage::ImageLoader`
impl From<Loader<Arc<dyn ImageViewAccess + Send + Sync>>> for EntryImage {
    fn from(o: Loader<Arc<dyn ImageViewAccess + Send + Sync>>) -> Self { EntryImage::ImageLoader(o) }
}

/// `AtlasBuilder` Result, used to get editor for last entry or just pass to next command
pub struct AtlasBuilderResult(Result<AtlasBuilder, AtlasError>);
impl AtlasBuilderResult {
    pub fn unwrap(self) -> AtlasBuilderEditor { AtlasBuilderEditor(self.0.unwrap()) }
    pub fn unwrap_and_next(self) -> AtlasBuilder { self.0.unwrap() }
}

/// `BuilderEntry` editor, result of `AtlasBuilderResult` unwrap
pub struct AtlasBuilderEditor(AtlasBuilder);
impl AtlasBuilderEditor {
    /// Set last entry absolute dimensions
    pub fn set_dim(mut self, dim: [u32; 2]) -> Self {
        self.0.entries.last_mut().unwrap().dims = dim;
        self
    }
    /// Proportionally change dimensions of last entry
    pub fn set_scl(mut self, scl: [f32; 2]) -> Self {
        let dim = self.0.entries.last().unwrap().dims;
        self.0.entries.last_mut().unwrap().dims = [
            (dim[0] as f32 * scl[0]) as u32,
            (dim[1] as f32 * scl[1]) as u32
        ];
        self
    }
    pub fn next(self) -> AtlasBuilder { self.0 }
}

/// Atlas Builder, collection of all data and build function required to build actual `TextureAtlas`
pub struct AtlasBuilder {
    max_dims: u32,
    padding: [u32; 2],
    background_color: [f32; 4],
    can_rotate: bool,
    format: Format,
    sampler: Option<Arc<Sampler>>, // If None(default) will become simple linear
    entries: Vec<BuilderEntry>,
}
impl AtlasBuilder {
    pub fn start() -> Self {
        AtlasBuilder {
            max_dims: 1024,
            padding: [2; 2],
            background_color: [0.0; 4],
            can_rotate: true,
            format: Format::R8G8B8A8Snorm,
            sampler: None,
            entries: Vec::new()
        }
    }

    // Params
    pub fn set_max_dims(mut self, s: u32) -> Self { self.max_dims = s; self }
    pub fn set_padding(mut self, px: u32, py: u32) -> Self { self.padding = [px, py]; self }
    pub fn set_background_color(mut self, r: f32, g: f32, b: f32, a: f32) -> Self { self.background_color = [r,g,b,a]; self }
    pub fn set_format(mut self, format: Format) -> Self { self.format = format; self }

    /// Add entry or return Err
    fn add_entry(mut self, entry: BuilderEntry) -> Result<Self, AtlasError> {
        if self.entries.iter().find(|x| x.name == entry.name).is_some() {
            Err(AtlasError::NameAlreadyInUse(entry.name))
        } else {
            self.entries.push(entry);
            Ok(self)
        }
    }

    /// Add already loaded image
    pub fn add_image<S: Into<String>>(mut self, name: S, image: Arc<dyn ImageViewAccess + Send + Sync>) -> AtlasBuilderResult {
        AtlasBuilderResult(self.add_entry(BuilderEntry {
            name: name.into(),
            dims: image.dimensions().width_height(),
            image: image.into(),
        }))
    }
    /// Add image future
    pub fn add_loader<S: Into<String>>(mut self, name: S, loader: Loader<Arc<dyn ImageViewAccess + Send + Sync>>) -> AtlasBuilderResult {
        AtlasBuilderResult(self.add_entry(BuilderEntry {
            name: name.into(),
            dims: loader.snapshot().unwrap().dimensions().width_height(),
            image: loader.into(),
        }))
    }
    /// Starts loading image for path
    pub fn add_data<S: Into<String>>(mut self, name: S, data: Cursor<Vec<u8>>) -> AtlasBuilderResult {
        let data = ImageContent::load_image_data(data);
        AtlasBuilderResult(self.add_entry(BuilderEntry {
            name: name.into(),
            dims: [data.dimensions.0, data.dimensions.1],
            image: EntryImage::Request(Some(data))
        }))
    }


    pub fn build(mut self, frame: &mut Frame) -> Result<Loader<TextureAtlas>, AtlasError> {
        let format = self.format;
        let queue = frame.queue.clone();
        let sampler = self.sampler.unwrap_or_else(||
            frame.sampler_pool.with_params(SamplerParams::simple_repeat())
        );
        let background_color = self.background_color;

        // Use rect_solver to map all images into rectangles and bin them with params
        let (min_dims, mut rects) = {
            use rect_solver::{ Solver, SolverError, Rect };
            let solver = Solver::with_params(self.max_dims, self.padding, self.can_rotate);
            let mut rects = self.entries.drain(..)
                .map(|x| {
                    let dims = x.dims;
                    Rect::new(x, dims[0], dims[1])
                })
                .collect();
            let dims = solver.solve(&mut rects)?;
            (dims, rects)
        };
        let dim = Dimensions::Dim2d { width: min_dims[0], height: min_dims[1] };

        /// Local, ane time use uniform builder for image
        struct LocalImageContent {
            image: Arc<dyn ImageViewAccess + Send + Sync>,
            sampler: Arc<Sampler>,
        }
        impl LocalImageContent {
            fn new(image_data: &EntryImage, sampler: Arc<Sampler>) -> Self {
                let image = match image_data {
                    EntryImage::Image(v) => v.clone(),
                    EntryImage::ImageLoader(l) => l.snapshot().unwrap(),
                    EntryImage::Request(_) => unimplemented!(),
                };
                Self {
                    image, sampler
                }
            }
        }
        impl ImageContentAbstract for LocalImageContent {
            fn get_uniform(&mut self, pipeline: &Arc<dyn GraphicsPipelineAbstract + Send + Sync>, set_id: usize) -> Arc<dyn DescriptorSet + Send + Sync> {
                Arc::new(PersistentDescriptorSet::start(pipeline.clone(), set_id)
                    .add_sampled_image(self.image.clone(), self.sampler.clone()).unwrap()
                    .build().unwrap()
                )
            }
        }

        // Resolve loading requests, into Loaders
        for l in rects.iter_mut() {
            let data = match &mut l.key.image {
                EntryImage::Request(r) => r.take(),
                _ => None,
            };
            if let Some(data) = data {
                let (i, f) = data.load_image(queue.clone(), format.clone());
                l.key.image = EntryImage::ImageLoader(
                    Loader::with_gpu_future(i as Arc<dyn ImageViewAccess + Send + Sync>, f)
                );
            }
        }

        // Transient image we render stuff into then copy it into `ImmutableImage` and delete this one
        let transient_image = vulkano::image::StorageImage::new(
            queue.device().clone(), dim, format, vec![queue.family()]
        ).unwrap();

        // Create copy task
        let (output_image, image_copy_command) = {
            let (image, init) = vulkano::image::ImmutableImage::uninitialized(
                queue.device().clone(),
                dim,
                format,
                MipmapsCount::One,
                ImageUsage {
                    sampled: true,
                    transfer_destination: true,
                    .. ImageUsage::none()
                },
                ImageLayout::ShaderReadOnlyOptimal,
                vec![queue.family()]
            ).unwrap();

            let cb = AutoCommandBufferBuilder::new(queue.device().clone(), queue.family()).unwrap()
                .copy_image(
                    transient_image.clone(), [0, 0, 0], 0, 0,
                    init, [0, 0, 0], 0, 0,
                    dim.width_height_depth(), 1).unwrap()
                .build().unwrap();
            (image, cb)
        };


        Ok(Loader::with_closure(move || {

            // Await on futures in `r.key.image`
            for r in rects.iter() {
                match &r.key.image {
                    // No need to wait on ready image
                    EntryImage::Image(_) => (),
                    // Wait on loader
                    EntryImage::ImageLoader(l) => l.wait(None).unwrap(),
                    // Requests should be resolved by this point
                    EntryImage::Request(_) => panic!("Request was not resolved"),
                }
            }

            // Render using Renderer2D
            let mut renderer = Renderer2D::new(queue.clone(), format, rects.len());
            renderer.set_viewport_window(dim.width() as f32, dim.height() as f32);
            renderer.clear_color = background_color;
            renderer.begin(transient_image.clone());

            let mut future = Box::new(vulkano::sync::now(queue.device().clone())) as Box<dyn GpuFuture + Send + Sync>;

            for r in rects.iter() {
                let mut content = LocalImageContent::new(&r.key.image, sampler.clone());
                let mut call = renderer.start_image_content(&mut content);
                let mut inst = {
                    let w = r.size[0] as f32;
                    let h = r.size[1] as f32;
                    let x = r.pos[0] as f32 + w*0.5;
                    let y = r.pos[1] as f32 + h*0.5;
                    let r = if r.rotated { 90.0 } else { 0.0 };
                    let mut inst = Renderer2D::prepare_instance(x, y, w, h, r);
                    inst.set_color(1.0, 1.0, 1.0, 1.0);
                    inst
                };
                call.render_instance(inst);
            }

            future = renderer.end(future);
            future
                .then_execute(queue, image_copy_command).unwrap()
                .then_signal_fence_and_flush().unwrap()
                .wait(None).unwrap();

            TextureAtlas::new(output_image, sampler, rects)
        }))
    }

}

/// Just contains relevant information to remap UVs
#[derive(Clone)]
pub struct TextureRegion {
    pub texture: Arc<dyn ImageViewAccess + Send + Sync>,
    pub sampler: Arc<Sampler>,
    pub uv_a: [f32; 2], // Lower UV corner
    pub uv_b: [f32; 2], // Upper UV corner
}
impl TextureRegion {
    pub fn from_image(texture: Arc<dyn ImageViewAccess + Send + Sync>, sampler: Arc<Sampler>) -> Self {
        Self {
            texture, sampler,
            uv_a: [0.0, 1.0],
            uv_b: [1.0, 0.0]
        }
    }
    fn from_rect(rect: rect_solver::Rect<BuilderEntry>, w: u32, h: u32, tex: Arc<dyn ImageViewAccess + Send + Sync>, sampler: Arc<Sampler>) -> Self {
        let iw = 1.0 / w as f32;
        let ih = 1.0 / h as f32;

        let (u0, u1) = {
            let u0 = rect.pos[0] as f32 * iw;
            let u1 = u0 + rect.size[0] as f32 * iw;
            (1.0 - u0, 1.0 - u1)
//            (u0, u1)
//            (0.0, 1.0)
        };
        let (v0, v1) = {
            let v0 = rect.pos[1] as f32 * ih;
            let v1 = v0 + rect.size[1] as f32 * ih;


            (1.0 - v0, 1.0 - v1)
//            (v0, v1)
//            (0.0, 1.0)
        };
        Self {
            texture: tex,
            sampler,
            uv_a: [u0, v0],
            uv_b: [u1, v1],
        }
    }
}
/// `TextureRegion` from image and sampler
impl From<(Arc<dyn ImageViewAccess + Send + Sync>, Arc<Sampler>)> for TextureRegion {
    fn from(o: (Arc<dyn ImageViewAccess + Send + Sync>, Arc<Sampler>)) -> Self { Self::from_image(o.0, o.1) }
}

/// Trait for resolving imagees then loading
pub trait ImageResolver {
    /// Receive region from resolver
    fn get(&mut self, usage: MaterialImageUsage, key: &String) -> Option<&TextureRegion>;
    /// Call resolver to flush any futures created while calling `get` if any accumulated
    /// Can and should block if any resource with associated future was returned by `get`
    fn flush(&mut self) {}
}

/// Resolves images from `TextureAtlas` to use in objects
pub struct AtlasImageResolver {
    regions: Arc<HashMap<String, TextureRegion>>, // regions are static, no reason to clone actual map, so RC it
}
impl AtlasImageResolver {
    pub fn new(atlas: &TextureAtlas) -> Box<dyn ImageResolver + Send + 'static> { Box::new(Self { regions: atlas.regions.clone() }) }
}
impl ImageResolver for AtlasImageResolver {
    fn get(&mut self, usage: MaterialImageUsage, key: &String) -> Option<&TextureRegion> {
        println!("AtlasImageResolver::get -> {:?}, {}", usage, key);
        self.regions.get(key)
    }
}

/// Resolves images from `directory` and loads them on the fly
pub struct DirectoryImageResolver {
    queue: Arc<Queue>,
    sampler: Arc<Sampler>,
    // base directory
    base_path: PathBuf,
    // Pooled images for (path, image, image_future)
    pooled: BTreeMap<String, (Loader<Arc<dyn ImageViewAccess + Send + Sync>>, TextureRegion)>
}
impl DirectoryImageResolver {
    pub fn new(path: &Path, queue: Arc<Queue>, sampler: Arc<Sampler>) -> Result<Box<Self>, std::io::Error> {
        Ok(Box::new(Self {
            queue,
            sampler,
            base_path: path.into(),
            pooled: BTreeMap::new(),
        }))
    }
}
impl ImageResolver for DirectoryImageResolver {
    fn get(&mut self, usage: MaterialImageUsage, key: &String) -> Option<&TextureRegion> {
        if !self.pooled.contains_key(key) {
            let path = {
                let mut p = self.base_path.clone();
                p.push(key);
                p
            };
            let bytes = std::fs::read(path.clone()).unwrap_or_else(|_| panic!("No file for path: {:?}", path));
            let loader = ImageContent::load_image(
                self.queue.clone(),
                Cursor::new(bytes),
                Format::R8G8B8A8Srgb
            );
            let region = {
                let image = loader.snapshot().unwrap();
                TextureRegion::from_image(image, self.sampler.clone())
            };
            self.pooled.insert(key.clone(), (loader, region));
        }

        Some(&self.pooled.get(key).as_ref().unwrap().1)
    }
}

/// Contains `ImageAccess` Arc and info about regions inside said image
/// Atlas is bound to one pipeline, clone atlas for use in a different pipeline (image is reused)
/// or clean uniform to recreate it with different pipeline
pub struct TextureAtlas {
    uniform: Option<Arc<dyn DescriptorSet + Send + Sync>>, // None by default or set to None if want to recreate
    image: Arc<dyn ImageViewAccess + Send + Sync>, // Atlas Image
    sampler: Arc<Sampler>, // Sampler, ya
    regions: Arc<HashMap<String, TextureRegion>>, // regions are static, no reason to clone actual map, so RC it
}
/// Clone
impl Clone for TextureAtlas {
    fn clone(&self) -> Self {
        Self {
            uniform: None,
            image: self.image.clone(),
            sampler: self.sampler.clone(),
            regions: self.regions.clone(),
        }
    }
}
/// Atlas stuff
impl TextureAtlas {

    pub fn start() -> AtlasBuilder { AtlasBuilder::start() }

    fn new(image: Arc<dyn ImageViewAccess + Send + Sync>, sampler: Arc<Sampler>, mut regions: Vec<rect_solver::Rect<BuilderEntry>>) -> Self {
        let (w, h) = {
            let d = image.dimensions().width_height();
            (d[0], d[1])
        };
        let mut map = regions
            .drain(..)
            .map(|x| (
                x.key.name.clone(),
                TextureRegion::from_rect(x, w, h, image.clone(), sampler.clone())
            ))
            .collect();
        Self {
            uniform: None,
            image,
            sampler,
            regions: Arc::new(map)
        }
    }

    /// Reset uniform
    #[inline] pub fn recreate_uniform(&mut self) { self.uniform = None; }

    /// Sets new sampler and reset uniform
    #[inline] pub fn set_sampler(&mut self, sampler: Arc<Sampler>) {
        self.sampler = sampler;
        self.recreate_uniform();
    }

    /// Return `Some(&TextureRegion)` for name Or `None`
    #[inline] pub fn get<T: Into<String>>(&self, key: T) -> Option<&TextureRegion> { self.regions.get(&key.into()) }

    /// Return clone of `Arc` instance of image used in this `TextureAtlas`
    #[inline] pub fn get_image(&self) -> Arc<dyn ImageViewAccess + Send + Sync> { self.image.clone() }

}
/// Index `TextureAtlas`, panics if no region for name
impl <T: Into<String>> std::ops::Index<T> for TextureAtlas {
    type Output = TextureRegion;
    fn index(&self, idx: T) -> &Self::Output { self.get(idx).unwrap() }
}
/// Yields uniform for image used in `TextureAtlas`
impl ImageContentAbstract for TextureAtlas {
    fn get_uniform(&mut self, pipeline: &Arc<dyn GraphicsPipelineAbstract + Send + Sync>, set_id: usize) -> Arc<dyn DescriptorSet + Send + Sync> {

        if self.uniform.is_none() {
            self.uniform = Some(Arc::new(PersistentDescriptorSet::start(pipeline.clone(), set_id)
                .add_sampled_image(self.image.clone(), self.sampler.clone()).unwrap()
                .build().unwrap()
            ));
        }

        self.uniform.clone().unwrap()
    }
}
