use std::sync::Arc;

use crate::graphics::object::{
    ScreenVertex, ScreenInstance
};
use crate::graphics::image::ImageContentAbstract;

use vulkano::{
    device::{ Queue },
    format::Format,

    image::{ ImageAccess, ImageViewAccess },

    buffer:: { BufferAccess, ImmutableBuffer, BufferUsage, CpuAccessibleBuffer },

    descriptor::{
        DescriptorSet
    },

    framebuffer::{ RenderPassAbstract, Subpass, FramebufferBuilder, Framebuffer },
    pipeline::{ GraphicsPipelineAbstract, GraphicsPipeline, viewport::Viewport, vertex::OneVertexOneInstanceDefinition },
    command_buffer::{ AutoCommandBufferBuilder, AutoCommandBuffer, DynamicState },

    sync::GpuFuture,
};

use cgmath::{Matrix4, SquareMatrix, vec3};
use vulkano::buffer::BufferSlice;
use blend::parsers::blend::Block::Rend;

pub mod cache;

use cache::{ Render2DCache, Render2DCacheError };

mod vs {
    vulkano_shaders::shader! {
        ty: "vertex",
        src: "
#version 450

layout(location = 0) in vec2 pos;
layout(location = 1) in vec4 col;
layout(location = 2) in vec2 uv;

// Instance Data
layout(location = 3) in mat4 inst_transform;
layout(location = 7) in vec4 inst_color;
layout(location = 8) in vec2 inst_uv_a;
layout(location = 9) in vec2 inst_uv_b;

layout(location = 0) out vec4 v_color;
layout(location = 1) out vec2 v_uv;

layout(push_constant) uniform PushData {
    mat4 viewport;
} push;

void main() {
    v_color = col * inst_color;
    v_uv = inst_uv_a + inst_uv_b * uv;

    gl_Position = push.viewport * inst_transform * vec4(pos, 0.0, 1.0);
}"
    }
}
mod fs {
    vulkano_shaders::shader!{
        ty: "fragment",
        src: "
#version 450

layout(location = 0) out vec4 f_color;

layout(location = 0) in vec4 v_color;
layout(location = 1) in vec2 v_uv;

layout(set = 0, binding = 0) uniform sampler2D u_sampler;

void main() {
    vec4 tex_col = texture(u_sampler, v_uv);
    f_color = tex_col * v_color; // vec4(tex_col.rgb, 1.0);
}"
    }
}

/// Begin call, renders multiple instances with single texture
pub struct Renderer2DCall<'f> {
    base: &'f mut Renderer2D,
    tex_set: Arc<dyn DescriptorSet + Send + Sync>,
}
impl <'f> Renderer2DCall<'f> {

    pub fn render_rect(&mut self, x: f32, y: f32, w: f32, h: f32, angle: f32) {
        self.render_instance(Renderer2D::prepare_instance(x, y, w, h, angle));
    }

    /// Add Single Instance
    #[inline] pub fn render_instance(&mut self, instance: ScreenInstance) {
        self.base.ibo_data.push(instance)
    }

    /// Add whole Vec into InstanceBufferObject
    #[inline] pub fn render_instances_vec(&mut self, mut data: Vec<ScreenInstance>) {
        self.base.ibo_data.append(&mut data);
    }

    /// End current draw call
    pub fn end_pass(self) { self.base.flush(self.tex_set); }

    /// End pass and rendering into output_image
    pub fn end_rendering(self, future: Box<dyn GpuFuture>) -> Box<dyn GpuFuture> {
        self.base.flush(self.tex_set);
        self.base.end(future)
    }

}

/// Base 2D renderer, manages renderpasses, pipelines
pub struct Renderer2D {
    // Basics
    queue: Arc<Queue>,
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>, // Pipeline with no textures
    dyn_state: DynamicState,

    vbo: Arc<dyn BufferAccess + Send + Sync>,

    ibo_data: Vec<ScreenInstance>,
    ibo: Arc<CpuAccessibleBuffer<[ScreenInstance]>>,

    pub clear_color: [f32; 4],

    viewport_mat: Matrix4<f32>,

    // Render CBB
    cbb: Option<AutoCommandBufferBuilder>,
}
impl Renderer2D {
    pub fn new(queue: Arc<Queue>, output_format: Format, capacity: usize) -> Self {
        assert!(capacity >= 1000, "Recommended capacity at least 1000 instances");
        let default_capacity = capacity;

        let render_pass = Arc::new(vulkano::ordered_passes_renderpass!(queue.device().clone(),
            attachments: {
                image: {
                    load: Clear,
                    store: Store,
                    format: output_format,
                    samples: 1,
                }
            },
            passes: [
                {
                    color: [image],
                    depth_stencil: {},
                    input: []
                }
            ]
        ).unwrap()) as Arc<dyn RenderPassAbstract + Send + Sync>;

        let flat_pipeline = {
            let vs = vs::Shader::load(queue.device().clone())
                .expect("failed to create shader module");
            let fs = fs::Shader::load(queue.device().clone())
                .expect("failed to create shader module");

            Arc::new(GraphicsPipeline::start()
                .vertex_input(OneVertexOneInstanceDefinition::<ScreenVertex, ScreenInstance>::new())
//                .vertex_input_single_buffer::<ScreenVertex>()
                .vertex_shader(vs.main_entry_point(), ())
                .triangle_list()
                .viewports_dynamic_scissors_irrelevant(1)
                .fragment_shader(fs.main_entry_point(), ())
                .blend_alpha_blending()
                .render_pass(Subpass::from(render_pass.clone(), 0).unwrap())
                .build(queue.device().clone())
                .unwrap()) as Arc<dyn GraphicsPipelineAbstract + Send + Sync>
        };

        let vbo = {
            let (a, b) = ImmutableBuffer::from_iter(vec![
                ScreenVertex::with_pos(-0.5, -0.5).uv(0.0, 0.0).uni_color(1.0, 1.0),
                ScreenVertex::with_pos( 0.5, -0.5).uv(1.0, 0.0).uni_color(1.0, 1.0),
                ScreenVertex::with_pos(-0.5,  0.5).uv(0.0, 1.0).uni_color(1.0, 1.0),

                ScreenVertex::with_pos( 0.5, -0.5).uv(1.0, 0.0).uni_color(1.0, 1.0),
                ScreenVertex::with_pos(-0.5,  0.5).uv(0.0, 1.0).uni_color(1.0, 1.0),
                ScreenVertex::with_pos( 0.5,  0.5).uv(1.0, 1.0).uni_color(1.0, 1.0),
            ].iter().cloned(), BufferUsage::vertex_buffer(), queue.clone()).unwrap();
            b.flush().unwrap();
            a
        };

        let ibo = unsafe {
            CpuAccessibleBuffer::uninitialized_array(queue.device().clone(), default_capacity, BufferUsage::all()).unwrap()
        };

        Self {
            queue,
            render_pass,
            pipeline: flat_pipeline,
            dyn_state: DynamicState::none(),

            clear_color: [1.0; 4],
            viewport_mat: Matrix4::identity(),

            vbo,
            ibo_data: Vec::with_capacity(default_capacity),
            ibo,

            cbb: None,
        }
    }

    pub fn set_viewport_window(&mut self, w: f32, h: f32) {
        self.viewport_mat = cgmath::ortho(0.0, w, 0.0, h, -1.0, 1.0);
    }

    /// for parallel, instanced drawing
    #[inline] pub fn prepare_instance(x: f32, y: f32, w: f32, h: f32, angle: f32) -> ScreenInstance {
        ScreenInstance {
            inst_transform: (Matrix4::from_translation(vec3(x, y, 0.0))
                * Matrix4::from_angle_z(cgmath::Deg(angle))
                * Matrix4::from_nonuniform_scale(w, h, 1.0)).into(),
            inst_color: [0.0; 4],
            inst_uv_a: [0.0, 0.0],
            inst_uv_b: [1.0, 1.0],
        }
    }

    /// Begin rendering into output_image
    pub fn begin<I>(&mut self, output_image: I)
        where
            I: ImageAccess + ImageViewAccess + Clone + Send + Sync + 'static,
    {
        assert!(self.cbb.is_none(), "Renderer already started");

        let img_dim = ImageAccess::dimensions(&output_image).width_height();
        self.dyn_state.viewports = Some(vec![Viewport {
            origin: [0.0, 0.0],
            dimensions: [img_dim[0] as _, img_dim[1] as _],
            depth_range: 0.0 .. 1.0
        }]);

        let fb = Arc::new(Framebuffer::start(self.render_pass.clone())
            .add(output_image.clone()).unwrap()
            .build().unwrap()
        );

        self.ibo_data.clear();
        self.cbb = Some(
            AutoCommandBufferBuilder::primary_one_time_submit(self.queue.device().clone(), self.queue.family()).unwrap()
                .begin_render_pass(fb.clone(), false, vec![
                    self.clear_color.into()
                ]).unwrap()
        );
    }

    /// Start RenderCall with new image uniform
    pub fn start_image_uniform(&mut self, image: Arc<dyn DescriptorSet + Send + Sync>) -> Renderer2DCall {
        Renderer2DCall {
            base: self,
            tex_set: image,
        }
    }

    /// Start RenderCall with ImageContentAccess
    pub fn start_image_content<I>(&mut self, image: &mut I) -> Renderer2DCall
        where I: ImageContentAbstract + Send + Sync
    {
        self.start_image_uniform(image.get_uniform(&self.pipeline, 0))
    }

    pub fn render_cache(&mut self, cache: &mut Render2DCache) {
        let (buff, tex) = cache.access(&self.pipeline, 0);
        // skip drawing if buffer slice is non (no instances)
        if buff.is_some() {
            self.cbb = Some(self.cbb.take().unwrap()
                .draw(self.pipeline.clone(), &self.dyn_state,
                      vec![self.vbo.clone(), buff.unwrap()], (tex), vs::ty::PushData {
                        viewport: self.viewport_mat.into()
                    }
                ).unwrap()
            );
        }
    }

    /// Render current batch and clear instance buffer
    fn flush(&mut self, texture: Arc<dyn DescriptorSet + Send + Sync>) {
        let mut cbb = self.cbb.take().unwrap();

        if self.ibo_data.len() > 0 {
            // write ibo_data to buffer
            {
                let mut writer = self.ibo.write().unwrap();
                for i in 0 .. self.ibo_data.len() {
                    writer[i] = self.ibo_data[i];
                }
            }

            let slice = BufferSlice::from_typed_buffer_access(self.ibo.clone()).slice(0 .. self.ibo_data.len()).unwrap();
            cbb = cbb.draw(self.pipeline.clone(), &self.dyn_state,
                           vec![self.vbo.clone(), Arc::new(slice)],
                           (texture), (vs::ty::PushData {
                    viewport: self.viewport_mat.into()
                })).unwrap()
        }

        self.cbb = Some(cbb);
        self.ibo_data.clear();
    }

    /// End rendering into output_image
    pub fn end(&mut self, prev_future: Box<dyn GpuFuture>) -> Box<dyn GpuFuture> {
        assert!(self.cbb.is_some(), "First need to begin renderer");

        let cb = self.cbb.take().unwrap()
            .end_render_pass().unwrap()
            .build().unwrap();
        Box::new(prev_future.then_execute(self.queue.clone(), cb).unwrap())
    }
}


