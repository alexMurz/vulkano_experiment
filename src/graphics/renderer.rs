
use std::sync::Arc;
use cgmath::{Matrix4, vec3};
use cgmath::SquareMatrix;
use cgmath::Vector3;
use vulkano::command_buffer::AutoCommandBufferBuilder;
use vulkano::command_buffer::CommandBuffer;
use vulkano::device::Queue;
use vulkano::format::Format;
use vulkano::framebuffer::Framebuffer;
use vulkano::framebuffer::FramebufferAbstract;
use vulkano::framebuffer::RenderPassAbstract;
use vulkano::framebuffer::Subpass;
use vulkano::image::{AttachmentImage, ImageInner};
use vulkano::image::ImageAccess;
use vulkano::image::ImageUsage;
use vulkano::image::ImageViewAccess;
use vulkano::sync::GpuFuture;
use std::cell::RefCell;
use std::rc::Rc;

use crate::graphics::Camera;
use std::any::Any;
use vulkano::buffer::{
    CpuAccessibleBuffer,
    DeviceLocalBuffer
};

pub enum RendererState<'f> {
    Drawing(RenderExecutor<'f>),
    Shadow(ShadowExecutor<'f>),
}

/// Provides access to execution chain
pub struct RenderExecutor<'f> {
    renderer: &'f mut Renderer,
    pub view_projection: Matrix4<f32>,
    framebuffer: Arc<dyn FramebufferAbstract + Send + Sync>,
}
impl <'f> RenderExecutor<'f> {

    #[inline]
    pub fn execute<C>(&mut self, cb: C) where C: CommandBuffer + Send + Sync + 'static {
        unsafe {
            self.renderer.cb = Some(self.renderer.cb.take().unwrap().execute_commands(cb).unwrap());
        }
    }

    #[inline]
    pub fn viewport(&self) -> [u32; 2] {
        let dims = self.framebuffer.dimensions();
        [dims[0], dims[1]]
    }
}

pub struct ShadowExecutor<'f> {
    renderer: &'f mut Renderer,
    pub view_projection: Matrix4<f32>,
    framebuffer: Arc<dyn FramebufferAbstract + Send + Sync>,
}
impl <'f> ShadowExecutor<'f> {

    #[inline]
    pub fn execute<C>(&mut self, cb: C) where C: CommandBuffer + Send + Sync + 'static {
        unsafe {
            self.renderer.shadow_cb = Some(self.renderer.shadow_cb.take().unwrap().execute_commands(cb).unwrap());
        }
    }

    #[inline]
    pub fn viewport(&self) -> [u32; 2] {
        let dims = self.framebuffer.dimensions();
        [dims[0], dims[1]]
    }
}

pub trait RendererDrawable { fn render(executor: &mut RenderExecutor); }

/// Provide multipass rendering
pub struct Renderer {
    pub(crate) camera: Camera,

    queue: Arc<Queue>,

    ////
    render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    diffuse_buffer: Arc<AttachmentImage>,   // Albedo
    normals_buffer: Arc<AttachmentImage>,   // Normals
    depth_buffer: Arc<AttachmentImage>,     // depths
    cb: Option<AutoCommandBufferBuilder>,

    ////
    shadow_render_pass: Arc<dyn RenderPassAbstract + Send + Sync>,
    shadow_depth_buffer: Arc<AttachmentImage>, // Shadow depth
    shadow_cb: Option<AutoCommandBufferBuilder>,

    ////
    light_pass: lighting_pass::LightingPass,
}

impl Renderer {
    pub fn new(queue: Arc<Queue>, final_output_format: Format, camera: Camera) -> Self {
        let render_pass = Arc::new(vulkano::ordered_passes_renderpass!(queue.device().clone(),
            attachments: {
                final_color: {
                    load: Clear,
                    store: Store,
                    format: final_output_format,
                    samples: 1,
                },
                // Will be bound to `self.diffuse_buffer`.
                diffuse: {
                    load: Clear,
                    store: DontCare,
                    format: Format::A2B10G10R10UnormPack32,
                    samples: 1,
                },
                // Will be bound to `self.normals_buffer`.
                normals: {
                    load: Clear,
                    store: DontCare,
                    format: Format::R16G16B16A16Sfloat,
                    samples: 1,
                },
                // Will be bound to `self.depth_buffer`.
                depth: {
                    load: Clear,
                    store: DontCare,
                    format: Format::D16Unorm,
                    samples: 1,
                }
            },
            passes: [
                // Write to the diffuse, normals and depth attachments.
                {
                    color: [diffuse, normals],
                    depth_stencil: {depth},
                    input: []
                },
                // Apply lighting by reading these three attachments and writing to `final_color`.
                {
                    color: [final_color],
                    depth_stencil: {},
                    input: [diffuse, normals, depth]
                }
            ]
        ).unwrap());

        let shadow_render_pass = Arc::new(vulkano::single_pass_renderpass!(queue.device().clone(),
            attachments: {
                depth: {
                    load: Clear,
                    store: Store,
                    format: Format::D16Unorm,
                    samples: 1,
                }
            },
            pass: {
                color: [],
                depth_stencil: {depth}
            }
        ).unwrap());

        let atch_usage = ImageUsage {
            transient_attachment: true,
            input_attachment: true,
            ..ImageUsage::none()
        };
        let diffuse_buffer = AttachmentImage::with_usage(
            queue.device().clone(), [1, 1], Format::A2B10G10R10UnormPack32, atch_usage
        ).unwrap();
        let normals_buffer = AttachmentImage::with_usage(
            queue.device().clone(), [1, 1], Format::R16G16B16A16Sfloat, atch_usage
        ).unwrap();
        let depth_buffer = AttachmentImage::with_usage(
            queue.device().clone(), [1, 1], Format::D16Unorm, atch_usage
        ).unwrap();


        let shadow_depth_buffer = AttachmentImage::with_usage(
            queue.device().clone(), [1024, 1024], Format::D16Unorm, ImageUsage {
                depth_stencil_attachment: true,
                sampled: true,
                ..ImageUsage::none()
            }).unwrap();


        let light_pass  = lighting_pass::LightingPass::new(
            queue.clone(),
            Subpass::from(render_pass.clone(), 1).unwrap(),
            Subpass::from(render_pass.clone(), 1).unwrap(),
        );

        Self {
            camera,
            queue,

            render_pass,
            diffuse_buffer,
            normals_buffer,
            depth_buffer,
            cb: None,

            shadow_render_pass,
            shadow_depth_buffer,
            shadow_cb: None,

            light_pass,
        }
    }

    #[inline]
    pub fn deferred_subpass(&self) -> Subpass<Arc<dyn RenderPassAbstract + Send + Sync>> {
        Subpass::from(self.render_pass.clone(), 0).unwrap()
    }

    #[inline]
    pub fn shadow_subpass(&self) -> Subpass<Arc<dyn RenderPassAbstract + Send + Sync>> {
        Subpass::from(self.shadow_render_pass.clone(), 0).unwrap()
    }

    pub fn render<F, I, Fn>(&mut self, before_future: F, final_image: I, mut drawcall: Fn) -> Box<dyn GpuFuture>
        where
            F: GpuFuture + 'static,
            I: ImageAccess + ImageViewAccess + Clone + Send + Sync + 'static,
            Fn: FnMut(RendererState)
    {
        // Recreate images if dims not matching
        let img_dims = ImageAccess::dimensions(&final_image).width_height();
        if ImageAccess::dimensions(&self.diffuse_buffer).width_height() != img_dims {
            let atch_usage = ImageUsage {
                transient_attachment: true,
                input_attachment: true,
                ..ImageUsage::none()
            };
            self.diffuse_buffer = AttachmentImage::with_usage(
                self.queue.device().clone(), img_dims, Format::A2B10G10R10UnormPack32, atch_usage)
                .unwrap();
            self.normals_buffer = AttachmentImage::with_usage(
                self.queue.device().clone(), img_dims, Format::R16G16B16A16Sfloat, atch_usage)
                .unwrap();
            self.depth_buffer = AttachmentImage::with_usage(
                self.queue.device().clone(), img_dims, Format::D16Unorm, atch_usage)
                .unwrap();
        }

        let framebuffer = Arc::new(Framebuffer::start(self.render_pass.clone())
            .add(final_image.clone()).unwrap()
            .add(self.diffuse_buffer.clone()).unwrap()
            .add(self.normals_buffer.clone()).unwrap()
            .add(self.depth_buffer.clone()).unwrap()
            .build().unwrap());

        let shadow_fb = Arc::new(Framebuffer::start(self.shadow_render_pass.clone())
            .add(self.shadow_depth_buffer.clone()).unwrap()
            .build().unwrap()
        );


        let cam_vp = self.camera.get_view_projection();
        let shadow_vp = {
//            cgmath::ortho(-5.0, 5.0, -5.0, 5.0, -50.0, 50.0) *
            cgmath::perspective(cgmath::Deg(80.0), 1.0, 1.0, 20.0) *
                Matrix4::look_at(cgmath::Point3::new(-0.0,-5.0, 5.0), cgmath::Point3::new(0.0, -1.0, 0.0), cgmath::vec3(0.0, 1.0, 0.0))
//                Matrix4::from_angle_x(cgmath::Deg(-45.0)) *
//                Matrix4::from_translation(cgmath::vec3(0.0, 0.0, -3.0))
        };
        let shadow_bias = {
            Matrix4::new(
                0.5, 0.0, 0.0, 0.0,
                0.0, 0.5, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.5, 0.5, 0.0, 1.0
            ) * shadow_vp
        };

        /* Render Shadows */
        {
            self.shadow_cb = Some(AutoCommandBufferBuilder::primary_one_time_submit(self.queue.device().clone(), self.queue.family()).unwrap()
                .begin_render_pass(shadow_fb.clone(),
                                   true,
                                   vec![1.0f32.into()]
                ).unwrap()
            );

            drawcall(RendererState::Shadow(ShadowExecutor {
                renderer: self,
                view_projection: shadow_vp,
                framebuffer: shadow_fb.clone()
            }));

            self.shadow_cb.take().unwrap()
                .end_render_pass().unwrap()
                .build().unwrap()
                .execute(self.queue.clone()).unwrap()
                .then_signal_fence_and_flush().unwrap()
                .wait(None).unwrap();
        }


//        /*
        self.cb = Some(
            AutoCommandBufferBuilder::primary_one_time_submit(self.queue.device().clone(), self.queue.family()).unwrap()
                .begin_render_pass(framebuffer.clone(),
                                   true,
                                   vec![
                                       [0.0, 0.0, 0.0, 0.0].into(), // Final image
                                       [0.0, 0.0, 0.0, 0.0].into(), // Diffuse
                                       [0.0, 0.0, 0.0, 0.0].into(), // Normals
                                       1.0f32.into(), // Depth
                                   ]).unwrap()
        );


        // Render objects
        {
            let mut executor = RenderExecutor {
                renderer: self,
                view_projection: cam_vp,
                framebuffer: framebuffer.clone()
            };
            drawcall(RendererState::Drawing(executor));
        }


        // Go to lighting pass
        self.cb = Some(self.cb.take().unwrap().next_subpass(true).unwrap());

        // Render lighting
        unsafe {
            let cb = self.light_pass.render(
                &img_dims,
                Matrix4::invert(&cam_vp).unwrap(),
                shadow_bias,
                self.diffuse_buffer.clone(),
                self.normals_buffer.clone(),
                self.depth_buffer.clone(),
                self.shadow_depth_buffer.clone()
            );
            self.cb = Some(self.cb.take().unwrap()
                .execute_commands(cb).unwrap()
            );
        }

        // Build buffer
        let command_buffer = self.cb.take().unwrap().end_render_pass().unwrap().build().unwrap();
//*/

        Box::new(before_future.then_execute(self.queue.clone(), command_buffer).unwrap())
    }
}


/// Lighting for renderer
/// Second subpass
/// accept (diffuse, normal, depth) buffers to generate output image
mod lighting_pass {
    use std::sync::Arc;
    use cgmath::Matrix4;
    use cgmath::SquareMatrix;
    use cgmath::Vector3;
    use vulkano::command_buffer::{AutoCommandBufferBuilder, DynamicState, AutoCommandBuffer};
    use vulkano::command_buffer::CommandBuffer;
    use vulkano::device::{Queue, DeviceOwned};
    use vulkano::format::Format;
    use vulkano::framebuffer::Framebuffer;
    use vulkano::framebuffer::FramebufferAbstract;
    use vulkano::framebuffer::RenderPassAbstract;
    use vulkano::framebuffer::Subpass;
    use vulkano::image::AttachmentImage;
    use vulkano::image::ImageAccess;
    use vulkano::image::ImageUsage;
    use vulkano::image::ImageViewAccess;
    use vulkano::sync::GpuFuture;
    use vulkano::buffer::{CpuAccessibleBuffer, BufferUsage};
    use vulkano::pipeline::{ GraphicsPipelineAbstract, GraphicsPipeline };
    use vulkano::pipeline::blend::{AttachmentBlend, BlendOp, BlendFactor};
    use crate::graphics::renderer::RenderExecutor;
    use vulkano::pipeline::viewport::Viewport;
    use vulkano::descriptor::descriptor_set::PersistentDescriptorSet;
    use vulkano::sampler::{Sampler, MipmapMode, SamplerAddressMode, Filter};
    use vulkano::descriptor::DescriptorSet;

    #[derive(Default, Debug, Clone)]
    struct Vertex {
        position: [f32; 2],
        uv: [f32; 2]
    }
    vulkano::impl_vertex!(Vertex, position, uv);

    mod vs {
        vulkano_shaders::shader!{
            ty: "vertex",
            src: "
    #version 450

    layout(location = 0) in vec2 position;
    layout(location = 1) in vec2 uv;
    layout(location = 0) out vec2 v_screen_coords;
    layout(location = 1) out vec2 v_uv;

    void main() {
        v_uv = uv;
        v_screen_coords = position;
        gl_Position = vec4(position, 0.0, 1.0);
    }"
        }
    }

    mod fs {
        vulkano_shaders::shader!{
        ty: "fragment",
        src: "
#version 450

// The `color_input` parameter of the `draw` method.
layout(input_attachment_index = 0, set = 0, binding = 0) uniform subpassInput u_diffuse;
// The `normals_input` parameter of the `draw` method.
layout(input_attachment_index = 1, set = 0, binding = 1) uniform subpassInput u_normals;
// The `depth_input` parameter of the `draw` method.
layout(input_attachment_index = 2, set = 0, binding = 2) uniform subpassInput u_depth;

layout(push_constant) uniform PushConstants {
    vec4 color;
    vec4 position;
} push_constants;

layout(location = 0) in vec2 v_screen_coords;
layout(location = 0) out vec4 f_color;

layout(location = 1) in vec2 v_uv;
layout(set = 0, binding = 3) uniform sampler2D u_shadow;

layout(set = 1, binding = 0) uniform Matrixes {
    mat4 screen_to_world;
    mat4 shadow_bias;
} mat;

float sampleShadow(vec4 shadow_coord, float bias) {
    float shade = 1.0;
    if (texture(u_shadow, shadow_coord.xy).r < shadow_coord.z - bias) shade = 0.0;
    return shade;
}

void main() {

    float in_depth = subpassLoad(u_depth).x;
    if (in_depth >= 1.0) { discard; }

    vec4 world = mat.screen_to_world * vec4(v_screen_coords, in_depth, 1.0);
    world /= world.w;


    vec3 in_normal = normalize(subpassLoad(u_normals).rgb);
    vec3 light_direction = normalize(push_constants.position.xyz - world.xyz);
    float cosTheta = dot(light_direction, in_normal);
    float light_percent = max(-cosTheta, 0.0);
    float light_distance = length(push_constants.position.xyz - world.xyz) / 20.0;
    light_percent *= 1.0 / exp(light_distance);

    vec4 shadow_coord = mat.shadow_bias * world;
    shadow_coord /= shadow_coord.w;
    float bias = clamp(0.05*tan(acos(cosTheta)), 0.0, 0.1);
    float shade = sampleShadow(shadow_coord, 0.005);

    vec3 in_diffuse = subpassLoad(u_diffuse).rgb;
    f_color.rgb = shade * push_constants.color.rgb * light_percent * in_diffuse;
    f_color.a = 1.0;

}"
    }
    }

    mod debug_vs {
        vulkano_shaders::shader!{
            ty: "vertex",
            src: "
    #version 450

    layout(location = 0) in vec2 position;
    layout(location = 1) in vec2 uv;
    layout(location = 0) out vec2 v_uv;

    void main() {
        v_uv = uv;
        gl_Position = vec4(position, 0.0, 1.0);
    }"
        }

    }
    mod debug_fs {
        vulkano_shaders::shader!{
            ty: "fragment",
            src: "
#version 450

layout(location = 0) in vec2 v_uv;
layout(location = 0) out vec4 f_color;

layout(set = 0, binding = 0) uniform sampler2D u_tex;

void main() {
    vec4 tex = texture(u_tex, v_uv);
    tex.rgb /= tex.a;
    float d = tex.r;
    if (d >= 1.0) f_color = vec4(0.5, 0.0, 0.0, 1.0);
    else f_color = vec4(vec3(d), 1.0);
}"
        }
    }

    pub struct LightingPass {
        queue: Arc<Queue>,

        vbo: Arc<CpuAccessibleBuffer<[Vertex]>>,
        pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

        debug_vbo: Arc<CpuAccessibleBuffer<[Vertex]>>,
        debug_pipeline: Arc<dyn GraphicsPipelineAbstract + Send + Sync>,

        matrix_buffer: Arc<CpuAccessibleBuffer<fs::ty::Matrixes>>,

        sampler: Arc<Sampler>
    }
    impl LightingPass {

        pub fn new<R>(queue: Arc<Queue>, subpass: Subpass<R>, dsubpass: Subpass<R>) -> Self
            where
                R: RenderPassAbstract + Send + Sync + 'static
        {
            let vertex_buffer = {
                CpuAccessibleBuffer::from_iter(queue.device().clone(), BufferUsage::all(), [
                    Vertex { position: [-1.0,-1.0], uv: [0.0, 0.0] },
                    Vertex { position: [-1.0, 1.0], uv: [0.0, 1.0] },
                    Vertex { position: [ 1.0,-1.0], uv: [1.0, 0.0] },

                    Vertex { position: [ 1.0,-1.0], uv: [1.0, 0.0] },
                    Vertex { position: [-1.0, 1.0], uv: [0.0, 1.0] },
                    Vertex { position: [ 1.0, 1.0], uv: [1.0, 1.0] },
                ].iter().cloned()).expect("failed to create buffer")
            };


            let pipeline = {
                let vs = vs::Shader::load(queue.device().clone())
                    .expect("failed to create shader module");
                let fs = fs::Shader::load(queue.device().clone())
                    .expect("failed to create shader module");

                Arc::new(GraphicsPipeline::start()
                    .vertex_input_single_buffer::<Vertex>()
                    .vertex_shader(vs.main_entry_point(), ())
                    .triangle_list()
                    .viewports_dynamic_scissors_irrelevant(1)
                    .fragment_shader(fs.main_entry_point(), ())
                    .blend_collective(AttachmentBlend {
                        enabled: true,
                        color_op: BlendOp::Add,
                        color_source: BlendFactor::One,
                        color_destination: BlendFactor::One,
                        alpha_op: BlendOp::Max,
                        alpha_source: BlendFactor::One,
                        alpha_destination: BlendFactor::One,
                        mask_red: true,
                        mask_green: true,
                        mask_blue: true,
                        mask_alpha: true,
                    })
                    .render_pass(subpass)
                    .build(queue.device().clone())
                    .unwrap()) as Arc<_>
            };

            let debug_vbo = {
                let l = -1.0;
                let u = -0.5;
                CpuAccessibleBuffer::from_iter(queue.device().clone(), BufferUsage::all(), [
                    Vertex { position: [ l, l], uv: [0.0, 0.0] },
                    Vertex { position: [ l, u], uv: [0.0, 1.0] },
                    Vertex { position: [ u, l], uv: [1.0, 0.0] },

                    Vertex { position: [ u, l], uv: [1.0, 0.0] },
                    Vertex { position: [ l, u], uv: [0.0, 1.0] },
                    Vertex { position: [ u, u], uv: [1.0, 1.0] },
                ].iter().cloned()).expect("Unable to create debug vbo")
            };
            let debug_pipeline = {
                let vs = debug_vs::Shader::load(queue.device().clone())
                    .expect("failed to create shader module");
                let fs = debug_fs::Shader::load(queue.device().clone())
                    .expect("failed to create shader module");

                Arc::new(GraphicsPipeline::start()
                    .vertex_input_single_buffer::<Vertex>()
                    .vertex_shader(vs.main_entry_point(), ())
                    .triangle_list()
                    .viewports_dynamic_scissors_irrelevant(1)
                    .fragment_shader(fs.main_entry_point(), ())
//                    .blend_collective(AttachmentBlend {
//                        enabled: true,
//                        color_op: BlendOp::Add,
//                        color_source: BlendFactor::One,
//                        color_destination: BlendFactor::One,
//                        alpha_op: BlendOp::Max,
//                        alpha_source: BlendFactor::One,
//                        alpha_destination: BlendFactor::One,
//                        mask_red: true,
//                        mask_green: true,
//                        mask_blue: true,
//                        mask_alpha: true,
//                    })
                    .render_pass(dsubpass)
                    .build(queue.device().clone())
                    .unwrap()) as Arc<_>
            };

            let sampler = Sampler::new(queue.device().clone(), Filter::Linear, Filter::Linear,
                                       MipmapMode::Nearest, SamplerAddressMode::Repeat, SamplerAddressMode::Repeat,
                                       SamplerAddressMode::Repeat, 0.0, 1.0, 0.0, 0.0).unwrap();

            let matrix_buffer = {
                CpuAccessibleBuffer::from_data(queue.device().clone(), BufferUsage::all(), fs::ty::Matrixes {
                    screen_to_world: Matrix4::identity().into(),
                    shadow_bias: Matrix4::identity().into(),
                }).unwrap()
            };

            Self {
                queue,

                vbo: vertex_buffer,
                pipeline,

                debug_vbo,
                debug_pipeline,

                matrix_buffer,

                sampler
            }
        }

        pub fn render<C, N, D, S>(&self, viewport: &[u32; 2], screen_to_world: Matrix4<f32>, shadow_bias: Matrix4<f32>,
                      color_input: C, normals_input: N, depth_input: D, shadow_input: S,
        ) -> AutoCommandBuffer
            where
                C: ImageViewAccess + Send + Sync + 'static,
                N: ImageViewAccess + Send + Sync + 'static,
                D: ImageViewAccess + Send + Sync + 'static,
                S: ImageViewAccess + Send + Sync + Clone + 'static,
        {

            let push_constants = fs::ty::PushConstants {
                color: [1.0, 1.0, 1.0, 1.0],
                position: [0.0, -5.0, 5.0, 1.0].into(),
            };

            {
                let matrix_data = fs::ty::Matrixes {
                    screen_to_world: screen_to_world.into(),
                    shadow_bias: shadow_bias.into(),
                };
                let mut writer = self.matrix_buffer.write().unwrap();
                *writer = matrix_data;
            }

            let descriptor_set = PersistentDescriptorSet::start(self.pipeline.clone(), 0)
                .add_image(color_input).unwrap()
                .add_image(normals_input).unwrap()
                .add_image(depth_input).unwrap()
                .add_sampled_image(shadow_input.clone(), self.sampler.clone()).unwrap()
                .build().unwrap();


            let matrix_set = PersistentDescriptorSet::start(self.pipeline.clone(), 1)
                .add_buffer(self.matrix_buffer.clone()).unwrap()
                .build().unwrap();

            let dynamic_state = DynamicState {
                viewports: Some(vec![Viewport {
                    origin: [0.0, 0.0],
                    dimensions: [viewport[0] as f32, viewport[1] as f32],
                    depth_range: 0.0 .. 1.0,
                }]),
                .. DynamicState::none()
            };

            let mut builder = AutoCommandBufferBuilder::secondary_graphics(
                self.queue.device().clone(),
                self.queue.family(),
                self.pipeline.clone().subpass()).unwrap()
                .draw(self.pipeline.clone(),
                      &dynamic_state,
                      vec![self.vbo.clone()],
                      (descriptor_set, matrix_set),
                      push_constants).unwrap();

            /* Debug render */ {
                let descriptor_set = PersistentDescriptorSet::start(self.debug_pipeline.clone(), 0)
                    .add_sampled_image(shadow_input, self.sampler.clone()).unwrap()
                    .build().unwrap();
                builder = builder.draw(self.debug_pipeline.clone(), &dynamic_state, vec![self.debug_vbo.clone()], (descriptor_set), ()).unwrap();
            }

            builder.build().unwrap()
        }

    }

}












