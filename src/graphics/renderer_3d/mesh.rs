
use vulkano::{
    device::Queue,
    buffer::{ BufferAccess, TypedBufferAccess, BufferUsage, ImmutableBuffer },
    descriptor::{
        PipelineLayoutAbstract,
        descriptor_set::{ DescriptorSet, PersistentDescriptorSet },
    },
    pipeline::{
        GraphicsPipelineAbstract,
    },
    sampler::Sampler,
    image::{ ImageViewAccess, ImageAccess },
    sync::GpuFuture,
};
use crate::{
    graphics::image::atlas::TextureRegion,
    loader::VertexInfo,
    sync::{ Loader, LoaderError },
};
use cgmath::{Matrix4, SquareMatrix, Vector3, Deg, Vector4, Matrix3, Matrix, BaseFloat, vec3};
use std::sync::{ Arc, Mutex };
use cgmath_culling::{FrustumCuller, BoundingBox, Intersection};
use vulkano::buffer::CpuAccessibleBuffer;
use vulkano::pipeline::input_assembly::Index;

#[derive(Default, Copy, Clone)]
pub struct Vertex3D {
    pub position: [f32; 3], // Vertex position in Model Space
    pub color: [f32; 4], // Vertex tint
    pub normal: [f32; 3], // Vertex Normal
    pub uv: [f32; 2], // Texture UV
}
impl Vertex3D {
    #[inline]
    pub fn empty() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            normal: [0.0, 0.0, 0.0],
            uv: [0.0, 0.0],
        }
    }
    #[inline]
    pub fn from_position(x: f32, y: f32, z: f32) -> Self {
        Self {
            position: [x, y, z],
            .. Vertex3D::empty()
        }
    }

    #[inline]pub fn position(mut self, x: f32, y: f32, z: f32) -> Self { self.position = [x, y, z];self }
    #[inline]pub fn normal(mut self, x: f32, y: f32, z: f32) -> Self { self.normal = [x, y, z];self }
    #[inline]pub fn color(mut self, r: f32, g: f32, b: f32, a: f32) -> Self { self.color = [r, g, b, a];self }
    #[inline]pub fn uv(mut self, u: f32, v: f32) -> Self { self.uv = [u, v];self }

}
/// Derive from `loader::VertexInfo`
impl From<VertexInfo> for Vertex3D {
    fn from(v: VertexInfo) -> Self {
        Self {
            position: v.pos,
            normal: v.norm,
            uv: v.uv,
            color: [1.0; 4],
        }
    }
}

// Impl for use in shader
vulkano::impl_vertex!(Vertex3D, position, color, normal, uv);

/// Type of MeshBuffer
type MeshVBOType = dyn TypedBufferAccess<Content = [Vertex3D]> + Send + Sync;
type MeshIBOType = dyn TypedBufferAccess<Content = [u32]> + Send + Sync;

/// Trait to access mesh data
pub trait MeshAccess {

    fn visible_in(&self, mat: Matrix4<f32>) -> bool { true }

    // Used for asynchronous loading, return false to skip drawing for this frame
    fn ready_for_use(&self) -> bool { true }

    // Return full typed VBO
    fn get_vbo(&self) -> Arc<MeshVBOType>;
    // Return BufferAccess of full VBO
    fn get_vbo_slice(&self) -> Arc<dyn BufferAccess + Send + Sync> { Arc::new(self.get_vbo().into_buffer_slice()) }

    fn has_ibo(&self) -> bool;
    fn get_ibo(&self) -> Arc<MeshIBOType>;
}

/// Axis Aligned Bounding Box
#[derive(Debug, Copy, Clone, PartialEq)]
pub struct MeshCulling {
    aabb: BoundingBox<f32>
}
impl MeshCulling {
    /// Constuct new AABB
    pub fn from_vec(data: &Vec<Vertex3D>) -> MeshCulling {
        let mut lo = data[0].position;
        let mut hi = data[0].position;
        for v in data {
            let p = v.position;
            for i in 0..3 {
                if p[i] < lo[i] { lo[i] = p[i]; }
                if p[i] > hi[i] { hi[i] = p[i]; }
            }
        }
        MeshCulling { aabb: BoundingBox {
            min: vec3(lo[0], lo[1], lo[2]),
            max: vec3(hi[0], hi[1], hi[2]),
        } }
    }


    /// Check if AABB, modified by model matrix, is visible by ViewProjection
    pub fn is_visible_by_matrixes(&self, model: Matrix4<f32>, vp: Matrix4<f32>) -> bool {
        return self.is_visible_by_matrix(vp * model);
    }
    pub fn is_visible_by_matrix(&self, mat: Matrix4<f32>) -> bool {
        let culler = FrustumCuller::from_matrix(mat);
        culler.test_bounding_box(self.aabb) != Intersection::Outside
    }

}

/// Describes mesh geometry with optional
/// Should be reused if possible
pub struct MeshData {
    vbo: Arc<MeshVBOType>,
    vbo_slice: Arc<dyn BufferAccess + Send + Sync + 'static>,
    ibo: Option<Arc<MeshIBOType>>,
    aabb: MeshCulling, // Rectangle describing model space AABB
}
impl MeshData {
    pub fn from_data(queue: Arc<Queue>, data: Vec<Vertex3D>, indices: Option<Vec<u32>>)
        -> Arc<dyn MeshAccess + Send + Sync>
    {
        let (a, b) = Self::from_data_later(queue, data, indices);
        b.flush().unwrap();
        a
    }
    pub fn from_data_later(queue: Arc<Queue>, data: Vec<Vertex3D>, indices: Option<Vec<u32>>)
        -> (Arc<dyn MeshAccess + Send + Sync>, Box<dyn GpuFuture + Send + Sync + 'static>)
    {
        let (vbo, vbo_future) = ImmutableBuffer::from_iter(
            data.iter().cloned(),
            BufferUsage::vertex_buffer(),
            queue.clone(),
        ).unwrap();

        let (ibo, future) = if let Some(idxs) = indices {
            let (ibo, ibo_future) = ImmutableBuffer::from_iter(
                idxs.iter().cloned(),
                BufferUsage::index_buffer(),
                queue.clone()
            ).unwrap();

            (
                Some(ibo as Arc<MeshIBOType>),
                Box::new(ibo_future.join(vbo_future)) as Box<dyn GpuFuture + Send + Sync>
            )
        } else {
            (None, Box::new(vbo_future) as Box<dyn GpuFuture + Send + Sync>)
        };

        (Arc::new(MeshData {
            vbo: vbo.clone(),
            vbo_slice: Arc::new(vbo.into_buffer_slice()),
            ibo,
            aabb: MeshCulling::from_vec(&data)
        }), future)
    }
}
impl MeshAccess for MeshData {
    fn visible_in(&self, mat: Matrix4<f32>) -> bool { self.aabb.is_visible_by_matrix(mat) }
    fn get_vbo(&self) -> Arc<MeshVBOType> { self.vbo.clone() }

    fn has_ibo(&self) -> bool { self.ibo.is_some() }
    fn get_ibo(&self) -> Arc<MeshIBOType> { self.ibo.as_ref().unwrap().clone() }
}

/// Draw mode, selected for pipeline, based on amount of textures
pub enum MaterialDrawMode {
    NoTexture,
    WithDiffuse
}

/// All kinds of renderer modes use same Matrial structure, just use any
type MaterialColor = crate::graphics::renderer_3d::geometry_pass::flat_fs::ty::Material;

/// Describes material info and textures
#[derive(Clone)]
pub struct MaterialData {
    recreate: bool,
    uniform: Option<Arc<dyn DescriptorSet + Send + Sync>>,

    // Material data
    material_dirty: bool,
    diffuse: [f32; 3],
    flat_shading: bool,

    // Attachment at 0, contains all material data
    material_buffer: Option<Arc<CpuAccessibleBuffer<MaterialColor>>>,

    // Attachment at 1
    diffuse_texture: Option<(Arc<dyn ImageViewAccess + Send + Sync>, Arc<Sampler>)>,
    diffuse_remap: [[f32; 2]; 2], // Remap UV to be from diffuse_remap[0] to diffuse_remap[1]
}
impl Default for MaterialData {
    fn default() -> Self { Self {
        material_dirty: false,
        diffuse: [1.0; 3],
        flat_shading: true,

        recreate: false,
        uniform: None,
        material_buffer: None,
        diffuse_texture: None,
        diffuse_remap: [[0.0, 0.0], [1.0, 1.0]],
    } }
}
impl MaterialData {
    pub fn new() -> Self { Self::default() }

    // #############
    // Material info
    pub fn set_diffuse(&mut self, r: f32, g: f32, b: f32) {
        self.diffuse = [r, g, b];
        self.material_dirty = true;
    }
    pub fn set_flat_shading(&mut self, flag: bool) {
        self.flat_shading = flag;
        self.material_dirty = true;
    }
    pub fn set_uv_remap(&mut self, remap: [[f32; 2]; 2]) {
        self.diffuse_remap = remap;
    }

    // Generate MaterialColor structure
    fn get_material_color(&self) -> MaterialColor {
        MaterialColor {
            _dummy0: [0; 4].into(),
            diffuse: self.diffuse.into(),
            uv_remap_a: self.diffuse_remap[0],
            uv_remap_b: self.diffuse_remap[1],
            flat_shading: if self.flat_shading { 1 } else { 0 },
        }
    }

    pub fn get_diffuse_remap_a(&self) -> [f32; 2] { self.diffuse_remap[0] }
    pub fn get_diffuse_remap_b(&self) -> [f32; 2] { self.diffuse_remap[1] }

    // #############
    // Textures
    #[inline] pub fn set_diffuse_texture_with_sampler(&mut self, texture: Arc<dyn ImageViewAccess + Send + Sync>, sampler: Arc<Sampler>) {
        self.recreate = true;
        self.diffuse_texture = Some((texture, sampler));
    }
    pub fn set_diffuse_region(&mut self, region: &TextureRegion) {
        self.set_diffuse_texture_with_sampler(region.texture.clone(), region.sampler.clone());
        self.diffuse_remap = [region.uv_a, region.uv_b];
        self.material_dirty = true;
    }

    #[inline] pub fn mode(&self) -> MaterialDrawMode {
        let diff = self.diffuse_texture.is_some();
        if diff { MaterialDrawMode::WithDiffuse }
        else { MaterialDrawMode::NoTexture }
    }

    /// Creates, updates and returns uniform according to TextureDrawMode
    pub fn get_uniform<Pl>(&mut self, pipeline: &Pl, set_id: usize) -> Arc<dyn DescriptorSet + Send + Sync>
        where Pl: PipelineLayoutAbstract + Clone + Send + Sync + 'static
    {
        // Create buffer with color info
        if self.material_buffer.is_none() {
            self.material_buffer = Some(CpuAccessibleBuffer::from_data(pipeline.device().clone(),
                BufferUsage::uniform_buffer(),
                self.get_material_color()
            ).unwrap());
            self.material_dirty = false;
        }

        // Update buffer if required
        if self.material_dirty {
            let mut writer = self.material_buffer.as_ref().unwrap().write().unwrap();
            *writer = self.get_material_color();
            self.material_dirty = false;
        }

        // Create/Recreate PersistentDescriptorSet for uniform
        if self.recreate || self.uniform.is_none() {
            match self.mode() {
                MaterialDrawMode::NoTexture => {
                    self.uniform = Some(Arc::new(
                        PersistentDescriptorSet::start(pipeline.clone(), set_id)
                            .add_buffer(self.material_buffer.clone().unwrap()).unwrap()
                            .build().unwrap()
                    ));
                },
                MaterialDrawMode::WithDiffuse => {
                    let (texture, sampler) = self.diffuse_texture.clone().unwrap();
                    self.uniform = Some(Arc::new(
                        PersistentDescriptorSet::start(pipeline.clone(), set_id)
                            .add_buffer(self.material_buffer.clone().unwrap()).unwrap()
                            .add_sampled_image(texture.clone(), sampler.clone()).unwrap()
                            .build().unwrap()
                    ));
                }
            }
            self.recreate = false;
        }

        self.uniform.clone().unwrap()
    }
}

/// Slice of some mesh with material attached
#[derive(Clone)]
pub struct MaterialMeshSlice {
    pub vbo_slice: Arc<dyn BufferAccess + Send + Sync>,
    pub ibo_slice: Option<Arc<MeshIBOType>>,
    pub material: MaterialData
}

/// Draw instance for mesh data
/// contains transformation of matrices
#[derive(Clone)]
pub struct ObjectInstance {
    // Vertex info
    pub mesh_data: Arc<dyn MeshAccess + Send + Sync>,

    // Mesh slices, sorted by mesh
    pub materials: Vec<MaterialMeshSlice>,

    // Model matrix info
    model: Matrix4<f32>,
    normal: Matrix4<f32>,
    dirty: bool,
    pos: [f32; 3],
    scl: [f32; 3],
    rot: [f32; 3], // Radians
}
impl ObjectInstance {

    pub fn new(mesh: Arc<dyn MeshAccess + Send + Sync>) -> Self { Self {
        mesh_data: mesh,
        materials: Vec::new(),
        model: Matrix4::identity(),
        normal: Matrix4::identity(),
        dirty: true,
        pos: [0.0, 0.0, 0.0],
        scl: [1.0, 1.0, 1.0],
        rot: [0.0, 0.0, 0.0],
    }}

    pub fn set_pos(&mut self, x: f32, y: f32, z: f32) {
        self.pos = [x, y, z];
        self.dirty = true;
    }

    pub fn set_rot(&mut self, x: f32, y: f32, z: f32) {
        self.rot = [x, y, z];
        self.dirty = true;
    }

    pub fn set_scl(&mut self, x: f32, y: f32, z: f32) {
        self.scl = [x, y, z];
        self.dirty = true;
    }

    pub fn update(&mut self) {
        if self.dirty {
            self.model = Matrix4::from_nonuniform_scale(self.scl[0], self.scl[1], self.scl[2])
                * Matrix4::from_angle_x(cgmath::Rad(self.rot[0]))
                * Matrix4::from_angle_y(cgmath::Rad(self.rot[1]))
                * Matrix4::from_angle_z(cgmath::Rad(self.rot[2]))
                * Matrix4::from_translation(cgmath::vec3(self.pos[0], self.pos[1], self.pos[2]));
            self.normal = self.model.invert().unwrap().transpose();
            self.dirty = false;
        }
    }

    pub fn model_matrix(&self) -> Matrix4<f32> { self.model }
    pub fn normal_matrix(&self) -> Matrix4<f32> { self.normal }


}