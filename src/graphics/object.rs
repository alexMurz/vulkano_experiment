
use vulkano::buffer::BufferAccess;
use vulkano::descriptor::DescriptorSet;

use cgmath::{Matrix4, SquareMatrix, Vector3, Deg, Vector4, Matrix3, Matrix, BaseFloat, vec3};
use std::sync::Arc;
use cgmath_culling::{FrustumCuller, BoundingBox, Intersection};

#[derive(Default, Copy, Clone)]
pub struct Vertex3D {
    position: [f32; 3], // Vertex position in Model Space
    color: [f32; 4], // Vertex tint
    normal: [f32; 3], // Vertex Normal
    // if 0 use flat shading
    // flat shading is calculated internally and does not use provided normal
    flat_shading: u32,
    uv: [f32; 2], // Texture UV
    material_id: u32, // Index of material
}
impl Vertex3D {
    #[inline]
    pub fn empty() -> Self {
        Self {
            position: [0.0, 0.0, 0.0],
            color: [1.0, 1.0, 1.0, 1.0],
            normal: [0.0, 0.0, 0.0],
            flat_shading: 0,
            uv: [0.0, 0.0],
            material_id: 0
        }
    }
    #[inline]
    pub fn from_position(x: f32, y: f32, z: f32) -> Self {
        Self {
            position: [x, y, z],
            .. Vertex3D::empty()
        }
    }

    #[inline]
    pub fn position(mut self, x: f32, y: f32, z: f32) -> Self {
        self.position = [x, y, z];
        self
    }
    #[inline]
    pub fn normal(mut self, x: f32, y: f32, z: f32) -> Self {
        self.normal = [x, y, z];
        self
    }
    #[inline]
    pub fn color(mut self, r: f32, g: f32, b: f32, a: f32) -> Self {
        self.color = [r, g, b, a];
        self
    }
    #[inline]
    pub fn flat_shading(mut self, flag: bool) -> Self {
        self.flat_shading = if flag { 1 } else { 0 };
        self
    }
    #[inline]
    pub fn uv(mut self, u: f32, v: f32) -> Self {
        self.uv = [u, v];
        self
    }
    #[inline]
    pub fn material(mut self, id: u32) -> Self {
        self.material_id = id;
        self
    }

}
vulkano::impl_vertex!(Vertex3D, position, color, normal, flat_shading, uv, material_id);

// Simplified vertex for working with screening
#[derive(Default, Copy, Clone)]
pub struct ScreenVertex {
    position: [f32; 2]
}
impl ScreenVertex {
    pub fn new(x: f32, y: f32) -> Self { Self { position: [x, y] } }
}
vulkano::impl_vertex!(ScreenVertex, position);

pub trait MeshAccess {
    fn visible_in(&self, mat: Matrix4<f32>) -> bool { true }
    fn get_vbo(&self) -> Arc<dyn BufferAccess + Send + Sync>;

    fn has_ibo(&self) -> bool;
    fn get_ibo(&self) -> Arc<dyn BufferAccess + Send + Sync>;
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
    pub vbo: Arc<dyn BufferAccess + Send + Sync + 'static>,
    pub ibo: Option<Arc<dyn BufferAccess + Send + Sync + 'static>>,
    pub aabb: MeshCulling, // Rectangle describing model space AABB
}
impl MeshAccess for MeshData {
    fn visible_in(&self, mat: Matrix4<f32>) -> bool { self.aabb.is_visible_by_matrix(mat) }
    fn get_vbo(&self) -> Arc<dyn BufferAccess + Send + Sync> { self.vbo.clone() }

    fn has_ibo(&self) -> bool { self.ibo.is_some() }
    fn get_ibo(&self) -> Arc<dyn BufferAccess + Send + Sync> { self.ibo.as_ref().unwrap().clone() }
}

/// Describes material information
pub struct Material {
    pub diffuse: [f32; 3],
}

pub struct ObjectInstance {
    // Vertex info
    pub mesh_data: Arc<dyn MeshAccess + Send + Sync>,

    // Model matrix info
    model: Matrix4<f32>,
    normal: Matrix4<f32>,
    dirty: bool,
    pos: [f32; 3],
    scl: [f32; 3],
    rot: [f32; 3], // Radians
}
/// Clone instance and reuse mesh data
impl Clone for ObjectInstance {
    fn clone(&self) -> Self {
        Self {
            mesh_data: self.mesh_data.clone(),
            model: self.model,
            normal: self.normal,
            dirty: self.dirty,
            pos: self.pos,
            scl: self.scl,
            rot: self.rot,
        }
    }
}
impl MeshAccess for ObjectInstance {
    fn visible_in(&self, mat: Matrix4<f32>) -> bool { self.mesh_data.visible_in(mat) }
    fn get_vbo(&self) -> Arc<dyn BufferAccess + Send + Sync> { self.mesh_data.get_vbo() }

    fn has_ibo(&self) -> bool { self.mesh_data.has_ibo() }
    fn get_ibo(&self) -> Arc<dyn BufferAccess + Send + Sync> { self.mesh_data.get_ibo() }
}
impl ObjectInstance {

    pub fn new(mesh: Arc<dyn MeshAccess + Send + Sync>) -> Self { Self {
        mesh_data: mesh,
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


mod test {
    use cgmath_culling::BoundingBox;

    /// Test AABB
    #[test] #[allow(non_snake_case)] fn test_MeshAABB() {
        use crate::graphics::object::{ Vertex3D, MeshCulling };
        use cgmath::{ vec3, Vector3, Matrix4, SquareMatrix };
        use cgmath_culling::FrustumCuller;

        let data = vec![
            Vertex3D::from_position(-2.0,-2.0, -2.0),
            Vertex3D::from_position( 2.0,-2.0,  0.0),
            Vertex3D::from_position(-2.0, 2.0,  0.0),
            Vertex3D::from_position( 2.0, 2.0,  2.0),
        ];

        // Is AABB Constructed Correctly
        let aabb = MeshCulling::from_vec(&data);
        assert_eq!(aabb, MeshCulling {
            aabb: BoundingBox {
                min: vec3(-2.0, -2.0, -2.0),
                max: vec3( 2.0,  2.0,  2.0),
            }
        });

        let vp = cgmath::ortho(-1.0, 1.0, -1.0, 1.0, -1.0, 1.0);

        // Visible
        assert_eq!(aabb.is_visible_by_matrixes(Matrix4::identity(), vp), true);

        // Invisible
        assert_eq!(aabb.is_visible_by_matrixes(Matrix4::from_translation(Vector3::new( 9.0,  0.0,  0.0)), vp), false);
        assert_eq!(aabb.is_visible_by_matrixes(Matrix4::from_translation(Vector3::new(-9.0,  0.0,  0.0)), vp), false);
        assert_eq!(aabb.is_visible_by_matrixes(Matrix4::from_translation(Vector3::new( 0.0,  9.0,  0.0)), vp), false);
        assert_eq!(aabb.is_visible_by_matrixes(Matrix4::from_translation(Vector3::new( 0.0, -9.0,  0.0)), vp), false);
        assert_eq!(aabb.is_visible_by_matrixes(Matrix4::from_translation(Vector3::new( 0.0,  0.0,  9.0)), vp), false);
        assert_eq!(aabb.is_visible_by_matrixes(Matrix4::from_translation(Vector3::new( 0.0,  0.0, -9.0)), vp), false);

        // Multiple corners
        assert_eq!(aabb.is_visible_by_matrixes(Matrix4::from_translation(Vector3::new(-9.0, -9.0, -9.0)), vp), false);
        assert_eq!(aabb.is_visible_by_matrixes(Matrix4::from_translation(Vector3::new( 0.0, -9.0, -9.0)), vp), false);
        assert_eq!(aabb.is_visible_by_matrixes(Matrix4::from_translation(Vector3::new( 0.0, -9.0, -9.0)), vp), false);


        // Rotated Rectangle
        assert_eq!(aabb.is_visible_by_matrixes({ Matrix4::from_angle_z(cgmath::Deg(-45.0)) *
            Matrix4::from_translation(Vector3::new(2.0, 2.0, 0.0)) }, vp), true);

    }

}










