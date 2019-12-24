use std::sync::Arc;
use crate::graphics::renderer_3d::mesh::{MeshVBOType, MeshIBOType, MeshCulling, MeshAccess, Vertex3D};
use vulkano::buffer::{BufferAccess, BufferUsage, ImmutableBuffer};
use vulkano::device::Queue;
use vulkano::sync::GpuFuture;
use cgmath::Matrix4;

/// Static `MeshData`, constant, doesn't change
/// Describes mesh geometry with optional IBO
/// Should be reused if possible
pub struct ImmutableMeshData {
    vbo: Arc<MeshVBOType>,
    vbo_slice: Arc<dyn BufferAccess + Send + Sync + 'static>,
    ibo: Option<Arc<MeshIBOType>>,
    aabb: MeshCulling, // Rectangle describing model space AABB
}
impl ImmutableMeshData {
    pub fn from_data(queue: Arc<Queue>, data: Vec<Vertex3D>, indices: Option<Vec<u32>>)
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

        (Arc::new(ImmutableMeshData {
            vbo: vbo.clone(),
            vbo_slice: Arc::new(vbo.into_buffer_slice()),
            ibo,
            aabb: MeshCulling::from_vec(&data)
        }), future)
    }
}
impl MeshAccess for ImmutableMeshData {
    fn visible_in(&self, mat: Matrix4<f32>) -> bool { self.aabb.is_visible_by_matrix(mat) }
    fn get_vbo(&self) -> Arc<MeshVBOType> { self.vbo.clone() }

    fn has_ibo(&self) -> bool { self.ibo.is_some() }
    fn get_ibo(&self) -> Arc<MeshIBOType> { self.ibo.as_ref().unwrap().clone() }
}
