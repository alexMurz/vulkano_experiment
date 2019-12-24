use std::sync::Arc;
use crate::graphics::renderer_3d::mesh::{MeshVBOType, MeshIBOType, MeshCulling, MeshAccess, Vertex3D};
use vulkano::buffer::{BufferAccess, BufferUsage, CpuAccessibleBuffer};
use vulkano::device::Queue;
use vulkano::sync::GpuFuture;
use cgmath::Matrix4;
use vulkano::buffer::cpu_access::{WriteLock, WriteLockError};

pub enum MutableMeshDataError {
    NoBuffer, // Then trying to write non existent buffer
    WriteError(WriteLockError),
}
impl std::error::Error for MutableMeshDataError {}
impl std::fmt::Debug for MutableMeshDataError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        match self {
            MutableMeshDataError::NoBuffer => write!(f, "Buffer does not exist"),
            MutableMeshDataError::WriteError(err) => write!(f, "Writer Error: {:?}", err),
            _ => write!(f, "Error not described"),
        }
    }
}
impl std::fmt::Display for MutableMeshDataError {
    fn fmt(&self, fmt: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        (self as &dyn std::fmt::Debug).fmt(fmt)
    }
}
impl From<WriteLockError> for MutableMeshDataError {
    fn from(e: WriteLockError) -> MutableMeshDataError { MutableMeshDataError::WriteError(e) }
}

/// Cpu Accessible MeshData, slower then Immutable one but can be changed without
/// recreating whole buffer
pub struct MutableMeshData {
    vbo: Arc<CpuAccessibleBuffer<[Vertex3D]>>,
    vbo_slice: Arc<dyn BufferAccess + Send + Sync + 'static>,
    ibo: Option<Arc<CpuAccessibleBuffer<[u32]>>>,
    aabb: MeshCulling, // Rectangle describing model space AABB
}
impl MutableMeshData {
    pub fn from_data(queue: Arc<Queue>, data: Vec<Vertex3D>, indices: Option<Vec<u32>>)
                     -> Arc<MutableMeshData>
    {
        let vbo = CpuAccessibleBuffer::from_iter(
            queue.device().clone(),
            BufferUsage::vertex_buffer(),
            data.iter().cloned()
        ).unwrap();

        let ibo = if let Some(idxs) = indices {
            Some(CpuAccessibleBuffer::from_iter(
                queue.device().clone(),
                BufferUsage::index_buffer(),
                idxs.iter().cloned()
            ).unwrap())
        } else {
            None
        };

        Arc::new(MutableMeshData {
            vbo: vbo.clone(),
            vbo_slice: Arc::new(vbo.into_buffer_slice()),
            ibo,
            aabb: MeshCulling::from_vec(&data)
        })
    }

    pub fn write_vbo(&self) -> Result<WriteLock<[Vertex3D]>, MutableMeshDataError> {
        self.vbo.write().map_err(|e| e.into())
    }

}
impl MeshAccess for MutableMeshData {
    fn visible_in(&self, mat: Matrix4<f32>) -> bool { self.aabb.is_visible_by_matrix(mat) }
    fn get_vbo(&self) -> Arc<MeshVBOType> { self.vbo.clone() }

    fn has_ibo(&self) -> bool { self.ibo.is_some() }
    fn get_ibo(&self) -> Arc<MeshIBOType> { self.ibo.as_ref().unwrap().clone() }
}