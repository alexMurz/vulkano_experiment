use std::path::Path;

use crate::loader::{
    obj::Error::NoObjectForName,
    ObjectInfo,
    MaterialSlice, MaterialInfo,
    VertexInfo
};
use std::collections::BTreeMap;

pub enum Error {
    LoadingError(tobj::LoadError),
    NoObjectForName(String),
}
impl std::error::Error for Error {}
impl std::fmt::Debug for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        match self {
            Error::LoadingError(err) => write!(f, "Unable to load files: {:?}", err),
            Error::NoObjectForName(name) => write!(f, "No object for name: {}", name),
        }
    }
}
impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        (self as &dyn std::fmt::Debug).fmt(f)
    }
}
impl From<tobj::LoadError> for Error {
    fn from(o: tobj::LoadError) -> Self { Error::LoadingError(o) }
}


/// Load objects with names into map (name, object) or error
pub fn load_objects<T: Into<String>>(path: &Path, mut names: Vec<T>) -> Result<BTreeMap<String, ObjectInfo>, Error> {
    let (models, materials) = tobj::load_obj(path)?;

    names.drain(..) // .iter()
        // Extract strings
        .map(|name| name.into())
        // Load object and map Result<ObjectInfo,_> into pair (name, object) to map onto BTreeMap
        .map(|name| load_object(&models, &materials, &name).map(|o| (name, o)) )
        .collect()
}

pub fn load_object(models: &Vec<tobj::Model>, materials: &Vec<tobj::Material>, target_name: &String) -> Result<ObjectInfo, Error> {
    let mut index_offset = 0usize; // Index buffer offset for objects with multiple materials
    let mut obj = ObjectInfo::new(target_name.clone());
    let mut exists = false;

    for m in models.iter() {
        if m.name == *target_name {
            exists = true;
            let mesh = &m.mesh;
            let vert_len = mesh.positions.len() / 3;
            let i_size = vert_len; // mesh.indices.len();
            for i in 0 .. vert_len {
                let mut vertex = VertexInfo::default();
                vertex.pos  = [ mesh.positions[i*3], mesh.positions[i*3+1], mesh.positions[i*3+2] ];
                vertex.norm = [ mesh.normals  [i*3], mesh.normals  [i*3+1], mesh.normals  [i*3+2] ];
                vertex.uv   = [ mesh.texcoords[i*2], mesh.texcoords[i*2+1] ];
                obj.vertices.push(vertex);
            }
            // Get vec of indices
            let mut indices = mesh.indices.iter()
                .map(|x| x + index_offset as u32)
                .collect::<Vec<u32>>();

            // Append to obj
            for i in indices.iter().cloned() { obj.indices.push(i) }

            obj.materials.push(MaterialSlice::WithIndices {
                material: {
                    let mut material = MaterialInfo::empty();
                    if m.mesh.material_id.is_some() {
                        let mat = &materials[m.mesh.material_id.unwrap()];
                        // Alpha
                        material.dissolve = mat.dissolve;

                        // Shadow test
                        material.cast_shadow = mat.illumination_model.unwrap_or(2) != 9;

                        // Diffuse color
                        if mat.diffuse_texture.is_empty() {
                            material.diffuse_color = mat.diffuse;
                        } else {
                            material.diffuse_tex = Some(mat.diffuse_texture.clone());
                        }

                    }
                    material
                },
                indices
            });

            index_offset += i_size;
        }
    }

    if !exists {
        Err(NoObjectForName(target_name.clone()))
    } else {
        obj.minimize();
        Ok(obj)
    }
}