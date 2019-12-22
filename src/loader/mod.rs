use std::ops::Range;

pub mod blender;
pub mod obj;

/// Contain meta information about material,
/// does not do actual loading, just contain information about what to load
#[derive(Default, Debug, Clone)] pub struct MaterialInfo {
    // Comment provide field context for Wavefront Obj format
    // Context Ref: http://paulbourke.net/dataformats/mtl/

    pub illumination_model: u32, // illum_*

    pub ambient_color: [f32; 3], // Ka
    pub ambient_tex: Option<String>, // map_Ka

    pub diffuse_color: [f32; 3], // Kd
    pub diffuse_tex: Option<String>, // map_Kd

    pub specular_color: [f32; 3], // Ks
    pub specular_tex: Option<String>, // map_Ks
    pub shininess: f32, // Ns

    pub emission_color: [f32; 3], // Ke
    pub emission_tex: Option<String>, // map_Ke

    pub transmission_filter: [f32; 3], // Tr

    pub displace_tex: [f32; 3], // disp
    pub displace_pow: f32, // disp

    pub optical_density: f32, // Ni
    pub dissolve: f32, // d aka alpha
}
impl MaterialInfo {
    fn empty() -> Self {
        Self {
            diffuse_color: [1.0, 1.0, 1.0],
            optical_density: 1.0,
            dissolve: 1.0,
            .. Self::default()
        }
    }
}

/// Use when resolving images
#[derive(Debug)] pub enum MaterialImageUsage {
    Diffuse,
}

/// Single vertex, can make up mesh face or used in index buffer
#[derive(Default, Debug, Clone, PartialEq)] pub struct VertexInfo {
    pub pos: [f32; 3],
    pub norm: [f32; 3],
    pub uv: [f32; 2],
}
impl VertexInfo {
    pub fn close_enough_to(&self, other: &VertexInfo) -> bool {
        for i in 0 .. 3 {
            if (self.pos[i] - other.pos[i]).abs() > 0.00001 { return false; }
            if (self.norm[i] - other.norm[i]).abs() > 0.00001 { return false; }
            if i <= 2 && (self.norm[i] - other.norm[i]).abs() > 0.00001 { return false; }
        }
        true
    }
}

/// Contains material and associated slice of vertices or indices
#[derive(Debug, Clone)] pub enum MaterialSlice {
    WithVertexSlice {
        material: MaterialInfo,
        vertex_slice: Range<usize>, // Range from vertices Vec, usually unsafe but in this case its okay
    },
    WithIndices {
        material: MaterialInfo,
        indices: Vec<u32>,
    }
}

/// Contains information about object
#[derive(Debug, Clone)] pub struct ObjectInfo {
    // Object name as specified by Obj file
    pub name: String,
    // Contains all mesh vertices
    pub vertices: Vec<VertexInfo>,
    // Indices to draw using mesh vertices, if empty, just use all vertices is series
    pub indices: Vec<u32>,
    // Pairs of material to indices used for this material
    pub materials: Vec<MaterialSlice>
}
impl ObjectInfo {
    pub fn new(name: String) -> Self { Self {
        name,
        vertices: Vec::new(),
        indices: Vec::new(),
        materials: Vec::new()
    } }

    /// Return names of all required textures and usages
    pub fn get_all_textures(&self) -> Vec<(MaterialImageUsage, String)> {
        let mut v = Vec::new();
        for m in self.materials.iter() { match m {
            MaterialSlice::WithVertexSlice { material, ..} |
            MaterialSlice::WithIndices { material, ..} => {
                if material.diffuse_tex.is_some() { v.push(
                    (MaterialImageUsage::Diffuse, material.diffuse_tex.clone().unwrap())
                ) }
            }
        } }
        v
    }

    /// Fold clone enough vertices into one and adjust indices
    pub fn minimize(&mut self) {

        // Create indexes if not present
        if self.indices.is_empty() {
            let mut as_vec = (0 .. self.vertices.len())
                .map(|x| x as u32)
                .collect::<Vec<_>>();
            self.indices.append(&mut as_vec);
        }

        // Also remap material to use generated indices
        self.materials = self.materials.drain(..).map(|mat| {
            match mat {
                MaterialSlice::WithVertexSlice { vertex_slice, material } => {
                    MaterialSlice::WithIndices {
                        material,
                        indices: vertex_slice.into_iter().map(|x| x as u32).collect()
                    }
                }
                _ => mat,
            }
        }).collect();

        /// Remap `idx` index `from` to `to`, assuming from is removed, hence all indexes over `from` will be decremented
        /// `from` > `to`
        fn edit_idx(idx: &mut u32, from: u32, to: u32) {
            if *idx > from { *idx -= 1 }
            else if *idx == from { *idx = to }
//            else { /* Do Nothing */ }
        }

        for i in (1 .. self.vertices.len()).rev() { for j in (0 .. i).rev() {
            // fold onto j if vertices are equal enough
            if self.vertices[i].close_enough_to(&self.vertices[j]) {
                // fold mesh indices
                for idx in self.indices.iter_mut() { edit_idx(idx, i as u32, j as u32) }
                // fold materials
                for m in self.materials.iter_mut() {
                    match m {
                        MaterialSlice::WithIndices { indices, .. } => {
                            for idx in indices.iter_mut() { edit_idx(idx, i as u32, j as u32) }
                        }
                        _ => (),
                    }
                }
                self.vertices.remove(i);
                break;
            }
        } }
    }
}


/// Old
#[derive(Debug, Clone)] pub struct Face {
    pub vert: [[f32; 3]; 3],
    pub norm: [[f32; 3]; 3],
    pub uv: [[f32; 2]; 3],
}