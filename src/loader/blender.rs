
extern crate blend;

use crate::loader::Face;
use cgmath::{vec3, Vector3, InnerSpace};

pub fn load_model_faces<M>(blend: &blend::Blend, model_name: &str, mut face_map: M)
    where M: FnMut(Face)
{
    let mesh = {
        let mut i: Option<blend::Instance> = None;
        for inst in blend.get_by_code(*b"OB") {
            if !inst.is_valid("data") { continue }
            let data = inst.get("data");
            if !(data.code() == *b"ME\0\0") { continue }
            if !(&inst.get("id").get_string("name")[2..] == model_name) { continue }
            i = Some(inst);
            break
        }

        i.expect(format!("No model named \"{}\"", model_name).as_ref())
    };

    let data = mesh.get("data");
    let faces = data.get_iter("mpoly").collect::<Vec<_>>();
    let loops = data.get_iter("mloop").collect::<Vec<_>>();
    let verts = data.get_iter("mvert").collect::<Vec<_>>();
    let uvs = data.get_iter("mloopuv").collect::<Vec<_>>();


    let mut vert_buff = [[0f32; 3]; 3];
    let mut norm_buff = [[0f32; 3]; 3];
    let mut uv_buff = [[0f32; 2]; 3];

    let mut index_count = 0;
    for face in &faces {
        let len = face.get_i32("totloop");
        let start = face.get_i32("loopstart");
        let mut indexi = 1;

        while indexi < len {
            let mut index;

            for l in 0..3 {
                if (indexi - 1) + l < len {
                    index = start + (indexi - 1) + l;
                } else {
                    index = start;
                }

                let v = loops[index as usize].get_i32("v");
                let vert = &verts[v as usize];

                let co = vert.get_f32_vec("co");
                let no = vert.get_i16_vec("no");
                let uv = uvs[index as usize].get_f32_vec("uv");

                let i = l as usize;
                vert_buff[i][0] = -co[0];
                vert_buff[i][1] = -co[2];
                vert_buff[i][2] = -co[1];

                norm_buff[i][0] = f32::from(-no[0]) / 32767.0;
                norm_buff[i][1] = f32::from(-no[1]) / 32767.0;
                norm_buff[i][2] = f32::from(-no[2]) / 32767.0;

                uv_buff[i][0] = uv[0];
                uv_buff[i][1] = uv[1];

                index_count += 1;
            }


            let face_normal = {
                // Points
                let a = vec3(vert_buff[0][0], vert_buff[0][1], vert_buff[0][2]);
                let b = vec3(vert_buff[1][0], vert_buff[1][1], vert_buff[1][2]);
                let c = vec3(vert_buff[2][0], vert_buff[2][1], vert_buff[2][2]);
                // Find face normal
                Vector3::normalize(Vector3::cross(b - a, c - a))
            };

            // Avg of vertex normals
            let avg_normal = {
                let a = vec3(norm_buff[0][0], norm_buff[0][1], norm_buff[0][2]);
                let b = vec3(norm_buff[1][0], norm_buff[1][1], norm_buff[1][2]);
                let c = vec3(norm_buff[2][0], norm_buff[2][1], norm_buff[2][2]);
                (a + b + c) / 3.0
            };

            // TODO: Spin vertices in right order
            // Reorder if required
//            if Vector3::dot(face_normal, avg_normal) < 0.0 {
//                vert_buff.swap(1, 2);
//                norm_buff.swap(1, 2);
//                uv_buff.swap(1, 2);
//            }
            vert_buff.swap(1, 2);
            norm_buff.swap(1, 2);
            uv_buff.swap(1, 2);

            face_map(Face {
                vert_count: 3,
                vert: &vert_buff,
                norm: &norm_buff,
                uv: &uv_buff
            });

            indexi += 2;
        }
    }

}
