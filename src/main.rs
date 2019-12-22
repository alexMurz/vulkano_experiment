
extern crate blend;
extern crate rayon;

use std::cell::{RefCell};
use std::sync::Arc;
use serializer::Peek;
use std::ops::Mul;
use crate::graphics::image::atlas::rect_solver::Rect;

mod utils;

mod graphics;
mod sync;
mod game_entry;
mod loader;
mod main_processor;

fn main() {
    start();

//    test_rect_solver();

//    test_wavefront_loader();

//    test_wrapper();
//    test_blend();
//    test_wavefront_obj();
//    test_serializer();
//    test_macro_derive();
}

fn test_rect_solver() {
    use graphics::image::atlas::rect_solver::{ Rect, Solver, SolverError };

    let mut rects = vec![
        Rect::new(0, 1, 3),
        Rect::new(1, 1, 3),
        Rect::new(2, 3, 2),
//        Rect::new(3, 2, 2),
    ];
    let mut solver = Solver::with_params(5, [0, 0], false);
    solver.solve(&mut rects).unwrap();
    dbg!(rects);
}

fn test_wrapper() {
    struct Ty(u8);
    impl Ty {
        pub fn test(&mut self) {
            println!("current value is: {}", self.0);
            self.0 += 1;
        }
    }

    use std::mem::{ ManuallyDrop };

    unsafe fn ref_copy<T>(v: &mut T) -> ManuallyDrop<Box<T>> {
        ManuallyDrop::new(Box::from_raw(v as *mut T))
    }

    let mut ty = Ty(0);
    let mut ty2 = unsafe { ref_copy(&mut ty) };
    let mut ty3 = unsafe { ref_copy(&mut ty) };
    ty.test();
    ty2.test();
    ty3.test();
    ty3.test();
    ty2.test();
    ty.test();
    ty2.test();
    ty3.test();

    fn get() -> i32 { 5 }

}

fn test_wavefront_loader() {
    use std::path::Path;
    let objs = loader::obj::load_objects(
        &Path::new("src/data/test.obj"),
        vec!["Plane"]
    ).unwrap();

    for o in objs.iter() {
        dbg!(o.0.clone());
//        dbg!(o.vertices.len());
//        dbg!(o.indices.len());
//        dbg!(o.vertices.clone());
//        dbg!(o.indices.clone());
//        dbg!(o.materials.clone());
    }

//    dbg!(objs);
}

fn test_wavefront_obj() {
    use tobj;
    use std::path::Path;
    let (models, materials) = tobj::load_obj(
        &Path::new("src/data/test.obj")
    ).unwrap();

    println!("Model Count: {}, Material Count: {}" , models.len(), materials.len());

    for m in models.iter() {
        println!("Model Name: {}", m.name);
        match m.mesh.material_id {
            Some(id) => {
                println!(" - Associated Material: {:?}", materials.get(id));
//              println!(" - Material with id: {}", id)
            },
            None => println!(" - No Material")
        }
    }
}

fn test_blend() {
    let blend = blend::Blend::from_path("src/data/test.blend");
    for inst in blend.get_by_code(*b"OB") {
        if !inst.is_valid("data") { continue }
        let data = inst.get("data");
        let data_code = &data.code()[..2];
        if data_code == *b"ME" {
            println!("Found MEsh")
        } else if data_code == *b"MA" {
            println!("Found MAterial: {:?}", data)
        }
    }
    for inst in blend.get_by_code(*b"MA") {
//        if !inst.is_valid("data") { continue }

        /*
Instance {
    type_name: "Material", fields: {
        "id": FieldTemplate {
            info: Value, type_index: 28, type_name: "ID", data_start: 0, data_len: 152, is_primitive: false
        },
        "adt": FieldTemplate { info: Pointer { indirection_count: 1 },
        type_index: 42,
        type_name: "AnimData",
        data_start: 152,
        data_len: 8,
         is_primitive: false
    },
    "flag": FieldTemplate {
        info: Value,
        type_index: 2,
        type_name: "short",
        data_start: 160,
        data_len: 2,
        is_primitive: true
    },
    "_pad1": FieldTemplate {
        info: ValueArray { len: 2, dimensions: [2] },
        type_index: 0,
        type_name: "char",
        data_start: 162,
        data_len: 2,
        is_primitive: true
    },
    "r": FieldTemplate { info: Value, type_index: 7, type_name: "float", data_start: 164, data_len: 4, is_primitive: true },
    "g": FieldTemplate { info: Value, type_index: 7, type_name: "float", data_start: 168, data_len: 4, is_primitive: true },
    "b": FieldTemplate { info: Value, type_index: 7, type_name: "float", data_start: 172, data_len: 4, is_primitive: true },
    "a": FieldTemplate { info: Value, type_index: 7, type_name: "float", data_start: 176, data_len: 4, is_primitive: true },
    "specr": FieldTemplate { info: Value, type_index: 7, type_name: "float", data_start: 180, data_len: 4, is_primitive: true },
    "specg": FieldTemplate { info: Value, type_index: 7, type_name: "float", data_start: 184, data_len: 4, is_primitive: true },
    "specb": FieldTemplate { info: Value, type_index: 7, type_name: "float", data_start: 188, data_len: 4, is_primitive: true },
    "alpha": FieldTemplate { info: Value, type_index: 7, type_name: "float", data_start: 192, data_len: 4, is_primitive: true },
    "ray_mirror": FieldTemplate { info: Value, type_index: 7, type_name: "float", data_start: 196, data_len: 4, is_primitive: true },
    "spec": FieldTemplate { info: Value, type_index: 7, type_name: "float", data_start: 200, data_len: 4, is_primitive: true },
    "gloss_mir": FieldTemplate { info: Value, type_index: 7, type_name: "float", data_start: 204, data_len: 4, is_primitive: true },
    "roughness": FieldTemplate { info: Value, type_index: 7, type_name: "float", data_start: 208, data_len: 4, is_primitive: true },
    "metallic": FieldTemplate { info: Value, type_index: 7, type_name: "float", data_start: 212, data_len: 4, is_primitive: true },
    "use_nodes": FieldTemplate { info: Value, type_index: 0, type_name: "char", data_start: 216, data_len: 1, is_primitive: true },
    "pr_type": FieldTemplate { info: Value, type_index: 0, type_name: "char", data_start: 217, data_len: 1, is_primitive: true },
    "pr_texture": FieldTemplate { info: Value, type_index: 2, type_name: "short", data_start: 218, data_len: 2, is_primitive: true },
    "pr_flag": FieldTemplate { info: Value, type_index: 2, type_name: "short", data_start: 220, data_len: 2, is_primitive: true },
    "index": FieldTemplate { info: Value, type_index: 2, type_name: "short", data_start: 222, data_len: 2, is_primitive: true },
    "nodetree": FieldTemplate { info: Pointer { indirection_count: 1 }, type_index: 73, type_name: "bNodeTree", data_start: 224, data_len: 8, is_primitive: false },
    "ipo": FieldTemplate { info: Pointer { indirection_count: 1 }, type_index: 39, type_name: "Ipo", data_start: 232, data_len: 8, is_primitive: false },
    "preview": FieldTemplate { info: Pointer { indirection_count: 1 }, type_index: 32, type_name: "PreviewImage", data_start: 240, data_len: 8, is_primitive: false },
    "line_col": FieldTemplate { info: ValueArray { len: 4, dimensions: [4] }, type_index: 7, type_name: "float", data_start: 248, data_len: 16, is_primitive: true },
    "line_priority": FieldTemplate { info: Value, type_index: 2, type_name: "short", data_start: 264, data_len: 2, is_primitive: true },
    "vcol_alpha": FieldTemplate { info: Value, type_index: 2, type_name: "short", data_start: 266, data_len: 2, is_primitive: true },
    "paint_active_slot": FieldTemplate { info: Value, type_index: 2, type_name: "short", data_start: 268, data_len: 2, is_primitive: true },
    "paint_clone_slot": FieldTemplate { info: Value, type_index: 2, type_name: "short", data_start: 270, data_len: 2, is_primitive: true },
    "tot_slots": FieldTemplate { info: Value, type_index: 2, type_name: "short", data_start: 272, data_len: 2, is_primitive: true },
    "_pad2": FieldTemplate { info: ValueArray { len: 2, dimensions: [2] }, type_index: 0, type_name: "char", data_start: 274, data_len: 2, is_primitive: true },
    "alpha_threshold": FieldTemplate { info: Value, type_index: 7, type_name: "float", data_start: 276, data_len: 4, is_primitive: true },
    "refract_depth": FieldTemplate { info: Value, type_index: 7, type_name: "float", data_start: 280, data_len: 4, is_primitive: true },
    "blend_method": FieldTemplate { info: Value, type_index: 0, type_name: "char", data_start: 284, data_len: 1, is_primitive: true },
    "blend_shadow": FieldTemplate { info: Value, type_index: 0, type_name: "char", data_start: 285, data_len: 1, is_primitive: true },
    "blend_flag": FieldTemplate { info: Value, type_index: 0, type_name: "char", data_start: 286, data_len: 1, is_primitive: true },
    "_pad3": FieldTemplate { info: ValueArray { len: 1, dimensions: [1] }, type_index: 0, type_name: "char", data_start: 287, data_len: 1, is_primitive: true },
    "texpaintslot": FieldTemplate { info: Pointer { indirection_count: 1 }, type_index: 77, type_name: "TexPaintSlot", data_start: 288, data_len: 8, is_primitive: false },
    "gpumaterial": FieldTemplate { info: Value, type_index: 14, type_name: "ListBase", data_start: 296, data_len: 16, is_primitive: false },
    "gp_style": FieldTemplate { info: Pointer { indirection_count: 1 }, type_index: 78, type_name: "MaterialGPencilStyle", data_start: 312, data_len: 8, is_primitive: false }} }



        */

        dbg!(inst.get("nodetree").get("nodes").get("last").get("inputs").get("first"));

//        println!("inst: {}", inst.is_valid("id"));
//        println!("inst: {:?}", inst);

//        let a = inst.get_iter("fields");
//        for v in a {
//            println!("v = {:?}", v)
//        }

    }
}

fn test_mutex() {
//use std::sync::{Arc, Mutex};
//    use std::thread;
//    use std::time::Duration;
//
//    let num = Arc::new(Mutex::new(5));
//    // allow `num` to be shared across threads (Arc) and modified
//    // (Mutex) safely without a data race.
//
//    let num_clone = num.clone();
//    // create a cloned reference before moving `num` into the thread.
//
//    thread::spawn(move || {
//        loop {
//            *num.lock().unwrap() += 1;
//            // modify the number.
//            thread::sleep(Duration::from_millis(1_500));
//        }
//    });
//
//    output(num_clone);
//
//
//    fn output(num: Arc<Mutex<i32>>) {
//        loop {
//            println!("{:?}", *num.lock().unwrap());
//            // read the number.
//            //  - lock(): obtains a mutable reference; may fail,
//            //    thus return a Result
//            //  - unwrap(): ignore the error and get the real
//            //    reference / cause panic on error.
//            thread::sleep(Duration::from_secs(1));
//        }
//    }
}

fn start() {
    use main_processor::{
        settings::{ GameSettings, WindowMode }
    };

    let settings = GameSettings {
//        window_mode: WindowMode::Borderless,
        .. GameSettings::default()
    };

    match main_processor::start_with_settings_and_listener(
        settings,
        |frame| { Box::new(game_entry::GameEntry::new(frame)) }
    ) {
        Ok(_) => (),
        Err(err) => println!("Finished with error: {}", err),
    }
}


mod test {

    #[test] fn test_arc() {
        use std::sync::Arc;

        let arc1 = Arc::new(5.0);
        let arc2 = arc1.clone();
        assert_eq!(arc1, arc2)
    }

    #[test] fn test_arc_vec() {
        use std::sync::Arc;

        let a = Arc::new(1.0);
        let b = Arc::new(2.0);
        let c = Arc::new(3.0);

        let mut vec = vec![a.clone(), b.clone(), c.clone()];
        vec.retain(|x| *x != b);
        assert_eq!(vec, vec![a, c]);
    }

    #[test] fn test_address_cmp() {
        use std::sync::Arc;
        use std::cell::RefCell;

        #[derive(Debug)]
        struct Data {}
        // Compare by address
        impl PartialEq for Data {
            fn eq(&self, other: &Data) -> bool { self as *const _ == other as *const _ }
        }

        let a = Arc::new(RefCell::new(Data{}));
        let b = a.clone();
        let c = Arc::new(RefCell::new(Data{}));

        assert_eq!(a, b);
        assert_ne!(a, c);
        assert_ne!(b, c);
    }

    #[test] fn test_obj_holder() {
        struct Holder<T> {
            pub obj: T
        }
        impl <T> Holder<T> {
            fn new(obj: T) -> Self { Self { obj } }
        }
        impl <T> std::ops::Deref for Holder<T> {
            type Target = T;
            fn deref(&self) -> &Self::Target { &self.obj }
        }
        impl <T> std::ops::DerefMut for Holder<T> {
            fn deref_mut(&mut self) -> &mut Self::Target { &mut self.obj }
        }

        let mut obj = Holder::new(5.0);
        *obj *= 3.0;
        assert_eq!(15.0, *obj);
    }
}

















