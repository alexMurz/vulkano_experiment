
use std::cell::{RefCell};
use std::sync::Arc;
use serializer::Peek;
use std::ops::Mul;

use gfx_lib;

mod game_entry;

fn main() { start(); }

fn start() {
    use gfx_lib::main_processor::{
        settings::{ GameSettings, WindowMode }
    };

    let settings = GameSettings {
//        window_mode: WindowMode::Borderless,
        .. GameSettings::default()
    };

    match gfx_lib::main_processor::start_with_settings_and_listener(
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

















