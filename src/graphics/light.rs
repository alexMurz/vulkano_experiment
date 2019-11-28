
use cgmath;
use cgmath::Matrix4;
use std::rc::Rc;
use std::cell::RefCell;

pub enum LightSourceKind {
    Point
}

pub struct LightSource {
    kind: LightSourceKind,
    dirty: bool, // Then system update required
    pow: f32, // Power
    pos: [f32; 3], // Position
    col: [f32; 3], // Color
}
impl LightSource {
    fn new(kind: LightSourceKind) -> Self {
        Self {
            kind,
            dirty: true,
            pow: 1.0,
            pos: [1.0, 1.0, 1.0],
            col: [1.0, 1.0, 1.0]
        }
    }
    fn pos(&mut self, x: f32, y: f32, z: f32) { self.pos = [x, y, z]; self.dirty = true; }
    fn col(&mut self, r: f32, g: f32, b: f32) { self.col = [r, g, b]; self.dirty = true; }
    fn pow(&mut self, pow: f32) { self.pow = pow; self.dirty = true; }
}

pub trait LightProvider {
    fn create_light(&mut self, kind: LightSourceKind) -> Rc<RefCell<LightSource>>;
}

pub struct LightSystem {
    dirty: bool,
    sources: Vec<Rc<RefCell<LightSource>>>,
}
impl LightSystem {
    pub fn new() -> Self {
        Self {
            dirty: true,
            sources: Vec::new(),
        }
    }

    pub fn update(&mut self) -> bool {
        let mut dirty = self.dirty;
        if !dirty {
            for v in self.sources.iter() {
                if v.borrow().dirty {
                    v.borrow_mut().dirty = false;
                    dirty = true;
                }
            }
        }
        dirty
    }
}
impl LightProvider for LightSystem {
    fn create_light(&mut self, kind: LightSourceKind) -> Rc<RefCell<LightSource>> {
        let l = Rc::new(RefCell::new(LightSource::new(kind)));
        self.sources.push(l.clone());
        l
    }
}
