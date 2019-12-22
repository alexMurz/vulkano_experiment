

/// Perform some operation *with* value, then drop it
#[inline] pub fn with<T, F, R>(val: T, mut f: F) -> R where F: FnMut(T) -> R { f(val) }

/// For numbers
pub trait NextPot {
    fn next_pot(&self) -> Self;
}

impl NextPot for u32 {
    fn next_pot(&self) -> Self {
        let mut i = 0;
        let mut v = 1;
        while v < *self {
            v <<= 1;
            i += 1;
        }
        i
    }
}