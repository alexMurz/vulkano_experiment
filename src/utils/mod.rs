

/// Perform some operation *with* value, then drop it
#[inline] pub fn with<T, F, R>(val: T, mut f: F) -> R where F: FnMut(T) -> R { f(val) }

