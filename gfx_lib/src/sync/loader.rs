

use vulkano::sync::GpuFuture;

use std::{
    ops::{ Deref, DerefMut },
    sync::{ Arc, Mutex },
    time::Duration,
};
use std::sync::atomic::AtomicPtr;
use std::mem::ManuallyDrop;

/// Errors that `Loader<T>` can return
pub enum LoaderError {
    Timeout
}
impl std::error::Error for LoaderError {}
impl std::fmt::Debug for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        match self {
            LoaderError::Timeout => write!(f, "Timeout reached"),
        }
    }
}
impl std::fmt::Display for LoaderError {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> Result<(), std::fmt::Error> {
        (self as &dyn std::fmt::Debug).fmt(f)
    }
}

/// Represents `vulkano::sync::GpuFuture` with resource which this future loads
/// When cloned will represent same loader with cloned value and same ref to `ready` value
#[derive(Clone)]
pub struct Loader<T> {
    obj: Box<Option<T>>,
    ready: Arc<Mutex<bool>>,
}
impl <T> Loader<T> {

    /// Create loader from `GpuFuture`
    pub fn with_gpu_future<F>(obj: T, future: F) -> Self
        where F: GpuFuture + Send + Sync + 'static
    {
        let ready = Arc::new(Mutex::new(false));
        let r = ready.clone();
        rayon::spawn(move || {
            future.flush().unwrap();
            *r.lock().unwrap() = true;
        });
        Self { obj: Box::new(Some(obj)), ready }
    }

    /// Loader with function, ready after closure returns
    pub fn with_closure<F>(mut func: F) -> Self
        where
            F: FnOnce() -> T + Send + 'static,
            T: 'static,
    {

        let ready = Arc::new(Mutex::new(false));
        let r = ready.clone();
        let mut obj = Box::new(None);
        let mut data = Self { obj, ready };

        unsafe {
            let mut ptr = AtomicPtr::new(data.obj.as_mut());
            rayon::spawn(move || {
//                std::thread::sleep(std::time::Duration::from_millis(3_000));
                let value = func();
                let v = &mut *ptr.into_inner();
                *v = Some(value);
                *r.lock().unwrap() = true;
            });
        }

        data
    }


    /// Return true when loading finished, otherwise false
    pub fn is_ready(&self) -> bool { *self.ready.lock().unwrap() }

    /// Await until loading finishes, or return Error if timeout reached
    /// TODO: Timeout currenly does nothing
    pub fn wait(&self, timeout: Option<Duration>) -> Result<(), LoaderError> {
        while !*self.ready.lock().unwrap() {}
        Ok(())
    }

    /// Get ref to underlying data, will block until loading finishes
    pub fn get_ref(&self) -> &T {
        if !self.is_ready() { self.wait(None).unwrap() }
        self.obj.as_ref().as_ref().unwrap()
    }

    /// Unwraps loader, returning underlying data, leaving None
    pub fn take(&mut self) -> T {
        if !self.is_ready() { self.wait(None).unwrap() }
        self.obj.take().unwrap()
    }

    /// Unwraps value and drops loader
    pub fn unwrap(mut self) -> T { self.take() }

}

/// Specials for clone capable
impl <T: Clone> Loader<T> {
    /// Clone current value and return it
    pub fn snapshot(&self) -> Option<T> { self.obj.as_ref().clone() }
}
/// To avoid writing to dealloc'ed memory, wait for loading to finish before actually dropping
impl <T> Drop for Loader<T> { fn drop(&mut self) { self.wait(None).unwrap(); } }

/// Deref for `T` in `Loader<T>`
impl <T> Deref for Loader<T> {
    type Target = T;
    fn deref(&self) -> &T { self.get_ref() }
}

/// Create loader from `GpuFuture`
impl <T, F: GpuFuture + Send + Sync + 'static> From<(T, F)> for Loader<T> {
    fn from(o: (T, F)) -> Self {
        Loader::with_gpu_future(o.0, o.1)
    }
}
