//! A small safe wrapper over mlx-c, Apple's C API for MLX.
//!
//! * [`Array`] owns one `mlx_array` handle and frees it on drop. Arrays are `!Send`: MLX builds
//!   graphs on one thread, and every op here runs on the MLX thread of a [`Worker`].
//! * Errors: mlx-c's default error handler prints and calls `exit(-1)`. The first use of this
//!   crate installs a handler that keeps the message instead, and every call returns a
//!   [`Result`] carrying it.
//! * Precision: only what Ollaya needs for fp32 parity is exposed. [`Worker::start`] checks that
//!   TF32 is off ([`disable_tf32`]) and that the Metal library loads.
//!
//! Without the `build` feature this crate is empty.

#![cfg(feature = "build")]

mod array;
mod metal;
mod worker;

use std::cell::RefCell;
use std::ffi::{CStr, c_char, c_int, c_void};
use std::sync::Once;

pub use array::{Array, Dtype};
pub use metal::{MetalInfo, disable_tf32, is_available};
pub use worker::Worker;

use ollaya_mlx_sys as sys;

/// Where this build installed `mlx.metallib`; development runs fall back to it.
pub const BUILD_METALLIB: &str = sys::BUILD_METALLIB;

#[derive(Debug, Clone, thiserror::Error)]
#[error("mlx: {0}")]
pub struct Error(pub String);

pub type Result<T> = std::result::Result<T, Error>;

thread_local! {
    /// The message of the last failed mlx-c call on this thread.
    static LAST_ERROR: RefCell<Option<String>> = const { RefCell::new(None) };
    /// The GPU stream of this thread's [`Worker`]; ops outside a worker thread fail.
    static STREAM: RefCell<Option<Stream>> = const { RefCell::new(None) };
}

/// mlx-c calls its handler from inside the failing function, on the calling thread.
unsafe extern "C" fn keep_error(msg: *const c_char, _data: *mut c_void) {
    let msg = if msg.is_null() {
        "unknown error".to_owned()
    } else {
        // SAFETY: mlx-c passes a NUL-terminated message that lives for this call.
        unsafe { CStr::from_ptr(msg) }
            .to_string_lossy()
            .into_owned()
    };
    // `try_with`: never panic across the FFI boundary, even during thread teardown.
    let _ = LAST_ERROR.try_with(|e| *e.borrow_mut() = Some(msg));
}

/// Install the error handler once per process, before the first mlx-c call.
pub(crate) fn init() {
    static INIT: Once = Once::new();
    INIT.call_once(|| {
        // SAFETY: a plain function pointer and no user data.
        unsafe { sys::mlx_set_error_handler(Some(keep_error), std::ptr::null_mut(), None) };
    });
}

/// Turn an mlx-c status into a [`Result`] carrying the handler's message.
pub(crate) fn check(status: c_int, what: &str) -> Result<()> {
    if status == 0 {
        return Ok(());
    }
    let msg = LAST_ERROR
        .with(|e| e.borrow_mut().take())
        .unwrap_or_else(|| format!("failed with status {status}"));
    Err(Error(format!("{what}: {msg}")))
}

/// An owned `mlx_stream` handle.
pub(crate) struct Stream(sys::mlx_stream);

impl Drop for Stream {
    fn drop(&mut self) {
        // SAFETY: the handle is owned and freed once.
        unsafe { sys::mlx_stream_free(self.0) };
    }
}

/// Run `f` with this thread's GPU stream.
pub(crate) fn with_stream<T>(f: impl FnOnce(sys::mlx_stream) -> Result<T>) -> Result<T> {
    STREAM.with(|s| match s.borrow().as_ref() {
        Some(stream) => f(stream.0),
        None => Err(Error(
            "MLX ops run only on the MLX thread (ollaya_mlx::Worker)".into(),
        )),
    })
}

/// Make the default GPU stream this thread's stream (the worker thread, once).
pub(crate) fn bind_gpu_stream() {
    init();
    // SAFETY: returns a new handle to the default GPU stream, owned by `Stream`.
    let stream = Stream(unsafe { sys::mlx_default_gpu_stream_new() });
    STREAM.with(|s| *s.borrow_mut() = Some(stream));
}

/// Release this thread's stream before the thread exits.
pub(crate) fn unbind_stream() {
    let _ = STREAM.try_with(|s| s.borrow_mut().take());
}
