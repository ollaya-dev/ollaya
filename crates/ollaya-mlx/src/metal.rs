//! Starting MLX on the Metal GPU: the environment, the Metal library and a self-check.

use std::ffi::{CStr, CString};
use std::path::{Path, PathBuf};

use crate::{Array, Dtype, Error, Result, check, init, sys};

/// `MLX_ENABLE_TF32` must be `0` before MLX first reads it (once, then cached).
const TF32_VAR: &str = "MLX_ENABLE_TF32";

/// How much freed GPU memory MLX keeps for reuse. Its default is its whole memory limit (1.5
/// times the GPU's recommended working set, at most 95% of RAM), so a runner that once saw a
/// long batch would hold on to that memory. The GPU shares RAM with everything else on a Mac, so a runner keeps 1 GiB:
/// enough for the buffers of the next batch, and it gives the rest back.
pub const CACHE_LIMIT: usize = 1 << 30;

/// What [`crate::Worker::start`] found.
#[derive(Debug, Clone)]
pub struct MetalInfo {
    /// The Metal library MLX loaded its kernels from.
    pub metallib: PathBuf,
    /// The GPU's name, e.g. `Apple M4 Pro`.
    pub device: String,
    /// Its architecture, e.g. `applegpu_g16s`.
    pub architecture: String,
}

/// Turn TF32 off for this process: MLX's default computes some fp32 matmuls with 10 mantissa
/// bits on GPUs that have the M5's neural accelerators, which breaks fp32 parity.
///
/// # Safety
///
/// Changes the process environment, so no other thread may run yet (call it first thing in
/// `main`).
pub unsafe fn disable_tf32() {
    // SAFETY: the caller guarantees the process is still single-threaded.
    unsafe { std::env::set_var(TF32_VAR, "0") };
}

/// Whether this Mac has a Metal GPU MLX can use (Apple silicon). This initialises Metal, so
/// only runner processes call it, never the daemon.
pub fn is_available() -> bool {
    init();
    let mut available = false;
    // SAFETY: writes one bool.
    let status = unsafe { sys::mlx_metal_is_available(&mut available) };
    status == 0 && available
}

/// Point MLX at `metallib`, check the GPU and run one kernel, on the MLX thread.
pub(crate) fn start(metallib: &Path) -> Result<MetalInfo> {
    if std::env::var(TF32_VAR).as_deref() != Ok("0") {
        return Err(Error(format!(
            "{TF32_VAR} must be 0 before MLX starts (see ollaya_mlx::disable_tf32)"
        )));
    }
    if !metallib.is_file() {
        return Err(Error(format!(
            "the MLX Metal library is missing: no {}",
            metallib.display()
        )));
    }
    if !is_available() {
        return Err(Error("no Metal GPU".into()));
    }
    let path = CString::new(metallib.as_os_str().as_encoded_bytes())
        .map_err(|_| Error(format!("bad path {}", metallib.display())))?;
    // SAFETY: MLX copies the path.
    check(
        unsafe { sys::mlx_metal_set_metallib_path(path.as_ptr()) },
        "set the Metal library path",
    )?;
    crate::bind_gpu_stream();
    let mut previous = 0;
    // SAFETY: writes one usize.
    check(
        unsafe { sys::mlx_set_cache_limit(&mut previous, CACHE_LIMIT) },
        "set the MLX cache limit",
    )?;

    // Loads the library and runs two kernels: a broken or mismatched metallib fails here, with
    // MLX's own message, instead of on the first request.
    let a = Array::arange(0, 64, Dtype::Float32)?;
    let b = a.multiply(&Array::scalar(0.5))?.add(&a)?;
    let got = b.to_f32_vec().map_err(|e| {
        Error(format!(
            "the MLX Metal library at {} does not load: {}",
            metallib.display(),
            e.0
        ))
    })?;
    if got.iter().enumerate().any(|(i, &v)| v != 1.5 * i as f32) {
        return Err(Error(format!(
            "the MLX self-check computed wrong values with {}",
            metallib.display()
        )));
    }
    let (device, architecture) = gpu_names()?;
    Ok(MetalInfo {
        metallib: metallib.to_path_buf(),
        device,
        architecture,
    })
}

fn gpu_names() -> Result<(String, String)> {
    // SAFETY: new owned handles, freed below; the strings are copied before the info is freed.
    unsafe {
        let dev = sys::mlx_device_new_type(sys::mlx_device_type_::MLX_GPU, 0);
        let mut info = sys::mlx_device_info_new();
        let status = sys::mlx_device_info_get(&mut info, dev);
        let get = |key: &CStr| {
            let mut value: *const std::ffi::c_char = std::ptr::null();
            if sys::mlx_device_info_get_string(&mut value, info, key.as_ptr()) == 0
                && !value.is_null()
            {
                CStr::from_ptr(value).to_string_lossy().into_owned()
            } else {
                String::new()
            }
        };
        let names = (get(c"device_name"), get(c"architecture"));
        sys::mlx_device_info_free(info);
        sys::mlx_device_free(dev);
        check(status, "device info")?;
        Ok(names)
    }
}
