//! The part of llama.cpp's C API the llama engine calls, loaded at run time from the pinned
//! upstream build (`scripts/llama-cpp.sh`, llama.cpp v0.5.0 = release build b11146).
//!
//! The declarations follow `include/llama.h` and `ggml/include/ggml-backend.h` at commit
//! `7fe450e19305b828c199d602c23a8337aaa1f03b`. llama.cpp's structs gain fields between releases, so
//! this file is tied to that build: bump both together, and [`Api::load`] checks the default
//! parameters the library reports against the ones this file expects before anything else runs.

use std::ffi::{c_char, c_int, c_void};
use std::path::Path;

use libloading::Library;

pub type Token = i32;
pub type Pos = i32;
pub type SeqId = i32;
/// `ggml_backend_dev_t`.
pub type Device = *mut c_void;

/// The llama.cpp version this file mirrors (`llama_version()` reports it, with `-dev` for the
/// nightly release builds).
pub const VERSION: &str = "0.5.0";

/// `ggml_backend_dev_type`.
pub const DEVICE_TYPE_GPU: c_int = 1;
pub const DEVICE_TYPE_IGPU: c_int = 2;

/// `enum ggml_log_level`.
pub const LOG_LEVEL_WARN: c_int = 3;
pub const LOG_LEVEL_ERROR: c_int = 4;

pub type LogCallback = unsafe extern "C" fn(level: c_int, text: *const c_char, user: *mut c_void);
/// `llama_tokenize(vocab, text, text_len, tokens, n_tokens_max, add_special, parse_special)`.
pub type TokenizeFn =
    unsafe extern "C" fn(*const c_void, *const c_char, i32, *mut Token, i32, bool, bool) -> i32;
/// `llama_token_to_piece(vocab, token, buf, length, lstrip, special)`.
pub type TokenToPieceFn =
    unsafe extern "C" fn(*const c_void, Token, *mut c_char, i32, i32, bool) -> i32;

/// `struct llama_model_params`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ModelParams {
    /// NULL-terminated list of devices to offload to; NULL means every device.
    pub devices: *mut Device,
    pub tensor_buft_overrides: *const c_void,
    pub n_gpu_layers: i32,
    pub split_mode: c_int,
    pub load_mode: c_int,
    pub lazy_mode: c_int,
    pub main_gpu: i32,
    pub tensor_split: *const f32,
    pub progress_callback: *const c_void,
    pub progress_callback_user_data: *mut c_void,
    pub kv_overrides: *const c_void,
    pub vocab_only: bool,
    pub check_tensors: bool,
    pub use_extra_bufts: bool,
    pub no_host: bool,
    pub no_alloc: bool,
    pub load_mtp: bool,
}

/// `struct llama_context_params`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct ContextParams {
    pub n_ctx: u32,
    pub n_batch: u32,
    pub n_ubatch: u32,
    pub n_seq_max: u32,
    pub n_rs_seq: u32,
    pub n_outputs_max: u32,
    pub n_outputs_max_per_seq: u32,
    pub n_threads: i32,
    pub n_threads_batch: i32,
    pub ctx_type: c_int,
    pub rope_scaling_type: c_int,
    pub pooling_type: c_int,
    pub attention_type: c_int,
    pub flash_attn_type: c_int,
    pub rope_freq_base: f32,
    pub rope_freq_scale: f32,
    pub yarn_ext_factor: f32,
    pub yarn_attn_factor: f32,
    pub yarn_beta_fast: f32,
    pub yarn_beta_slow: f32,
    pub yarn_orig_ctx: u32,
    pub defrag_thold: f32,
    pub cb_eval: *const c_void,
    pub cb_eval_user_data: *mut c_void,
    pub type_k: c_int,
    pub type_v: c_int,
    pub abort_callback: *const c_void,
    pub abort_callback_data: *mut c_void,
    pub embeddings: bool,
    pub offload_kqv: bool,
    pub no_perf: bool,
    pub op_offload: bool,
    pub swa_full: bool,
    pub kv_unified: bool,
    pub samplers: *mut c_void,
    pub n_samplers: usize,
    pub ctx_other: *mut c_void,
}

/// `struct llama_batch`: arrays owned by the caller, `n_tokens` long.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Batch {
    pub n_tokens: i32,
    pub token: *mut Token,
    pub embd: *mut f32,
    pub pos: *mut Pos,
    pub n_seq_id: *mut i32,
    pub seq_id: *mut *mut SeqId,
    pub logits: *mut i8,
}

/// The loaded libraries and the functions the engine calls. The libraries stay loaded for the
/// life of the process (the runner serves one model), so the function pointers stay valid.
pub struct Api {
    _libs: Vec<Library>,
    pub llama_version: unsafe extern "C" fn() -> *const c_char,
    pub llama_backend_init: unsafe extern "C" fn(),
    pub llama_log_set: unsafe extern "C" fn(LogCallback, *mut c_void),
    pub llama_model_default_params: unsafe extern "C" fn() -> ModelParams,
    pub llama_context_default_params: unsafe extern "C" fn() -> ContextParams,
    pub llama_model_load_from_file: unsafe extern "C" fn(*const c_char, ModelParams) -> *mut c_void,
    pub llama_model_free: unsafe extern "C" fn(*mut c_void),
    pub llama_init_from_model: unsafe extern "C" fn(*mut c_void, ContextParams) -> *mut c_void,
    pub llama_free: unsafe extern "C" fn(*mut c_void),
    pub llama_model_get_vocab: unsafe extern "C" fn(*const c_void) -> *const c_void,
    pub llama_model_desc: unsafe extern "C" fn(*const c_void, *mut c_char, usize) -> i32,
    pub llama_get_memory: unsafe extern "C" fn(*const c_void) -> *mut c_void,
    pub llama_memory_clear: unsafe extern "C" fn(*mut c_void, bool),
    pub llama_memory_seq_rm: unsafe extern "C" fn(*mut c_void, SeqId, Pos, Pos) -> bool,
    pub llama_decode: unsafe extern "C" fn(*mut c_void, Batch) -> i32,
    pub llama_get_logits_ith: unsafe extern "C" fn(*mut c_void, i32) -> *mut f32,
    pub llama_n_ctx: unsafe extern "C" fn(*const c_void) -> u32,
    pub llama_vocab_n_tokens: unsafe extern "C" fn(*const c_void) -> i32,
    pub llama_vocab_bos: unsafe extern "C" fn(*const c_void) -> Token,
    pub llama_vocab_get_add_bos: unsafe extern "C" fn(*const c_void) -> bool,
    pub llama_tokenize: TokenizeFn,
    pub llama_token_to_piece: TokenToPieceFn,
    pub ggml_backend_load: unsafe extern "C" fn(*const c_char) -> *mut c_void,
    pub ggml_backend_load_all_from_path: unsafe extern "C" fn(*const c_char),
    pub ggml_backend_dev_count: unsafe extern "C" fn() -> usize,
    pub ggml_backend_dev_get: unsafe extern "C" fn(usize) -> Device,
    pub ggml_backend_dev_name: unsafe extern "C" fn(Device) -> *const c_char,
    pub ggml_backend_dev_description: unsafe extern "C" fn(Device) -> *const c_char,
    pub ggml_backend_dev_type: unsafe extern "C" fn(Device) -> c_int,
    pub ggml_backend_dev_memory: unsafe extern "C" fn(Device, *mut usize, *mut usize),
}

/// File names of the three libraries, in load order.
#[cfg(target_os = "linux")]
pub const LIBRARIES: [&str; 3] = ["libggml-base.so.0", "libggml.so.0", "libllama.so.0"];
#[cfg(target_os = "macos")]
pub const LIBRARIES: [&str; 3] = [
    "libggml-base.0.dylib",
    "libggml.0.dylib",
    "libllama.0.dylib",
];
#[cfg(windows)]
pub const LIBRARIES: [&str; 3] = ["ggml-base.dll", "ggml.dll", "llama.dll"];

fn open(path: &Path) -> Result<Library, String> {
    // SAFETY: loading llama.cpp runs its library initializers, which only register backends.
    #[cfg(windows)]
    let lib = unsafe {
        // LOAD_WITH_ALTERED_SEARCH_PATH: the library's own dependencies are found next to it.
        libloading::os::windows::Library::load_with_flags(path, 0x0000_0008).map(Library::from)
    };
    #[cfg(not(windows))]
    let lib = unsafe { Library::new(path) };
    lib.map_err(|e| {
        let msg = e.to_string();
        let hint = if msg.contains("libgomp") {
            " (install your distribution's libgomp1 package: GCC's OpenMP runtime)"
        } else {
            ""
        };
        format!("cannot load {}: {msg}{hint}", path.display())
    })
}

impl Api {
    /// Load libggml-base, libggml and libllama from `dir`, resolve every function, and check that
    /// the library is the build this file was written for.
    pub fn load(dir: &Path) -> Result<Api, String> {
        #[cfg(windows)]
        set_dll_directory(dir);
        let libs = LIBRARIES
            .iter()
            .map(|name| open(&dir.join(name)))
            .collect::<Result<Vec<_>, _>>()?;
        let (base, ggml, llama) = (&libs[0], &libs[1], &libs[2]);
        macro_rules! sym {
            ($lib:expr, $name:ident) => {{
                // SAFETY: the type is the declaration in llama.h / ggml-backend.h at the pinned
                // commit; `Api::load` checks the build before any of them is called.
                let s = unsafe { $lib.get(concat!(stringify!($name), "\0").as_bytes()) }
                    .map_err(|e| format!("llama.cpp: no symbol {}: {e}", stringify!($name)))?;
                *s
            }};
        }
        let api = Api {
            llama_version: sym!(llama, llama_version),
            llama_backend_init: sym!(llama, llama_backend_init),
            llama_log_set: sym!(llama, llama_log_set),
            llama_model_default_params: sym!(llama, llama_model_default_params),
            llama_context_default_params: sym!(llama, llama_context_default_params),
            llama_model_load_from_file: sym!(llama, llama_model_load_from_file),
            llama_model_free: sym!(llama, llama_model_free),
            llama_init_from_model: sym!(llama, llama_init_from_model),
            llama_free: sym!(llama, llama_free),
            llama_model_get_vocab: sym!(llama, llama_model_get_vocab),
            llama_model_desc: sym!(llama, llama_model_desc),
            llama_get_memory: sym!(llama, llama_get_memory),
            llama_memory_clear: sym!(llama, llama_memory_clear),
            llama_memory_seq_rm: sym!(llama, llama_memory_seq_rm),
            llama_decode: sym!(llama, llama_decode),
            llama_get_logits_ith: sym!(llama, llama_get_logits_ith),
            llama_n_ctx: sym!(llama, llama_n_ctx),
            llama_vocab_n_tokens: sym!(llama, llama_vocab_n_tokens),
            llama_vocab_bos: sym!(llama, llama_vocab_bos),
            llama_vocab_get_add_bos: sym!(llama, llama_vocab_get_add_bos),
            llama_tokenize: sym!(llama, llama_tokenize),
            llama_token_to_piece: sym!(llama, llama_token_to_piece),
            ggml_backend_load: sym!(ggml, ggml_backend_load),
            ggml_backend_load_all_from_path: sym!(ggml, ggml_backend_load_all_from_path),
            ggml_backend_dev_count: sym!(ggml, ggml_backend_dev_count),
            ggml_backend_dev_get: sym!(ggml, ggml_backend_dev_get),
            ggml_backend_dev_name: sym!(base, ggml_backend_dev_name),
            ggml_backend_dev_description: sym!(base, ggml_backend_dev_description),
            ggml_backend_dev_type: sym!(base, ggml_backend_dev_type),
            ggml_backend_dev_memory: sym!(base, ggml_backend_dev_memory),
            _libs: libs,
        };
        api.check()?;
        Ok(api)
    }

    /// The build must be the one this file mirrors: its version, and its default parameters at
    /// the offsets this file reads them from. A different build could lay the structs out
    /// differently, and passing them by value would then corrupt memory.
    fn check(&self) -> Result<(), String> {
        // SAFETY: llama_version returns a static NUL-terminated string.
        let version = unsafe { cstr((self.llama_version)()) };
        let mismatch = |what: &str| {
            Err(format!(
                "the llama.cpp library in use (version {version}) is not the build this ollaya \
                 was made for ({VERSION}): {what}. Reinstall ollaya"
            ))
        };
        if version.split('-').next() != Some(VERSION) {
            return mismatch("version");
        }
        // SAFETY: both functions take nothing and return their structs by value.
        let (m, c) = unsafe {
            (
                (self.llama_model_default_params)(),
                (self.llama_context_default_params)(),
            )
        };
        let model_ok = m.devices.is_null()
            && m.tensor_buft_overrides.is_null()
            && m.n_gpu_layers == -1
            && m.split_mode == 1
            && m.load_mode == -1
            && m.lazy_mode == 1
            && m.main_gpu == 0
            && m.tensor_split.is_null()
            && m.progress_callback.is_null()
            && m.kv_overrides.is_null()
            && (m.vocab_only, m.check_tensors, m.use_extra_bufts) == (false, false, true)
            && (m.no_host, m.no_alloc, m.load_mtp) == (false, false, false);
        let context_ok = (c.n_ctx, c.n_batch, c.n_ubatch, c.n_seq_max) == (512, 2048, 512, 1)
            && (c.n_rs_seq, c.n_outputs_max, c.n_outputs_max_per_seq) == (0, 0, 1)
            && (c.ctx_type, c.rope_scaling_type, c.pooling_type) == (0, -1, -1)
            && (c.attention_type, c.flash_attn_type) == (-1, -1)
            && c.yarn_ext_factor == -1.0
            && c.defrag_thold == -1.0
            && c.cb_eval.is_null()
            && (c.type_k, c.type_v) == (1, 1)
            && c.abort_callback.is_null()
            && (c.embeddings, c.offload_kqv, c.no_perf) == (false, true, true)
            && (c.op_offload, c.swa_full, c.kv_unified) == (true, true, false)
            && c.samplers.is_null()
            && c.n_samplers == 0
            && c.ctx_other.is_null();
        if !model_ok {
            return mismatch("llama_model_params");
        }
        if !context_ok {
            return mismatch("llama_context_params");
        }
        Ok(())
    }
}

/// A NUL-terminated C string as Rust text (lossy); `""` for NULL.
///
/// # Safety
/// `p` is NULL or points to a NUL-terminated string that outlives the call.
pub unsafe fn cstr(p: *const c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: the caller guarantees a valid NUL-terminated string.
    unsafe { std::ffi::CStr::from_ptr(p) }
        .to_string_lossy()
        .into_owned()
}

/// Let Windows find the libraries' own dependencies (the OpenMP runtime next to the CPU backends)
/// in `dir`: ggml loads its backends with `LoadLibraryW`, which searches the application's
/// directory (the runner's `bin`), not the library's.
#[cfg(windows)]
fn set_dll_directory(dir: &Path) {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn SetDllDirectoryW(path: *const u16) -> i32;
    }
    let wide: Vec<u16> = dir.as_os_str().encode_wide().chain([0]).collect();
    // SAFETY: a NUL-terminated wide string; the runner process serves this one model.
    unsafe {
        SetDllDirectoryW(wide.as_ptr());
    }
}

#[cfg(test)]
mod tests {
    use std::mem::{offset_of, size_of};

    use super::*;

    /// The layouts clang computes for llama.h at the pinned commit on 64-bit targets.
    #[test]
    fn struct_layouts_match_llama_h() {
        assert_eq!(size_of::<ModelParams>(), 80);
        assert_eq!(offset_of!(ModelParams, n_gpu_layers), 16);
        assert_eq!(offset_of!(ModelParams, tensor_split), 40);
        assert_eq!(offset_of!(ModelParams, kv_overrides), 64);
        assert_eq!(offset_of!(ModelParams, vocab_only), 72);
        assert_eq!(offset_of!(ModelParams, load_mtp), 77);
        assert_eq!(size_of::<ContextParams>(), 160);
        assert_eq!(offset_of!(ContextParams, flash_attn_type), 52);
        assert_eq!(offset_of!(ContextParams, cb_eval), 88);
        assert_eq!(offset_of!(ContextParams, type_k), 104);
        assert_eq!(offset_of!(ContextParams, abort_callback), 112);
        assert_eq!(offset_of!(ContextParams, embeddings), 128);
        assert_eq!(offset_of!(ContextParams, kv_unified), 133);
        assert_eq!(offset_of!(ContextParams, samplers), 136);
        assert_eq!(offset_of!(ContextParams, ctx_other), 152);
        assert_eq!(size_of::<Batch>(), 56);
        assert_eq!(offset_of!(Batch, token), 8);
        assert_eq!(offset_of!(Batch, logits), 48);
    }
}
