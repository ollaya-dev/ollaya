//! Still-image support through libmtmd from the same pinned release as libllama.
//! All operations run under LlamaModel's context mutex. PNG decoding uses Ollaya's
//! existing decoder; no URLs, paths, video subprocesses or audio are accepted.
use super::model_error;
use crate::{Error, vision};
use libloading::Library;
use std::ffi::{CStr, CString, c_char, c_int, c_void};
use std::path::Path;
use std::ptr::NonNull;

// Mirrors mtmd_context_params in the pinned b11146 tools/mtmd/mtmd.h (llama.cpp
// v0.5.0). The load-time llama_version check enforces the libllama version pin;
// libmtmd must be loaded from that same release so this layout matches its ABI.
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    use_gpu: bool,
    device: *mut c_void,
    print_timings: bool,
    n_threads: c_int,
    image_marker: *const c_char,
    media_marker: *const c_char,
    flash_attn_type: c_int,
    warmup: bool,
    image_min_tokens: c_int,
    image_max_tokens: c_int,
    cb_eval: *const c_void,
    cb_eval_user_data: *mut c_void,
    batch_max_tokens: i32,
    progress_callback: *const c_void,
    progress_callback_user_data: *mut c_void,
}
#[repr(C)]
struct Text {
    text: *const c_char,
    text_len: usize,
    add_special: bool,
    parse_special: bool,
}
// mtmd's default logger prints even DEBUG messages (including complete prompts) to
// stderr. Never forward those messages, even when Ollaya's own debug logs are enabled.
unsafe extern "C" fn log(level: c_int, text: *const c_char, user: *mut c_void) {
    if matches!(
        level,
        super::ffi::LOG_LEVEL_WARN | super::ffi::LOG_LEVEL_ERROR
    ) {
        // SAFETY: mtmd supplies the same NUL-terminated ggml log contract as llama.cpp.
        unsafe { super::log(level, text, user) };
    }
}

struct Api {
    _lib: Library,
    defaults: unsafe extern "C" fn() -> Params,
    init: unsafe extern "C" fn(*const c_char, *const c_void, Params) -> *mut c_void,
    free: unsafe extern "C" fn(*mut c_void),
    marker: unsafe extern "C" fn(*const c_void) -> *const c_char,
    supports_vision: unsafe extern "C" fn(*const c_void) -> bool,
    bitmap: unsafe extern "C" fn(u32, u32, *const u8) -> *mut c_void,
    bitmap_free: unsafe extern "C" fn(*mut c_void),
    chunks: unsafe extern "C" fn() -> *mut c_void,
    chunks_free: unsafe extern "C" fn(*mut c_void),
    tokenize: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *const Text,
        *const *const c_void,
        usize,
    ) -> i32,
    positions: unsafe extern "C" fn(*const c_void) -> i32,
    count: unsafe extern "C" fn(*const c_void) -> usize,
    chunk: unsafe extern "C" fn(*const c_void, usize) -> *const c_void,
    chunk_type: unsafe extern "C" fn(*const c_void) -> c_int,
    chunk_tokens: unsafe extern "C" fn(*const c_void) -> usize,
    non_causal: unsafe extern "C" fn(*const c_void, *const c_void) -> bool,
    text_tokens: unsafe extern "C" fn(*const c_void, *mut usize) -> *const i32,
    chunk_positions: unsafe extern "C" fn(*const c_void) -> i32,
    batch_init: unsafe extern "C" fn(*mut c_void) -> *mut c_void,
    batch_free: unsafe extern "C" fn(*mut c_void),
    batch_add: unsafe extern "C" fn(*mut c_void, *const c_void) -> i32,
    batch_encode: unsafe extern "C" fn(*mut c_void) -> i32,
    batch_output: unsafe extern "C" fn(*mut c_void, *const c_void) -> *mut f32,
    decode_image: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *const c_void,
        *mut f32,
        i32,
        i32,
        i32,
        *mut i32,
        *const c_void,
        *mut c_void,
    ) -> i32,
    eval: unsafe extern "C" fn(
        *mut c_void,
        *mut c_void,
        *const c_void,
        i32,
        i32,
        i32,
        bool,
        *mut i32,
    ) -> i32,
}
impl Api {
    fn load(dir: &Path) -> Result<Self, Error> {
        let name = if cfg!(target_os = "windows") {
            "mtmd.dll"
        } else if cfg!(target_os = "macos") {
            "libmtmd.0.dylib"
        } else {
            "libmtmd.so.0"
        };
        // SAFETY: these signatures mirror tools/mtmd/{mtmd,mtmd-helper}.h at
        // 7fe450e19305b828c199d602c23a8337aaa1f03b. Library lives as long as its symbols.
        unsafe {
            let lib = Library::new(dir.join(name)).map_err(|e| {
                model_error(format!(
                    "load {name}: {e}; reinstall llama.cpp libraries with multimodal support"
                ))
            })?;
            macro_rules! symbol {
                ($name:literal) => {
                    *lib.get(concat!($name, "\0").as_bytes())
                        .map_err(|e| model_error(format!("libmtmd: {e}")))?
                };
            }
            // The helper setter configures both helper and encoder loggers. Install it
            // before initialization; the callback itself lives in the runner executable.
            let set_log: unsafe extern "C" fn(super::ffi::LogCallback, *mut c_void) =
                symbol!("mtmd_helper_log_set");
            set_log(log, std::ptr::null_mut());
            Ok(Self {
                defaults: symbol!("mtmd_context_params_default"),
                init: symbol!("mtmd_init_from_file"),
                free: symbol!("mtmd_free"),
                marker: symbol!("mtmd_get_marker"),
                supports_vision: symbol!("mtmd_support_vision"),
                bitmap: symbol!("mtmd_bitmap_init"),
                bitmap_free: symbol!("mtmd_bitmap_free"),
                chunks: symbol!("mtmd_input_chunks_init"),
                chunks_free: symbol!("mtmd_input_chunks_free"),
                tokenize: symbol!("mtmd_tokenize"),
                positions: symbol!("mtmd_helper_get_n_pos"),
                count: symbol!("mtmd_input_chunks_size"),
                chunk: symbol!("mtmd_input_chunks_get"),
                chunk_type: symbol!("mtmd_input_chunk_get_type"),
                chunk_tokens: symbol!("mtmd_input_chunk_get_n_tokens"),
                non_causal: symbol!("mtmd_decode_use_non_causal"),
                batch_init: symbol!("mtmd_batch_init"),
                batch_free: symbol!("mtmd_batch_free"),
                batch_add: symbol!("mtmd_batch_add_chunk"),
                batch_encode: symbol!("mtmd_batch_encode"),
                batch_output: symbol!("mtmd_batch_get_output_embd"),
                decode_image: symbol!("mtmd_helper_decode_image_chunk"),
                text_tokens: symbol!("mtmd_input_chunk_get_tokens_text"),
                chunk_positions: symbol!("mtmd_input_chunk_get_n_pos"),
                eval: symbol!("mtmd_helper_eval_chunk_single"),
                _lib: lib,
            })
        }
    }
}
pub(super) struct Vision {
    api: Api,
    ptr: NonNull<c_void>,
}
impl Drop for Vision {
    fn drop(&mut self) {
        // SAFETY: initialized once, freed before the text model it borrows.
        unsafe {
            (self.api.free)(self.ptr.as_ptr());
        }
    }
}
impl Vision {
    pub(super) fn load(
        dir: &Path,
        projector: &Path,
        model: *const c_void,
        device: super::ffi::Device,
        threads: i32,
    ) -> Result<Self, Error> {
        let api = Api::load(dir)?;
        let path = super::cpath(projector).map_err(Error::Model)?;
        // SAFETY: model and path outlive initialization; returned handle owned here.
        let ptr = unsafe {
            let mut params = (api.defaults)();
            params.use_gpu = !device.is_null();
            params.device = device;
            params.n_threads = threads;
            params.warmup = false;
            NonNull::new((api.init)(path.as_ptr(), model, params))
                .ok_or_else(|| model_error("could not load matching vision projector"))?
        };
        let vision = Self { api, ptr };
        // SAFETY: valid projector handle.
        if !unsafe { (vision.api.supports_vision)(vision.ptr.as_ptr()) } {
            return Err(model_error("projector does not support vision"));
        }
        Ok(vision)
    }
    pub(super) fn prefix(&self, text: &str, images: &[Vec<u8>]) -> Result<Chunks<'_>, Error> {
        if images.len() > 16 {
            return Err(
                ollaya_decision::Error::invalid("GGUF vision accepts at most 16 images").into(),
            );
        }
        let mut bitmaps = Vec::new();
        for bytes in images {
            let image = vision::decode(bytes)?;
            // SAFETY: RGB buffer is width*height*3 bytes; mtmd copies it.
            let ptr = unsafe {
                (self.api.bitmap)(image.width as u32, image.height as u32, image.data.as_ptr())
            };
            bitmaps.push(Bitmap {
                api: &self.api,
                ptr: NonNull::new(ptr)
                    .ok_or_else(|| model_error("mtmd bitmap allocation failed"))?,
            });
        }
        // SAFETY: marker belongs to the live projector and is NUL terminated.
        let marker = unsafe { CStr::from_ptr((self.api.marker)(self.ptr.as_ptr())) }
            .to_str()
            .map_err(|e| model_error(e.to_string()))?;
        let marked = text.replace("<__media__>", marker);
        let text = CString::new(marked).map_err(|e| model_error(e.to_string()))?;
        let input = Text {
            text: text.as_ptr(),
            text_len: text.as_bytes().len(),
            add_special: true,
            parse_special: true,
        };
        let pointers: Vec<*const c_void> = bitmaps
            .iter()
            .map(|b| b.ptr.as_ptr().cast_const())
            .collect();
        // SAFETY: chunks and bitmap handles remain live through tokenization/evaluation.
        let ptr = unsafe {
            NonNull::new((self.api.chunks)())
                .ok_or_else(|| model_error("mtmd chunks allocation failed"))?
        };
        let chunks = Chunks {
            vision: self,
            ptr,
            _bitmaps: bitmaps,
        };
        let rc = unsafe {
            (self.api.tokenize)(
                self.ptr.as_ptr(),
                ptr.as_ptr(),
                &input,
                pointers.as_ptr(),
                pointers.len(),
            )
        };
        if rc != 0 {
            return Err(model_error(format!("image tokenization failed ({rc})")));
        }
        // SAFETY: each chunk belongs to this successfully tokenized list. Non-causal
        // image attention must fit one physical microbatch, as in the author's server.
        unsafe {
            for i in 0..(self.api.count)(ptr.as_ptr()) {
                let chunk = (self.api.chunk)(ptr.as_ptr(), i);
                if (self.api.chunk_type)(chunk) == 1
                    && (self.api.non_causal)(self.ptr.as_ptr(), chunk)
                    && (self.api.chunk_tokens)(chunk) > super::N_UBATCH as usize
                {
                    return Err(ollaya_decision::Error::invalid(
                        "image exceeds the vision microbatch; resize the image",
                    )
                    .into());
                }
            }
        }
        Ok(chunks)
    }
}
struct Bitmap<'a> {
    api: &'a Api,
    ptr: NonNull<c_void>,
}
impl Drop for Bitmap<'_> {
    fn drop(&mut self) {
        // SAFETY: uniquely owned bitmap.
        unsafe {
            (self.api.bitmap_free)(self.ptr.as_ptr());
        }
    }
}
pub(super) struct Chunks<'a> {
    vision: &'a Vision,
    ptr: NonNull<c_void>,
    _bitmaps: Vec<Bitmap<'a>>,
}
impl Drop for Chunks<'_> {
    fn drop(&mut self) {
        // SAFETY: uniquely owned chunks, freed before their bitmaps.
        unsafe {
            (self.vision.api.chunks_free)(self.ptr.as_ptr());
        }
    }
}
impl Chunks<'_> {
    pub(super) fn positions(&self) -> usize {
        // SAFETY: chunks returned by successful tokenization.
        unsafe { (self.vision.api.positions)(self.ptr.as_ptr()).max(0) as usize }
    }
    /// Text after the final image; tokenization must include the question to preserve
    /// whitespace merges at the state/question boundary, just as stock llama-server does.
    pub(super) fn tail(&self) -> Result<(usize, Vec<i32>), Error> {
        // SAFETY: handles belong to this live, tokenized chunk list.
        unsafe {
            let count = (self.vision.api.count)(self.ptr.as_ptr());
            if count == 0 {
                return Err(model_error("empty multimodal prompt"));
            }
            let last = (self.vision.api.chunk)(self.ptr.as_ptr(), count - 1);
            if (self.vision.api.chunk_type)(last) != 0 {
                return Err(model_error("multimodal prompt must end in text"));
            }
            let mut n = 0;
            let tokens = (self.vision.api.text_tokens)(last, &mut n);
            if tokens.is_null() || n == 0 {
                return Err(model_error("empty multimodal text tail"));
            }
            let before = self.positions() - (self.vision.api.chunk_positions)(last) as usize;
            Ok((before, std::slice::from_raw_parts(tokens, n).to_vec()))
        }
    }
    /// Evaluate every chunk preceding the final text tail, once per request.
    pub(super) fn evaluate_head(&self, context: *mut c_void) -> Result<usize, Error> {
        let mut end = 0;
        let api = &self.vision.api;
        let mut batch: Option<VisionBatch<'_>> = None;
        // SAFETY: caller owns context mutex; chunks and batch remain live through decode.
        unsafe {
            let count = (api.count)(self.ptr.as_ptr());
            for i in 0..count.saturating_sub(1) {
                let chunk = (api.chunk)(self.ptr.as_ptr(), i);
                let rc = if (api.chunk_type)(chunk) == 1 {
                    let mut embd = batch.as_ref().map_or(std::ptr::null_mut(), |b| {
                        (api.batch_output)(b.ptr.as_ptr(), chunk)
                    });
                    if embd.is_null() {
                        let ptr = NonNull::new((api.batch_init)(self.vision.ptr.as_ptr()))
                            .ok_or_else(|| model_error("vision batch allocation failed"))?;
                        let next = VisionBatch { api, ptr };
                        if (api.batch_add)(ptr.as_ptr(), chunk) != 0 {
                            return Err(model_error("could not add image to empty vision batch"));
                        }
                        // Match stock llama-server: batch subsequent compatible image chunks,
                        // including across intervening text, until mtmd's batch limit is reached.
                        for j in i + 1..count {
                            let candidate = (api.chunk)(self.ptr.as_ptr(), j);
                            if (api.chunk_type)(candidate) == 1
                                && (api.batch_add)(ptr.as_ptr(), candidate) != 0
                            {
                                break;
                            }
                        }
                        if (api.batch_encode)(ptr.as_ptr()) != 0 {
                            return Err(model_error("vision batch encoding failed"));
                        }
                        embd = (api.batch_output)(ptr.as_ptr(), chunk);
                        batch = Some(next);
                    }
                    if embd.is_null() {
                        return Err(model_error("missing image embeddings"));
                    }
                    (api.decode_image)(
                        self.vision.ptr.as_ptr(),
                        context,
                        chunk,
                        embd,
                        end,
                        0,
                        super::N_BATCH as i32,
                        &mut end,
                        std::ptr::null(),
                        std::ptr::null_mut(),
                    )
                } else {
                    (api.eval)(
                        self.vision.ptr.as_ptr(),
                        context,
                        chunk,
                        end,
                        0,
                        super::N_BATCH as i32,
                        false,
                        &mut end,
                    )
                };
                if rc != 0 {
                    return Err(model_error(format!(
                        "image prefix evaluation failed ({rc})"
                    )));
                }
            }
        }
        let (expected, _) = self.tail()?;
        if end < 0 || end as usize != expected {
            return Err(model_error("image prefix position mismatch"));
        }
        Ok(end as usize)
    }
}

struct VisionBatch<'a> {
    api: &'a Api,
    ptr: NonNull<c_void>,
}
impl Drop for VisionBatch<'_> {
    fn drop(&mut self) {
        // SAFETY: uniquely owned batch; borrowed chunks outlive it.
        unsafe {
            (self.api.batch_free)(self.ptr.as_ptr());
        }
    }
}
