//! [`Array`]: an owned `mlx_array`, and the ops Ollaya's encoders use.
//!
//! Every op builds a lazy graph node on the calling thread's GPU stream (see [`crate::Worker`]);
//! nothing runs until [`Array::eval`] or a read such as [`Array::to_f32_vec`].

use std::ffi::{CString, c_int};

use crate::{Error, Result, check, init, sys, with_stream};

/// Element types used by Ollaya's models.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    Bool,
    Int32,
    Float16,
    BFloat16,
    Float32,
}

impl Dtype {
    fn raw(self) -> sys::mlx_dtype {
        match self {
            Dtype::Bool => sys::mlx_dtype_::MLX_BOOL,
            Dtype::Int32 => sys::mlx_dtype_::MLX_INT32,
            Dtype::Float16 => sys::mlx_dtype_::MLX_FLOAT16,
            Dtype::BFloat16 => sys::mlx_dtype_::MLX_BFLOAT16,
            Dtype::Float32 => sys::mlx_dtype_::MLX_FLOAT32,
        }
    }

    fn from_raw(raw: sys::mlx_dtype) -> Option<Dtype> {
        Some(match raw {
            sys::mlx_dtype_::MLX_BOOL => Dtype::Bool,
            sys::mlx_dtype_::MLX_INT32 => Dtype::Int32,
            sys::mlx_dtype_::MLX_FLOAT16 => Dtype::Float16,
            sys::mlx_dtype_::MLX_BFLOAT16 => Dtype::BFloat16,
            sys::mlx_dtype_::MLX_FLOAT32 => Dtype::Float32,
            _ => return None,
        })
    }

    /// Bytes per element.
    pub fn size(self) -> usize {
        match self {
            Dtype::Bool => 1,
            Dtype::Float16 | Dtype::BFloat16 => 2,
            Dtype::Int32 | Dtype::Float32 => 4,
        }
    }
}

/// Rust types that map to an MLX element type one to one.
pub trait Element: Copy {
    const DTYPE: Dtype;
}

impl Element for f32 {
    const DTYPE: Dtype = Dtype::Float32;
}
impl Element for i32 {
    const DTYPE: Dtype = Dtype::Int32;
}
impl Element for bool {
    const DTYPE: Dtype = Dtype::Bool;
}

/// An owned MLX array handle. Cloning shares the underlying buffer (MLX arrays are immutable).
pub struct Array {
    raw: sys::mlx_array,
}

impl Drop for Array {
    fn drop(&mut self) {
        // SAFETY: the handle is owned by this value and freed once.
        unsafe { sys::mlx_array_free(self.raw) };
    }
}

impl Clone for Array {
    fn clone(&self) -> Self {
        let mut out = Array::out();
        // SAFETY: both handles are valid; `set` takes a new reference to the same array.
        unsafe { sys::mlx_array_set(&mut out.raw, self.raw) };
        out
    }
}

impl std::fmt::Debug for Array {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Array({:?}, {:?})", self.shape(), self.dtype())
    }
}

fn elements(shape: &[i32]) -> Result<usize> {
    shape.iter().try_fold(1usize, |n, &d| {
        usize::try_from(d)
            .ok()
            .and_then(|d| n.checked_mul(d))
            .ok_or_else(|| Error(format!("bad shape {shape:?}")))
    })
}

impl Array {
    /// An empty handle for an op to write its result into.
    fn out() -> Array {
        init();
        // SAFETY: returns a new, owned, empty handle.
        Array {
            raw: unsafe { sys::mlx_array_new() },
        }
    }

    /// A null handle: mlx-c's "no array" for optional arguments.
    fn null() -> sys::mlx_array {
        sys::mlx_array {
            ctx: std::ptr::null_mut(),
        }
    }

    /// An array holding a copy of `data`, row-major with `shape`.
    pub fn from_slice<T: Element>(data: &[T], shape: &[i32]) -> Result<Array> {
        if elements(shape)? != data.len() {
            return Err(Error(format!(
                "{} elements for shape {shape:?}",
                data.len()
            )));
        }
        init();
        // SAFETY: `data` holds exactly product(shape) elements of `T::DTYPE`; MLX copies them.
        let raw = unsafe {
            sys::mlx_array_new_data(
                data.as_ptr().cast(),
                shape.as_ptr(),
                shape.len() as c_int,
                T::DTYPE.raw(),
            )
        };
        Ok(Array { raw })
    }

    /// An array holding a copy of little-endian `bytes` of `dtype`, row-major with `shape`.
    pub fn from_bytes(bytes: &[u8], shape: &[i32], dtype: Dtype) -> Result<Array> {
        if elements(shape)? * dtype.size() != bytes.len() {
            return Err(Error(format!(
                "{} bytes for shape {shape:?} of {dtype:?}",
                bytes.len()
            )));
        }
        init();
        // SAFETY: `bytes` holds exactly product(shape) elements of `dtype`; MLX copies them.
        let raw = unsafe {
            sys::mlx_array_new_data(
                bytes.as_ptr().cast(),
                shape.as_ptr(),
                shape.len() as c_int,
                dtype.raw(),
            )
        };
        Ok(Array { raw })
    }

    /// A float32 scalar.
    pub fn scalar(v: f32) -> Array {
        init();
        // SAFETY: returns a new, owned handle.
        Array {
            raw: unsafe { sys::mlx_array_new_float32(v) },
        }
    }

    pub fn shape(&self) -> Vec<i32> {
        // SAFETY: the handle is valid; the shape pointer holds `ndim` entries while it lives.
        unsafe {
            let n = sys::mlx_array_ndim(self.raw);
            if n == 0 {
                return Vec::new();
            }
            std::slice::from_raw_parts(sys::mlx_array_shape(self.raw), n).to_vec()
        }
    }

    pub fn dtype(&self) -> Option<Dtype> {
        // SAFETY: the handle is valid.
        Dtype::from_raw(unsafe { sys::mlx_array_dtype(self.raw) })
    }

    /// Run the graph that produces this array.
    pub fn eval(&self) -> Result<()> {
        // SAFETY: the handle is valid.
        check(unsafe { sys::mlx_array_eval(self.raw) }, "eval")
    }

    /// Evaluate and copy out a float32 array, row-major.
    pub fn to_f32_vec(&self) -> Result<Vec<f32>> {
        if self.dtype() != Some(Dtype::Float32) {
            return Err(Error(format!("to_f32_vec on {:?}", self.dtype())));
        }
        let dense = self.contiguous()?;
        dense.eval()?;
        let n = elements(&dense.shape())?;
        // SAFETY: an evaluated, row-contiguous float32 array of `n` elements.
        let data = unsafe { sys::mlx_array_data_float32(dense.raw) };
        if data.is_null() {
            return Err(Error("no data after eval".into()));
        }
        Ok(unsafe { std::slice::from_raw_parts(data, n) }.to_vec())
    }

    fn contiguous(&self) -> Result<Array> {
        op(
            |res, s| unsafe { sys::mlx_contiguous(res, self.raw, false, s) },
            "contiguous",
        )
    }
}

/// Run one mlx-c op that writes into `res` on the thread's stream.
fn op(f: impl FnOnce(*mut sys::mlx_array, sys::mlx_stream) -> c_int, what: &str) -> Result<Array> {
    let mut out = Array::out();
    with_stream(|s| check(f(&mut out.raw, s), what))?;
    Ok(out)
}

macro_rules! unary {
    ($($name:ident => $f:ident),* $(,)?) => {
        impl Array {
            $(
                pub fn $name(&self) -> Result<Array> {
                    // SAFETY: valid handles; mlx-c writes a new array into `res`.
                    op(|res, s| unsafe { sys::$f(res, self.raw, s) }, stringify!($name))
                }
            )*
        }
    };
}

macro_rules! binary {
    ($($name:ident => $f:ident),* $(,)?) => {
        impl Array {
            $(
                pub fn $name(&self, other: &Array) -> Result<Array> {
                    // SAFETY: valid handles; mlx-c writes a new array into `res`.
                    op(|res, s| unsafe { sys::$f(res, self.raw, other.raw, s) }, stringify!($name))
                }
            )*
        }
    };
}

unary! {
    erf => mlx_erf,
    exp => mlx_exp,
    log => mlx_log,
    abs => mlx_abs,
    logical_not => mlx_logical_not,
}

binary! {
    add => mlx_add,
    subtract => mlx_subtract,
    multiply => mlx_multiply,
    divide => mlx_divide,
    matmul => mlx_matmul,
    maximum => mlx_maximum,
    minimum => mlx_minimum,
    logical_and => mlx_logical_and,
    logical_or => mlx_logical_or,
    equal => mlx_equal,
    greater_equal => mlx_greater_equal,
    less_equal => mlx_less_equal,
}

impl Array {
    pub fn reshape(&self, shape: &[i32]) -> Result<Array> {
        // SAFETY: valid handle; `shape` outlives the call.
        op(
            |res, s| unsafe { sys::mlx_reshape(res, self.raw, shape.as_ptr(), shape.len(), s) },
            "reshape",
        )
    }

    pub fn transpose(&self, axes: &[i32]) -> Result<Array> {
        // SAFETY: valid handle; `axes` outlives the call.
        op(
            |res, s| unsafe {
                sys::mlx_transpose_axes(res, self.raw, axes.as_ptr(), axes.len(), s)
            },
            "transpose",
        )
    }

    pub fn astype(&self, dtype: Dtype) -> Result<Array> {
        // SAFETY: valid handle.
        op(
            |res, s| unsafe { sys::mlx_astype(res, self.raw, dtype.raw(), s) },
            "astype",
        )
    }

    pub fn broadcast_to(&self, shape: &[i32]) -> Result<Array> {
        // SAFETY: valid handle; `shape` outlives the call.
        op(
            |res, s| unsafe {
                sys::mlx_broadcast_to(res, self.raw, shape.as_ptr(), shape.len(), s)
            },
            "broadcast_to",
        )
    }

    pub fn expand_dims(&self, axis: i32) -> Result<Array> {
        // SAFETY: valid handle.
        op(
            |res, s| unsafe { sys::mlx_expand_dims(res, self.raw, axis, s) },
            "expand_dims",
        )
    }

    pub fn squeeze(&self, axis: i32) -> Result<Array> {
        // SAFETY: valid handle.
        op(
            |res, s| unsafe { sys::mlx_squeeze_axis(res, self.raw, axis, s) },
            "squeeze",
        )
    }

    /// Gather whole slices along `axis` (an embedding lookup with `axis` 0).
    pub fn take(&self, indices: &Array, axis: i32) -> Result<Array> {
        // SAFETY: valid handles.
        op(
            |res, s| unsafe { sys::mlx_take_axis(res, self.raw, indices.raw, axis, s) },
            "take",
        )
    }

    pub fn take_along_axis(&self, indices: &Array, axis: i32) -> Result<Array> {
        // SAFETY: valid handles.
        op(
            |res, s| unsafe { sys::mlx_take_along_axis(res, self.raw, indices.raw, axis, s) },
            "take_along_axis",
        )
    }

    /// `self[start[0]..stop[0], start[1]..stop[1], ...]`, step 1.
    pub fn slice(&self, start: &[i32], stop: &[i32]) -> Result<Array> {
        let strides = vec![1; start.len()];
        // SAFETY: valid handle; the three slices outlive the call.
        op(
            |res, s| unsafe {
                sys::mlx_slice(
                    res,
                    self.raw,
                    start.as_ptr(),
                    start.len(),
                    stop.as_ptr(),
                    stop.len(),
                    strides.as_ptr(),
                    strides.len(),
                    s,
                )
            },
            "slice",
        )
    }

    pub fn sum(&self, axis: i32, keepdims: bool) -> Result<Array> {
        // SAFETY: valid handle.
        op(
            |res, s| unsafe { sys::mlx_sum_axis(res, self.raw, axis, keepdims, s) },
            "sum",
        )
    }

    /// Softmax along `axis`, accumulated in float32.
    pub fn softmax(&self, axis: i32) -> Result<Array> {
        // SAFETY: valid handle.
        op(
            |res, s| unsafe { sys::mlx_softmax_axis(res, self.raw, axis, true, s) },
            "softmax",
        )
    }

    /// Ascending sort along `axis`.
    pub fn sort(&self, axis: i32) -> Result<Array> {
        // SAFETY: valid handle.
        op(
            |res, s| unsafe { sys::mlx_sort_axis(res, self.raw, axis, s) },
            "sort",
        )
    }

    /// `n` equal parts along `axis`.
    pub fn split(&self, n: i32, axis: i32) -> Result<Vec<Array>> {
        init();
        // SAFETY: a new, owned vector handle, freed below.
        let mut parts = unsafe { sys::mlx_vector_array_new() };
        let status =
            with_stream(|s| Ok(unsafe { sys::mlx_split(&mut parts, self.raw, n, axis, s) }));
        let result = status.and_then(|st| check(st, "split")).and_then(|()| {
            // SAFETY: `parts` is valid; each `get` writes a new reference into an owned handle.
            let len = unsafe { sys::mlx_vector_array_size(parts) };
            (0..len)
                .map(|i| {
                    let mut out = Array::out();
                    check(
                        unsafe { sys::mlx_vector_array_get(&mut out.raw, parts, i) },
                        "split",
                    )?;
                    Ok(out)
                })
                .collect()
        });
        // SAFETY: owned above, freed once.
        unsafe { sys::mlx_vector_array_free(parts) };
        result
    }

    /// Join `arrays` along `axis`.
    pub fn concatenate(arrays: &[&Array], axis: i32) -> Result<Array> {
        init();
        let raws: Vec<sys::mlx_array> = arrays.iter().map(|a| a.raw).collect();
        // SAFETY: a new vector holding new references to the arrays, freed below.
        let vec = unsafe { sys::mlx_vector_array_new_data(raws.as_ptr(), raws.len()) };
        let out = op(
            |res, s| unsafe { sys::mlx_concatenate_axis(res, vec, axis, s) },
            "concatenate",
        );
        // SAFETY: owned above, freed once.
        unsafe { sys::mlx_vector_array_free(vec) };
        out
    }

    /// `condition ? x : y`, elementwise with broadcasting.
    pub fn select(condition: &Array, x: &Array, y: &Array) -> Result<Array> {
        // SAFETY: valid handles.
        op(
            |res, s| unsafe { sys::mlx_where(res, condition.raw, x.raw, y.raw, s) },
            "where",
        )
    }

    /// `[start, stop)` with step 1.
    pub fn arange(start: i32, stop: i32, dtype: Dtype) -> Result<Array> {
        // SAFETY: plain values.
        op(
            |res, s| unsafe {
                sys::mlx_arange(res, f64::from(start), f64::from(stop), 1.0, dtype.raw(), s)
            },
            "arange",
        )
    }

    /// Layer normalisation over the last axis, as `torch.nn.LayerNorm`.
    pub fn layer_norm(
        &self,
        weight: Option<&Array>,
        bias: Option<&Array>,
        eps: f32,
    ) -> Result<Array> {
        let w = weight.map_or(Array::null(), |a| a.raw);
        let b = bias.map_or(Array::null(), |a| a.raw);
        // SAFETY: valid or null handles, as mlx-c documents.
        op(
            |res, s| unsafe { sys::mlx_fast_layer_norm(res, self.raw, w, b, eps, s) },
            "layer_norm",
        )
    }

    /// Rotary embedding of the last `dims` features, positions `offset..` along axis -2.
    /// `traditional = false` rotates the two halves (GPT-NeoX and Hugging Face style).
    pub fn rope(
        &self,
        dims: i32,
        traditional: bool,
        base: f32,
        scale: f32,
        offset: i32,
    ) -> Result<Array> {
        let base = sys::mlx_optional_float {
            value: base,
            has_value: true,
        };
        // SAFETY: valid handle; no custom frequencies.
        op(
            |res, s| unsafe {
                sys::mlx_fast_rope(
                    res,
                    self.raw,
                    dims,
                    traditional,
                    base,
                    scale,
                    offset,
                    Array::null(),
                    s,
                )
            },
            "rope",
        )
    }

    /// Fused attention `softmax(q k^T * scale + mask) v` for `[batch, heads, seq, dim]` inputs.
    /// A boolean `mask` keeps `true` positions. Every query row must keep at least one key:
    /// a fully masked row gives NaN.
    pub fn attention(
        q: &Array,
        k: &Array,
        v: &Array,
        scale: f32,
        mask: Option<&Array>,
    ) -> Result<Array> {
        let mode = CString::new(if mask.is_some() { "array" } else { "" }).expect("no NUL");
        let m = mask.map_or(Array::null(), |a| a.raw);
        // SAFETY: valid or null handles; `mode` outlives the call.
        op(
            |res, s| unsafe {
                sys::mlx_fast_scaled_dot_product_attention(
                    res,
                    q.raw,
                    k.raw,
                    v.raw,
                    scale,
                    mode.as_ptr(),
                    m,
                    Array::null(),
                    false,
                    s,
                )
            },
            "attention",
        )
    }
}
