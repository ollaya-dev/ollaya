//! Image input for vision decision models: decoding, Qwen2-VL's preprocessing and the vision
//! tower's grid tables, value for value as `transformers`' PIL backend computes them
//! (`Qwen2VLImageProcessorPil`, `ollaya_convert.families.decider_vision.ref`).
//!
//! 1. Decode a PNG to 8-bit RGB (alpha dropped, not composited; grayscale replicated; palette
//!    expanded), as `PIL.Image.convert("RGB")` does.
//! 2. `smart_resize`: the nearest size whose sides are multiples of `patch * merge` and whose pixel
//!    count is within [min_pixels, max_pixels].
//! 3. Resize with PIL's bicubic filter in its 8-bit fixed-point form (`libImaging/Resample.c`).
//! 4. Rescale (`x * rescale_factor` in f64, then f32), normalise (`(x - mean) / std` in f32).
//! 5. Cut `patch x patch` patches in merge-block order, each repeated over the temporal pair.

use serde::Deserialize;

/// The image section of a vision model's `decision.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct ImageConfig {
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub min_pixels: usize,
    pub max_pixels: usize,
    pub rescale_factor: f64,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    /// Side of the learned square position table (48 for a 2,304-entry table).
    pub position_table_side: usize,
}

/// An 8-bit RGB image, rows top to bottom, `RGBRGB...`.
#[derive(Debug, Clone, PartialEq)]
pub struct Rgb {
    pub width: usize,
    pub height: usize,
    pub data: Vec<u8>,
}

/// One image ready for the vision graph.
#[derive(Debug, Clone)]
pub struct Patches {
    /// [n, 3 * temporal * patch * patch], in merge-block order.
    pub values: Vec<f32>,
    pub dim: usize,
    /// Patch grid (rows, columns); `n = grid_h * grid_w`.
    pub grid_h: usize,
    pub grid_w: usize,
    /// Bilinear taps into the position table: [n, 4] indices and weights.
    pub pos_idx: Vec<i64>,
    pub pos_w: Vec<f32>,
    /// Each patch's (row, column), [n, 2].
    pub rot_ids: Vec<i64>,
}

impl Patches {
    pub fn count(&self) -> usize {
        self.grid_h * self.grid_w
    }

    /// Visual tokens after the merge.
    pub fn tokens(&self, merge: usize) -> usize {
        self.count() / (merge * merge)
    }
}

/// Why an image cannot be used.
#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    #[error("the image is not a PNG ({0}); this model reads PNG images")]
    Format(String),
    #[error("the image could not be decoded: {0}")]
    Decode(String),
    #[error("the image is {width}x{height}; its aspect ratio must be below 200")]
    Aspect { width: usize, height: usize },
    #[error("{0} images given; this model reads one image per request")]
    Count(usize),
    #[error(
        "the image is {width}x{height}, which the model resizes to {}x{} ({patches} patches); \
         it reads at most {max} patches, so send an image of at most about 1 megapixel (1024x1024)",
        resized.0, resized.1
    )]
    TooLarge {
        width: usize,
        height: usize,
        resized: (usize, usize),
        patches: usize,
        max: usize,
    },
    #[error("this model does not read images")]
    Unsupported,
    #[error("an image must be base64 or a base64 data URL: {0}")]
    Base64(String),
}

/// An image as the API carries it: base64, or a `data:<type>;base64,` URL.
pub fn from_base64(text: &str) -> Result<Vec<u8>, ImageError> {
    use base64::Engine as _;
    let data = match text.strip_prefix("data:") {
        Some(url) => url
            .split_once(";base64,")
            .map(|(_, data)| data)
            .ok_or_else(|| ImageError::Base64("a data URL must be base64-encoded".into()))?,
        None => text,
    };
    let data: String = data.chars().filter(|c| !c.is_ascii_whitespace()).collect();
    base64::engine::general_purpose::STANDARD
        .decode(data)
        .map_err(|e| ImageError::Base64(e.to_string()))
}

/// Encode an RGB image as PNG (warm-up requests and tests).
pub fn encode_png(img: &Rgb) -> Vec<u8> {
    let mut out = Vec::new();
    let mut encoder = png::Encoder::new(&mut out, img.width as u32, img.height as u32);
    encoder.set_color(png::ColorType::Rgb);
    encoder.set_depth(png::BitDepth::Eight);
    let mut writer = encoder.write_header().expect("PNG header to memory");
    writer
        .write_image_data(&img.data)
        .expect("PNG data to memory");
    writer.finish().expect("PNG end to memory");
    out
}

/// Decode a PNG to RGB, the way `PIL.Image.open(...).convert("RGB")` reads it.
pub fn decode(bytes: &[u8]) -> Result<Rgb, ImageError> {
    const MAGIC: &[u8] = b"\x89PNG\r\n\x1a\n";
    if !bytes.starts_with(MAGIC) {
        let kind = if bytes.starts_with(b"\xff\xd8\xff") {
            "JPEG"
        } else {
            "unknown format"
        };
        return Err(ImageError::Format(kind.into()));
    }
    let mut decoder = png::Decoder::new(std::io::Cursor::new(bytes));
    // Palette to RGB, tRNS to alpha, 16 bits to 8 (the high byte, as PIL's RGB;16B unpacker).
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder
        .read_info()
        .map_err(|e| ImageError::Decode(e.to_string()))?;
    let size = reader
        .output_buffer_size()
        .ok_or_else(|| ImageError::Decode("image too large".into()))?;
    let mut buf = vec![0; size];
    let info = reader
        .next_frame(&mut buf)
        .map_err(|e| ImageError::Decode(e.to_string()))?;
    let (w, h) = (info.width as usize, info.height as usize);
    let src = &buf[..info.buffer_size()];
    let channels = match info.color_type {
        png::ColorType::Grayscale => 1,
        png::ColorType::GrayscaleAlpha => 2,
        png::ColorType::Rgb => 3,
        png::ColorType::Rgba => 4,
        png::ColorType::Indexed => return Err(ImageError::Decode("palette not expanded".into())),
    };
    let stride = info.line_size;
    let mut data = Vec::with_capacity(w * h * 3);
    for y in 0..h {
        let row = &src[y * stride..y * stride + w * channels];
        for px in row.chunks_exact(channels) {
            match channels {
                1 | 2 => data.extend([px[0], px[0], px[0]]),
                _ => data.extend(&px[..3]),
            }
        }
    }
    Ok(Rgb {
        width: w,
        height: h,
        data,
    })
}

/// Qwen2-VL's `smart_resize` -> (height, width).
pub fn smart_resize(
    height: usize,
    width: usize,
    factor: usize,
    min_pixels: usize,
    max_pixels: usize,
) -> Result<(usize, usize), ImageError> {
    let (h, w, f) = (height as f64, width as f64, factor as f64);
    if h.max(w) / h.min(w) > 200.0 {
        return Err(ImageError::Aspect { width, height });
    }
    // Python's round() rounds halves to even.
    let mut hb = (h / f).round_ties_even() * f;
    let mut wb = (w / f).round_ties_even() * f;
    if hb * wb > max_pixels as f64 {
        let beta = ((h * w) / max_pixels as f64).sqrt();
        hb = f.max((h / beta / f).floor() * f);
        wb = f.max((w / beta / f).floor() * f);
    } else if hb * wb < min_pixels as f64 {
        let beta = (min_pixels as f64 / (h * w)).sqrt();
        hb = (h * beta / f).ceil() * f;
        wb = (w * beta / f).ceil() * f;
    }
    Ok((hb as usize, wb as usize))
}

/// PIL's bicubic kernel (a = -0.5).
fn bicubic(x: f64) -> f64 {
    const A: f64 = -0.5;
    let x = x.abs();
    if x < 1.0 {
        ((A + 2.0) * x - (A + 3.0)) * x * x + 1.0
    } else if x < 2.0 {
        (((x - 5.0) * x + 8.0) * x - 4.0) * A
    } else {
        0.0
    }
}

const PRECISION_BITS: u32 = 32 - 8 - 2;

/// `precompute_coeffs` + `normalize_coeffs_8bpc`: per output pixel, the first input pixel, the
/// number of taps and the fixed-point weights (`ksize` per pixel).
fn coefficients(in_size: usize, out_size: usize) -> (usize, Vec<(usize, usize)>, Vec<i32>) {
    let scale = in_size as f64 / out_size as f64;
    let filterscale = scale.max(1.0);
    let support = 2.0 * filterscale;
    let ksize = support.ceil() as usize * 2 + 1;
    let mut bounds = Vec::with_capacity(out_size);
    let mut kk = vec![0i32; out_size * ksize];
    let mut k = vec![0f64; ksize];
    for xx in 0..out_size {
        let center = (xx as f64 + 0.5) * scale;
        let ss = 1.0 / filterscale;
        // C's (int) cast truncates toward zero.
        let xmin = ((center - support + 0.5) as i64).max(0) as usize;
        let xmax = ((center + support + 0.5) as i64).min(in_size as i64) as usize - xmin;
        let mut ww = 0.0;
        for (x, kx) in k.iter_mut().enumerate().take(xmax) {
            let w = bicubic((x as f64 + xmin as f64 - center + 0.5) * ss);
            *kx = w;
            ww += w;
        }
        for (x, kx) in k.iter_mut().enumerate() {
            let w = if x < xmax && ww != 0.0 {
                *kx / ww
            } else if x < xmax {
                *kx
            } else {
                0.0
            };
            let fixed = w * f64::from(1u32 << PRECISION_BITS);
            kk[xx * ksize + x] = if w < 0.0 {
                (-0.5 + fixed) as i32
            } else {
                (0.5 + fixed) as i32
            };
        }
        bounds.push((xmin, xmax));
    }
    (ksize, bounds, kk)
}

fn clip8(v: i32) -> u8 {
    if v >= (1 << PRECISION_BITS << 8) {
        255
    } else if v <= 0 {
        0
    } else {
        (v >> PRECISION_BITS) as u8
    }
}

/// `PIL.Image.resize(size, BICUBIC)` for 8-bit RGB: a horizontal pass over the rows the
/// vertical pass needs, then the vertical pass, each rounding to 8 bits.
pub fn resize_bicubic(img: &Rgb, width: usize, height: usize) -> Rgb {
    if img.width == width && img.height == height {
        return img.clone();
    }
    let (ksize_v, mut bounds_v, kk_v) = coefficients(img.height, height);
    let need_h = width != img.width;
    let need_v = height != img.height;
    let mut cur = img.clone();
    if need_h {
        let first = bounds_v[0].0;
        let last = bounds_v[height - 1].0 + bounds_v[height - 1].1;
        for b in &mut bounds_v {
            b.0 -= first;
        }
        let (ksize, bounds, kk) = coefficients(img.width, width);
        let rows = last - first;
        let mut out = vec![0u8; width * rows * 3];
        for y in 0..rows {
            let src = &img.data[(y + first) * img.width * 3..(y + first + 1) * img.width * 3];
            for (xx, &(xmin, n)) in bounds.iter().enumerate() {
                let k = &kk[xx * ksize..xx * ksize + n];
                for c in 0..3 {
                    let mut ss: i32 = 1 << (PRECISION_BITS - 1);
                    for (x, &w) in k.iter().enumerate() {
                        ss = ss.wrapping_add(i32::from(src[(x + xmin) * 3 + c]).wrapping_mul(w));
                    }
                    out[(y * width + xx) * 3 + c] = clip8(ss);
                }
            }
        }
        cur = Rgb {
            width,
            height: rows,
            data: out,
        };
    }
    if need_v {
        let w = cur.width;
        let mut out = vec![0u8; w * height * 3];
        for (yy, &(ymin, n)) in bounds_v.iter().enumerate() {
            let k = &kk_v[yy * ksize_v..yy * ksize_v + n];
            for x in 0..w {
                for c in 0..3 {
                    let mut ss: i32 = 1 << (PRECISION_BITS - 1);
                    for (y, &kw) in k.iter().enumerate() {
                        let v = cur.data[((y + ymin) * w + x) * 3 + c];
                        ss = ss.wrapping_add(i32::from(v).wrapping_mul(kw));
                    }
                    out[(yy * w + x) * 3 + c] = clip8(ss);
                }
            }
        }
        cur = Rgb {
            width: w,
            height,
            data: out,
        };
    }
    cur
}

/// The resized image and everything the vision graph needs for it.
pub fn preprocess(img: &Rgb, cfg: &ImageConfig) -> Result<(Rgb, Patches), ImageError> {
    let (p, m, t) = (cfg.patch_size, cfg.merge_size, cfg.temporal_patch_size);
    let (h, w) = smart_resize(img.height, img.width, p * m, cfg.min_pixels, cfg.max_pixels)?;
    let resized = resize_bicubic(img, w, h);
    let (gh, gw) = (h / p, w / p);
    // Rescale in f64, then f32; normalise in f32.
    let norm = |v: u8, c: usize| {
        let x = (f64::from(v) * cfg.rescale_factor) as f32;
        (x - cfg.image_mean[c]) / cfg.image_std[c]
    };
    let dim = 3 * t * p * p;
    let n = gh * gw;
    let mut values = Vec::with_capacity(n * dim);
    let mut rot_ids = Vec::with_capacity(n * 2);
    let mut pos_idx = Vec::with_capacity(n * 4);
    let mut pos_w = Vec::with_capacity(n * 4);
    let side = cfg.position_table_side;
    // Bilinear taps with align_corners along one axis of `size` patches (f32, as torch computes).
    let taps = |i: usize, size: usize| {
        let src = (i as f32 * (side - 1) as f32) / ((size.max(2) - 1) as f32);
        let floor = src.floor();
        let t0 = (floor as i64).clamp(0, side as i64 - 1);
        let t1 = (floor as i64 + 1).clamp(0, side as i64 - 1);
        let w0 = (1.0 - (src - floor).abs()).max(0.0);
        let w1 = (1.0 - (src - floor - 1.0).abs()).max(0.0);
        ([t0, t1], [w0, w1])
    };
    for bh in 0..gh / m {
        for bw in 0..gw / m {
            for mh in 0..m {
                for mw in 0..m {
                    let (row, col) = (bh * m + mh, bw * m + mw);
                    for c in 0..3 {
                        for _ in 0..t {
                            for py in 0..p {
                                let y = row * p + py;
                                for px in 0..p {
                                    let x = col * p + px;
                                    values.push(norm(resized.data[(y * w + x) * 3 + c], c));
                                }
                            }
                        }
                    }
                    rot_ids.extend([row as i64, col as i64]);
                    let (ht, hw) = taps(row, gh);
                    let (wt, ww) = taps(col, gw);
                    for a in 0..2 {
                        for b in 0..2 {
                            pos_idx.push(ht[a] * side as i64 + wt[b]);
                            pos_w.push(hw[a] * ww[b]);
                        }
                    }
                }
            }
        }
    }
    Ok((
        resized,
        Patches {
            values,
            dim,
            grid_h: gh,
            grid_w: gw,
            pos_idx,
            pos_w,
            rot_ids,
        },
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smart_resize_matches_qwen() {
        // 256x240 rounds 7.5 to 8 (half to even), below the minimum it scales up.
        assert_eq!(
            smart_resize(240, 256, 32, 65536, 16777216).unwrap(),
            (256, 256)
        );
        assert_eq!(
            smart_resize(30, 40, 32, 65536, 16777216).unwrap(),
            (224, 320)
        );
        assert_eq!(
            smart_resize(480, 640, 32, 65536, 16777216).unwrap(),
            (480, 640)
        );
        assert!(smart_resize(1, 500, 32, 65536, 16777216).is_err());
    }

    #[test]
    fn png_round_trips_and_data_urls_decode() {
        let img = Rgb {
            width: 3,
            height: 2,
            data: (0..18).map(|v| v * 10).collect(),
        };
        let png = encode_png(&img);
        assert_eq!(decode(&png).unwrap(), img);
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&png);
        assert_eq!(from_base64(&b64).unwrap(), png);
        let url = format!("data:image/png;base64,{b64}");
        assert_eq!(from_base64(&url).unwrap(), png);
        assert!(from_base64("data:image/png,abc").is_err());
        assert!(matches!(
            decode(b"\xff\xd8\xff\xe0"),
            Err(ImageError::Format(_))
        ));
    }

    #[test]
    fn resize_keeps_a_flat_image_flat() {
        let img = Rgb {
            width: 7,
            height: 5,
            data: [10u8, 200, 30].repeat(35),
        };
        let out = resize_bicubic(&img, 16, 12);
        assert_eq!(out.data, [10u8, 200, 30].repeat(16 * 12));
    }
}
