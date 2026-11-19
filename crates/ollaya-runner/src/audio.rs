//! Audio input for models whose projector has an audio encoder (Gemma 4's, through libmtmd).
//!
//! 1. Decode a WAV, FLAC or MP3 file to `f32` samples (symphonia), every channel.
//! 2. Mix to mono: the mean of the channels, as `librosa.to_mono` does.
//! 3. Resample to the encoder's rate (16 kHz for Gemma 4) with a Kaiser-windowed sinc filter.
//!    A clip already at that rate is passed through untouched, so a 16 kHz mono float WAV reaches
//!    the encoder sample for sample. Other rates are close to, not identical with, `librosa.load`'s
//!    soxr resampler.
//!
//! Clips are limited to [`MAX_SECONDS`]; mtmd's own feature extractor does the rest.

use std::f64::consts::PI;

/// Longest clip accepted, in seconds (Gemma 4's audio encoder reads up to 30 s).
pub const MAX_SECONDS: f64 = 30.0;

/// Why a clip cannot be used.
#[derive(Debug, thiserror::Error)]
pub enum AudioError {
    #[error("audio must be base64 or a base64 data URL: {0}")]
    Base64(String),
    #[error("the audio could not be decoded ({0}); send a WAV, FLAC or MP3 file")]
    Decode(String),
    #[error("the audio has no samples")]
    Empty,
    #[error(
        "the audio is {seconds:.1} s long; this model reads at most {max:.0} s, so cut the clip first"
    )]
    TooLong { seconds: f64, max: f64 },
    #[error("this model does not read audio")]
    Unsupported,
}

/// A decoded clip: mono samples at the file's own rate.
#[derive(Debug, Clone, PartialEq)]
pub struct Clip {
    pub rate: u32,
    pub samples: Vec<f32>,
}

impl Clip {
    pub fn seconds(&self) -> f64 {
        self.samples.len() as f64 / self.rate as f64
    }

    /// The samples at `rate` (unchanged when the clip is already at that rate).
    pub fn at(&self, rate: u32) -> Vec<f32> {
        if rate == self.rate {
            return self.samples.clone();
        }
        resample(&self.samples, self.rate, rate)
    }
}

/// A clip as the API carries it: base64, or a `data:<type>;base64,` URL.
pub fn from_base64(text: &str) -> Result<Vec<u8>, AudioError> {
    crate::vision::from_base64(text).map_err(|e| match e {
        crate::vision::ImageError::Base64(reason) => AudioError::Base64(reason),
        other => AudioError::Base64(other.to_string()),
    })
}

/// Decode a WAV, FLAC or MP3 file to a mono clip, at most [`MAX_SECONDS`] long.
pub fn decode(bytes: &[u8]) -> Result<Clip, AudioError> {
    use symphonia::core::audio::SampleBuffer;
    use symphonia::core::codecs::{CODEC_TYPE_NULL, DecoderOptions};
    use symphonia::core::errors::Error as SymError;
    use symphonia::core::formats::FormatOptions;
    use symphonia::core::io::MediaSourceStream;
    use symphonia::core::meta::MetadataOptions;
    use symphonia::core::probe::Hint;

    let err = |e: SymError| AudioError::Decode(e.to_string());
    let source = MediaSourceStream::new(
        Box::new(std::io::Cursor::new(bytes.to_vec())),
        Default::default(),
    );
    let probed = symphonia::default::get_probe()
        .format(
            &Hint::new(),
            source,
            &FormatOptions::default(),
            &MetadataOptions::default(),
        )
        .map_err(err)?;
    let mut format = probed.format;
    let track = format
        .tracks()
        .iter()
        .find(|t| t.codec_params.codec != CODEC_TYPE_NULL)
        .ok_or_else(|| AudioError::Decode("no audio track".into()))?;
    let track_id = track.id;
    let mut decoder = symphonia::default::get_codecs()
        .make(&track.codec_params, &DecoderOptions::default())
        .map_err(err)?;
    let mut rate = track.codec_params.sample_rate;
    let max_frames = |rate: u32| (MAX_SECONDS * rate as f64).floor() as usize;
    let mut samples: Vec<f32> = Vec::new();
    let mut buf: Option<SampleBuffer<f32>> = None;
    loop {
        let packet = match format.next_packet() {
            Ok(p) => p,
            Err(SymError::IoError(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof => break,
            Err(SymError::ResetRequired) => break,
            Err(e) => return Err(err(e)),
        };
        if packet.track_id() != track_id {
            continue;
        }
        let decoded = match decoder.decode(&packet) {
            Ok(d) => d,
            // A damaged frame (common at the end of MP3 files) is skipped, as ffmpeg does.
            Err(SymError::DecodeError(_)) => continue,
            Err(e) => return Err(err(e)),
        };
        let spec = *decoded.spec();
        rate = Some(spec.rate);
        let channels = spec.channels.count().max(1);
        let b = buf.get_or_insert_with(|| SampleBuffer::new(decoded.capacity() as u64, spec));
        if b.capacity() < decoded.capacity() * channels {
            *b = SampleBuffer::new(decoded.capacity() as u64, spec);
        }
        b.copy_interleaved_ref(decoded);
        mix_into(&mut samples, b.samples(), channels);
        // Stop decoding early: anything past the limit is rejected anyway.
        if samples.len() > max_frames(spec.rate) + 1 {
            break;
        }
    }
    let rate = rate.ok_or_else(|| AudioError::Decode("unknown sample rate".into()))?;
    if rate == 0 {
        return Err(AudioError::Decode("sample rate 0".into()));
    }
    if samples.is_empty() {
        return Err(AudioError::Empty);
    }
    let clip = Clip { rate, samples };
    if clip.samples.len() > max_frames(rate) {
        // The exact length is unknown when decoding stopped early; report what is known.
        let seconds = format
            .default_track()
            .and_then(|t| Some(t.codec_params.n_frames? as f64 / rate as f64))
            .unwrap_or(clip.seconds());
        return Err(AudioError::TooLong {
            seconds,
            max: MAX_SECONDS,
        });
    }
    Ok(clip)
}

/// Append interleaved frames as their channel mean.
fn mix_into(out: &mut Vec<f32>, interleaved: &[f32], channels: usize) {
    if channels == 1 {
        out.extend_from_slice(interleaved);
        return;
    }
    out.extend(
        interleaved
            .chunks_exact(channels)
            .map(|f| f.iter().sum::<f32>() / channels as f32),
    );
}

/// Zero crossings of the sinc on each side of an output sample (at the lower of the two rates).
const HALF_WIDTH: f64 = 24.0;
/// Kaiser window shape: about 90 dB of stop-band attenuation.
const BETA: f64 = 9.0;
/// Pass band edge as a fraction of the lower Nyquist frequency.
const ROLLOFF: f64 = 0.95;

/// Band-limited resampling with a Kaiser-windowed sinc, evaluated per output sample in `f64`.
/// Samples outside the clip count as silence.
pub fn resample(x: &[f32], from: u32, to: u32) -> Vec<f32> {
    if x.is_empty() || from == to {
        return x.to_vec();
    }
    let ratio = to as f64 / from as f64;
    // Low-pass at the lower Nyquist; the filter is stretched over input samples when downsampling.
    let cutoff = ROLLOFF * ratio.min(1.0);
    let half = HALF_WIDTH / cutoff;
    let n_out = ((x.len() as f64) * ratio).round() as usize;
    let i0_beta = bessel_i0(BETA);
    let mut out = Vec::with_capacity(n_out);
    for n in 0..n_out {
        let t = n as f64 / ratio;
        let lo = (t - half).ceil().max(0.0) as usize;
        let hi = ((t + half).floor() as usize).min(x.len() - 1);
        let mut acc = 0.0f64;
        for (k, &xk) in x.iter().enumerate().take(hi + 1).skip(lo) {
            let d = t - k as f64;
            let w = 1.0 - (d / half).powi(2);
            if w <= 0.0 {
                continue;
            }
            let window = bessel_i0(BETA * w.sqrt()) / i0_beta;
            acc += xk as f64 * cutoff * sinc(cutoff * d) * window;
        }
        out.push(acc as f32);
    }
    out
}

fn sinc(x: f64) -> f64 {
    if x.abs() < 1e-12 {
        1.0
    } else {
        (PI * x).sin() / (PI * x)
    }
}

/// The modified Bessel function of the first kind, order 0 (its power series).
fn bessel_i0(x: f64) -> f64 {
    let (mut sum, mut term, q) = (1.0, 1.0, x * x / 4.0);
    for k in 1..64 {
        term *= q / (k * k) as f64;
        sum += term;
        if term < sum * 1e-17 {
            break;
        }
    }
    sum
}

/// Encode mono `f32` samples as a 32-bit float WAV file (tests and warm-up requests).
pub fn encode_wav(rate: u32, samples: &[f32]) -> Vec<u8> {
    let data = (samples.len() * 4) as u32;
    let mut out = Vec::with_capacity(44 + data as usize);
    out.extend_from_slice(b"RIFF");
    out.extend_from_slice(&(36 + data).to_le_bytes());
    out.extend_from_slice(b"WAVEfmt ");
    out.extend_from_slice(&16u32.to_le_bytes());
    out.extend_from_slice(&3u16.to_le_bytes()); // IEEE float
    out.extend_from_slice(&1u16.to_le_bytes()); // mono
    out.extend_from_slice(&rate.to_le_bytes());
    out.extend_from_slice(&(rate * 4).to_le_bytes());
    out.extend_from_slice(&4u16.to_le_bytes());
    out.extend_from_slice(&32u16.to_le_bytes());
    out.extend_from_slice(b"data");
    out.extend_from_slice(&data.to_le_bytes());
    for s in samples {
        out.extend_from_slice(&s.to_le_bytes());
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tone(rate: u32, hz: f64, seconds: f64) -> Vec<f32> {
        (0..(rate as f64 * seconds) as usize)
            .map(|i| (2.0 * PI * hz * i as f64 / rate as f64).sin() as f32 * 0.5)
            .collect()
    }

    fn pcm16_wav(rate: u32, channels: u16, frames: &[i16]) -> Vec<u8> {
        let data = (frames.len() * 2) as u32;
        let mut out = Vec::new();
        out.extend_from_slice(b"RIFF");
        out.extend_from_slice(&(36 + data).to_le_bytes());
        out.extend_from_slice(b"WAVEfmt ");
        out.extend_from_slice(&16u32.to_le_bytes());
        out.extend_from_slice(&1u16.to_le_bytes());
        out.extend_from_slice(&channels.to_le_bytes());
        out.extend_from_slice(&rate.to_le_bytes());
        out.extend_from_slice(&(rate * 2 * channels as u32).to_le_bytes());
        out.extend_from_slice(&(2 * channels).to_le_bytes());
        out.extend_from_slice(&16u16.to_le_bytes());
        out.extend_from_slice(b"data");
        out.extend_from_slice(&data.to_le_bytes());
        for s in frames {
            out.extend_from_slice(&s.to_le_bytes());
        }
        out
    }

    #[test]
    fn float_wav_at_16k_round_trips_sample_for_sample() {
        let x = tone(16_000, 440.0, 0.5);
        let clip = decode(&encode_wav(16_000, &x)).unwrap();
        assert_eq!(clip.rate, 16_000);
        assert_eq!(clip.samples, x);
        assert_eq!(clip.at(16_000), x);
    }

    #[test]
    fn stereo_pcm_is_mixed_to_the_channel_mean() {
        // Frames (L, R): (1000, 3000), (-2000, 0).
        let clip = decode(&pcm16_wav(8_000, 2, &[1000, 3000, -2000, 0])).unwrap();
        assert_eq!(clip.rate, 8_000);
        let want = [2000.0 / 32768.0, -1000.0 / 32768.0];
        for (a, b) in clip.samples.iter().zip(want) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn clips_over_thirty_seconds_are_rejected() {
        let x = vec![0.0f32; 8_000 * 31];
        match decode(&encode_wav(8_000, &x)) {
            Err(AudioError::TooLong { seconds, .. }) => assert!((seconds - 31.0).abs() < 0.1),
            other => panic!("{other:?}"),
        }
        assert!(decode(&encode_wav(8_000, &vec![0.0f32; 8_000 * 30])).is_ok());
    }

    /// The dominant frequency of `x` in Hz, from zero crossings in its middle half (MP3 encoders
    /// pad both ends with silence).
    fn pitch(x: &[f32], rate: u32) -> f64 {
        let x = &x[x.len() / 4..x.len() * 3 / 4];
        let crossings = x.windows(2).filter(|w| w[0] < 0.0 && w[1] >= 0.0).count();
        crossings as f64 * rate as f64 / x.len() as f64
    }

    #[test]
    fn flac_and_mp3_decode_to_mono_at_their_own_rate() {
        // 0.5 s of 440 Hz from ffmpeg: FLAC 44.1 kHz stereo, MP3 22.05 kHz mono at 32 kb/s.
        let flac = decode(include_bytes!("testdata/tone-44k-stereo.flac")).unwrap();
        assert_eq!(flac.rate, 44_100);
        assert_eq!(flac.samples.len(), 22_050);
        assert!((pitch(&flac.samples, flac.rate) - 440.0).abs() < 5.0);
        let mp3 = decode(include_bytes!("testdata/tone-22k-mono.mp3")).unwrap();
        assert_eq!(mp3.rate, 22_050);
        assert!((mp3.seconds() - 0.5).abs() < 0.1, "{} s", mp3.seconds());
        assert!((pitch(&mp3.samples, mp3.rate) - 440.0).abs() < 10.0);
        // Both reach the encoder at 16 kHz with the tone intact.
        let at16 = flac.at(16_000);
        assert_eq!(at16.len(), 8_000);
        assert!((pitch(&at16, 16_000) - 440.0).abs() < 5.0);
    }

    #[test]
    fn base64_errors_name_the_audio() {
        let e = from_base64("%%%").unwrap_err().to_string();
        assert!(e.starts_with("audio must be base64"), "{e}");
        assert!(!e.contains("image"), "{e}");
        let wav = encode_wav(16_000, &[0.25, -0.25]);
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&wav);
        assert_eq!(
            from_base64(&format!("data:audio/wav;base64,{b64}")).unwrap(),
            wav
        );
    }

    #[test]
    fn junk_and_empty_files_are_rejected() {
        assert!(matches!(
            decode(b"not audio at all"),
            Err(AudioError::Decode(_))
        ));
        assert!(matches!(
            decode(&encode_wav(16_000, &[])),
            Err(AudioError::Empty | AudioError::Decode(_))
        ));
    }

    #[test]
    fn resampling_keeps_a_tone_and_removes_what_the_new_rate_cannot_hold() {
        // 1 kHz survives 48 kHz -> 16 kHz; 12 kHz (above the new Nyquist) is filtered out.
        let rms =
            |v: &[f32]| (v.iter().map(|x| (x * x) as f64).sum::<f64>() / v.len() as f64).sqrt();
        let keep = resample(&tone(48_000, 1_000.0, 1.0), 48_000, 16_000);
        assert_eq!(keep.len(), 16_000);
        let want = tone(16_000, 1_000.0, 1.0);
        let mid = 2_000..14_000; // away from the edges, where the clip is padded with silence
        let err: f64 = keep[mid.clone()]
            .iter()
            .zip(&want[mid.clone()])
            .map(|(a, b)| ((a - b) as f64).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-3, "pass band error {err}");
        let gone = resample(&tone(48_000, 12_000.0, 1.0), 48_000, 16_000);
        assert!(rms(&gone[mid]) < 1e-3, "alias rms {}", rms(&gone));
        // Upsampling 8 kHz -> 16 kHz keeps the tone too.
        let up = resample(&tone(8_000, 1_000.0, 1.0), 8_000, 16_000);
        let err: f64 = up[2_000..14_000]
            .iter()
            .zip(&want[2_000..14_000])
            .map(|(a, b)| ((a - b) as f64).abs())
            .fold(0.0, f64::max);
        assert!(err < 1e-3, "upsampled error {err}");
    }
}
