//! MLX on this Mac's GPU, through the worker thread. Needs Apple silicon and the `build` feature.

#![cfg(feature = "build")]

use std::path::Path;

use ollaya_mlx::{Array, BUILD_METALLIB, Dtype, Worker};

fn worker() -> Worker<()> {
    Worker::start("mlx-test", Path::new(BUILD_METALLIB), || Ok(())).expect("MLX starts")
}

/// Row-major `a [m, k] @ b [k, n]` in f64.
fn matmul(a: &[f32], b: &[f32], m: usize, k: usize, n: usize) -> Vec<f64> {
    let mut out = vec![0.0; m * n];
    for i in 0..m {
        for j in 0..n {
            out[i * n + j] = (0..k)
                .map(|t| f64::from(a[i * k + t]) * f64::from(b[t * n + j]))
                .sum();
        }
    }
    out
}

#[test]
fn self_check_reports_the_gpu() {
    let w = worker();
    assert!(!w.info().device.is_empty());
    assert_eq!(w.info().metallib, Path::new(BUILD_METALLIB));
}

#[test]
fn fp32_matmul_matches_f64() {
    let (m, k, n) = (37, 300, 29);
    let a: Vec<f32> = (0..m * k)
        .map(|i| ((i * 7919 % 211) as f32 - 105.0) / 97.0)
        .collect();
    let b: Vec<f32> = (0..k * n)
        .map(|i| ((i * 104_729 % 199) as f32 - 99.0) / 89.0)
        .collect();
    let want = matmul(&a, &b, m, k, n);
    let got = worker()
        .run(move |_| {
            let x = Array::from_slice(&a, &[m as i32, k as i32])?;
            let y = Array::from_slice(&b, &[k as i32, n as i32])?;
            x.matmul(&y)?.to_f32_vec()
        })
        .unwrap();
    let err = got
        .iter()
        .zip(&want)
        .map(|(&g, &w)| (f64::from(g) - w).abs())
        .fold(0.0, f64::max);
    // fp32 accumulation over 300 terms of magnitude <= 1.3: TF32 would be ~1e-2 off.
    assert!(err < 1e-4, "max error {err:e}");
}

#[test]
fn layer_norm_attention_and_gathers() {
    let got = worker()
        .run(|_| {
            let x = Array::from_slice(&[1.0f32, 2.0, 3.0, 4.0, -1.0, 0.0, 1.0, 6.0], &[2, 4])?;
            let ln = x.layer_norm(None, None, 1e-5)?.to_f32_vec()?;
            // One query that may only see key 1: attention returns value row 1.
            let q = Array::from_slice(&[1.0f32, 0.0], &[1, 1, 1, 2])?;
            let k = Array::from_slice(&[1.0f32, 0.0, 0.0, 1.0], &[1, 1, 2, 2])?;
            let v = Array::from_slice(&[10.0f32, 20.0, 30.0, 40.0], &[1, 1, 2, 2])?;
            let mask = Array::from_slice(&[false, true], &[1, 1, 1, 2])?;
            let att = Array::attention(&q, &k, &v, 1.0, Some(&mask))?.to_f32_vec()?;
            let idx = Array::from_slice(&[3i32, 0], &[2])?;
            let taken = x.take(&idx, 1)?.to_f32_vec()?;
            let parts = x.split(2, 1)?;
            let joined = Array::concatenate(&[&parts[1], &parts[0]], 1)?.to_f32_vec()?;
            Ok((ln, att, taken, joined))
        })
        .unwrap();
    let (ln, att, taken, joined) = got;
    let row0 = [-1.341_636, -0.447_212, 0.447_212, 1.341_636];
    for (g, w) in ln[..4].iter().zip(row0) {
        assert!((g - w).abs() < 1e-5, "{ln:?}");
    }
    assert_eq!(att, [30.0, 40.0]);
    assert_eq!(taken, [4.0, 1.0, 6.0, -1.0]);
    assert_eq!(joined, [3.0, 4.0, 1.0, 2.0, 1.0, 6.0, -1.0, 0.0]);
    assert_eq!(Dtype::Float32.size(), 4);
}

#[test]
fn errors_are_returned_not_fatal() {
    let err = worker()
        .run(|_| {
            let x = Array::from_slice(&[1.0f32, 2.0, 3.0], &[3])?;
            // Arrays never leave the MLX thread (they are not `Send`); only the outcome does.
            x.reshape(&[2, 2]).map(drop)
        })
        .unwrap_err();
    assert!(err.0.contains("reshape"), "{err}");
    // The thread is still usable after a failed op.
    let w = worker();
    assert_eq!(
        w.run(|_| Array::scalar(2.0).add(&Array::scalar(3.0))?.to_f32_vec())
            .unwrap(),
        [5.0]
    );
}

#[test]
fn ops_outside_the_mlx_thread_fail() {
    let err = Array::scalar(1.0).add(&Array::scalar(1.0)).unwrap_err();
    assert!(err.0.contains("MLX thread"), "{err}");
}

#[test]
fn a_missing_metal_library_is_a_clear_error() {
    let err = match Worker::start(
        "mlx-test",
        Path::new("/nonexistent/mlx.metallib"),
        || Ok(()),
    ) {
        Ok(_) => panic!("started without a Metal library"),
        Err(e) => e,
    };
    assert!(err.0.contains("/nonexistent/mlx.metallib"), "{err}");
}
