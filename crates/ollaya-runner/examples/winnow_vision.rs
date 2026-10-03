//! GPU/CPU smoke and cache regression check with real author-hosted weights.
//! cargo run -p ollaya-runner --example winnow_vision -- MODEL DECISION LIBDIR PROJECTOR [CUDA_LIB]
use ollaya_runner::{
    Engine,
    llama::{Libraries, LlamaModel, Target},
    vision::{Rgb, encode_png},
};
use serde_json::json;
use std::path::PathBuf;
fn main() -> anyhow::Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    anyhow::ensure!(
        (4..=5).contains(&args.len()),
        "MODEL DECISION LIBDIR PROJECTOR [CUDA_LIB]"
    );
    let libs = Libraries {
        dir: PathBuf::from(&args[2]),
        cuda: args.get(4).map(PathBuf::from),
    };
    let mut model = LlamaModel::load(
        &PathBuf::from(&args[0]),
        &PathBuf::from(&args[1]),
        &libs,
        &Target::Auto,
        Some(4),
    )?;
    let state = json!("Inspect the supplied image.");
    let questions = json!({"color":{"type":"choice","instructions":"What is the dominant color in the image?","criteria":{"red":"Red","green":"Green","blue":"Blue"}},"red":{"type":"noul","instructions":"Is the image predominantly red?"}});
    let before = model.run_json(&state, &questions)?;
    model.load_projector(&PathBuf::from(&args[3]), &libs, Some(4))?;
    let mut outputs = Vec::new();
    for (name, pixel) in [
        ("red", [255, 0, 0]),
        ("green", [0, 128, 0]),
        ("blue", [0, 0, 255]),
        ("red", [255, 0, 0]),
    ] {
        let image = encode_png(&Rgb {
            width: 128,
            height: 128,
            data: pixel.repeat(128 * 128),
        });
        let output = model.run_images(&state, &questions, &[image])?;
        let expected = match name {
            "red" => 0,
            "green" => 1,
            "blue" => 2,
            _ => unreachable!(),
        };
        let color = &output.questions[0].logits;
        let best = color
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i);
        anyhow::ensure!(
            best == Some(expected),
            "incorrect {name} image classification"
        );
        println!("{name}: {:?}", output.questions);
        outputs.push(output);
    }
    for (a, b) in outputs[0].questions.iter().zip(&outputs[3].questions) {
        for (a, b) in a.logits.iter().zip(&b.logits) {
            anyhow::ensure!(
                (a - b).abs() <= 1e-5,
                "image cache changed repeated request logits"
            );
        }
    }
    let after = model.run_json(&state, &questions)?;
    for (a, b) in before.questions.iter().zip(&after.questions) {
        for (a, b) in a.logits.iter().zip(&b.logits) {
            anyhow::ensure!(
                (a - b).abs() <= 1e-5,
                "image cache changed text-only logits"
            );
        }
    }
    anyhow::ensure!(
        model
            .run_images(&state, &questions, &[vec![1, 2, 3]])
            .is_err(),
        "invalid image accepted"
    );
    println!(
        "passed on {}: multiple questions, changed/repeated images, text after images, invalid input",
        model.device
    );
    Ok(())
}
