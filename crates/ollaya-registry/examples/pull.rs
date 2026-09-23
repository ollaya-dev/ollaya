//! Pull a model into a store without the daemon (development tool).
//!
//!     cargo run --release -p ollaya-registry --example pull -- <name> [store-dir]

use std::time::Instant;

use ollaya_registry::{ModelName, Puller, Store};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let name = ModelName::parse(args.get(1).ok_or("usage: pull <name> [store-dir]")?)?;
    let store = Store::open(
        args.get(2)
            .map(Into::into)
            .unwrap_or_else(Store::default_root),
    )?;
    let puller = Puller::new(store)?;
    let started = Instant::now();
    let last = std::sync::Mutex::new(String::new());
    puller
        .pull(&name, &|p| {
            let line = match (p.completed, p.total) {
                (Some(c), Some(t)) => {
                    format!("{} {:>5.1}%", p.status, 100.0 * c as f64 / t.max(1) as f64)
                }
                _ => p.status.clone(),
            };
            let mut last = last.lock().unwrap();
            if *last != line
                && (p.completed.is_none() || p.completed == p.total || line.ends_with("0%"))
            {
                println!("{line}");
            }
            *last = line;
        })
        .await?;
    println!("pulled {name} in {:.1}s", started.elapsed().as_secs_f64());
    Ok(())
}
