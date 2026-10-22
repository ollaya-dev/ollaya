#!/bin/sh
# Build crates/ollaya-web to web/pkg (ES module + .wasm). Needs the wasm32-unknown-unknown target
# and the wasm-bindgen CLI of the version the workspace pins:
#   rustup target add wasm32-unknown-unknown
#   cargo install wasm-bindgen-cli --version 0.2.128 --locked
set -eu
cd "$(dirname "$0")/.."
cargo build --release --locked -p ollaya-web --target wasm32-unknown-unknown
wasm-bindgen --target web --out-dir web/pkg target/wasm32-unknown-unknown/release/ollaya_web.wasm
