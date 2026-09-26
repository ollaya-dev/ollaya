//! Builds MLX and mlx-c from pinned sources, links them statically and generates the bindings.
//!
//! Nothing happens without the `build` feature, so the default workspace build needs no CMake.
//!
//! * **Pins.** MLX v0.32.2 and mlx-c `ebc88f10`, the versions Ollama runs. Both tarballs are
//!   checked against their sha256 before anything is built.
//! * **Deployment target.** macOS 14.0, MLX's minimum. That also leaves out the M5 "NAX" kernels,
//!   which need 26.2 and are the only ones that ever compute fp32 as TF32.
//! * **Metal kernels.** Ahead of time into `mlx.metallib` (no `MLX_METAL_JIT`): kernels compiled
//!   at run time use fast math on macOS 15 and later (mlx#4553), which breaks fp32 parity.
//! * **Cache.** The install prefix is `$OLLAYA_MLX_DIR/<key>` when that is set (CI caches it),
//!   otherwise `$OUT_DIR/mlx`. A finished prefix has a `.complete` stamp holding the key, and is
//!   reused as is. The CMake build tree and the sources are deleted once installed.
//! * **The metallib.** It is copied next to the binaries (`target/<profile>/mlx.metallib`), as
//!   `ort`'s `copy-dylibs` does for its providers; `scripts/package.sh` takes it from there. Its
//!   install path is also exported as `ollaya_mlx_sys::BUILD_METALLIB`.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=OLLAYA_MLX_DIR");
    #[cfg(feature = "build")]
    imp::main();
}

#[cfg(feature = "build")]
mod imp {
    use std::fs;
    use std::io::Write as _;
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use sha2::Digest as _;

    const MLX_VERSION: &str = "0.32.2";
    const MLX_URL: &str = "https://github.com/ml-explore/mlx/archive/refs/tags/v0.32.2.tar.gz";
    const MLX_SHA256: &str = "bf892386ccac1d249acc9da21574fd7b95a995f4c58b03310105fe05f077b175";
    const MLX_C_COMMIT: &str = "ebc88f10caa1b625e6b581437a8dea6df8a70085";
    const MLX_C_SHA256: &str = "0559a688fe6ef20f6bcb6aac3b419e878fcbc295e19bef816afd51e4b3134e25";
    const DEPLOYMENT_TARGET: &str = "14.0";

    fn env(name: &str) -> String {
        std::env::var(name).unwrap_or_else(|_| panic!("{name} is not set"))
    }

    /// `<version>-<mlx-c>-<target>`: a prefix built for other pins is never reused.
    fn key() -> String {
        format!(
            "mlx-{MLX_VERSION}-mlxc-{}-macos{DEPLOYMENT_TARGET}",
            &MLX_C_COMMIT[..8]
        )
    }

    fn run(cmd: &mut Command) {
        let shown = format!("{cmd:?}");
        let status = cmd
            .status()
            .unwrap_or_else(|e| panic!("could not run {shown}: {e}"));
        if !status.success() {
            panic!("{shown} failed with {status}");
        }
    }

    fn output(cmd: &mut Command) -> String {
        let shown = format!("{cmd:?}");
        let out = cmd
            .output()
            .unwrap_or_else(|e| panic!("could not run {shown}: {e}"));
        if !out.status.success() {
            panic!(
                "{shown} failed: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            );
        }
        String::from_utf8(out.stdout)
            .expect("utf-8 output")
            .trim()
            .to_owned()
    }

    pub fn main() {
        let target = env("TARGET");
        if target != "aarch64-apple-darwin" {
            panic!(
                "MLX builds only for aarch64-apple-darwin (Apple silicon), not {target}; \
                 build without the `mlx` feature"
            );
        }
        // The binary's own minimum macOS comes from rustc (11.0 unless this is set), not from
        // MLX's libraries: a release build must say 14.0, or it claims to run where MLX can't.
        println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
        let deployment = std::env::var("MACOSX_DEPLOYMENT_TARGET").ok();
        let major = deployment
            .as_deref()
            .and_then(|v| v.split('.').next()?.parse::<u32>().ok());
        if major.is_none_or(|m| m < 14) {
            println!(
                "cargo:warning=MLX needs macOS {DEPLOYMENT_TARGET} but MACOSX_DEPLOYMENT_TARGET is {}; \
                 set it to {DEPLOYMENT_TARGET} for binaries you ship",
                deployment.as_deref().unwrap_or("unset (rustc's 11.0)")
            );
        }
        let prefix = match std::env::var_os("OLLAYA_MLX_DIR") {
            Some(root) => PathBuf::from(root).join(key()),
            None => PathBuf::from(env("OUT_DIR")).join("mlx"),
        };
        ensure_built(&prefix);
        link(&prefix);
        bindings(&prefix);
        let metallib = prefix.join("lib/mlx.metallib");
        copy_next_to_binaries(&metallib);
        println!(
            "cargo:rustc-env=OLLAYA_MLX_BUILD_METALLIB={}",
            metallib.display()
        );
        // `DEP_MLX_METALLIB` for crates that depend on this one.
        println!("cargo:metallib={}", metallib.display());
    }

    fn complete(prefix: &Path) -> bool {
        let stamp = fs::read_to_string(prefix.join(".complete")).unwrap_or_default();
        stamp.trim() == key()
            && [
                "lib/libmlx.a",
                "lib/libmlxc.a",
                "lib/mlx.metallib",
                "include/mlx/c/mlx.h",
            ]
            .iter()
            .all(|f| prefix.join(f).is_file())
    }

    fn ensure_built(prefix: &Path) {
        if complete(prefix) {
            return;
        }
        let parent = prefix.parent().expect("prefix has a parent");
        fs::create_dir_all(parent).expect("create the MLX directory");
        // Two builds (cargo check and cargo build, or two worktrees) may share one cache: the
        // first builds, the others wait and then find the stamp.
        let lock = fs::File::create(parent.join(format!("{}.lock", key())))
            .expect("create the MLX build lock");
        lock.lock().expect("lock the MLX build");
        if complete(prefix) {
            return;
        }
        let work = parent.join(format!("{}.work", key()));
        let _ = fs::remove_dir_all(&work);
        let _ = fs::remove_dir_all(prefix);
        fs::create_dir_all(&work).expect("create the MLX work directory");

        let mlx_c_url =
            format!("https://github.com/ml-explore/mlx-c/archive/{MLX_C_COMMIT}.tar.gz");
        let mlx = fetch(MLX_URL, MLX_SHA256, &work, "mlx");
        let mlx_c = fetch(&mlx_c_url, MLX_C_SHA256, &work, "mlx-c");
        let build = work.join("build");
        let jobs = std::env::var("NUM_JOBS").unwrap_or_else(|_| "4".into());
        run(Command::new("cmake")
            .env("MACOSX_DEPLOYMENT_TARGET", DEPLOYMENT_TARGET)
            .arg("-S")
            .arg(&mlx_c)
            .arg("-B")
            .arg(&build)
            .arg("-DCMAKE_BUILD_TYPE=Release")
            .arg(format!("-DCMAKE_OSX_DEPLOYMENT_TARGET={DEPLOYMENT_TARGET}"))
            .arg("-DCMAKE_OSX_ARCHITECTURES=arm64")
            .arg(format!("-DCMAKE_INSTALL_PREFIX={}", prefix.display()))
            .arg(format!("-DFETCHCONTENT_SOURCE_DIR_MLX={}", mlx.display()))
            .arg("-DBUILD_SHARED_LIBS=OFF")
            .arg("-DMLX_C_BUILD_EXAMPLES=OFF")
            .arg("-DMLX_BUILD_TESTS=OFF")
            .arg("-DMLX_BUILD_EXAMPLES=OFF")
            .arg("-DMLX_BUILD_BENCHMARKS=OFF")
            .arg("-DMLX_BUILD_PYTHON_BINDINGS=OFF")
            .arg("-DMLX_BUILD_PYTHON_STUBS=OFF")
            .arg("-DMLX_BUILD_GGUF=OFF")
            .arg("-DMLX_METAL_JIT=OFF")
            .arg("-DMLX_USE_CCACHE=OFF"));
        run(Command::new("cmake")
            .env("MACOSX_DEPLOYMENT_TARGET", DEPLOYMENT_TARGET)
            .arg("--build")
            .arg(&build)
            .arg("--config")
            .arg("Release")
            .arg("--parallel")
            .arg(&jobs));
        run(Command::new("cmake").arg("--install").arg(&build));
        // JACCL (MLX's RDMA backend, built with the macOS 26.2 SDK or newer) is a second static
        // library that `libmlx.a` depends on; the install rules leave it out.
        let jaccl = build.join("jaccl/libjaccl.a");
        if jaccl.is_file() {
            fs::copy(&jaccl, prefix.join("lib/libjaccl.a")).expect("copy libjaccl.a");
        }
        fs::remove_dir_all(&work).expect("remove the MLX work directory");
        let mut stamp =
            fs::File::create(prefix.join(".complete")).expect("write the MLX build stamp");
        writeln!(stamp, "{}", key()).expect("write the MLX build stamp");
        assert!(complete(prefix), "MLX installed incompletely in {prefix:?}");
    }

    /// Download `url` into `work`, check its sha256 and unpack it into `work/<name>`.
    fn fetch(url: &str, sha256: &str, work: &Path, name: &str) -> PathBuf {
        let archive = work.join(format!("{name}.tar.gz"));
        run(Command::new("curl")
            .args(["-fsSL", "--retry", "3", "-o"])
            .arg(&archive)
            .arg(url));
        let bytes = fs::read(&archive).expect("read the downloaded archive");
        let got = hex(&sha2::Sha256::digest(&bytes));
        if got != sha256 {
            panic!("sha256 mismatch for {url}: got {got}, want {sha256}");
        }
        let dir = work.join(name);
        fs::create_dir_all(&dir).expect("create the source directory");
        run(Command::new("tar")
            .arg("-xzf")
            .arg(&archive)
            .arg("--strip-components=1")
            .arg("-C")
            .arg(&dir));
        dir
    }

    fn hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn link(prefix: &Path) {
        let lib = prefix.join("lib");
        println!("cargo:rustc-link-search=native={}", lib.display());
        println!("cargo:rustc-link-lib=static=mlxc");
        println!("cargo:rustc-link-lib=static=mlx");
        if lib.join("libjaccl.a").is_file() {
            println!("cargo:rustc-link-lib=static=jaccl");
        }
        for framework in ["Metal", "Foundation", "QuartzCore", "Accelerate"] {
            println!("cargo:rustc-link-lib=framework={framework}");
        }
        println!("cargo:rustc-link-lib=c++");
        // MLX's `__builtin_available` checks call `__isPlatformVersionAtLeast`, which lives in
        // clang's compiler runtime; rustc's link line does not include it.
        let resources = output(Command::new("xcrun").args(["clang", "--print-resource-dir"]));
        let darwin = Path::new(&resources).join("lib/darwin");
        if !darwin.join("libclang_rt.osx.a").is_file() {
            panic!("no libclang_rt.osx.a in {}", darwin.display());
        }
        println!("cargo:rustc-link-search=native={}", darwin.display());
        println!("cargo:rustc-link-lib=static=clang_rt.osx");
    }

    fn bindings(prefix: &Path) {
        let include = prefix.join("include");
        let out = PathBuf::from(env("OUT_DIR")).join("bindings.rs");
        bindgen::Builder::default()
            .header(include.join("mlx/c/mlx.h").display().to_string())
            .clang_arg(format!("-I{}", include.display()))
            .allowlist_function("mlx_.*")
            .allowlist_type("mlx_.*")
            .allowlist_var("MLX_.*")
            // A `static` in the header: nothing to link against.
            .blocklist_item("mlx_array_empty")
            .default_enum_style(bindgen::EnumVariation::Rust {
                non_exhaustive: false,
            })
            .derive_default(true)
            .layout_tests(false)
            .generate()
            .expect("generate the mlx-c bindings")
            .write_to_file(&out)
            .expect("write the mlx-c bindings");
    }

    /// `target/<profile>/mlx.metallib`, next to the `ollaya` binary a `cargo build` produces.
    fn copy_next_to_binaries(metallib: &Path) {
        // OUT_DIR is <target>/<profile>/build/<package>-<hash>/out.
        let out = PathBuf::from(env("OUT_DIR"));
        let Some(profile_dir) = out.ancestors().nth(3) else {
            return;
        };
        let dest = profile_dir.join("mlx.metallib");
        let same = fs::metadata(&dest)
            .ok()
            .zip(fs::metadata(metallib).ok())
            .is_some_and(|(a, b)| a.len() == b.len() && a.modified().ok() >= b.modified().ok());
        if !same {
            let _ = fs::remove_file(&dest);
            if fs::hard_link(metallib, &dest).is_err() {
                fs::copy(metallib, &dest).expect("copy mlx.metallib next to the binaries");
            }
        }
    }
}
