//! Windows: the executable's icon (Explorer, the Start menu shortcut, Apps & features) and its
//! version details, so Task Manager lists it as Ollaya.

fn main() {
    println!("cargo::rerun-if-changed=icons/icon.ico");
    if std::env::var("CARGO_CFG_TARGET_OS").as_deref() == Ok("windows") {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("icons/icon.ico")
            .set("ProductName", "Ollaya")
            .set("FileDescription", "Ollaya");
        res.compile().expect("compile the Windows resources");
    }
}
