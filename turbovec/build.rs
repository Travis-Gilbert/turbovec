// Emit BLAS link directives so downstream binaries (tests, examples, and
// crate users) resolve `cblas_sgemm` without needing `extern crate blas_src;`.
//
// ndarray's `blas` feature calls into C-BLAS for matrix multiplication but
// doesn't pick a provider; the provider lives in an external native library
// (OpenBLAS on Linux, Apple's Accelerate on macOS). Emitting the link flag
// from this crate's build script attaches the directive to `turbovec` itself,
// so any binary that depends on `turbovec` inherits it — bypassing the
// "blas-src must be referenced in the final binary" footgun.
//
// Windows falls through to ndarray's pure-Rust matrixmultiply fallback.
fn main() {
    println!("cargo:rustc-check-cfg=cfg(turbovec_nightly)");

    if rustc_is_nightly() {
        println!("cargo:rustc-cfg=turbovec_nightly");
    }

    match std::env::var("CARGO_CFG_TARGET_OS").as_deref() {
        Ok("linux") => println!("cargo:rustc-link-lib=openblas"),
        Ok("macos") => println!("cargo:rustc-link-lib=framework=Accelerate"),
        _ => {}
    }
}

fn rustc_is_nightly() -> bool {
    let Ok(rustc) = std::env::var("RUSTC") else {
        return false;
    };
    let Ok(output) = std::process::Command::new(rustc).arg("--version").output() else {
        return false;
    };
    String::from_utf8_lossy(&output.stdout).contains("nightly")
}
