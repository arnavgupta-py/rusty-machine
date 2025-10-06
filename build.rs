use std::process::Command;

fn main() {
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cusolver");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");

    // --- OPTIMIZED NVCC COMMAND ---
    // We add two crucial flags:
    // 1. -O3: The highest level of compiler optimization.
    // 2. -arch=sm_75: Generate code specifically for modern GPUs (NVIDIA Turing/RTX 20-series and newer).
    //    This allows the compiler to use modern instructions for maximum performance.
    let status = Command::new("nvcc")
        .arg("-O3")
        .arg("-arch=sm_75")
        .arg("-ptx")
        .arg("-o")
        .arg("src/kernels.ptx")
        .arg("src/kernels.cu")
        .status()
        .expect("Failed to execute nvcc. Is it in your PATH?");

    if !status.success() {
        panic!("nvcc failed to compile the CUDA kernel");
    }

    // Tell Cargo to rerun this script if the CUDA kernel changes.
    println!("cargo:rerun-if-changed=src/kernels.cu");
}