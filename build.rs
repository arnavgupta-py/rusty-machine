use std::process::Command;

fn main() {
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cusolver");
    println!("cargo:rustc-link-lib=cublas");
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");

    let status = Command::new("nvcc")
        .arg("-O3")
        .arg("-arch=sm_75")
        .arg("-ptx")
        .arg("-o")
        .arg("src/kernels.ptx")
        .arg("src/kernels.cu")
        .status()
        .expect("Failed to execute nvcc. Ensure the CUDA toolkit is installed and nvcc is in your PATH.");

    if !status.success() {
        panic!("nvcc failed to compile the CUDA kernel.");
    }

    println!("cargo:rerun-if-changed=src/kernels.cu");
}