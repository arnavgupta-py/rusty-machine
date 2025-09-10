use std::process::Command;

fn main() {
    println!("cargo:rustc-link-lib=cuda");
    println!("cargo:rustc-link-lib=cusolver"); 
    println!("cargo:rustc-link-search=native=/usr/local/cuda/lib64");

    let status = Command::new("nvcc")
        .arg("-ptx")
        .arg("-o")
        .arg("src/kernels.ptx")
        .arg("src/kernels.cu")
        .status()
        .expect("Failed to execute nvcc. Is it in your PATH?");

    if !status.success() {
        panic!("nvcc failed to compile the CUDA kernel");
    }
}