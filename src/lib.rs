use cust::prelude::*;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::types::PyModule;
use std::os::raw::c_int;
use cust::memory::DeviceMemory;

/// FFI bindings module for libcusolver.so
mod ffi {
    use super::c_int;
    
    #[repr(C)]
    pub struct CusolverDnContext { _private: [u8; 0] }
    pub type CusolverDnHandle = *mut CusolverDnContext;

    #[link(name = "cusolver")]
    extern "C" {
        pub fn cusolverDnCreate(handle: *mut CusolverDnHandle) -> c_int;
        pub fn cusolverDnDestroy(handle: CusolverDnHandle) -> c_int;
        pub fn cusolverDnSgetrf_bufferSize(
            handle: CusolverDnHandle, m: c_int, n: c_int, A: *mut f32, lda: c_int, Lwork: *mut c_int
        ) -> c_int;
        pub fn cusolverDnSgetrf(
            handle: CusolverDnHandle, m: c_int, n: c_int, A: *mut f32, lda: c_int,
            workspace: *mut f32, devIpiv: *mut c_int, devInfo: *mut c_int
        ) -> c_int;
        pub fn cusolverDnSgetrs(
            handle: CusolverDnHandle, trans: c_int, n: c_int, nrhs: c_int,
            A: *const f32, lda: c_int, devIpiv: *const c_int,
            B: *mut f32, ldb: c_int, devInfo: *mut c_int
        ) -> c_int;
    }
}

#[pyfunction]
fn gpu_matrix_multiply(a: Vec<f32>, b: Vec<f32>, m: usize, n: usize, k: usize) -> PyResult<Vec<f32>> {
    let to_py_err = |e: cust::error::CudaError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string());
    cust::init(CudaFlags::empty()).map_err(to_py_err)?;
    let device = Device::get_device(0).map_err(to_py_err)?;
    let _context = Context::new(device).map_err(to_py_err)?;
    let module = Module::from_ptx(include_str!("kernels.ptx"), &[]).map_err(to_py_err)?;
    let stream = Stream::new(StreamFlags::DEFAULT, None).map_err(to_py_err)?;
    let a_dev = DeviceBuffer::from_slice(&a).map_err(to_py_err)?;
    let b_dev = DeviceBuffer::from_slice(&b).map_err(to_py_err)?;
    let mut c_dev = unsafe { DeviceBuffer::uninitialized(m * k).map_err(to_py_err)? };
    let func = module.get_function("matmul").map_err(to_py_err)?;
    let grid_dims = ((k as u32 + 15) / 16, (m as u32 + 15) / 16, 1);
    let block_dims = (16, 16, 1);
    unsafe {
        launch!(func<<<grid_dims, block_dims, 0, stream>>>(
            a_dev.as_device_ptr(), b_dev.as_device_ptr(), c_dev.as_device_ptr(),
            m as i32, n as i32, k as i32
        )).map_err(to_py_err)?;
    }
    stream.synchronize().map_err(to_py_err)?;
    let mut c_host = vec![0.0f32; m * k];
    c_dev.copy_to(&mut c_host).map_err(to_py_err)?;
    Ok(c_host)
}

#[pyfunction]
fn gpu_transpose(input: Vec<f32>, m: usize, n: usize) -> PyResult<Vec<f32>> {
    let to_py_err = |e: cust::error::CudaError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string());
    cust::init(CudaFlags::empty()).map_err(to_py_err)?;
    let device = Device::get_device(0).map_err(to_py_err)?;
    let _context = Context::new(device).map_err(to_py_err)?;
    let module = Module::from_ptx(include_str!("kernels.ptx"), &[]).map_err(to_py_err)?;
    let stream = Stream::new(StreamFlags::DEFAULT, None).map_err(to_py_err)?;
    let in_dev = DeviceBuffer::from_slice(&input).map_err(to_py_err)?;
    let mut out_dev = unsafe { DeviceBuffer::uninitialized(n * m).map_err(to_py_err)? };
    let func = module.get_function("transpose").map_err(to_py_err)?;
    let grid_dims = ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1);
    let block_dims = (16, 16, 1);
    unsafe {
        launch!(func<<<grid_dims, block_dims, 0, stream>>>(
            in_dev.as_device_ptr(), out_dev.as_device_ptr(), m as i32, n as i32
        )).map_err(to_py_err)?;
    }
    stream.synchronize().map_err(to_py_err)?;
    let mut out_host = vec![0.0f32; n * m];
    out_dev.copy_to(&mut out_host).map_err(to_py_err)?;
    Ok(out_host)
}

#[pyfunction]
fn gpu_inverse(input: Vec<f32>, n: usize) -> PyResult<Vec<f32>> {
    let to_py_err = |e: cust::error::CudaError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string());
    cust::init(CudaFlags::empty()).map_err(to_py_err)?;
    let device = Device::get_device(0).map_err(to_py_err)?;
    let _context = Context::new(device).map_err(to_py_err)?;
    let mut a_dev = DeviceBuffer::from_slice(&input).map_err(to_py_err)?;
    let mut handle: ffi::CusolverDnHandle = std::ptr::null_mut();
    let identity = {
        let mut id = vec![0.0f32; n * n];
        for i in 0..n { id[i * n + i] = 1.0; }
        id
    };
    let mut b_dev = DeviceBuffer::from_slice(&identity).map_err(to_py_err)?;
    unsafe {
        ffi::cusolverDnCreate(&mut handle);
        let mut ipiv_dev = DeviceBuffer::<c_int>::uninitialized(n).map_err(to_py_err)?;
        let mut info_dev = DeviceBuffer::<c_int>::uninitialized(1).map_err(to_py_err)?;
        let mut lwork: c_int = 0;
        ffi::cusolverDnSgetrf_bufferSize(handle, n as c_int, n as c_int, a_dev.as_raw_ptr() as *mut f32, n as c_int, &mut lwork);
        let mut work_dev = DeviceBuffer::<f32>::uninitialized(lwork as usize).map_err(to_py_err)?;
        ffi::cusolverDnSgetrf(handle, n as c_int, n as c_int, a_dev.as_raw_ptr() as *mut f32, n as c_int, work_dev.as_raw_ptr() as *mut f32, ipiv_dev.as_raw_ptr() as *mut c_int, info_dev.as_raw_ptr() as *mut c_int);
        ffi::cusolverDnSgetrs(handle, 0, n as c_int, n as c_int, a_dev.as_raw_ptr() as *const f32, n as c_int, ipiv_dev.as_raw_ptr() as *const c_int, b_dev.as_raw_ptr() as *mut f32, n as c_int, info_dev.as_raw_ptr() as *mut c_int);
        ffi::cusolverDnDestroy(handle);
    }
    let mut inv_host = vec![0.0f32; n * n];
    b_dev.copy_to(&mut inv_host).map_err(to_py_err)?;
    Ok(inv_host)
}

/// Solves the Normal Equation: theta = (X^T * X)^-1 * X^T * y
#[pyfunction]
fn solve_normal_equation(x: Vec<f32>, y: Vec<f32>, samples: usize, features: usize) -> PyResult<Vec<f32>> {
    // This function orchestrates our existing primitives.
    // Note: This version is clear but not maximally efficient, as data is copied
    // back to the CPU after each step. A future optimization would be to create
    // internal-only functions that operate purely on GPU device memory.

    // --- Step 1: Transpose X ---
    // X is (samples x features) -> x_t is (features x samples)
    let x_t = gpu_transpose(x.clone(), samples, features)?;

    // --- Step 2: Calculate X^T * X ---
    // (features x samples) @ (samples x features) -> (features x features)
    let xtx = gpu_matrix_multiply(x_t.clone(), x, features, samples, features)?;

    // --- Step 3: Invert (X^T * X) ---
    // (features x features) -> (features x features)
    let xtx_inv = gpu_inverse(xtx, features)?;

    // --- Step 4: Calculate (X^T * X)^-1 * X^T ---
    // (features x features) @ (features x samples) -> (features x samples)
    let intermediate = gpu_matrix_multiply(xtx_inv, x_t, features, features, samples)?;

    // --- Step 5: Calculate final theta = intermediate * y ---
    // (features x samples) @ (samples x 1) -> (features x 1)
    let theta = gpu_matrix_multiply(intermediate, y, features, samples, 1)?;

    Ok(theta)
}

#[pymodule]
fn rusty_machine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_wrapped(wrap_pyfunction!(gpu_matrix_multiply))?;
    m.add_wrapped(wrap_pyfunction!(gpu_transpose))?;
    m.add_wrapped(wrap_pyfunction!(gpu_inverse))?;
    // ADDED: Expose the new solver function to Python
    m.add_wrapped(wrap_pyfunction!(solve_normal_equation))?;
    Ok(())
}