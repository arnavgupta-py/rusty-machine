use cust::prelude::*;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::types::PyModule;
use std::os::raw::c_int;
use cust::memory::{DeviceMemory, DevicePointer, DeviceSlice};

/// FFI bindings module for libcusolver.so.
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
fn gpu_matrix_multiply(a_ptr: u64, b_ptr: u64, c_ptr: u64, m: usize, n: usize, k: usize) -> PyResult<()> {
    let to_py_err = |e: cust::error::CudaError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string());
    
    cust::init(CudaFlags::empty()).map_err(to_py_err)?;
    let device = Device::get_device(0).map_err(to_py_err)?;
    let _context = Context::new(device).map_err(to_py_err)?;
    let module = Module::from_ptx(include_str!("kernels.ptx"), &[]).map_err(to_py_err)?;
    let stream = Stream::new(StreamFlags::DEFAULT, None).map_err(to_py_err)?;

    let a_dev = unsafe { DevicePointer::<f32>::from_raw(a_ptr) };
    let b_dev = unsafe { DevicePointer::<f32>::from_raw(b_ptr) };
    let c_dev = unsafe { DevicePointer::<f32>::from_raw(c_ptr) };

    let func = module.get_function("matmul").map_err(to_py_err)?;
    let grid_dims = ((k as u32 + 15) / 16, (m as u32 + 15) / 16, 1);
    let block_dims = (16, 16, 1);

    unsafe {
        launch!(func<<<grid_dims, block_dims, 0, stream>>>(
            a_dev, b_dev, c_dev,
            m as i32, n as i32, k as i32
        )).map_err(to_py_err)?;
    }
    stream.synchronize().map_err(to_py_err)?;
    Ok(())
}

#[pyfunction]
fn gpu_transpose(in_ptr: u64, out_ptr: u64, m: usize, n: usize) -> PyResult<()> {
    let to_py_err = |e: cust::error::CudaError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string());

    cust::init(CudaFlags::empty()).map_err(to_py_err)?;
    let device = Device::get_device(0).map_err(to_py_err)?;
    let _context = Context::new(device).map_err(to_py_err)?;
    let module = Module::from_ptx(include_str!("kernels.ptx"), &[]).map_err(to_py_err)?;
    let stream = Stream::new(StreamFlags::DEFAULT, None).map_err(to_py_err)?;

    let in_dev = unsafe { DevicePointer::<f32>::from_raw(in_ptr) };
    let out_dev = unsafe { DevicePointer::<f32>::from_raw(out_ptr) };

    let func = module.get_function("transpose").map_err(to_py_err)?;
    let grid_dims = ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1);
    let block_dims = (16, 16, 1);

    unsafe {
        launch!(func<<<grid_dims, block_dims, 0, stream>>>(
            in_dev, out_dev, m as i32, n as i32
        )).map_err(to_py_err)?;
    }
    stream.synchronize().map_err(to_py_err)?;
    Ok(())
}

#[pyfunction]
fn gpu_inverse(a_ptr: u64, inv_ptr: u64, n: usize) -> PyResult<()> {
    let to_py_err = |e: cust::error::CudaError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string());

    cust::init(CudaFlags::empty()).map_err(to_py_err)?;
    let device = Device::get_device(0).map_err(to_py_err)?;
    let _context = Context::new(device).map_err(to_py_err)?;

    let mut handle: ffi::CusolverDnHandle = std::ptr::null_mut();
    
    let identity = {
        let mut id = vec![0.0f32; n * n];
        for i in 0..n { id[i * n + i] = 1.0; }
        id
    };
    let mut inv_dev_temp = DeviceBuffer::from_slice(&identity).map_err(to_py_err)?;
    let a_dev_ptr = a_ptr as *mut f32;

    unsafe {
        ffi::cusolverDnCreate(&mut handle);
        let mut ipiv_dev = DeviceBuffer::<c_int>::uninitialized(n).map_err(to_py_err)?;
        let mut info_dev = DeviceBuffer::<c_int>::uninitialized(1).map_err(to_py_err)?;
        let mut lwork: c_int = 0;

        ffi::cusolverDnSgetrf_bufferSize(handle, n as c_int, n as c_int, a_dev_ptr, n as c_int, &mut lwork);
        let mut work_dev = DeviceBuffer::<f32>::uninitialized(lwork as usize).map_err(to_py_err)?;

        ffi::cusolverDnSgetrf(handle, n as c_int, n as c_int, a_dev_ptr, n as c_int, work_dev.as_raw_ptr() as *mut f32, ipiv_dev.as_raw_ptr() as *mut i32, info_dev.as_raw_ptr() as *mut i32);
        ffi::cusolverDnSgetrs(handle, 0, n as c_int, n as c_int, a_dev_ptr as *const f32, n as c_int, ipiv_dev.as_raw_ptr() as *const i32, inv_dev_temp.as_raw_ptr() as *mut f32, n as c_int, info_dev.as_raw_ptr() as *mut i32);
        
        ffi::cusolverDnDestroy(handle);

        let inv_out_ptr = DevicePointer::<f32>::from_raw(inv_ptr);
        let mut inv_out_slice = DeviceSlice::from_raw_parts(inv_out_ptr, n * n);
        inv_dev_temp.copy_to(&mut inv_out_slice).map_err(to_py_err)?;
    }
    Ok(())
}

#[pyfunction]
fn solve_normal_equation_device(x_ptr: u64, y_ptr: u64, theta_ptr: u64, samples: usize, features: usize) -> PyResult<()> {
    let to_py_err = |e: cust::error::CudaError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string());
    cust::init(CudaFlags::empty()).map_err(to_py_err)?;
    let device = Device::get_device(0).map_err(to_py_err)?;
    let _context = Context::new(device).map_err(to_py_err)?;

    unsafe {
        let x_t_dev = DeviceBuffer::<f32>::uninitialized(features * samples).map_err(to_py_err)?;
        let mut xtx_dev = DeviceBuffer::<f32>::uninitialized(features * features).map_err(to_py_err)?;
        let xtx_inv_dev = DeviceBuffer::<f32>::uninitialized(features * features).map_err(to_py_err)?;
        let intermediate_dev = DeviceBuffer::<f32>::uninitialized(features * samples).map_err(to_py_err)?;

        gpu_transpose(x_ptr, x_t_dev.as_raw_ptr() as u64, samples, features)?;

        gpu_matrix_multiply(
            x_t_dev.as_raw_ptr() as u64,
            x_ptr,
            xtx_dev.as_raw_ptr() as u64,
            features, samples, features
        )?;

        gpu_inverse(xtx_dev.as_raw_ptr() as u64, xtx_inv_dev.as_raw_ptr() as u64, features)?;

        gpu_matrix_multiply(
            xtx_inv_dev.as_raw_ptr() as u64,
            x_t_dev.as_raw_ptr() as u64,
            intermediate_dev.as_raw_ptr() as u64,
            features, features, samples
        )?;

        gpu_matrix_multiply(
            intermediate_dev.as_raw_ptr() as u64,
            y_ptr,
            theta_ptr,
            features, samples, 1
        )?;
    }
    
    Ok(())
}

#[pymodule]
fn rusty_machine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gpu_matrix_multiply, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_transpose, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_normal_equation_device, m)?)?;
    Ok(())
}