use cust::prelude::*;
use cust::memory::{DeviceBuffer, DevicePointer, DeviceSlice, DeviceMemory};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::sync::OnceLock;
use std::os::raw::c_int;

// FFI definitions for cuSOLVER
mod ffi {
    use super::c_int;
    #[repr(C)]
    pub struct CusolverDnContext { _private: [u8; 0] }
    pub type CusolverDnHandle = *mut CusolverDnContext;
    #[link(name = "cusolver")]
    extern "C" {
        pub fn cusolverDnCreate(handle: *mut CusolverDnHandle) -> c_int;
        pub fn cusolverDnDestroy(handle: CusolverDnHandle) -> c_int;
        pub fn cusolverDnSgetrf_bufferSize(handle: CusolverDnHandle, m: c_int, n: c_int, A: *mut f32, lda: c_int, Lwork: *mut c_int) -> c_int;
        pub fn cusolverDnSgetrf(handle: CusolverDnHandle, m: c_int, n: c_int, A: *mut f32, lda: c_int, workspace: *mut f32, devIpiv: *mut c_int, devInfo: *mut c_int) -> c_int;
        pub fn cusolverDnSgetrs(handle: CusolverDnHandle, trans: c_int, n: c_int, nrhs: c_int, A: *const f32, lda: c_int, devIpiv: *const c_int, B: *mut f32, ldb: c_int, devInfo: *mut c_int) -> c_int;
    }
}

// Global context for CUDA
struct GpuContext {
    _ctx: Context,
    stream: Stream,
    module: Module,
}

static GLOBAL_CTX: OnceLock<GpuContext> = OnceLock::new();

fn to_py_err<E: std::fmt::Display>(e: E) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

fn get_gpu_context() -> PyResult<&'static GpuContext> {
    if let Some(ctx) = GLOBAL_CTX.get() {
        return Ok(ctx);
    }
    let created = (|| -> Result<GpuContext, PyErr> {
        cust::init(CudaFlags::empty()).map_err(to_py_err)?;
        let device = Device::get_device(0).map_err(to_py_err)?;
        let ctx = Context::new(device).map_err(to_py_err)?;
        let stream = Stream::new(StreamFlags::NON_BLOCKING, None).map_err(to_py_err)?;
        let module = Module::from_ptx(include_str!("kernels.ptx"), &[]).map_err(to_py_err)?;
        Ok(GpuContext { _ctx: ctx, stream, module })
    })();
    match created {
        Ok(ctx_val) => {
            GLOBAL_CTX.set(ctx_val).map_err(|_| to_py_err("Failed to set GLOBAL_CTX"))?;
            Ok(GLOBAL_CTX.get().expect("GLOBAL_CTX set but not present"))
        }
        Err(e) => Err(e),
    }
}

fn device_ptr_from_u64(ptr: u64) -> DevicePointer<f32> {
    DevicePointer::from_raw(ptr)
}

#[pyfunction]
fn gpu_matrix_multiply(a_ptr: u64, b_ptr: u64, c_ptr: u64, m: usize, n: usize, k: usize) -> PyResult<()> {
    let ctx = get_gpu_context()?;
    let func = ctx.module.get_function("matmul_tiled").map_err(to_py_err)?;
    let grid_dims = ((k as u32 + 15) / 16, (m as u32 + 15) / 16, 1);
    let block_dims = (16, 16, 1);
    let stream = &ctx.stream;
    unsafe {
        launch!(func<<<grid_dims, block_dims, 0, stream>>>(
            device_ptr_from_u64(a_ptr), device_ptr_from_u64(b_ptr), device_ptr_from_u64(c_ptr),
            m as i32, n as i32, k as i32
        )).map_err(to_py_err)?;
    }
    Ok(())
}

#[pyfunction]
fn gpu_transpose(in_ptr: u64, out_ptr: u64, m: usize, n: usize) -> PyResult<()> {
    let ctx = get_gpu_context()?;
    let func = ctx.module.get_function("transpose").map_err(to_py_err)?;
    let grid_dims = ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1);
    let block_dims = (16, 16, 1);
    let stream = &ctx.stream;
    unsafe {
        launch!(func<<<grid_dims, block_dims, 0, stream>>>(
            device_ptr_from_u64(in_ptr), device_ptr_from_u64(out_ptr), m as i32, n as i32
        )).map_err(to_py_err)?;
    }
    Ok(())
}

#[pyfunction]
fn gpu_inverse(a_ptr: u64, inv_ptr: u64, n: usize) -> PyResult<()> {
    let _ctx = get_gpu_context()?;
    let mut identity = vec![0f32; n * n];
    for i in 0..n { identity[i * n + i] = 1.0f32; }
    let inv_dev_temp = DeviceBuffer::from_slice(&identity).map_err(to_py_err)?;
    
    unsafe {
        let mut handle: ffi::CusolverDnHandle = std::ptr::null_mut();
        ffi::cusolverDnCreate(&mut handle);
        let ipiv_dev = DeviceBuffer::<c_int>::uninitialized(n).map_err(to_py_err)?;
        let info_dev = DeviceBuffer::<c_int>::uninitialized(1).map_err(to_py_err)?;
        let mut lwork: c_int = 0;
        let a_dev_ptr = a_ptr as *mut f32;
        ffi::cusolverDnSgetrf_bufferSize(handle, n as c_int, n as c_int, a_dev_ptr, n as c_int, &mut lwork);
        let work_dev = DeviceBuffer::<f32>::uninitialized(lwork as usize).map_err(to_py_err)?;
        ffi::cusolverDnSgetrf(handle, n as c_int, n as c_int, a_dev_ptr, n as c_int, work_dev.as_raw_ptr() as *mut f32, ipiv_dev.as_raw_ptr() as *mut i32, info_dev.as_raw_ptr() as *mut i32);
        ffi::cusolverDnSgetrs(handle, 0, n as c_int, n as c_int, a_dev_ptr as *const f32, n as c_int, ipiv_dev.as_raw_ptr() as *const i32, inv_dev_temp.as_raw_ptr() as *mut f32, n as c_int, info_dev.as_raw_ptr() as *mut i32);
        ffi::cusolverDnDestroy(handle);
        let mut inv_out_slice = DeviceSlice::from_raw_parts_mut(device_ptr_from_u64(inv_ptr), n * n);
        inv_dev_temp.copy_to(&mut inv_out_slice).map_err(to_py_err)?;
    }
    Ok(())
}

#[pyfunction]
fn solve_normal_equation_device(x_ptr: u64, y_ptr: u64, theta_ptr: u64, samples: usize, features: usize) -> PyResult<()> {
    let ctx = get_gpu_context()?;
    unsafe {
        let x_t_dev = DeviceBuffer::<f32>::uninitialized(features * samples).map_err(to_py_err)?;
        let xtx_dev = DeviceBuffer::<f32>::uninitialized(features * features).map_err(to_py_err)?;
        let mut xtx_inv_dev = DeviceBuffer::<f32>::uninitialized(features * features).map_err(to_py_err)?;
        let intermediate_dev = DeviceBuffer::<f32>::uninitialized(features * samples).map_err(to_py_err)?;
        
        gpu_transpose(x_ptr, x_t_dev.as_raw_ptr() as u64, samples, features)?;
        gpu_matrix_multiply(x_t_dev.as_raw_ptr() as u64, x_ptr, xtx_dev.as_raw_ptr() as u64, features, samples, features)?;
        
        xtx_dev.copy_to(&mut xtx_inv_dev).map_err(to_py_err)?;
        gpu_inverse(xtx_inv_dev.as_raw_ptr() as u64, xtx_inv_dev.as_raw_ptr() as u64, features)?;
        
        gpu_matrix_multiply(xtx_inv_dev.as_raw_ptr() as u64, x_t_dev.as_raw_ptr() as u64, intermediate_dev.as_raw_ptr() as u64, features, features, samples)?;
        gpu_matrix_multiply(intermediate_dev.as_raw_ptr() as u64, y_ptr, theta_ptr, features, samples, 1)?;
    }
    ctx.stream.synchronize().map_err(to_py_err)?;
    Ok(())
}

// ✅ **THE FIX**: Restored the full implementation for the original GD solver.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn train_logistic_gpu(x_ptr: u64, y_ptr: u64, theta_ptr: u64, samples: usize, features: usize, max_iter: usize, lr: f32, tol: f32, l1: f32, l2: f32) -> PyResult<()> {
    let ctx = get_gpu_context()?;
    let matmul_f = ctx.module.get_function("matmul_tiled").map_err(to_py_err)?;
    let sigmoid_f = ctx.module.get_function("elementwise_sigmoid").map_err(to_py_err)?;
    let sub_f = ctx.module.get_function("elementwise_sub").map_err(to_py_err)?;
    let transpose_f = ctx.module.get_function("transpose").map_err(to_py_err)?;
    let axpy_f = ctx.module.get_function("axpy").map_err(to_py_err)?;
    let sum_sq_f = ctx.module.get_function("sum_of_squares_reduction").map_err(to_py_err)?;
    let l1_f = ctx.module.get_function("proximal_update_l1").map_err(to_py_err)?;

    let grid256 = |n: usize| ((n as u32 + 255) / 256, 1, 1);
    let block256 = (256, 1, 1);
    let grid16 = |m: usize, k: usize| ((k as u32 + 15) / 16, (m as u32 + 15) / 16, 1);
    let block16 = (16, 16, 1);
    let stream = &ctx.stream;
    let theta_dev_ptr = device_ptr_from_u64(theta_ptr);

    unsafe {
        let h_dev = DeviceBuffer::<f32>::uninitialized(samples).map_err(to_py_err)?;
        let error_dev = DeviceBuffer::<f32>::uninitialized(samples).map_err(to_py_err)?;
        let x_t_dev = DeviceBuffer::<f32>::uninitialized(features * samples).map_err(to_py_err)?;
        let grad_dev = DeviceBuffer::<f32>::uninitialized(features).map_err(to_py_err)?;
        let num_blocks = (features + 255) / 256;
        let partial_sums_dev = DeviceBuffer::<f32>::uninitialized(num_blocks).map_err(to_py_err)?;
        let mut grad_norm_sq_host = vec![0.0f32; num_blocks];

        launch!(transpose_f<<<grid16(samples, features), block16, 0, stream>>>(
            device_ptr_from_u64(x_ptr),
            x_t_dev.as_device_ptr(),
            samples as i32,
            features as i32
        )).map_err(to_py_err)?;

        for i in 0..max_iter {
            launch!(matmul_f<<<grid16(samples, 1), block16, 0, stream>>>(device_ptr_from_u64(x_ptr), theta_dev_ptr, h_dev.as_device_ptr(), samples as i32, features as i32, 1)).map_err(to_py_err)?;
            launch!(sigmoid_f<<<grid256(samples), block256, 0, stream>>>(h_dev.as_device_ptr(), samples as i32)).map_err(to_py_err)?;
            launch!(sub_f<<<grid256(samples), block256, 0, stream>>>(h_dev.as_device_ptr(), device_ptr_from_u64(y_ptr), error_dev.as_device_ptr(), samples as i32)).map_err(to_py_err)?;
            launch!(matmul_f<<<grid16(features, 1), block16, 0, stream>>>(x_t_dev.as_device_ptr(), error_dev.as_device_ptr(), grad_dev.as_device_ptr(), features as i32, samples as i32, 1)).map_err(to_py_err)?;

            if l2 != 0.0 {
                launch!(axpy_f<<<grid256(features), block256, 0, stream>>>(l2, theta_dev_ptr, grad_dev.as_device_ptr(), features as i32)).map_err(to_py_err)?;
            }
            
            if i % 50 == 0 {
                launch!(sum_sq_f<<<grid256(features), block256, 256 * 4, stream>>>(
                    grad_dev.as_device_ptr(),
                    partial_sums_dev.as_device_ptr(),
                    features as i32
                )).map_err(to_py_err)?;
                
                partial_sums_dev.copy_to(&mut grad_norm_sq_host).map_err(to_py_err)?;
                stream.synchronize().map_err(to_py_err)?;

                let grad_norm = grad_norm_sq_host.iter().sum::<f32>().sqrt();
                if grad_norm < tol {
                    break;
                }
            }
            
            let alpha = -lr / (samples as f32);
            launch!(axpy_f<<<grid256(features), block256, 0, stream>>>(alpha, grad_dev.as_device_ptr(), theta_dev_ptr, features as i32)).map_err(to_py_err)?;

            if l1 != 0.0 {
                let threshold = lr * l1 / samples as f32;
                launch!(l1_f<<<grid256(features), block256, 0, stream>>>(
                    theta_dev_ptr,
                    threshold,
                    features as i32
                )).map_err(to_py_err)?;
            }
        }
    }
    ctx.stream.synchronize().map_err(to_py_err)?;
    Ok(())
}

fn get_column_ptr(matrix_ptr: DevicePointer<f32>, col_idx: usize, num_rows: usize) -> DevicePointer<f32> {
    unsafe { matrix_ptr.offset((col_idx * num_rows) as isize) }
}

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn train_logistic_sgd_l1_gpu(
    x_col_major_ptr: u64,
    y_ptr: u64,
    theta_ptr: u64,
    samples: usize,
    features: usize,
    epochs: usize,
    lr: f32,
    l1_penalty: f32,
) -> PyResult<()> {
    let ctx = get_gpu_context()?;
    let stream = &ctx.stream;

    let matmul_f = ctx.module.get_function("matmul_tiled").map_err(to_py_err)?;
    let grad_f = ctx.module.get_function("fused_gradient_from_logits").map_err(to_py_err)?;
    let update_f = ctx.module.get_function("update_theta_and_z").map_err(to_py_err)?;

    let x_col_major_dev_ptr = device_ptr_from_u64(x_col_major_ptr);
    let y_dev_ptr = device_ptr_from_u64(y_ptr);
    let theta_dev_ptr = device_ptr_from_u64(theta_ptr);

    // ✅ **THE FIX**: Removed `mut` from z_dev as it's not reassigned.
    let (z_dev, mut grad_j_dev, x_row_major_dev) = unsafe {
        (
            DeviceBuffer::<f32>::uninitialized(samples).map_err(to_py_err)?,
            DeviceBuffer::<f32>::uninitialized(1).map_err(to_py_err)?,
            DeviceBuffer::<f32>::uninitialized(samples * features).map_err(to_py_err)?,
        )
    };
    
    gpu_transpose(x_col_major_ptr, x_row_major_dev.as_raw_ptr() as u64, samples, features)?;

    let block_size = 256u32;
    let grid_size_samples = (samples as u32 + block_size - 1) / block_size;
    
    unsafe {
        launch!(matmul_f<<<((1 + 15) / 16, (samples as u32 + 15) / 16, 1), (16, 16, 1), 0, stream>>>(
            x_row_major_dev.as_device_ptr(),
            theta_dev_ptr,
            z_dev.as_device_ptr(),
            samples as i32, features as i32, 1
        )).map_err(to_py_err)?;
    }

    for _ in 0..epochs {
        for j in 0..features {
            let x_col_ptr = get_column_ptr(x_col_major_dev_ptr, j, samples);
            
            grad_j_dev.copy_from(&[0.0f32]).map_err(to_py_err)?;
            
            unsafe {
                launch!(grad_f<<<grid_size_samples, block_size, block_size * 4, stream>>>(
                    x_col_ptr,
                    y_dev_ptr,
                    z_dev.as_device_ptr(),
                    grad_j_dev.as_device_ptr(),
                    samples as i32
                )).map_err(to_py_err)?;

                launch!(update_f<<<grid_size_samples, block_size, 0, stream>>>(
                    theta_dev_ptr,
                    z_dev.as_device_ptr(),
                    x_col_ptr,
                    grad_j_dev.as_device_ptr(),
                    j as i32,
                    samples as i32,
                    lr,
                    l1_penalty
                )).map_err(to_py_err)?;
            }
        }
    }
    ctx.stream.synchronize().map_err(to_py_err)?;
    Ok(())
}

#[pymodule]
fn rusty_machine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(gpu_matrix_multiply, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_transpose, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_inverse, m)?)?;
    m.add_function(wrap_pyfunction!(solve_normal_equation_device, m)?)?;
    m.add_function(wrap_pyfunction!(train_logistic_gpu, m)?)?; 
    m.add_function(wrap_pyfunction!(train_logistic_sgd_l1_gpu, m)?)?;
    Ok(())
}