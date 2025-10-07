use cust::prelude::*;
use cust::memory::{DeviceBuffer, DevicePointer, DeviceCopy, DeviceMemory};
use pyo3::prelude::*;
use pyo3::wrap_pyfunction;
use std::sync::OnceLock;
use std::os::raw::{c_int, c_void};

mod ffi {
    use super::{c_int, c_void};
    #[repr(C)]
    pub struct CublasContext { _private: [u8; 0] }
    pub type CublasHandle = *mut CublasContext;

    #[repr(C)]
    pub struct CusolverDnContext { _private: [u8; 0] }
    pub type CusolverDnHandle = *mut CusolverDnContext;

    #[link(name = "cublas")]
    extern "C" {
        pub fn cublasCreate_v2(handle: *mut CublasHandle) -> c_int;
        pub fn cublasDestroy_v2(handle: CublasHandle) -> c_int;
        pub fn cublasSgemv_v2(
            handle: CublasHandle,
            trans: c_int,
            m: c_int, n: c_int,
            alpha: *const f32,
            A: *const c_void, lda: c_int,
            x: *const c_void, incx: c_int,
            beta: *const f32,
            y: *mut c_void, incy: c_int,
        ) -> c_int;
    }

    #[link(name = "cusolver")]
    extern "C" {
        pub fn cusolverDnCreate(handle: *mut CusolverDnHandle) -> c_int;
        pub fn cusolverDnDestroy(handle: CusolverDnHandle) -> c_int;
        pub fn cusolverDnSgetrf_bufferSize(handle: CusolverDnHandle, m: c_int, n: c_int, A: *mut f32, lda: c_int, Lwork: *mut c_int) -> c_int;
        pub fn cusolverDnSgetrf(handle: CusolverDnHandle, m: c_int, n: c_int, A: *mut f32, lda: c_int, workspace: *mut f32, devIpiv: *mut c_int, devInfo: *mut c_int) -> c_int;
        pub fn cusolverDnSgetrs(handle: CusolverDnHandle, trans: c_int, n: c_int, nrhs: c_int, A: *const f32, lda: c_int, devIpiv: *const c_int, B: *mut f32, ldb: c_int, devInfo: *mut c_int) -> c_int;
    }
}

struct SyncCublasHandle(ffi::CublasHandle);
unsafe impl Send for SyncCublasHandle {}
unsafe impl Sync for SyncCublasHandle {}

struct SyncCusolverHandle(ffi::CusolverDnHandle);
unsafe impl Send for SyncCusolverHandle {}
unsafe impl Sync for SyncCusolverHandle {}

struct GpuContext {
    _ctx: Context,
    stream: Stream,
    module: Module,
    cublas_handle: SyncCublasHandle,
    cusolver_handle: SyncCusolverHandle,
}

impl Drop for GpuContext {
    fn drop(&mut self) {
        unsafe {
            ffi::cublasDestroy_v2(self.cublas_handle.0);
            ffi::cusolverDnDestroy(self.cusolver_handle.0);
        }
    }
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
        
        let mut cublas_handle: ffi::CublasHandle = std::ptr::null_mut();
        unsafe { ffi::cublasCreate_v2(&mut cublas_handle); }

        let mut cusolver_handle: ffi::CusolverDnHandle = std::ptr::null_mut();
        unsafe { ffi::cusolverDnCreate(&mut cusolver_handle); }

        Ok(GpuContext { _ctx: ctx, stream, module, cublas_handle: SyncCublasHandle(cublas_handle), cusolver_handle: SyncCusolverHandle(cusolver_handle) })
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
    let ctx = get_gpu_context()?;
    let mut identity = vec![0f32; n * n];
    for i in 0..n { identity[i * n + i] = 1.0f32; }
    let inv_dev_temp = DeviceBuffer::from_slice(&identity).map_err(to_py_err)?;
    
    unsafe {
        let handle = ctx.cusolver_handle.0;
        let ipiv_dev = DeviceBuffer::<c_int>::uninitialized(n).map_err(to_py_err)?;
        let info_dev = DeviceBuffer::<c_int>::uninitialized(1).map_err(to_py_err)?;
        let mut lwork: c_int = 0;
        let a_dev_ptr = a_ptr as *mut f32;
        ffi::cusolverDnSgetrf_bufferSize(handle, n as c_int, n as c_int, a_dev_ptr, n as c_int, &mut lwork);
        let work_dev = DeviceBuffer::<f32>::uninitialized(lwork as usize).map_err(to_py_err)?;
        ffi::cusolverDnSgetrf(handle, n as c_int, n as c_int, a_dev_ptr, n as c_int, work_dev.as_raw_ptr() as *mut f32, ipiv_dev.as_raw_ptr() as *mut i32, info_dev.as_raw_ptr() as *mut i32);
        ffi::cusolverDnSgetrs(handle, 0, n as c_int, n as c_int, a_dev_ptr as *const f32, n as c_int, ipiv_dev.as_raw_ptr() as *const i32, inv_dev_temp.as_raw_ptr() as *mut f32, n as c_int, info_dev.as_raw_ptr() as *mut i32);
        
        let mut inv_out_slice = cust::memory::DeviceSlice::from_raw_parts_mut(device_ptr_from_u64(inv_ptr), n * n);
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

#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn train_logistic_minibatch_gpu(
    x_ptr: u64, y_ptr: u64, theta_ptr: u64,
    samples: usize, features: usize, _epochs: usize,
    lr: f32, batch_size: usize,
) -> PyResult<()> {
    let ctx = get_gpu_context()?;
    let stream = &ctx.stream;

    let fused_sigmoid_sub_f = ctx.module.get_function("fused_sigmoid_sub").map_err(to_py_err)?;
    let axpy_f = ctx.module.get_function("axpy").map_err(to_py_err)?;

    let x_dev_ptr: DevicePointer<f32> = device_ptr_from_u64(x_ptr);
    let y_dev_ptr: DevicePointer<f32> = device_ptr_from_u64(y_ptr);
    let theta_dev_ptr: DevicePointer<f32> = device_ptr_from_u64(theta_ptr);

    let (z_dev, error_dev, grad_dev) = unsafe {
        (
            DeviceBuffer::<f32>::uninitialized(batch_size).map_err(to_py_err)?,
            DeviceBuffer::<f32>::uninitialized(batch_size).map_err(to_py_err)?,
            DeviceBuffer::<f32>::uninitialized(features).map_err(to_py_err)?,
        )
    };
    
    let alpha = 1.0f32;
    let beta = 0.0f32;

    let num_batches = (samples + batch_size - 1) / batch_size;

    for i in 0..num_batches {
        let current_batch_size = if i == num_batches - 1 { samples - i * batch_size } else { batch_size };

        let x_batch_ptr = unsafe { x_dev_ptr.offset((i * batch_size * features) as isize) };
        let y_batch_ptr = unsafe { y_dev_ptr.offset((i * batch_size) as isize) };

        unsafe {
            ffi::cublasSgemv_v2(
                ctx.cublas_handle.0,
                1, 
                features as i32,
                current_batch_size as i32,
                &alpha,
                x_batch_ptr.as_raw() as *const c_void,
                features as i32,
                theta_dev_ptr.as_raw() as *const c_void,
                1,
                &beta,
                z_dev.as_device_ptr().as_raw() as *mut c_void,
                1,
            );

            launch!(fused_sigmoid_sub_f<<<((current_batch_size as u32 + 255)/256, 1, 1), (256, 1, 1), 0, stream>>>(
                z_dev.as_device_ptr(), y_batch_ptr, error_dev.as_device_ptr(), current_batch_size as i32
            )).map_err(to_py_err)?;

            ffi::cublasSgemv_v2(
                ctx.cublas_handle.0,
                0,
                features as i32,
                current_batch_size as i32,
                &alpha,
                x_batch_ptr.as_raw() as *const c_void,
                features as i32,
                error_dev.as_device_ptr().as_raw() as *const c_void,
                1,
                &beta,
                grad_dev.as_device_ptr().as_raw() as *mut c_void,
                1,
            );

            let update_alpha = -lr / (current_batch_size as f32);
            launch!(axpy_f<<<((features as u32 + 255)/256, 1, 1), (256, 1, 1), 0, stream>>>(
                update_alpha, grad_dev.as_device_ptr(), theta_dev_ptr, features as i32
            )).map_err(to_py_err)?;
        }
    }
    ctx.stream.synchronize().map_err(to_py_err)?;
    Ok(())
}

#[pymodule]
fn rusty_machine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(solve_normal_equation_device, m)?)?;
    m.add_function(wrap_pyfunction!(train_logistic_minibatch_gpu, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_matrix_multiply, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_transpose, m)?)?;
    m.add_function(wrap_pyfunction!(gpu_inverse, m)?)?;
    Ok(())
}