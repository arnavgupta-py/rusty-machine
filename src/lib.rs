use cust::prelude::*;
use pyo3::prelude::*;
use pyo3::Bound;
use pyo3::types::PyModule;
use std::os::raw::c_int;
use cust::memory::{DeviceMemory, DevicePointer, DeviceSlice};
use lbfgs::Lbfgs;

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

struct GpuObjective<'a> {
    samples: usize,
    features: usize,
    x_ptr: u64,
    y_ptr: u64,
    module: &'a Module,
    stream: &'a Stream,
}

fn objective_function(instance: &mut GpuObjective, x: &[f64], g: &mut [f64]) -> f64 {
    let theta_f32: Vec<f32> = x.iter().map(|&val| val as f32).collect();
    let mut grad_f32 = vec![0.0f32; instance.features];
    let cost;

    unsafe {
        let theta_dev = DeviceBuffer::from_slice(&theta_f32).unwrap();
        let grad_dev = DeviceBuffer::from_slice(&mut grad_f32).unwrap();
        let stream = instance.stream;

        let h_dev = DeviceBuffer::<f32>::uninitialized(instance.samples).unwrap();
        gpu_matrix_multiply(instance.x_ptr, theta_dev.as_raw_ptr() as u64, h_dev.as_raw_ptr() as u64, instance.samples, instance.features, 1).unwrap();

        let sigmoid_func = instance.module.get_function("elementwise_sigmoid").unwrap();
        let grid_dims = ((instance.samples as u32 + 255) / 256, 1, 1);
        let block_dims = (256, 1, 1);
        launch!(sigmoid_func<<<grid_dims, block_dims, 0, stream>>>(h_dev.as_device_ptr(), instance.samples as i32)).unwrap();
        
        let error_dev = DeviceBuffer::<f32>::uninitialized(instance.samples).unwrap();
        let sub_func = instance.module.get_function("elementwise_sub").unwrap();
        launch!(sub_func<<<grid_dims, block_dims, 0, stream>>>(h_dev.as_device_ptr(), DevicePointer::<f32>::from_raw(instance.y_ptr), error_dev.as_device_ptr(), instance.samples as i32)).unwrap();
        
        let x_t_dev = DeviceBuffer::<f32>::uninitialized(instance.features * instance.samples).unwrap();
        gpu_transpose(instance.x_ptr, x_t_dev.as_raw_ptr() as u64, instance.samples, instance.features).unwrap();
        
        gpu_matrix_multiply(x_t_dev.as_raw_ptr() as u64, error_dev.as_raw_ptr() as u64, grad_dev.as_raw_ptr() as u64, instance.features, instance.samples, 1).unwrap();
        
        let h_log_dev = h_dev.clone();
        let one_minus_h_dev = DeviceBuffer::from_slice(&vec![1.0f32; instance.samples]).unwrap();
        let axpy_func = instance.module.get_function("axpy").unwrap();
        launch!(axpy_func<<<grid_dims, block_dims, 0, stream>>>(-1.0f32, h_dev.as_device_ptr(), one_minus_h_dev.as_device_ptr(), instance.samples as i32)).unwrap();
        let log_func = instance.module.get_function("elementwise_log").unwrap();
        launch!(log_func<<<grid_dims, block_dims, 0, stream>>>(h_log_dev.as_device_ptr(), instance.samples as i32)).unwrap();
        launch!(log_func<<<grid_dims, block_dims, 0, stream>>>(one_minus_h_dev.as_device_ptr(), instance.samples as i32)).unwrap();
        
        let cost_vals_dev = DeviceBuffer::<f32>::uninitialized(instance.samples).unwrap();
        let cost_kernel_func = instance.module.get_function("cost_kernel").unwrap();
        launch!(cost_kernel_func<<<grid_dims, block_dims, 0, stream>>>(DevicePointer::<f32>::from_raw(instance.y_ptr), h_log_dev.as_device_ptr(), one_minus_h_dev.as_device_ptr(), cost_vals_dev.as_device_ptr(), instance.samples as i32)).unwrap();
        let num_blocks = (instance.samples + 255) / 256;
        let partial_sums_dev = DeviceBuffer::<f32>::uninitialized(num_blocks).unwrap();
        let sum_reduction_func = instance.module.get_function("sum_reduction").unwrap();
        launch!(sum_reduction_func<<<grid_dims, block_dims, 0, stream>>>(cost_vals_dev.as_device_ptr(), partial_sums_dev.as_device_ptr(), instance.samples as i32)).unwrap();

        let mut host_sums = vec![0.0f32; num_blocks];
        stream.synchronize().unwrap();
        partial_sums_dev.copy_to(&mut host_sums).unwrap();
        grad_dev.copy_to(&mut grad_f32).unwrap();
        
        cost = host_sums.iter().sum::<f32>() / instance.samples as f32;
    }

    for i in 0..instance.features {
        g[i] = (grad_f32[i] / instance.samples as f32) as f64;
    }
    cost as f64
}

#[pyfunction]
fn train_logistic_lbfgs(x_ptr: u64, y_ptr: u64, theta_ptr: u64, samples: usize, features: usize, max_iter: usize, _tol: f32) -> PyResult<()> {
    let to_py_err = |e: cust::error::CudaError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string());
    
    cust::init(CudaFlags::empty()).map_err(to_py_err)?;
    let device = Device::get_device(0).map_err(to_py_err)?;
    let _context = Context::new(device).map_err(to_py_err)?;
    let module = Module::from_ptx(include_str!("kernels.ptx"), &[]).map_err(to_py_err)?;
    let stream = Stream::new(StreamFlags::DEFAULT, None).map_err(to_py_err)?;

    let mut objective = GpuObjective {
        samples, features, x_ptr, y_ptr,
        module: &module, stream: &stream,
    };

    let mut x = vec![0.0f64; features];
    let mut lbfgs = Lbfgs::new(features, 10);
    let mut prev_x = x.clone();
    let mut prev_gx = vec![0.0f64; features];
    
    for iteration in 0..max_iter {
        let mut gx = vec![0.0f64; features];
        let fx = objective_function(&mut objective, &x, &mut gx);

        let grad_norm: f64 = gx.iter().map(|g| g * g).sum::<f64>().sqrt();
        if grad_norm < 1e-5 {
            break;
        }
        
        if iteration > 0 {
            let s: Vec<f64> = x.iter().zip(prev_x.iter()).map(|(a, b)| a - b).collect();
            let y: Vec<f64> = gx.iter().zip(prev_gx.iter()).map(|(a, b)| a - b).collect();
            lbfgs.update_hessian(&s, &y);
        }
        
        let mut direction = gx.clone();
        lbfgs.apply_hessian(&mut direction);
        
        for d in direction.iter_mut() {
            *d = -*d;
        }
        
        let mut step_size = 1.0;
        let c1 = 1e-4;
        let grad_dot_dir: f64 = gx.iter().zip(direction.iter()).map(|(g, d)| g * d).sum();
        
        prev_x.copy_from_slice(&x);
        prev_gx.copy_from_slice(&gx);
        
        for _ in 0..20 {
            for i in 0..features {
                x[i] = prev_x[i] + step_size * direction[i];
            }
            
            let mut dummy_grad = vec![0.0f64; features];
            let new_fx = objective_function(&mut objective, &x, &mut dummy_grad);
            
            if new_fx <= fx + c1 * step_size * grad_dot_dir {
                break;
            }
            step_size *= 0.5;
        }
    }
    
    let final_theta_f32: Vec<f32> = x.iter().map(|&val| val as f32).collect();

    unsafe {
        let final_theta_dev = DeviceBuffer::from_slice(&final_theta_f32).map_err(to_py_err)?;
        let mut theta_out_slice = DeviceSlice::from_raw_parts_mut(DevicePointer::<f32>::from_raw(theta_ptr), features);
        final_theta_dev.copy_to(&mut theta_out_slice).map_err(to_py_err)?;
    }

    stream.synchronize().map_err(to_py_err)?;
    Ok(())
}

#[pyfunction]
fn gpu_matrix_multiply(a_ptr: u64, b_ptr: u64, c_ptr: u64, m: usize, n: usize, k: usize) -> PyResult<()> {
    let to_py_err = |e: cust::error::CudaError| pyo3::exceptions::PyRuntimeError::new_err(e.to_string());
    
    cust::init(CudaFlags::empty()).map_err(to_py_err)?;
    let device = Device::get_device(0).map_err(to_py_err)?;
    let _context = Context::new(device).map_err(to_py_err)?;
    let module = Module::from_ptx(include_str!("kernels.ptx"), &[]).map_err(to_py_err)?;
    let stream = Stream::new(StreamFlags::DEFAULT, None).map_err(to_py_err)?;

    let func = module.get_function("matmul").map_err(to_py_err)?;
    let grid_dims = ((k as u32 + 15) / 16, (m as u32 + 15) / 16, 1);
    let block_dims = (16, 16, 1);
    let stream_ref = &stream;

    unsafe {
        let a_dev = DevicePointer::<f32>::from_raw(a_ptr);
        let b_dev = DevicePointer::<f32>::from_raw(b_ptr);
        let c_dev = DevicePointer::<f32>::from_raw(c_ptr);

        launch!(func<<<grid_dims, block_dims, 0, stream_ref>>>(
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

    let func = module.get_function("transpose").map_err(to_py_err)?;
    let grid_dims = ((n as u32 + 15) / 16, (m as u32 + 15) / 16, 1);
    let block_dims = (16, 16, 1);
    let stream_ref = &stream;
    
    unsafe {
        let in_dev = DevicePointer::<f32>::from_raw(in_ptr);
        let out_dev = DevicePointer::<f32>::from_raw(out_ptr);

        launch!(func<<<grid_dims, block_dims, 0, stream_ref>>>(
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
    let inv_dev_temp = DeviceBuffer::from_slice(&identity).map_err(to_py_err)?;
    
    unsafe {
        let a_dev_ptr = a_ptr as *mut f32;
        ffi::cusolverDnCreate(&mut handle);
        let ipiv_dev = DeviceBuffer::<c_int>::uninitialized(n).map_err(to_py_err)?;
        let info_dev = DeviceBuffer::<c_int>::uninitialized(1).map_err(to_py_err)?;
        let mut lwork: c_int = 0;

        ffi::cusolverDnSgetrf_bufferSize(handle, n as c_int, n as c_int, a_dev_ptr, n as c_int, &mut lwork);
        let work_dev = DeviceBuffer::<f32>::uninitialized(lwork as usize).map_err(to_py_err)?;

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
        let xtx_dev = DeviceBuffer::<f32>::uninitialized(features * features).map_err(to_py_err)?;
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
    m.add_function(wrap_pyfunction!(train_logistic_lbfgs, m)?)?;
    Ok(())
}