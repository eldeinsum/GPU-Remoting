use crate::types::cublas::*;
use codegen::{cuda_custom_hook, cuda_hook};
use std::os::raw::*;

#[cuda_hook(proc_id = 1100)]
fn cublasCreate_v2(handle: *mut cublasHandle_t) -> cublasStatus_t {
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            CUBLAS_CACHE
                .write()
                .unwrap()
                .pointer_modes
                .insert(*handle, cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        }
    }
}

#[cuda_hook(proc_id = 1101)]
fn cublasDestroy_v2(handle: cublasHandle_t) -> cublasStatus_t {
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            CUBLAS_CACHE.write().unwrap().pointer_modes.remove(&handle);
        }
    }
}

#[cuda_hook(proc_id = 1102)]
fn cublasGetPointerMode_v2(
    handle: cublasHandle_t,
    mode: *mut cublasPointerMode_t,
) -> cublasStatus_t {
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            CUBLAS_CACHE
                .write()
                .unwrap()
                .pointer_modes
                .insert(handle, *mode);
        }
    }
}

#[cuda_hook(proc_id = 1103)]
fn cublasSetPointerMode_v2(handle: cublasHandle_t, mode: cublasPointerMode_t) -> cublasStatus_t {
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            CUBLAS_CACHE
                .write()
                .unwrap()
                .pointer_modes
                .insert(handle, mode);
        }
    }
}

#[cuda_hook(proc_id = 1104, async_api)]
fn cublasSetStream_v2(handle: cublasHandle_t, streamId: cudaStream_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1106)]
fn cublasGetVersion_v2(handle: cublasHandle_t, version: *mut c_int) -> cublasStatus_t;

#[cuda_hook(proc_id = 1107)]
fn cublasGetProperty(type_: libraryPropertyType, value: *mut c_int) -> cublasStatus_t;

#[cuda_custom_hook] // local: derived from remoted runtime version query
fn cublasGetCudartVersion() -> usize;

#[cuda_hook(proc_id = 1108)]
fn cublasGetStream_v2(handle: cublasHandle_t, streamId: *mut cudaStream_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1109)]
fn cublasGetAtomicsMode(handle: cublasHandle_t, mode: *mut cublasAtomicsMode_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1110)]
fn cublasSetAtomicsMode(handle: cublasHandle_t, mode: cublasAtomicsMode_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1113)]
fn cublasSetVector(
    n: c_int,
    elemSize: c_int,
    #[skip] x: *const c_void,
    incx: c_int,
    #[device] devicePtr: *mut c_void,
    incy: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let n_usize = if n > 0 {
            usize::try_from(n).unwrap()
        } else {
            0
        };
        let elem_size = if elemSize > 0 {
            usize::try_from(elemSize).unwrap()
        } else {
            0
        };
        let incx_isize = isize::try_from(incx).unwrap();
        let elem_size_isize = isize::try_from(elemSize).unwrap();
        let mut packed = vec![0u8; n_usize * elem_size];
        if !packed.is_empty() {
            assert!(!x.is_null());
            assert!(incx != 0);
            for i in 0..n_usize {
                let src = unsafe {
                    x.cast::<u8>()
                        .offset(isize::try_from(i).unwrap() * incx_isize * elem_size_isize)
                };
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src,
                        packed.as_mut_ptr().add(i * elem_size),
                        elem_size,
                    );
                }
            }
        }
    }
    'client_extra_send: {
        send_slice(&packed, channel_sender).unwrap();
    }
    'server_extra_recv: {
        let packed = recv_slice::<u8, _>(channel_receiver).unwrap();
        let x_arg = if packed.is_empty() {
            std::ptr::null()
        } else {
            packed.as_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe { cublasSetVector(n, elemSize, x_arg, 1, devicePtr, incy) };
    }
}

#[cuda_hook(proc_id = 1114)]
fn cublasGetVector(
    n: c_int,
    elemSize: c_int,
    #[device] x: *const c_void,
    incx: c_int,
    #[skip] y: *mut c_void,
    incy: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let n_usize = if n > 0 {
            usize::try_from(n).unwrap()
        } else {
            0
        };
        let elem_size = if elemSize > 0 {
            usize::try_from(elemSize).unwrap()
        } else {
            0
        };
        let incy_isize = isize::try_from(incy).unwrap();
        let elem_size_isize = isize::try_from(elemSize).unwrap();
        if n_usize * elem_size > 0 {
            assert!(!y.is_null());
            assert!(incy != 0);
        }
    }
    'server_extra_recv: {
        let n_usize = if n > 0 {
            usize::try_from(n).unwrap()
        } else {
            0
        };
        let elem_size = if elemSize > 0 {
            usize::try_from(elemSize).unwrap()
        } else {
            0
        };
        let mut packed = vec![0u8; n_usize * elem_size];
        let y_arg = if packed.is_empty() {
            std::ptr::null_mut()
        } else {
            packed.as_mut_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe { cublasGetVector(n, elemSize, x, incx, y_arg, 1) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            send_slice(&packed, channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            let packed = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert_eq!(packed.len(), n_usize * elem_size);
            for i in 0..n_usize {
                let dst = unsafe {
                    y.cast::<u8>()
                        .offset(isize::try_from(i).unwrap() * incy_isize * elem_size_isize)
                };
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        packed.as_ptr().add(i * elem_size),
                        dst,
                        elem_size,
                    );
                }
            }
        }
    }
}

#[cuda_hook(proc_id = 1119, async_api)]
fn cublasSetMathMode(handle: cublasHandle_t, mode: cublasMath_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1121)]
fn cublasSetVector_64(
    n: i64,
    elemSize: i64,
    #[skip] x: *const c_void,
    incx: i64,
    #[device] devicePtr: *mut c_void,
    incy: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let n_usize = if n > 0 {
            usize::try_from(n).unwrap()
        } else {
            0
        };
        let elem_size = if elemSize > 0 {
            usize::try_from(elemSize).unwrap()
        } else {
            0
        };
        let incx_isize = isize::try_from(incx).unwrap();
        let elem_size_isize = isize::try_from(elemSize).unwrap();
        let mut packed = vec![0u8; n_usize * elem_size];
        if !packed.is_empty() {
            assert!(!x.is_null());
            assert!(incx != 0);
            for i in 0..n_usize {
                let src = unsafe {
                    x.cast::<u8>()
                        .offset(isize::try_from(i).unwrap() * incx_isize * elem_size_isize)
                };
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        src,
                        packed.as_mut_ptr().add(i * elem_size),
                        elem_size,
                    );
                }
            }
        }
    }
    'client_extra_send: {
        send_slice(&packed, channel_sender).unwrap();
    }
    'server_extra_recv: {
        let packed = recv_slice::<u8, _>(channel_receiver).unwrap();
        let x_arg = if packed.is_empty() {
            std::ptr::null()
        } else {
            packed.as_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe { cublasSetVector_64(n, elemSize, x_arg, 1, devicePtr, incy) };
    }
}

#[cuda_hook(proc_id = 1122)]
fn cublasGetVector_64(
    n: i64,
    elemSize: i64,
    #[device] x: *const c_void,
    incx: i64,
    #[skip] y: *mut c_void,
    incy: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let n_usize = if n > 0 {
            usize::try_from(n).unwrap()
        } else {
            0
        };
        let elem_size = if elemSize > 0 {
            usize::try_from(elemSize).unwrap()
        } else {
            0
        };
        let incy_isize = isize::try_from(incy).unwrap();
        let elem_size_isize = isize::try_from(elemSize).unwrap();
        if n_usize * elem_size > 0 {
            assert!(!y.is_null());
            assert!(incy != 0);
        }
    }
    'server_extra_recv: {
        let n_usize = if n > 0 {
            usize::try_from(n).unwrap()
        } else {
            0
        };
        let elem_size = if elemSize > 0 {
            usize::try_from(elemSize).unwrap()
        } else {
            0
        };
        let mut packed = vec![0u8; n_usize * elem_size];
        let y_arg = if packed.is_empty() {
            std::ptr::null_mut()
        } else {
            packed.as_mut_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe { cublasGetVector_64(n, elemSize, x, incx, y_arg, 1) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            send_slice(&packed, channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            let packed = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert_eq!(packed.len(), n_usize * elem_size);
            for i in 0..n_usize {
                let dst = unsafe {
                    y.cast::<u8>()
                        .offset(isize::try_from(i).unwrap() * incy_isize * elem_size_isize)
                };
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        packed.as_ptr().add(i * elem_size),
                        dst,
                        elem_size,
                    );
                }
            }
        }
    }
}

#[cuda_hook(proc_id = 1123)]
fn cublasGetSmCountTarget(handle: cublasHandle_t, smCountTarget: *mut c_int) -> cublasStatus_t;

#[cuda_hook(proc_id = 1124)]
fn cublasSetSmCountTarget(handle: cublasHandle_t, smCountTarget: c_int) -> cublasStatus_t;

#[cuda_hook(proc_id = 1125)]
fn cublasGetVersion(version: *mut c_int) -> cublasStatus_t;

#[cuda_custom_hook] // local: returns a client-owned C string
fn cublasGetStatusName(status: cublasStatus_t) -> *const c_char;

#[cuda_custom_hook] // local: returns a client-owned C string
fn cublasGetStatusString(status: cublasStatus_t) -> *const c_char;

#[cuda_hook(proc_id = 1300, async_api)]
fn cublasSgemm_v2(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: c_int,
    #[device] B: *const f32,
    ldb: c_int,
    #[skip] beta: *const f32,
    #[device] C: *mut f32,
    ldc: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f32;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = 0.0f32;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f32
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const f32
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasSgemm_v2(
                handle, transa, transb, m, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1240, async_api)]
fn cublasSgemm_v2_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    k: i64,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: i64,
    #[device] B: *const f32,
    ldb: i64,
    #[skip] beta: *const f32,
    #[device] C: *mut f32,
    ldc: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f32;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = 0.0f32;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f32
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const f32
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasSgemm_v2_64(
                handle, transa, transb, m, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1241, async_api)]
fn cublasDgemm_v2(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: c_int,
    #[device] B: *const f64,
    ldb: c_int,
    #[skip] beta: *const f64,
    #[device] C: *mut f64,
    ldc: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f64;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = 0.0f64;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f64
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const f64
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasDgemm_v2(
                handle, transa, transb, m, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1242, async_api)]
fn cublasDgemm_v2_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    k: i64,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: i64,
    #[device] B: *const f64,
    ldb: i64,
    #[skip] beta: *const f64,
    #[device] C: *mut f64,
    ldc: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f64;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = 0.0f64;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f64
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const f64
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasDgemm_v2_64(
                handle, transa, transb, m, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1243, async_api)]
fn cublasCgemm_v2(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: c_int,
    #[device] B: *const cuComplex,
    ldb: c_int,
    #[skip] beta: *const cuComplex,
    #[device] C: *mut cuComplex,
    ldc: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            cuComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = cuComplex { x: 0.0, y: 0.0 };
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuComplex
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const cuComplex
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasCgemm_v2(
                handle, transa, transb, m, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1244, async_api)]
fn cublasCgemm_v2_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    k: i64,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: i64,
    #[device] B: *const cuComplex,
    ldb: i64,
    #[skip] beta: *const cuComplex,
    #[device] C: *mut cuComplex,
    ldc: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            cuComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = cuComplex { x: 0.0, y: 0.0 };
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuComplex
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const cuComplex
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasCgemm_v2_64(
                handle, transa, transb, m, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1245, async_api)]
fn cublasZgemm_v2(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
    #[device] B: *const cuDoubleComplex,
    ldb: c_int,
    #[skip] beta: *const cuDoubleComplex,
    #[device] C: *mut cuDoubleComplex,
    ldc: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuDoubleComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            cuDoubleComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuDoubleComplex
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const cuDoubleComplex
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasZgemm_v2(
                handle, transa, transb, m, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1246, async_api)]
fn cublasZgemm_v2_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    k: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: i64,
    #[device] B: *const cuDoubleComplex,
    ldb: i64,
    #[skip] beta: *const cuDoubleComplex,
    #[device] C: *mut cuDoubleComplex,
    ldc: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuDoubleComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            cuDoubleComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuDoubleComplex
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const cuDoubleComplex
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasZgemm_v2_64(
                handle, transa, transb, m, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1313, async_api)]
fn cublasSgemmStridedBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: c_int,
    strideA: c_longlong,
    #[device] B: *const f32,
    ldb: c_int,
    strideB: c_longlong,
    #[skip] beta: *const f32,
    #[device] C: *mut f32,
    ldc: c_int,
    strideC: c_longlong,
    batchCount: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f32;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = 0.0f32;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f32
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const f32
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasSgemmStridedBatched(
                handle, transa, transb, m, n, k, alpha_arg, A, lda, strideA, B, ldb, strideB,
                beta_arg, C, ldc, strideC, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1247, async_api)]
fn cublasSgemmStridedBatched_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    k: i64,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: i64,
    strideA: c_longlong,
    #[device] B: *const f32,
    ldb: i64,
    strideB: c_longlong,
    #[skip] beta: *const f32,
    #[device] C: *mut f32,
    ldc: i64,
    strideC: c_longlong,
    batchCount: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f32;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = 0.0f32;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f32
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const f32
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasSgemmStridedBatched_64(
                handle, transa, transb, m, n, k, alpha_arg, A, lda, strideA, B, ldb, strideB,
                beta_arg, C, ldc, strideC, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1248, async_api)]
fn cublasDgemmStridedBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: c_int,
    strideA: c_longlong,
    #[device] B: *const f64,
    ldb: c_int,
    strideB: c_longlong,
    #[skip] beta: *const f64,
    #[device] C: *mut f64,
    ldc: c_int,
    strideC: c_longlong,
    batchCount: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f64;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = 0.0f64;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f64
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const f64
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasDgemmStridedBatched(
                handle, transa, transb, m, n, k, alpha_arg, A, lda, strideA, B, ldb, strideB,
                beta_arg, C, ldc, strideC, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1249, async_api)]
fn cublasDgemmStridedBatched_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    k: i64,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: i64,
    strideA: c_longlong,
    #[device] B: *const f64,
    ldb: i64,
    strideB: c_longlong,
    #[skip] beta: *const f64,
    #[device] C: *mut f64,
    ldc: i64,
    strideC: c_longlong,
    batchCount: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f64;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = 0.0f64;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f64
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const f64
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasDgemmStridedBatched_64(
                handle, transa, transb, m, n, k, alpha_arg, A, lda, strideA, B, ldb, strideB,
                beta_arg, C, ldc, strideC, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1250, async_api)]
fn cublasCgemmStridedBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: c_int,
    strideA: c_longlong,
    #[device] B: *const cuComplex,
    ldb: c_int,
    strideB: c_longlong,
    #[skip] beta: *const cuComplex,
    #[device] C: *mut cuComplex,
    ldc: c_int,
    strideC: c_longlong,
    batchCount: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            cuComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = cuComplex { x: 0.0, y: 0.0 };
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuComplex
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const cuComplex
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasCgemmStridedBatched(
                handle, transa, transb, m, n, k, alpha_arg, A, lda, strideA, B, ldb, strideB,
                beta_arg, C, ldc, strideC, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1251, async_api)]
fn cublasCgemmStridedBatched_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    k: i64,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: i64,
    strideA: c_longlong,
    #[device] B: *const cuComplex,
    ldb: i64,
    strideB: c_longlong,
    #[skip] beta: *const cuComplex,
    #[device] C: *mut cuComplex,
    ldc: i64,
    strideC: c_longlong,
    batchCount: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            cuComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = cuComplex { x: 0.0, y: 0.0 };
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuComplex
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const cuComplex
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasCgemmStridedBatched_64(
                handle, transa, transb, m, n, k, alpha_arg, A, lda, strideA, B, ldb, strideB,
                beta_arg, C, ldc, strideC, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1252, async_api)]
fn cublasZgemmStridedBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
    strideA: c_longlong,
    #[device] B: *const cuDoubleComplex,
    ldb: c_int,
    strideB: c_longlong,
    #[skip] beta: *const cuDoubleComplex,
    #[device] C: *mut cuDoubleComplex,
    ldc: c_int,
    strideC: c_longlong,
    batchCount: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuDoubleComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            cuDoubleComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuDoubleComplex
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const cuDoubleComplex
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasZgemmStridedBatched(
                handle, transa, transb, m, n, k, alpha_arg, A, lda, strideA, B, ldb, strideB,
                beta_arg, C, ldc, strideC, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1253, async_api)]
fn cublasZgemmStridedBatched_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    k: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: i64,
    strideA: c_longlong,
    #[device] B: *const cuDoubleComplex,
    ldb: i64,
    strideB: c_longlong,
    #[skip] beta: *const cuDoubleComplex,
    #[device] C: *mut cuDoubleComplex,
    ldc: i64,
    strideC: c_longlong,
    batchCount: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuDoubleComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            cuDoubleComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuDoubleComplex
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const cuDoubleComplex
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasZgemmStridedBatched_64(
                handle, transa, transb, m, n, k, alpha_arg, A, lda, strideA, B, ldb, strideB,
                beta_arg, C, ldc, strideC, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1260, async_api)]
fn cublasSgemmBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const f32,
    #[device] Aarray: *const *const f32,
    lda: c_int,
    #[device] Barray: *const *const f32,
    ldb: c_int,
    #[skip] beta: *const f32,
    #[device] Carray: *const *mut f32,
    ldc: c_int,
    batchCount: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f32;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = 0.0f32;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f32
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const f32
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasSgemmBatched(
                handle, transa, transb, m, n, k, alpha_arg, Aarray, lda, Barray, ldb, beta_arg,
                Carray, ldc, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1261, async_api)]
fn cublasSgemmBatched_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    k: i64,
    #[skip] alpha: *const f32,
    #[device] Aarray: *const *const f32,
    lda: i64,
    #[device] Barray: *const *const f32,
    ldb: i64,
    #[skip] beta: *const f32,
    #[device] Carray: *const *mut f32,
    ldc: i64,
    batchCount: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f32;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = 0.0f32;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f32
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const f32
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasSgemmBatched_64(
                handle, transa, transb, m, n, k, alpha_arg, Aarray, lda, Barray, ldb, beta_arg,
                Carray, ldc, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1262, async_api)]
fn cublasDgemmBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const f64,
    #[device] Aarray: *const *const f64,
    lda: c_int,
    #[device] Barray: *const *const f64,
    ldb: c_int,
    #[skip] beta: *const f64,
    #[device] Carray: *const *mut f64,
    ldc: c_int,
    batchCount: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f64;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = 0.0f64;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f64
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const f64
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasDgemmBatched(
                handle, transa, transb, m, n, k, alpha_arg, Aarray, lda, Barray, ldb, beta_arg,
                Carray, ldc, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1263, async_api)]
fn cublasDgemmBatched_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    k: i64,
    #[skip] alpha: *const f64,
    #[device] Aarray: *const *const f64,
    lda: i64,
    #[device] Barray: *const *const f64,
    ldb: i64,
    #[skip] beta: *const f64,
    #[device] Carray: *const *mut f64,
    ldc: i64,
    batchCount: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f64;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = 0.0f64;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f64
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const f64
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasDgemmBatched_64(
                handle, transa, transb, m, n, k, alpha_arg, Aarray, lda, Barray, ldb, beta_arg,
                Carray, ldc, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1264, async_api)]
fn cublasCgemmBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] Aarray: *const *const cuComplex,
    lda: c_int,
    #[device] Barray: *const *const cuComplex,
    ldb: c_int,
    #[skip] beta: *const cuComplex,
    #[device] Carray: *const *mut cuComplex,
    ldc: c_int,
    batchCount: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            cuComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = cuComplex { x: 0.0, y: 0.0 };
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuComplex
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const cuComplex
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasCgemmBatched(
                handle, transa, transb, m, n, k, alpha_arg, Aarray, lda, Barray, ldb, beta_arg,
                Carray, ldc, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1265, async_api)]
fn cublasCgemmBatched_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    k: i64,
    #[skip] alpha: *const cuComplex,
    #[device] Aarray: *const *const cuComplex,
    lda: i64,
    #[device] Barray: *const *const cuComplex,
    ldb: i64,
    #[skip] beta: *const cuComplex,
    #[device] Carray: *const *mut cuComplex,
    ldc: i64,
    batchCount: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            cuComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = cuComplex { x: 0.0, y: 0.0 };
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuComplex
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const cuComplex
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasCgemmBatched_64(
                handle, transa, transb, m, n, k, alpha_arg, Aarray, lda, Barray, ldb, beta_arg,
                Carray, ldc, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1266, async_api)]
fn cublasZgemmBatched(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] Aarray: *const *const cuDoubleComplex,
    lda: c_int,
    #[device] Barray: *const *const cuDoubleComplex,
    ldb: c_int,
    #[skip] beta: *const cuDoubleComplex,
    #[device] Carray: *const *mut cuDoubleComplex,
    ldc: c_int,
    batchCount: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuDoubleComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            cuDoubleComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuDoubleComplex
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const cuDoubleComplex
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasZgemmBatched(
                handle, transa, transb, m, n, k, alpha_arg, Aarray, lda, Barray, ldb, beta_arg,
                Carray, ldc, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1267, async_api)]
fn cublasZgemmBatched_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    k: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] Aarray: *const *const cuDoubleComplex,
    lda: i64,
    #[device] Barray: *const *const cuDoubleComplex,
    ldb: i64,
    #[skip] beta: *const cuDoubleComplex,
    #[device] Carray: *const *mut cuDoubleComplex,
    ldc: i64,
    batchCount: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuDoubleComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            cuDoubleComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuDoubleComplex
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const cuDoubleComplex
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasZgemmBatched_64(
                handle, transa, transb, m, n, k, alpha_arg, Aarray, lda, Barray, ldb, beta_arg,
                Carray, ldc, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1120)]
fn cublasGetMathMode(handle: cublasHandle_t, mode: *mut cublasMath_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1254, async_api)]
fn cublasSgemmEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const f32,
    #[device] A: *const c_void,
    Atype: cudaDataType,
    lda: c_int,
    #[device] B: *const c_void,
    Btype: cudaDataType,
    ldb: c_int,
    #[skip] beta: *const f32,
    #[device] C: *mut c_void,
    Ctype: cudaDataType,
    ldc: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f32;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = 0.0f32;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f32
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const f32
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasSgemmEx(
                handle, transa, transb, m, n, k, alpha_arg, A, Atype, lda, B, Btype, ldb, beta_arg,
                C, Ctype, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1255, async_api)]
fn cublasSgemmEx_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    k: i64,
    #[skip] alpha: *const f32,
    #[device] A: *const c_void,
    Atype: cudaDataType,
    lda: i64,
    #[device] B: *const c_void,
    Btype: cudaDataType,
    ldb: i64,
    #[skip] beta: *const f32,
    #[device] C: *mut c_void,
    Ctype: cudaDataType,
    ldc: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f32;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = 0.0f32;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f32
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const f32
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasSgemmEx_64(
                handle, transa, transb, m, n, k, alpha_arg, A, Atype, lda, B, Btype, ldb, beta_arg,
                C, Ctype, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1441, async_api)]
fn cublasGemmEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const c_void,
    #[device] A: *const c_void,
    Atype: cudaDataType,
    lda: c_int,
    #[device] B: *const c_void,
    Btype: cudaDataType,
    ldb: c_int,
    #[skip] beta: *const c_void,
    #[device] C: *mut c_void,
    Ctype: cudaDataType,
    ldc: c_int,
    computeType: cublasComputeType_t,
    algo: cublasGemmAlgo_t,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha.cast::<f32>() }
        };
        let beta_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!beta.is_null());
            unsafe { *beta.cast::<f32>() }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f32;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = 0.0f32;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const c_void
        } else {
            (&raw const alpha_value).cast::<c_void>()
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const c_void
        } else {
            (&raw const beta_value).cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasGemmEx(
                handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha_arg,
                A,
                Atype,
                lda,
                B,
                Btype,
                ldb,
                beta_arg,
                C,
                Ctype,
                ldc,
                computeType,
                algo,
            )
        };
    }
}

#[cuda_hook(proc_id = 1256, async_api)]
fn cublasGemmEx_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    k: i64,
    #[skip] alpha: *const c_void,
    #[device] A: *const c_void,
    Atype: cudaDataType,
    lda: i64,
    #[device] B: *const c_void,
    Btype: cudaDataType,
    ldb: i64,
    #[skip] beta: *const c_void,
    #[device] C: *mut c_void,
    Ctype: cudaDataType,
    ldc: i64,
    computeType: cublasComputeType_t,
    algo: cublasGemmAlgo_t,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha.cast::<f32>() }
        };
        let beta_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!beta.is_null());
            unsafe { *beta.cast::<f32>() }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f32;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = 0.0f32;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const c_void
        } else {
            (&raw const alpha_value).cast::<c_void>()
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const c_void
        } else {
            (&raw const beta_value).cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasGemmEx_64(
                handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha_arg,
                A,
                Atype,
                lda,
                B,
                Btype,
                ldb,
                beta_arg,
                C,
                Ctype,
                ldc,
                computeType,
                algo,
            )
        };
    }
}

#[cuda_hook(proc_id = 1268, async_api)]
fn cublasGemmBatchedEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const c_void,
    #[device] Aarray: *const *const c_void,
    Atype: cudaDataType,
    lda: c_int,
    #[device] Barray: *const *const c_void,
    Btype: cudaDataType,
    ldb: c_int,
    #[skip] beta: *const c_void,
    #[device] Carray: *const *mut c_void,
    Ctype: cudaDataType,
    ldc: c_int,
    batchCount: c_int,
    computeType: cublasComputeType_t,
    algo: cublasGemmAlgo_t,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha.cast::<f32>() }
        };
        let beta_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!beta.is_null());
            unsafe { *beta.cast::<f32>() }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f32;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = 0.0f32;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const c_void
        } else {
            (&raw const alpha_value).cast::<c_void>()
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const c_void
        } else {
            (&raw const beta_value).cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasGemmBatchedEx(
                handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha_arg,
                Aarray,
                Atype,
                lda,
                Barray,
                Btype,
                ldb,
                beta_arg,
                Carray,
                Ctype,
                ldc,
                batchCount,
                computeType,
                algo,
            )
        };
    }
}

#[cuda_hook(proc_id = 1269, async_api)]
fn cublasGemmBatchedEx_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    k: i64,
    #[skip] alpha: *const c_void,
    #[device] Aarray: *const *const c_void,
    Atype: cudaDataType,
    lda: i64,
    #[device] Barray: *const *const c_void,
    Btype: cudaDataType,
    ldb: i64,
    #[skip] beta: *const c_void,
    #[device] Carray: *const *mut c_void,
    Ctype: cudaDataType,
    ldc: i64,
    batchCount: i64,
    computeType: cublasComputeType_t,
    algo: cublasGemmAlgo_t,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha.cast::<f32>() }
        };
        let beta_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!beta.is_null());
            unsafe { *beta.cast::<f32>() }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f32;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = 0.0f32;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const c_void
        } else {
            (&raw const alpha_value).cast::<c_void>()
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const c_void
        } else {
            (&raw const beta_value).cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasGemmBatchedEx_64(
                handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha_arg,
                Aarray,
                Atype,
                lda,
                Barray,
                Btype,
                ldb,
                beta_arg,
                Carray,
                Ctype,
                ldc,
                batchCount,
                computeType,
                algo,
            )
        };
    }
}

#[cuda_hook(proc_id = 1270, async_api)]
fn cublasSgemmGroupedBatched(
    handle: cublasHandle_t,
    #[host(len = group_count as usize)] transa_array: *const cublasOperation_t,
    #[host(len = group_count as usize)] transb_array: *const cublasOperation_t,
    #[host(len = group_count as usize)] m_array: *const c_int,
    #[host(len = group_count as usize)] n_array: *const c_int,
    #[host(len = group_count as usize)] k_array: *const c_int,
    #[host(len = group_count as usize)] alpha_array: *const f32,
    #[device] Aarray: *const *const f32,
    #[host(len = group_count as usize)] lda_array: *const c_int,
    #[device] Barray: *const *const f32,
    #[host(len = group_count as usize)] ldb_array: *const c_int,
    #[host(len = group_count as usize)] beta_array: *const f32,
    #[device] Carray: *const *mut f32,
    #[host(len = group_count as usize)] ldc_array: *const c_int,
    group_count: c_int,
    #[host(len = group_count as usize)] group_size: *const c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1271, async_api)]
fn cublasSgemmGroupedBatched_64(
    handle: cublasHandle_t,
    #[host(len = group_count as usize)] transa_array: *const cublasOperation_t,
    #[host(len = group_count as usize)] transb_array: *const cublasOperation_t,
    #[host(len = group_count as usize)] m_array: *const i64,
    #[host(len = group_count as usize)] n_array: *const i64,
    #[host(len = group_count as usize)] k_array: *const i64,
    #[host(len = group_count as usize)] alpha_array: *const f32,
    #[device] Aarray: *const *const f32,
    #[host(len = group_count as usize)] lda_array: *const i64,
    #[device] Barray: *const *const f32,
    #[host(len = group_count as usize)] ldb_array: *const i64,
    #[host(len = group_count as usize)] beta_array: *const f32,
    #[device] Carray: *const *mut f32,
    #[host(len = group_count as usize)] ldc_array: *const i64,
    group_count: i64,
    #[host(len = group_count as usize)] group_size: *const i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1272, async_api)]
fn cublasDgemmGroupedBatched(
    handle: cublasHandle_t,
    #[host(len = group_count as usize)] transa_array: *const cublasOperation_t,
    #[host(len = group_count as usize)] transb_array: *const cublasOperation_t,
    #[host(len = group_count as usize)] m_array: *const c_int,
    #[host(len = group_count as usize)] n_array: *const c_int,
    #[host(len = group_count as usize)] k_array: *const c_int,
    #[host(len = group_count as usize)] alpha_array: *const f64,
    #[device] Aarray: *const *const f64,
    #[host(len = group_count as usize)] lda_array: *const c_int,
    #[device] Barray: *const *const f64,
    #[host(len = group_count as usize)] ldb_array: *const c_int,
    #[host(len = group_count as usize)] beta_array: *const f64,
    #[device] Carray: *const *mut f64,
    #[host(len = group_count as usize)] ldc_array: *const c_int,
    group_count: c_int,
    #[host(len = group_count as usize)] group_size: *const c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1273, async_api)]
fn cublasDgemmGroupedBatched_64(
    handle: cublasHandle_t,
    #[host(len = group_count as usize)] transa_array: *const cublasOperation_t,
    #[host(len = group_count as usize)] transb_array: *const cublasOperation_t,
    #[host(len = group_count as usize)] m_array: *const i64,
    #[host(len = group_count as usize)] n_array: *const i64,
    #[host(len = group_count as usize)] k_array: *const i64,
    #[host(len = group_count as usize)] alpha_array: *const f64,
    #[device] Aarray: *const *const f64,
    #[host(len = group_count as usize)] lda_array: *const i64,
    #[device] Barray: *const *const f64,
    #[host(len = group_count as usize)] ldb_array: *const i64,
    #[host(len = group_count as usize)] beta_array: *const f64,
    #[device] Carray: *const *mut f64,
    #[host(len = group_count as usize)] ldc_array: *const i64,
    group_count: i64,
    #[host(len = group_count as usize)] group_size: *const i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1274, async_api)]
fn cublasGemmGroupedBatchedEx(
    handle: cublasHandle_t,
    #[host(len = group_count as usize)] transa_array: *const cublasOperation_t,
    #[host(len = group_count as usize)] transb_array: *const cublasOperation_t,
    #[host(len = group_count as usize)] m_array: *const c_int,
    #[host(len = group_count as usize)] n_array: *const c_int,
    #[host(len = group_count as usize)] k_array: *const c_int,
    #[host(len = group_count as usize * std::mem::size_of::<f32>())] alpha_array: *const c_void,
    #[device] Aarray: *const *const c_void,
    Atype: cudaDataType_t,
    #[host(len = group_count as usize)] lda_array: *const c_int,
    #[device] Barray: *const *const c_void,
    Btype: cudaDataType_t,
    #[host(len = group_count as usize)] ldb_array: *const c_int,
    #[host(len = group_count as usize * std::mem::size_of::<f32>())] beta_array: *const c_void,
    #[device] Carray: *const *mut c_void,
    Ctype: cudaDataType_t,
    #[host(len = group_count as usize)] ldc_array: *const c_int,
    group_count: c_int,
    #[host(len = group_count as usize)] group_size: *const c_int,
    computeType: cublasComputeType_t,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1275, async_api)]
fn cublasGemmGroupedBatchedEx_64(
    handle: cublasHandle_t,
    #[host(len = group_count as usize)] transa_array: *const cublasOperation_t,
    #[host(len = group_count as usize)] transb_array: *const cublasOperation_t,
    #[host(len = group_count as usize)] m_array: *const i64,
    #[host(len = group_count as usize)] n_array: *const i64,
    #[host(len = group_count as usize)] k_array: *const i64,
    #[host(len = group_count as usize * std::mem::size_of::<f32>())] alpha_array: *const c_void,
    #[device] Aarray: *const *const c_void,
    Atype: cudaDataType_t,
    #[host(len = group_count as usize)] lda_array: *const i64,
    #[device] Barray: *const *const c_void,
    Btype: cudaDataType_t,
    #[host(len = group_count as usize)] ldb_array: *const i64,
    #[host(len = group_count as usize * std::mem::size_of::<f32>())] beta_array: *const c_void,
    #[device] Carray: *const *mut c_void,
    Ctype: cudaDataType_t,
    #[host(len = group_count as usize)] ldc_array: *const i64,
    group_count: i64,
    #[host(len = group_count as usize)] group_size: *const i64,
    computeType: cublasComputeType_t,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1257, async_api)]
fn cublasCgemmEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const c_void,
    Atype: cudaDataType,
    lda: c_int,
    #[device] B: *const c_void,
    Btype: cudaDataType,
    ldb: c_int,
    #[skip] beta: *const cuComplex,
    #[device] C: *mut c_void,
    Ctype: cudaDataType,
    ldc: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            cuComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = cuComplex { x: 0.0, y: 0.0 };
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuComplex
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const cuComplex
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasCgemmEx(
                handle, transa, transb, m, n, k, alpha_arg, A, Atype, lda, B, Btype, ldb, beta_arg,
                C, Ctype, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1258, async_api)]
fn cublasCgemmEx_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    k: i64,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const c_void,
    Atype: cudaDataType,
    lda: i64,
    #[device] B: *const c_void,
    Btype: cudaDataType,
    ldb: i64,
    #[skip] beta: *const cuComplex,
    #[device] C: *mut c_void,
    Ctype: cudaDataType,
    ldc: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            cuComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!beta.is_null());
            unsafe { *beta }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = cuComplex { x: 0.0, y: 0.0 };
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuComplex
        } else {
            &raw const alpha_value
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const cuComplex
        } else {
            &raw const beta_value
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasCgemmEx_64(
                handle, transa, transb, m, n, k, alpha_arg, A, Atype, lda, B, Btype, ldb, beta_arg,
                C, Ctype, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1443, async_api)]
fn cublasGemmStridedBatchedEx(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const c_void,
    #[device] A: *const c_void,
    Atype: cudaDataType,
    lda: c_int,
    strideA: c_longlong,
    #[device] B: *const c_void,
    Btype: cudaDataType,
    ldb: c_int,
    strideB: c_longlong,
    #[skip] beta: *const c_void,
    #[device] C: *mut c_void,
    Ctype: cudaDataType,
    ldc: c_int,
    strideC: c_longlong,
    batchCount: c_int,
    computeType: cublasComputeType_t,
    algo: cublasGemmAlgo_t,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha.cast::<f32>() }
        };
        let beta_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!beta.is_null());
            unsafe { *beta.cast::<f32>() }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f32;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = 0.0f32;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const c_void
        } else {
            (&raw const alpha_value).cast::<c_void>()
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const c_void
        } else {
            (&raw const beta_value).cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasGemmStridedBatchedEx(
                handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha_arg,
                A,
                Atype,
                lda,
                strideA,
                B,
                Btype,
                ldb,
                strideB,
                beta_arg,
                C,
                Ctype,
                ldc,
                strideC,
                batchCount,
                computeType,
                algo,
            )
        };
    }
}

#[cuda_hook(proc_id = 1259, async_api)]
fn cublasGemmStridedBatchedEx_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    k: i64,
    #[skip] alpha: *const c_void,
    #[device] A: *const c_void,
    Atype: cudaDataType,
    lda: i64,
    strideA: c_longlong,
    #[device] B: *const c_void,
    Btype: cudaDataType,
    ldb: i64,
    strideB: c_longlong,
    #[skip] beta: *const c_void,
    #[device] C: *mut c_void,
    Ctype: cudaDataType,
    ldc: i64,
    strideC: c_longlong,
    batchCount: i64,
    computeType: cublasComputeType_t,
    algo: cublasGemmAlgo_t,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha.cast::<f32>() }
        };
        let beta_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!beta.is_null());
            unsafe { *beta.cast::<f32>() }
        };
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        beta_value.send(channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f32;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let mut beta_value = 0.0f32;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const c_void
        } else {
            (&raw const alpha_value).cast::<c_void>()
        };
        let beta_arg = if device_pointer_mode {
            beta_addr as *const c_void
        } else {
            (&raw const beta_value).cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasGemmStridedBatchedEx_64(
                handle,
                transa,
                transb,
                m,
                n,
                k,
                alpha_arg,
                A,
                Atype,
                lda,
                strideA,
                B,
                Btype,
                ldb,
                strideB,
                beta_arg,
                C,
                Ctype,
                ldc,
                strideC,
                batchCount,
                computeType,
                algo,
            )
        };
    }
}

#[cuda_hook(proc_id = 1105)]
fn cublasSetWorkspace_v2(
    handle: cublasHandle_t,
    #[device] workspace: *mut c_void,
    workspaceSizeInBytes: usize,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1111)]
fn cublasSetMatrix(
    rows: c_int,
    cols: c_int,
    elemSize: c_int,
    #[host(len = rows * cols * elemSize)] A: *const c_void,
    lda: c_int,
    #[device] B: *mut c_void,
    ldb: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1115, async_api)]
fn cublasSetMatrixAsync(
    rows: c_int,
    cols: c_int,
    elemSize: c_int,
    #[host(len = rows * cols * elemSize)] A: *const c_void,
    lda: c_int,
    #[device] B: *mut c_void,
    ldb: c_int,
    stream: cudaStream_t,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1112)]
fn cublasGetMatrix(
    rows: c_int,
    cols: c_int,
    elemSize: c_int,
    #[device] A: *const c_void,
    lda: c_int,
    #[host(output, len = rows * cols * elemSize)] B: *mut c_void,
    ldb: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1126)]
fn cublasScopy_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const f32,
    incx: c_int,
    #[device] y: *mut f32,
    incy: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1127)]
fn cublasScopy_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const f32,
    incx: i64,
    #[device] y: *mut f32,
    incy: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1128)]
fn cublasDcopy_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const f64,
    incx: c_int,
    #[device] y: *mut f64,
    incy: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1129)]
fn cublasDcopy_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const f64,
    incx: i64,
    #[device] y: *mut f64,
    incy: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1130)]
fn cublasCcopy_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[device] y: *mut cuComplex,
    incy: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1131)]
fn cublasCcopy_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const cuComplex,
    incx: i64,
    #[device] y: *mut cuComplex,
    incy: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1132)]
fn cublasZcopy_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[device] y: *mut cuDoubleComplex,
    incy: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1133)]
fn cublasZcopy_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[device] y: *mut cuDoubleComplex,
    incy: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1134)]
fn cublasSswap_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *mut f32,
    incx: c_int,
    #[device] y: *mut f32,
    incy: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1135)]
fn cublasSswap_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *mut f32,
    incx: i64,
    #[device] y: *mut f32,
    incy: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1136)]
fn cublasDswap_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *mut f64,
    incx: c_int,
    #[device] y: *mut f64,
    incy: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1137)]
fn cublasDswap_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *mut f64,
    incx: i64,
    #[device] y: *mut f64,
    incy: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1138)]
fn cublasCswap_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *mut cuComplex,
    incx: c_int,
    #[device] y: *mut cuComplex,
    incy: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1139)]
fn cublasCswap_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *mut cuComplex,
    incx: i64,
    #[device] y: *mut cuComplex,
    incy: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1140)]
fn cublasZswap_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *mut cuDoubleComplex,
    incx: c_int,
    #[device] y: *mut cuDoubleComplex,
    incy: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1141)]
fn cublasZswap_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *mut cuDoubleComplex,
    incx: i64,
    #[device] y: *mut cuDoubleComplex,
    incy: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1182)]
fn cublasSscal_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[skip] alpha: *const f32,
    #[device] x: *mut f32,
    incx: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let alpha_addr = alpha as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f32;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f32
        } else {
            &raw const alpha_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasSscal_v2(handle, n, alpha_arg, x, incx) };
    }
}

#[cuda_hook(proc_id = 1142)]
fn cublasSscal_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[skip] alpha: *const f32,
    #[device] x: *mut f32,
    incx: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let alpha_addr = alpha as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f32;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f32
        } else {
            &raw const alpha_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasSscal_v2_64(handle, n, alpha_arg, x, incx) };
    }
}

#[cuda_hook(proc_id = 1143)]
fn cublasDscal_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[skip] alpha: *const f64,
    #[device] x: *mut f64,
    incx: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let alpha_addr = alpha as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f64;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f64
        } else {
            &raw const alpha_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasDscal_v2(handle, n, alpha_arg, x, incx) };
    }
}

#[cuda_hook(proc_id = 1144)]
fn cublasDscal_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[skip] alpha: *const f64,
    #[device] x: *mut f64,
    incx: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let alpha_addr = alpha as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f64;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f64
        } else {
            &raw const alpha_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasDscal_v2_64(handle, n, alpha_arg, x, incx) };
    }
}

#[cuda_hook(proc_id = 1145)]
fn cublasCscal_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] x: *mut cuComplex,
    incx: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let alpha_addr = alpha as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuComplex
        } else {
            &raw const alpha_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasCscal_v2(handle, n, alpha_arg, x, incx) };
    }
}

#[cuda_hook(proc_id = 1146)]
fn cublasCscal_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[skip] alpha: *const cuComplex,
    #[device] x: *mut cuComplex,
    incx: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let alpha_addr = alpha as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuComplex
        } else {
            &raw const alpha_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasCscal_v2_64(handle, n, alpha_arg, x, incx) };
    }
}

#[cuda_hook(proc_id = 1147)]
fn cublasCsscal_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[skip] alpha: *const f32,
    #[device] x: *mut cuComplex,
    incx: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let alpha_addr = alpha as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f32;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f32
        } else {
            &raw const alpha_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasCsscal_v2(handle, n, alpha_arg, x, incx) };
    }
}

#[cuda_hook(proc_id = 1148)]
fn cublasCsscal_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[skip] alpha: *const f32,
    #[device] x: *mut cuComplex,
    incx: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let alpha_addr = alpha as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f32;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f32
        } else {
            &raw const alpha_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasCsscal_v2_64(handle, n, alpha_arg, x, incx) };
    }
}

#[cuda_hook(proc_id = 1149)]
fn cublasZscal_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] x: *mut cuDoubleComplex,
    incx: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuDoubleComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let alpha_addr = alpha as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuDoubleComplex
        } else {
            &raw const alpha_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasZscal_v2(handle, n, alpha_arg, x, incx) };
    }
}

#[cuda_hook(proc_id = 1150)]
fn cublasZscal_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] x: *mut cuDoubleComplex,
    incx: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuDoubleComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let alpha_addr = alpha as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuDoubleComplex
        } else {
            &raw const alpha_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasZscal_v2_64(handle, n, alpha_arg, x, incx) };
    }
}

#[cuda_hook(proc_id = 1151)]
fn cublasZdscal_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[skip] alpha: *const f64,
    #[device] x: *mut cuDoubleComplex,
    incx: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let alpha_addr = alpha as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f64;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f64
        } else {
            &raw const alpha_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasZdscal_v2(handle, n, alpha_arg, x, incx) };
    }
}

#[cuda_hook(proc_id = 1152)]
fn cublasZdscal_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[skip] alpha: *const f64,
    #[device] x: *mut cuDoubleComplex,
    incx: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let alpha_addr = alpha as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f64;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f64
        } else {
            &raw const alpha_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasZdscal_v2_64(handle, n, alpha_arg, x, incx) };
    }
}

#[cuda_hook(proc_id = 1153)]
fn cublasSaxpy_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[skip] alpha: *const f32,
    #[device] x: *const f32,
    incx: c_int,
    #[device] y: *mut f32,
    incy: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let alpha_addr = alpha as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f32;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f32
        } else {
            &raw const alpha_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasSaxpy_v2(handle, n, alpha_arg, x, incx, y, incy) };
    }
}

#[cuda_hook(proc_id = 1154)]
fn cublasSaxpy_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[skip] alpha: *const f32,
    #[device] x: *const f32,
    incx: i64,
    #[device] y: *mut f32,
    incy: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let alpha_addr = alpha as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f32;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f32
        } else {
            &raw const alpha_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasSaxpy_v2_64(handle, n, alpha_arg, x, incx, y, incy) };
    }
}

#[cuda_hook(proc_id = 1155)]
fn cublasDaxpy_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[skip] alpha: *const f64,
    #[device] x: *const f64,
    incx: c_int,
    #[device] y: *mut f64,
    incy: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let alpha_addr = alpha as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f64;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f64
        } else {
            &raw const alpha_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasDaxpy_v2(handle, n, alpha_arg, x, incx, y, incy) };
    }
}

#[cuda_hook(proc_id = 1156)]
fn cublasDaxpy_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[skip] alpha: *const f64,
    #[device] x: *const f64,
    incx: i64,
    #[device] y: *mut f64,
    incy: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            0.0
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let alpha_addr = alpha as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = 0.0f64;
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const f64
        } else {
            &raw const alpha_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasDaxpy_v2_64(handle, n, alpha_arg, x, incx, y, incy) };
    }
}

#[cuda_hook(proc_id = 1157)]
fn cublasCaxpy_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[device] y: *mut cuComplex,
    incy: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let alpha_addr = alpha as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuComplex
        } else {
            &raw const alpha_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasCaxpy_v2(handle, n, alpha_arg, x, incx, y, incy) };
    }
}

#[cuda_hook(proc_id = 1158)]
fn cublasCaxpy_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[skip] alpha: *const cuComplex,
    #[device] x: *const cuComplex,
    incx: i64,
    #[device] y: *mut cuComplex,
    incy: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let alpha_addr = alpha as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuComplex
        } else {
            &raw const alpha_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasCaxpy_v2_64(handle, n, alpha_arg, x, incx, y, incy) };
    }
}

#[cuda_hook(proc_id = 1159)]
fn cublasZaxpy_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[device] y: *mut cuDoubleComplex,
    incy: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuDoubleComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let alpha_addr = alpha as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuDoubleComplex
        } else {
            &raw const alpha_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasZaxpy_v2(handle, n, alpha_arg, x, incx, y, incy) };
    }
}

#[cuda_hook(proc_id = 1160)]
fn cublasZaxpy_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[device] y: *mut cuDoubleComplex,
    incy: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        let alpha_value = if device_pointer_mode {
            cuDoubleComplex { x: 0.0, y: 0.0 }
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let alpha_addr = alpha as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_value.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        alpha_value.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuDoubleComplex
        } else {
            &raw const alpha_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasZaxpy_v2_64(handle, n, alpha_arg, x, incx, y, incy) };
    }
}

#[cuda_hook(proc_id = 1161)]
fn cublasSnrm2_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const f32,
    incx: c_int,
    #[skip] result_ptr: *mut f32,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0.0f32;
        let result_arg = if device_pointer_mode {
            result_addr as *mut f32
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasSnrm2_v2(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0.0f32;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1162)]
fn cublasSnrm2_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const f32,
    incx: i64,
    #[skip] result_ptr: *mut f32,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0.0f32;
        let result_arg = if device_pointer_mode {
            result_addr as *mut f32
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasSnrm2_v2_64(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0.0f32;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1163)]
fn cublasDnrm2_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const f64,
    incx: c_int,
    #[skip] result_ptr: *mut f64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0.0f64;
        let result_arg = if device_pointer_mode {
            result_addr as *mut f64
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasDnrm2_v2(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0.0f64;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1164)]
fn cublasDnrm2_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const f64,
    incx: i64,
    #[skip] result_ptr: *mut f64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0.0f64;
        let result_arg = if device_pointer_mode {
            result_addr as *mut f64
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasDnrm2_v2_64(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0.0f64;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1165)]
fn cublasScnrm2_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[skip] result_ptr: *mut f32,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0.0f32;
        let result_arg = if device_pointer_mode {
            result_addr as *mut f32
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasScnrm2_v2(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0.0f32;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1166)]
fn cublasScnrm2_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const cuComplex,
    incx: i64,
    #[skip] result_ptr: *mut f32,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0.0f32;
        let result_arg = if device_pointer_mode {
            result_addr as *mut f32
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasScnrm2_v2_64(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0.0f32;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1167)]
fn cublasDznrm2_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[skip] result_ptr: *mut f64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0.0f64;
        let result_arg = if device_pointer_mode {
            result_addr as *mut f64
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasDznrm2_v2(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0.0f64;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1168)]
fn cublasDznrm2_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[skip] result_ptr: *mut f64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0.0f64;
        let result_arg = if device_pointer_mode {
            result_addr as *mut f64
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasDznrm2_v2_64(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0.0f64;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1169)]
fn cublasSasum_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const f32,
    incx: c_int,
    #[skip] result_ptr: *mut f32,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0.0f32;
        let result_arg = if device_pointer_mode {
            result_addr as *mut f32
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasSasum_v2(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0.0f32;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1170)]
fn cublasSasum_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const f32,
    incx: i64,
    #[skip] result_ptr: *mut f32,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0.0f32;
        let result_arg = if device_pointer_mode {
            result_addr as *mut f32
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasSasum_v2_64(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0.0f32;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1171)]
fn cublasDasum_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const f64,
    incx: c_int,
    #[skip] result_ptr: *mut f64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0.0f64;
        let result_arg = if device_pointer_mode {
            result_addr as *mut f64
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasDasum_v2(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0.0f64;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1172)]
fn cublasDasum_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const f64,
    incx: i64,
    #[skip] result_ptr: *mut f64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0.0f64;
        let result_arg = if device_pointer_mode {
            result_addr as *mut f64
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasDasum_v2_64(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0.0f64;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1173)]
fn cublasScasum_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[skip] result_ptr: *mut f32,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0.0f32;
        let result_arg = if device_pointer_mode {
            result_addr as *mut f32
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasScasum_v2(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0.0f32;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1174)]
fn cublasScasum_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const cuComplex,
    incx: i64,
    #[skip] result_ptr: *mut f32,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0.0f32;
        let result_arg = if device_pointer_mode {
            result_addr as *mut f32
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasScasum_v2_64(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0.0f32;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1175)]
fn cublasDzasum_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[skip] result_ptr: *mut f64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0.0f64;
        let result_arg = if device_pointer_mode {
            result_addr as *mut f64
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasDzasum_v2(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0.0f64;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1176)]
fn cublasDzasum_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[skip] result_ptr: *mut f64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0.0f64;
        let result_arg = if device_pointer_mode {
            result_addr as *mut f64
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasDzasum_v2_64(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0.0f64;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1190)]
fn cublasSdot_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const f32,
    incx: c_int,
    #[device] y: *const f32,
    incy: c_int,
    #[skip] result_ptr: *mut f32,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0.0f32;
        let result_arg = if device_pointer_mode {
            result_addr as *mut f32
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasSdot_v2(handle, n, x, incx, y, incy, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0.0f32;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1191)]
fn cublasSdot_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const f32,
    incx: i64,
    #[device] y: *const f32,
    incy: i64,
    #[skip] result_ptr: *mut f32,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0.0f32;
        let result_arg = if device_pointer_mode {
            result_addr as *mut f32
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasSdot_v2_64(handle, n, x, incx, y, incy, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0.0f32;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1192)]
fn cublasDdot_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const f64,
    incx: c_int,
    #[device] y: *const f64,
    incy: c_int,
    #[skip] result_ptr: *mut f64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0.0f64;
        let result_arg = if device_pointer_mode {
            result_addr as *mut f64
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasDdot_v2(handle, n, x, incx, y, incy, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0.0f64;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1193)]
fn cublasDdot_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const f64,
    incx: i64,
    #[device] y: *const f64,
    incy: i64,
    #[skip] result_ptr: *mut f64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0.0f64;
        let result_arg = if device_pointer_mode {
            result_addr as *mut f64
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasDdot_v2_64(handle, n, x, incx, y, incy, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0.0f64;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1194)]
fn cublasCdotu_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[device] y: *const cuComplex,
    incy: c_int,
    #[skip] result_ptr: *mut cuComplex,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = cuComplex { x: 0.0, y: 0.0 };
        let result_arg = if device_pointer_mode {
            result_addr as *mut cuComplex
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasCdotu_v2(handle, n, x, incx, y, incy, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = cuComplex { x: 0.0, y: 0.0 };
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1195)]
fn cublasCdotu_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const cuComplex,
    incx: i64,
    #[device] y: *const cuComplex,
    incy: i64,
    #[skip] result_ptr: *mut cuComplex,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = cuComplex { x: 0.0, y: 0.0 };
        let result_arg = if device_pointer_mode {
            result_addr as *mut cuComplex
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasCdotu_v2_64(handle, n, x, incx, y, incy, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = cuComplex { x: 0.0, y: 0.0 };
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1196)]
fn cublasCdotc_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[device] y: *const cuComplex,
    incy: c_int,
    #[skip] result_ptr: *mut cuComplex,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = cuComplex { x: 0.0, y: 0.0 };
        let result_arg = if device_pointer_mode {
            result_addr as *mut cuComplex
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasCdotc_v2(handle, n, x, incx, y, incy, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = cuComplex { x: 0.0, y: 0.0 };
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1197)]
fn cublasCdotc_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const cuComplex,
    incx: i64,
    #[device] y: *const cuComplex,
    incy: i64,
    #[skip] result_ptr: *mut cuComplex,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = cuComplex { x: 0.0, y: 0.0 };
        let result_arg = if device_pointer_mode {
            result_addr as *mut cuComplex
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasCdotc_v2_64(handle, n, x, incx, y, incy, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = cuComplex { x: 0.0, y: 0.0 };
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1198)]
fn cublasZdotu_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[device] y: *const cuDoubleComplex,
    incy: c_int,
    #[skip] result_ptr: *mut cuDoubleComplex,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        let result_arg = if device_pointer_mode {
            result_addr as *mut cuDoubleComplex
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasZdotu_v2(handle, n, x, incx, y, incy, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = cuDoubleComplex { x: 0.0, y: 0.0 };
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1199)]
fn cublasZdotu_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[device] y: *const cuDoubleComplex,
    incy: i64,
    #[skip] result_ptr: *mut cuDoubleComplex,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        let result_arg = if device_pointer_mode {
            result_addr as *mut cuDoubleComplex
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasZdotu_v2_64(handle, n, x, incx, y, incy, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = cuDoubleComplex { x: 0.0, y: 0.0 };
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1200)]
fn cublasZdotc_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[device] y: *const cuDoubleComplex,
    incy: c_int,
    #[skip] result_ptr: *mut cuDoubleComplex,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        let result_arg = if device_pointer_mode {
            result_addr as *mut cuDoubleComplex
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasZdotc_v2(handle, n, x, incx, y, incy, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = cuDoubleComplex { x: 0.0, y: 0.0 };
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1201)]
fn cublasZdotc_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[device] y: *const cuDoubleComplex,
    incy: i64,
    #[skip] result_ptr: *mut cuDoubleComplex,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        let result_arg = if device_pointer_mode {
            result_addr as *mut cuDoubleComplex
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasZdotc_v2_64(handle, n, x, incx, y, incy, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = cuDoubleComplex { x: 0.0, y: 0.0 };
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1202)]
fn cublasIsamax_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const f32,
    incx: c_int,
    #[skip] result_ptr: *mut c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0 as c_int;
        let result_arg = if device_pointer_mode {
            result_addr as *mut c_int
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasIsamax_v2(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0 as c_int;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1203)]
fn cublasIsamax_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const f32,
    incx: i64,
    #[skip] result_ptr: *mut i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0i64;
        let result_arg = if device_pointer_mode {
            result_addr as *mut i64
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasIsamax_v2_64(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0i64;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1204)]
fn cublasIdamax_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const f64,
    incx: c_int,
    #[skip] result_ptr: *mut c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0 as c_int;
        let result_arg = if device_pointer_mode {
            result_addr as *mut c_int
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasIdamax_v2(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0 as c_int;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1205)]
fn cublasIdamax_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const f64,
    incx: i64,
    #[skip] result_ptr: *mut i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0i64;
        let result_arg = if device_pointer_mode {
            result_addr as *mut i64
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasIdamax_v2_64(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0i64;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1206)]
fn cublasIcamax_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[skip] result_ptr: *mut c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0 as c_int;
        let result_arg = if device_pointer_mode {
            result_addr as *mut c_int
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasIcamax_v2(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0 as c_int;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1207)]
fn cublasIcamax_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const cuComplex,
    incx: i64,
    #[skip] result_ptr: *mut i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0i64;
        let result_arg = if device_pointer_mode {
            result_addr as *mut i64
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasIcamax_v2_64(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0i64;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1208)]
fn cublasIzamax_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[skip] result_ptr: *mut c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0 as c_int;
        let result_arg = if device_pointer_mode {
            result_addr as *mut c_int
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasIzamax_v2(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0 as c_int;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1209)]
fn cublasIzamax_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[skip] result_ptr: *mut i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0i64;
        let result_arg = if device_pointer_mode {
            result_addr as *mut i64
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasIzamax_v2_64(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0i64;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1210)]
fn cublasIsamin_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const f32,
    incx: c_int,
    #[skip] result_ptr: *mut c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0 as c_int;
        let result_arg = if device_pointer_mode {
            result_addr as *mut c_int
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasIsamin_v2(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0 as c_int;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1211)]
fn cublasIsamin_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const f32,
    incx: i64,
    #[skip] result_ptr: *mut i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0i64;
        let result_arg = if device_pointer_mode {
            result_addr as *mut i64
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasIsamin_v2_64(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0i64;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1212)]
fn cublasIdamin_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const f64,
    incx: c_int,
    #[skip] result_ptr: *mut c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0 as c_int;
        let result_arg = if device_pointer_mode {
            result_addr as *mut c_int
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasIdamin_v2(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0 as c_int;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1213)]
fn cublasIdamin_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const f64,
    incx: i64,
    #[skip] result_ptr: *mut i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0i64;
        let result_arg = if device_pointer_mode {
            result_addr as *mut i64
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasIdamin_v2_64(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0i64;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1214)]
fn cublasIcamin_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[skip] result_ptr: *mut c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0 as c_int;
        let result_arg = if device_pointer_mode {
            result_addr as *mut c_int
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasIcamin_v2(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0 as c_int;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1215)]
fn cublasIcamin_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const cuComplex,
    incx: i64,
    #[skip] result_ptr: *mut i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0i64;
        let result_arg = if device_pointer_mode {
            result_addr as *mut i64
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasIcamin_v2_64(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0i64;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1216)]
fn cublasIzamin_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[skip] result_ptr: *mut c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0 as c_int;
        let result_arg = if device_pointer_mode {
            result_addr as *mut c_int
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasIzamin_v2(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0 as c_int;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1217)]
fn cublasIzamin_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[skip] result_ptr: *mut i64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && result_ptr.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        let mut host_result_value = 0i64;
        let result_arg = if device_pointer_mode {
            result_addr as *mut i64
        } else {
            &raw mut host_result_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasIzamin_v2_64(handle, n, x, incx, result_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_result_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_result_value = 0i64;
            host_result_value.recv(channel_receiver).unwrap();
            unsafe {
                *result_ptr = host_result_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1218)]
fn cublasSrot_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *mut f32,
    incx: c_int,
    #[device] y: *mut f32,
    incy: c_int,
    #[skip] c: *const f32,
    #[skip] s: *const f32,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && (c.is_null() || s.is_null()) {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let c_addr = c as usize;
        let s_addr = s as usize;
        let mut host_c_value = 0.0f32;
        let mut host_s_value = 0.0f32;
        if !device_pointer_mode {
            unsafe {
                host_c_value = *c;
                host_s_value = *s;
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        c_addr.send(channel_sender).unwrap();
        s_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            host_c_value.send(channel_sender).unwrap();
            host_s_value.send(channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut c_addr = 0usize;
        c_addr.recv(channel_receiver).unwrap();
        let mut s_addr = 0usize;
        s_addr.recv(channel_receiver).unwrap();
        let mut host_c_value = 0.0f32;
        let mut host_s_value = 0.0f32;
        if !device_pointer_mode {
            host_c_value.recv(channel_receiver).unwrap();
            host_s_value.recv(channel_receiver).unwrap();
        }
        let c_arg = if device_pointer_mode {
            c_addr as *const f32
        } else {
            &raw const host_c_value
        };
        let s_arg = if device_pointer_mode {
            s_addr as *const f32
        } else {
            &raw const host_s_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasSrot_v2(handle, n, x, incx, y, incy, c_arg, s_arg) };
    }
}

#[cuda_hook(proc_id = 1219)]
fn cublasSrot_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *mut f32,
    incx: i64,
    #[device] y: *mut f32,
    incy: i64,
    #[skip] c: *const f32,
    #[skip] s: *const f32,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && (c.is_null() || s.is_null()) {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let c_addr = c as usize;
        let s_addr = s as usize;
        let mut host_c_value = 0.0f32;
        let mut host_s_value = 0.0f32;
        if !device_pointer_mode {
            unsafe {
                host_c_value = *c;
                host_s_value = *s;
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        c_addr.send(channel_sender).unwrap();
        s_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            host_c_value.send(channel_sender).unwrap();
            host_s_value.send(channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut c_addr = 0usize;
        c_addr.recv(channel_receiver).unwrap();
        let mut s_addr = 0usize;
        s_addr.recv(channel_receiver).unwrap();
        let mut host_c_value = 0.0f32;
        let mut host_s_value = 0.0f32;
        if !device_pointer_mode {
            host_c_value.recv(channel_receiver).unwrap();
            host_s_value.recv(channel_receiver).unwrap();
        }
        let c_arg = if device_pointer_mode {
            c_addr as *const f32
        } else {
            &raw const host_c_value
        };
        let s_arg = if device_pointer_mode {
            s_addr as *const f32
        } else {
            &raw const host_s_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasSrot_v2_64(handle, n, x, incx, y, incy, c_arg, s_arg) };
    }
}

#[cuda_hook(proc_id = 1220)]
fn cublasDrot_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *mut f64,
    incx: c_int,
    #[device] y: *mut f64,
    incy: c_int,
    #[skip] c: *const f64,
    #[skip] s: *const f64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && (c.is_null() || s.is_null()) {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let c_addr = c as usize;
        let s_addr = s as usize;
        let mut host_c_value = 0.0f64;
        let mut host_s_value = 0.0f64;
        if !device_pointer_mode {
            unsafe {
                host_c_value = *c;
                host_s_value = *s;
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        c_addr.send(channel_sender).unwrap();
        s_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            host_c_value.send(channel_sender).unwrap();
            host_s_value.send(channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut c_addr = 0usize;
        c_addr.recv(channel_receiver).unwrap();
        let mut s_addr = 0usize;
        s_addr.recv(channel_receiver).unwrap();
        let mut host_c_value = 0.0f64;
        let mut host_s_value = 0.0f64;
        if !device_pointer_mode {
            host_c_value.recv(channel_receiver).unwrap();
            host_s_value.recv(channel_receiver).unwrap();
        }
        let c_arg = if device_pointer_mode {
            c_addr as *const f64
        } else {
            &raw const host_c_value
        };
        let s_arg = if device_pointer_mode {
            s_addr as *const f64
        } else {
            &raw const host_s_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasDrot_v2(handle, n, x, incx, y, incy, c_arg, s_arg) };
    }
}

#[cuda_hook(proc_id = 1221)]
fn cublasDrot_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *mut f64,
    incx: i64,
    #[device] y: *mut f64,
    incy: i64,
    #[skip] c: *const f64,
    #[skip] s: *const f64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && (c.is_null() || s.is_null()) {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let c_addr = c as usize;
        let s_addr = s as usize;
        let mut host_c_value = 0.0f64;
        let mut host_s_value = 0.0f64;
        if !device_pointer_mode {
            unsafe {
                host_c_value = *c;
                host_s_value = *s;
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        c_addr.send(channel_sender).unwrap();
        s_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            host_c_value.send(channel_sender).unwrap();
            host_s_value.send(channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut c_addr = 0usize;
        c_addr.recv(channel_receiver).unwrap();
        let mut s_addr = 0usize;
        s_addr.recv(channel_receiver).unwrap();
        let mut host_c_value = 0.0f64;
        let mut host_s_value = 0.0f64;
        if !device_pointer_mode {
            host_c_value.recv(channel_receiver).unwrap();
            host_s_value.recv(channel_receiver).unwrap();
        }
        let c_arg = if device_pointer_mode {
            c_addr as *const f64
        } else {
            &raw const host_c_value
        };
        let s_arg = if device_pointer_mode {
            s_addr as *const f64
        } else {
            &raw const host_s_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasDrot_v2_64(handle, n, x, incx, y, incy, c_arg, s_arg) };
    }
}

#[cuda_hook(proc_id = 1222)]
fn cublasCrot_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *mut cuComplex,
    incx: c_int,
    #[device] y: *mut cuComplex,
    incy: c_int,
    #[skip] c: *const f32,
    #[skip] s: *const cuComplex,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && (c.is_null() || s.is_null()) {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let c_addr = c as usize;
        let s_addr = s as usize;
        let mut host_c_value = 0.0f32;
        let mut host_s_value = cuComplex { x: 0.0, y: 0.0 };
        if !device_pointer_mode {
            unsafe {
                host_c_value = *c;
                host_s_value = *s;
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        c_addr.send(channel_sender).unwrap();
        s_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            host_c_value.send(channel_sender).unwrap();
            host_s_value.send(channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut c_addr = 0usize;
        c_addr.recv(channel_receiver).unwrap();
        let mut s_addr = 0usize;
        s_addr.recv(channel_receiver).unwrap();
        let mut host_c_value = 0.0f32;
        let mut host_s_value = cuComplex { x: 0.0, y: 0.0 };
        if !device_pointer_mode {
            host_c_value.recv(channel_receiver).unwrap();
            host_s_value.recv(channel_receiver).unwrap();
        }
        let c_arg = if device_pointer_mode {
            c_addr as *const f32
        } else {
            &raw const host_c_value
        };
        let s_arg = if device_pointer_mode {
            s_addr as *const cuComplex
        } else {
            &raw const host_s_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasCrot_v2(handle, n, x, incx, y, incy, c_arg, s_arg) };
    }
}

#[cuda_hook(proc_id = 1223)]
fn cublasCrot_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *mut cuComplex,
    incx: i64,
    #[device] y: *mut cuComplex,
    incy: i64,
    #[skip] c: *const f32,
    #[skip] s: *const cuComplex,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && (c.is_null() || s.is_null()) {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let c_addr = c as usize;
        let s_addr = s as usize;
        let mut host_c_value = 0.0f32;
        let mut host_s_value = cuComplex { x: 0.0, y: 0.0 };
        if !device_pointer_mode {
            unsafe {
                host_c_value = *c;
                host_s_value = *s;
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        c_addr.send(channel_sender).unwrap();
        s_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            host_c_value.send(channel_sender).unwrap();
            host_s_value.send(channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut c_addr = 0usize;
        c_addr.recv(channel_receiver).unwrap();
        let mut s_addr = 0usize;
        s_addr.recv(channel_receiver).unwrap();
        let mut host_c_value = 0.0f32;
        let mut host_s_value = cuComplex { x: 0.0, y: 0.0 };
        if !device_pointer_mode {
            host_c_value.recv(channel_receiver).unwrap();
            host_s_value.recv(channel_receiver).unwrap();
        }
        let c_arg = if device_pointer_mode {
            c_addr as *const f32
        } else {
            &raw const host_c_value
        };
        let s_arg = if device_pointer_mode {
            s_addr as *const cuComplex
        } else {
            &raw const host_s_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasCrot_v2_64(handle, n, x, incx, y, incy, c_arg, s_arg) };
    }
}

#[cuda_hook(proc_id = 1224)]
fn cublasCsrot_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *mut cuComplex,
    incx: c_int,
    #[device] y: *mut cuComplex,
    incy: c_int,
    #[skip] c: *const f32,
    #[skip] s: *const f32,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && (c.is_null() || s.is_null()) {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let c_addr = c as usize;
        let s_addr = s as usize;
        let mut host_c_value = 0.0f32;
        let mut host_s_value = 0.0f32;
        if !device_pointer_mode {
            unsafe {
                host_c_value = *c;
                host_s_value = *s;
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        c_addr.send(channel_sender).unwrap();
        s_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            host_c_value.send(channel_sender).unwrap();
            host_s_value.send(channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut c_addr = 0usize;
        c_addr.recv(channel_receiver).unwrap();
        let mut s_addr = 0usize;
        s_addr.recv(channel_receiver).unwrap();
        let mut host_c_value = 0.0f32;
        let mut host_s_value = 0.0f32;
        if !device_pointer_mode {
            host_c_value.recv(channel_receiver).unwrap();
            host_s_value.recv(channel_receiver).unwrap();
        }
        let c_arg = if device_pointer_mode {
            c_addr as *const f32
        } else {
            &raw const host_c_value
        };
        let s_arg = if device_pointer_mode {
            s_addr as *const f32
        } else {
            &raw const host_s_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasCsrot_v2(handle, n, x, incx, y, incy, c_arg, s_arg) };
    }
}

#[cuda_hook(proc_id = 1225)]
fn cublasCsrot_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *mut cuComplex,
    incx: i64,
    #[device] y: *mut cuComplex,
    incy: i64,
    #[skip] c: *const f32,
    #[skip] s: *const f32,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && (c.is_null() || s.is_null()) {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let c_addr = c as usize;
        let s_addr = s as usize;
        let mut host_c_value = 0.0f32;
        let mut host_s_value = 0.0f32;
        if !device_pointer_mode {
            unsafe {
                host_c_value = *c;
                host_s_value = *s;
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        c_addr.send(channel_sender).unwrap();
        s_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            host_c_value.send(channel_sender).unwrap();
            host_s_value.send(channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut c_addr = 0usize;
        c_addr.recv(channel_receiver).unwrap();
        let mut s_addr = 0usize;
        s_addr.recv(channel_receiver).unwrap();
        let mut host_c_value = 0.0f32;
        let mut host_s_value = 0.0f32;
        if !device_pointer_mode {
            host_c_value.recv(channel_receiver).unwrap();
            host_s_value.recv(channel_receiver).unwrap();
        }
        let c_arg = if device_pointer_mode {
            c_addr as *const f32
        } else {
            &raw const host_c_value
        };
        let s_arg = if device_pointer_mode {
            s_addr as *const f32
        } else {
            &raw const host_s_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasCsrot_v2_64(handle, n, x, incx, y, incy, c_arg, s_arg) };
    }
}

#[cuda_hook(proc_id = 1226)]
fn cublasZrot_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *mut cuDoubleComplex,
    incx: c_int,
    #[device] y: *mut cuDoubleComplex,
    incy: c_int,
    #[skip] c: *const f64,
    #[skip] s: *const cuDoubleComplex,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && (c.is_null() || s.is_null()) {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let c_addr = c as usize;
        let s_addr = s as usize;
        let mut host_c_value = 0.0f64;
        let mut host_s_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        if !device_pointer_mode {
            unsafe {
                host_c_value = *c;
                host_s_value = *s;
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        c_addr.send(channel_sender).unwrap();
        s_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            host_c_value.send(channel_sender).unwrap();
            host_s_value.send(channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut c_addr = 0usize;
        c_addr.recv(channel_receiver).unwrap();
        let mut s_addr = 0usize;
        s_addr.recv(channel_receiver).unwrap();
        let mut host_c_value = 0.0f64;
        let mut host_s_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        if !device_pointer_mode {
            host_c_value.recv(channel_receiver).unwrap();
            host_s_value.recv(channel_receiver).unwrap();
        }
        let c_arg = if device_pointer_mode {
            c_addr as *const f64
        } else {
            &raw const host_c_value
        };
        let s_arg = if device_pointer_mode {
            s_addr as *const cuDoubleComplex
        } else {
            &raw const host_s_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasZrot_v2(handle, n, x, incx, y, incy, c_arg, s_arg) };
    }
}

#[cuda_hook(proc_id = 1227)]
fn cublasZrot_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *mut cuDoubleComplex,
    incx: i64,
    #[device] y: *mut cuDoubleComplex,
    incy: i64,
    #[skip] c: *const f64,
    #[skip] s: *const cuDoubleComplex,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && (c.is_null() || s.is_null()) {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let c_addr = c as usize;
        let s_addr = s as usize;
        let mut host_c_value = 0.0f64;
        let mut host_s_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        if !device_pointer_mode {
            unsafe {
                host_c_value = *c;
                host_s_value = *s;
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        c_addr.send(channel_sender).unwrap();
        s_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            host_c_value.send(channel_sender).unwrap();
            host_s_value.send(channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut c_addr = 0usize;
        c_addr.recv(channel_receiver).unwrap();
        let mut s_addr = 0usize;
        s_addr.recv(channel_receiver).unwrap();
        let mut host_c_value = 0.0f64;
        let mut host_s_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        if !device_pointer_mode {
            host_c_value.recv(channel_receiver).unwrap();
            host_s_value.recv(channel_receiver).unwrap();
        }
        let c_arg = if device_pointer_mode {
            c_addr as *const f64
        } else {
            &raw const host_c_value
        };
        let s_arg = if device_pointer_mode {
            s_addr as *const cuDoubleComplex
        } else {
            &raw const host_s_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasZrot_v2_64(handle, n, x, incx, y, incy, c_arg, s_arg) };
    }
}

#[cuda_hook(proc_id = 1228)]
fn cublasZdrot_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *mut cuDoubleComplex,
    incx: c_int,
    #[device] y: *mut cuDoubleComplex,
    incy: c_int,
    #[skip] c: *const f64,
    #[skip] s: *const f64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && (c.is_null() || s.is_null()) {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let c_addr = c as usize;
        let s_addr = s as usize;
        let mut host_c_value = 0.0f64;
        let mut host_s_value = 0.0f64;
        if !device_pointer_mode {
            unsafe {
                host_c_value = *c;
                host_s_value = *s;
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        c_addr.send(channel_sender).unwrap();
        s_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            host_c_value.send(channel_sender).unwrap();
            host_s_value.send(channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut c_addr = 0usize;
        c_addr.recv(channel_receiver).unwrap();
        let mut s_addr = 0usize;
        s_addr.recv(channel_receiver).unwrap();
        let mut host_c_value = 0.0f64;
        let mut host_s_value = 0.0f64;
        if !device_pointer_mode {
            host_c_value.recv(channel_receiver).unwrap();
            host_s_value.recv(channel_receiver).unwrap();
        }
        let c_arg = if device_pointer_mode {
            c_addr as *const f64
        } else {
            &raw const host_c_value
        };
        let s_arg = if device_pointer_mode {
            s_addr as *const f64
        } else {
            &raw const host_s_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasZdrot_v2(handle, n, x, incx, y, incy, c_arg, s_arg) };
    }
}

#[cuda_hook(proc_id = 1229)]
fn cublasZdrot_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *mut cuDoubleComplex,
    incx: i64,
    #[device] y: *mut cuDoubleComplex,
    incy: i64,
    #[skip] c: *const f64,
    #[skip] s: *const f64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && (c.is_null() || s.is_null()) {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let c_addr = c as usize;
        let s_addr = s as usize;
        let mut host_c_value = 0.0f64;
        let mut host_s_value = 0.0f64;
        if !device_pointer_mode {
            unsafe {
                host_c_value = *c;
                host_s_value = *s;
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        c_addr.send(channel_sender).unwrap();
        s_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            host_c_value.send(channel_sender).unwrap();
            host_s_value.send(channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut c_addr = 0usize;
        c_addr.recv(channel_receiver).unwrap();
        let mut s_addr = 0usize;
        s_addr.recv(channel_receiver).unwrap();
        let mut host_c_value = 0.0f64;
        let mut host_s_value = 0.0f64;
        if !device_pointer_mode {
            host_c_value.recv(channel_receiver).unwrap();
            host_s_value.recv(channel_receiver).unwrap();
        }
        let c_arg = if device_pointer_mode {
            c_addr as *const f64
        } else {
            &raw const host_c_value
        };
        let s_arg = if device_pointer_mode {
            s_addr as *const f64
        } else {
            &raw const host_s_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasZdrot_v2_64(handle, n, x, incx, y, incy, c_arg, s_arg) };
    }
}

#[cuda_hook(proc_id = 1230)]
fn cublasSrotg_v2(
    handle: cublasHandle_t,
    #[skip] a: *mut f32,
    #[skip] b: *mut f32,
    #[skip] c: *mut f32,
    #[skip] s: *mut f32,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && (a.is_null() || b.is_null() || c.is_null() || s.is_null()) {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let a_addr = a as usize;
        let b_addr = b as usize;
        let c_addr = c as usize;
        let s_addr = s as usize;
        let mut host_a_value = 0.0f32;
        let mut host_b_value = 0.0f32;
        if !device_pointer_mode {
            unsafe {
                host_a_value = *a;
                host_b_value = *b;
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        a_addr.send(channel_sender).unwrap();
        b_addr.send(channel_sender).unwrap();
        c_addr.send(channel_sender).unwrap();
        s_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            host_a_value.send(channel_sender).unwrap();
            host_b_value.send(channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut a_addr = 0usize;
        a_addr.recv(channel_receiver).unwrap();
        let mut b_addr = 0usize;
        b_addr.recv(channel_receiver).unwrap();
        let mut c_addr = 0usize;
        c_addr.recv(channel_receiver).unwrap();
        let mut s_addr = 0usize;
        s_addr.recv(channel_receiver).unwrap();
        let mut host_a_value = 0.0f32;
        let mut host_b_value = 0.0f32;
        let mut host_c_value = 0.0f32;
        let mut host_s_value = 0.0f32;
        if !device_pointer_mode {
            host_a_value.recv(channel_receiver).unwrap();
            host_b_value.recv(channel_receiver).unwrap();
        }
        let a_arg = if device_pointer_mode {
            a_addr as *mut f32
        } else {
            &raw mut host_a_value
        };
        let b_arg = if device_pointer_mode {
            b_addr as *mut f32
        } else {
            &raw mut host_b_value
        };
        let c_arg = if device_pointer_mode {
            c_addr as *mut f32
        } else {
            &raw mut host_c_value
        };
        let s_arg = if device_pointer_mode {
            s_addr as *mut f32
        } else {
            &raw mut host_s_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasSrotg_v2(handle, a_arg, b_arg, c_arg, s_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_a_value.send(channel_sender).unwrap();
            host_b_value.send(channel_sender).unwrap();
            host_c_value.send(channel_sender).unwrap();
            host_s_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_a_value = 0.0f32;
            let mut host_b_value = 0.0f32;
            let mut host_c_value = 0.0f32;
            let mut host_s_value = 0.0f32;
            host_a_value.recv(channel_receiver).unwrap();
            host_b_value.recv(channel_receiver).unwrap();
            host_c_value.recv(channel_receiver).unwrap();
            host_s_value.recv(channel_receiver).unwrap();
            unsafe {
                *a = host_a_value;
                *b = host_b_value;
                *c = host_c_value;
                *s = host_s_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1231)]
fn cublasDrotg_v2(
    handle: cublasHandle_t,
    #[skip] a: *mut f64,
    #[skip] b: *mut f64,
    #[skip] c: *mut f64,
    #[skip] s: *mut f64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && (a.is_null() || b.is_null() || c.is_null() || s.is_null()) {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let a_addr = a as usize;
        let b_addr = b as usize;
        let c_addr = c as usize;
        let s_addr = s as usize;
        let mut host_a_value = 0.0f64;
        let mut host_b_value = 0.0f64;
        if !device_pointer_mode {
            unsafe {
                host_a_value = *a;
                host_b_value = *b;
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        a_addr.send(channel_sender).unwrap();
        b_addr.send(channel_sender).unwrap();
        c_addr.send(channel_sender).unwrap();
        s_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            host_a_value.send(channel_sender).unwrap();
            host_b_value.send(channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut a_addr = 0usize;
        a_addr.recv(channel_receiver).unwrap();
        let mut b_addr = 0usize;
        b_addr.recv(channel_receiver).unwrap();
        let mut c_addr = 0usize;
        c_addr.recv(channel_receiver).unwrap();
        let mut s_addr = 0usize;
        s_addr.recv(channel_receiver).unwrap();
        let mut host_a_value = 0.0f64;
        let mut host_b_value = 0.0f64;
        let mut host_c_value = 0.0f64;
        let mut host_s_value = 0.0f64;
        if !device_pointer_mode {
            host_a_value.recv(channel_receiver).unwrap();
            host_b_value.recv(channel_receiver).unwrap();
        }
        let a_arg = if device_pointer_mode {
            a_addr as *mut f64
        } else {
            &raw mut host_a_value
        };
        let b_arg = if device_pointer_mode {
            b_addr as *mut f64
        } else {
            &raw mut host_b_value
        };
        let c_arg = if device_pointer_mode {
            c_addr as *mut f64
        } else {
            &raw mut host_c_value
        };
        let s_arg = if device_pointer_mode {
            s_addr as *mut f64
        } else {
            &raw mut host_s_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasDrotg_v2(handle, a_arg, b_arg, c_arg, s_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_a_value.send(channel_sender).unwrap();
            host_b_value.send(channel_sender).unwrap();
            host_c_value.send(channel_sender).unwrap();
            host_s_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_a_value = 0.0f64;
            let mut host_b_value = 0.0f64;
            let mut host_c_value = 0.0f64;
            let mut host_s_value = 0.0f64;
            host_a_value.recv(channel_receiver).unwrap();
            host_b_value.recv(channel_receiver).unwrap();
            host_c_value.recv(channel_receiver).unwrap();
            host_s_value.recv(channel_receiver).unwrap();
            unsafe {
                *a = host_a_value;
                *b = host_b_value;
                *c = host_c_value;
                *s = host_s_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1232)]
fn cublasCrotg_v2(
    handle: cublasHandle_t,
    #[skip] a: *mut cuComplex,
    #[skip] b: *mut cuComplex,
    #[skip] c: *mut f32,
    #[skip] s: *mut cuComplex,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && (a.is_null() || b.is_null() || c.is_null() || s.is_null()) {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let a_addr = a as usize;
        let b_addr = b as usize;
        let c_addr = c as usize;
        let s_addr = s as usize;
        let mut host_a_value = cuComplex { x: 0.0, y: 0.0 };
        let mut host_b_value = cuComplex { x: 0.0, y: 0.0 };
        if !device_pointer_mode {
            unsafe {
                host_a_value = *a;
                host_b_value = *b;
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        a_addr.send(channel_sender).unwrap();
        b_addr.send(channel_sender).unwrap();
        c_addr.send(channel_sender).unwrap();
        s_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            host_a_value.send(channel_sender).unwrap();
            host_b_value.send(channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut a_addr = 0usize;
        a_addr.recv(channel_receiver).unwrap();
        let mut b_addr = 0usize;
        b_addr.recv(channel_receiver).unwrap();
        let mut c_addr = 0usize;
        c_addr.recv(channel_receiver).unwrap();
        let mut s_addr = 0usize;
        s_addr.recv(channel_receiver).unwrap();
        let mut host_a_value = cuComplex { x: 0.0, y: 0.0 };
        let mut host_b_value = cuComplex { x: 0.0, y: 0.0 };
        let mut host_c_value = 0.0f32;
        let mut host_s_value = cuComplex { x: 0.0, y: 0.0 };
        if !device_pointer_mode {
            host_a_value.recv(channel_receiver).unwrap();
            host_b_value.recv(channel_receiver).unwrap();
        }
        let a_arg = if device_pointer_mode {
            a_addr as *mut cuComplex
        } else {
            &raw mut host_a_value
        };
        let b_arg = if device_pointer_mode {
            b_addr as *mut cuComplex
        } else {
            &raw mut host_b_value
        };
        let c_arg = if device_pointer_mode {
            c_addr as *mut f32
        } else {
            &raw mut host_c_value
        };
        let s_arg = if device_pointer_mode {
            s_addr as *mut cuComplex
        } else {
            &raw mut host_s_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasCrotg_v2(handle, a_arg, b_arg, c_arg, s_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_a_value.send(channel_sender).unwrap();
            host_b_value.send(channel_sender).unwrap();
            host_c_value.send(channel_sender).unwrap();
            host_s_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_a_value = cuComplex { x: 0.0, y: 0.0 };
            let mut host_b_value = cuComplex { x: 0.0, y: 0.0 };
            let mut host_c_value = 0.0f32;
            let mut host_s_value = cuComplex { x: 0.0, y: 0.0 };
            host_a_value.recv(channel_receiver).unwrap();
            host_b_value.recv(channel_receiver).unwrap();
            host_c_value.recv(channel_receiver).unwrap();
            host_s_value.recv(channel_receiver).unwrap();
            unsafe {
                *a = host_a_value;
                *b = host_b_value;
                *c = host_c_value;
                *s = host_s_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1233)]
fn cublasZrotg_v2(
    handle: cublasHandle_t,
    #[skip] a: *mut cuDoubleComplex,
    #[skip] b: *mut cuDoubleComplex,
    #[skip] c: *mut f64,
    #[skip] s: *mut cuDoubleComplex,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && (a.is_null() || b.is_null() || c.is_null() || s.is_null()) {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let a_addr = a as usize;
        let b_addr = b as usize;
        let c_addr = c as usize;
        let s_addr = s as usize;
        let mut host_a_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        let mut host_b_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        if !device_pointer_mode {
            unsafe {
                host_a_value = *a;
                host_b_value = *b;
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        a_addr.send(channel_sender).unwrap();
        b_addr.send(channel_sender).unwrap();
        c_addr.send(channel_sender).unwrap();
        s_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            host_a_value.send(channel_sender).unwrap();
            host_b_value.send(channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut a_addr = 0usize;
        a_addr.recv(channel_receiver).unwrap();
        let mut b_addr = 0usize;
        b_addr.recv(channel_receiver).unwrap();
        let mut c_addr = 0usize;
        c_addr.recv(channel_receiver).unwrap();
        let mut s_addr = 0usize;
        s_addr.recv(channel_receiver).unwrap();
        let mut host_a_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        let mut host_b_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        let mut host_c_value = 0.0f64;
        let mut host_s_value = cuDoubleComplex { x: 0.0, y: 0.0 };
        if !device_pointer_mode {
            host_a_value.recv(channel_receiver).unwrap();
            host_b_value.recv(channel_receiver).unwrap();
        }
        let a_arg = if device_pointer_mode {
            a_addr as *mut cuDoubleComplex
        } else {
            &raw mut host_a_value
        };
        let b_arg = if device_pointer_mode {
            b_addr as *mut cuDoubleComplex
        } else {
            &raw mut host_b_value
        };
        let c_arg = if device_pointer_mode {
            c_addr as *mut f64
        } else {
            &raw mut host_c_value
        };
        let s_arg = if device_pointer_mode {
            s_addr as *mut cuDoubleComplex
        } else {
            &raw mut host_s_value
        };
    }
    'server_execution: {
        let result = unsafe { cublasZrotg_v2(handle, a_arg, b_arg, c_arg, s_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_a_value.send(channel_sender).unwrap();
            host_b_value.send(channel_sender).unwrap();
            host_c_value.send(channel_sender).unwrap();
            host_s_value.send(channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_a_value = cuDoubleComplex { x: 0.0, y: 0.0 };
            let mut host_b_value = cuDoubleComplex { x: 0.0, y: 0.0 };
            let mut host_c_value = 0.0f64;
            let mut host_s_value = cuDoubleComplex { x: 0.0, y: 0.0 };
            host_a_value.recv(channel_receiver).unwrap();
            host_b_value.recv(channel_receiver).unwrap();
            host_c_value.recv(channel_receiver).unwrap();
            host_s_value.recv(channel_receiver).unwrap();
            unsafe {
                *a = host_a_value;
                *b = host_b_value;
                *c = host_c_value;
                *s = host_s_value;
            }
        }
    }
}

#[cuda_hook(proc_id = 1234)]
fn cublasSrotm_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *mut f32,
    incx: c_int,
    #[device] y: *mut f32,
    incy: c_int,
    #[skip] param: *const f32,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && param.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let param_addr = param as usize;
        let mut host_param_value = [0.0f32; 5];
        if !device_pointer_mode {
            unsafe {
                std::ptr::copy_nonoverlapping(param, host_param_value.as_mut_ptr(), 5);
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        param_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            send_slice(&host_param_value, channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut param_addr = 0usize;
        param_addr.recv(channel_receiver).unwrap();
        let host_param_value = if device_pointer_mode {
            Box::<[f32]>::default()
        } else {
            recv_slice::<f32, _>(channel_receiver).unwrap()
        };
        let param_arg = if device_pointer_mode {
            param_addr as *const f32
        } else {
            host_param_value.as_ptr()
        };
    }
    'server_execution: {
        let result = unsafe { cublasSrotm_v2(handle, n, x, incx, y, incy, param_arg) };
    }
}

#[cuda_hook(proc_id = 1235)]
fn cublasSrotm_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *mut f32,
    incx: i64,
    #[device] y: *mut f32,
    incy: i64,
    #[skip] param: *const f32,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && param.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let param_addr = param as usize;
        let mut host_param_value = [0.0f32; 5];
        if !device_pointer_mode {
            unsafe {
                std::ptr::copy_nonoverlapping(param, host_param_value.as_mut_ptr(), 5);
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        param_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            send_slice(&host_param_value, channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut param_addr = 0usize;
        param_addr.recv(channel_receiver).unwrap();
        let host_param_value = if device_pointer_mode {
            Box::<[f32]>::default()
        } else {
            recv_slice::<f32, _>(channel_receiver).unwrap()
        };
        let param_arg = if device_pointer_mode {
            param_addr as *const f32
        } else {
            host_param_value.as_ptr()
        };
    }
    'server_execution: {
        let result = unsafe { cublasSrotm_v2_64(handle, n, x, incx, y, incy, param_arg) };
    }
}

#[cuda_hook(proc_id = 1236)]
fn cublasDrotm_v2(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *mut f64,
    incx: c_int,
    #[device] y: *mut f64,
    incy: c_int,
    #[skip] param: *const f64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && param.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let param_addr = param as usize;
        let mut host_param_value = [0.0f64; 5];
        if !device_pointer_mode {
            unsafe {
                std::ptr::copy_nonoverlapping(param, host_param_value.as_mut_ptr(), 5);
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        param_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            send_slice(&host_param_value, channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut param_addr = 0usize;
        param_addr.recv(channel_receiver).unwrap();
        let host_param_value = if device_pointer_mode {
            Box::<[f64]>::default()
        } else {
            recv_slice::<f64, _>(channel_receiver).unwrap()
        };
        let param_arg = if device_pointer_mode {
            param_addr as *const f64
        } else {
            host_param_value.as_ptr()
        };
    }
    'server_execution: {
        let result = unsafe { cublasDrotm_v2(handle, n, x, incx, y, incy, param_arg) };
    }
}

#[cuda_hook(proc_id = 1237)]
fn cublasDrotm_v2_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *mut f64,
    incx: i64,
    #[device] y: *mut f64,
    incy: i64,
    #[skip] param: *const f64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode && param.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let param_addr = param as usize;
        let mut host_param_value = [0.0f64; 5];
        if !device_pointer_mode {
            unsafe {
                std::ptr::copy_nonoverlapping(param, host_param_value.as_mut_ptr(), 5);
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        param_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            send_slice(&host_param_value, channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut param_addr = 0usize;
        param_addr.recv(channel_receiver).unwrap();
        let host_param_value = if device_pointer_mode {
            Box::<[f64]>::default()
        } else {
            recv_slice::<f64, _>(channel_receiver).unwrap()
        };
        let param_arg = if device_pointer_mode {
            param_addr as *const f64
        } else {
            host_param_value.as_ptr()
        };
    }
    'server_execution: {
        let result = unsafe { cublasDrotm_v2_64(handle, n, x, incx, y, incy, param_arg) };
    }
}

#[cuda_hook(proc_id = 1238)]
fn cublasSrotmg_v2(
    handle: cublasHandle_t,
    #[skip] d1: *mut f32,
    #[skip] d2: *mut f32,
    #[skip] x1: *mut f32,
    #[skip] y1: *const f32,
    #[skip] param: *mut f32,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode
            && (d1.is_null() || d2.is_null() || x1.is_null() || y1.is_null() || param.is_null())
        {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let d1_addr = d1 as usize;
        let d2_addr = d2 as usize;
        let x1_addr = x1 as usize;
        let y1_addr = y1 as usize;
        let param_addr = param as usize;
        let mut host_d1_value = 0.0f32;
        let mut host_d2_value = 0.0f32;
        let mut host_x1_value = 0.0f32;
        let mut host_y1_value = 0.0f32;
        if !device_pointer_mode {
            unsafe {
                host_d1_value = *d1;
                host_d2_value = *d2;
                host_x1_value = *x1;
                host_y1_value = *y1;
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        d1_addr.send(channel_sender).unwrap();
        d2_addr.send(channel_sender).unwrap();
        x1_addr.send(channel_sender).unwrap();
        y1_addr.send(channel_sender).unwrap();
        param_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            host_d1_value.send(channel_sender).unwrap();
            host_d2_value.send(channel_sender).unwrap();
            host_x1_value.send(channel_sender).unwrap();
            host_y1_value.send(channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut d1_addr = 0usize;
        d1_addr.recv(channel_receiver).unwrap();
        let mut d2_addr = 0usize;
        d2_addr.recv(channel_receiver).unwrap();
        let mut x1_addr = 0usize;
        x1_addr.recv(channel_receiver).unwrap();
        let mut y1_addr = 0usize;
        y1_addr.recv(channel_receiver).unwrap();
        let mut param_addr = 0usize;
        param_addr.recv(channel_receiver).unwrap();
        let mut host_d1_value = 0.0f32;
        let mut host_d2_value = 0.0f32;
        let mut host_x1_value = 0.0f32;
        let mut host_y1_value = 0.0f32;
        let mut host_param_value = [0.0f32; 5];
        if !device_pointer_mode {
            host_d1_value.recv(channel_receiver).unwrap();
            host_d2_value.recv(channel_receiver).unwrap();
            host_x1_value.recv(channel_receiver).unwrap();
            host_y1_value.recv(channel_receiver).unwrap();
        }
        let d1_arg = if device_pointer_mode {
            d1_addr as *mut f32
        } else {
            &raw mut host_d1_value
        };
        let d2_arg = if device_pointer_mode {
            d2_addr as *mut f32
        } else {
            &raw mut host_d2_value
        };
        let x1_arg = if device_pointer_mode {
            x1_addr as *mut f32
        } else {
            &raw mut host_x1_value
        };
        let y1_arg = if device_pointer_mode {
            y1_addr as *const f32
        } else {
            &raw const host_y1_value
        };
        let param_arg = if device_pointer_mode {
            param_addr as *mut f32
        } else {
            host_param_value.as_mut_ptr()
        };
    }
    'server_execution: {
        let result = unsafe { cublasSrotmg_v2(handle, d1_arg, d2_arg, x1_arg, y1_arg, param_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_d1_value.send(channel_sender).unwrap();
            host_d2_value.send(channel_sender).unwrap();
            host_x1_value.send(channel_sender).unwrap();
            send_slice(&host_param_value, channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_d1_value = 0.0f32;
            let mut host_d2_value = 0.0f32;
            let mut host_x1_value = 0.0f32;
            host_d1_value.recv(channel_receiver).unwrap();
            host_d2_value.recv(channel_receiver).unwrap();
            host_x1_value.recv(channel_receiver).unwrap();
            let host_param_value = recv_slice::<f32, _>(channel_receiver).unwrap();
            assert_eq!(host_param_value.len(), 5);
            unsafe {
                *d1 = host_d1_value;
                *d2 = host_d2_value;
                *x1 = host_x1_value;
                std::ptr::copy_nonoverlapping(host_param_value.as_ptr(), param, 5);
            }
        }
    }
}

#[cuda_hook(proc_id = 1239)]
fn cublasDrotmg_v2(
    handle: cublasHandle_t,
    #[skip] d1: *mut f64,
    #[skip] d2: *mut f64,
    #[skip] x1: *mut f64,
    #[skip] y1: *const f64,
    #[skip] param: *mut f64,
) -> cublasStatus_t {
    'client_before_send: {
        let pointer_mode = CUBLAS_CACHE
            .read()
            .unwrap()
            .pointer_modes
            .get(&handle)
            .copied()
            .unwrap_or(cublasPointerMode_t::CUBLAS_POINTER_MODE_HOST);
        let device_pointer_mode = pointer_mode == cublasPointerMode_t::CUBLAS_POINTER_MODE_DEVICE;
        if !device_pointer_mode
            && (d1.is_null() || d2.is_null() || x1.is_null() || y1.is_null() || param.is_null())
        {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let d1_addr = d1 as usize;
        let d2_addr = d2 as usize;
        let x1_addr = x1 as usize;
        let y1_addr = y1 as usize;
        let param_addr = param as usize;
        let mut host_d1_value = 0.0f64;
        let mut host_d2_value = 0.0f64;
        let mut host_x1_value = 0.0f64;
        let mut host_y1_value = 0.0f64;
        if !device_pointer_mode {
            unsafe {
                host_d1_value = *d1;
                host_d2_value = *d2;
                host_x1_value = *x1;
                host_y1_value = *y1;
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        d1_addr.send(channel_sender).unwrap();
        d2_addr.send(channel_sender).unwrap();
        x1_addr.send(channel_sender).unwrap();
        y1_addr.send(channel_sender).unwrap();
        param_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            host_d1_value.send(channel_sender).unwrap();
            host_d2_value.send(channel_sender).unwrap();
            host_x1_value.send(channel_sender).unwrap();
            host_y1_value.send(channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut d1_addr = 0usize;
        d1_addr.recv(channel_receiver).unwrap();
        let mut d2_addr = 0usize;
        d2_addr.recv(channel_receiver).unwrap();
        let mut x1_addr = 0usize;
        x1_addr.recv(channel_receiver).unwrap();
        let mut y1_addr = 0usize;
        y1_addr.recv(channel_receiver).unwrap();
        let mut param_addr = 0usize;
        param_addr.recv(channel_receiver).unwrap();
        let mut host_d1_value = 0.0f64;
        let mut host_d2_value = 0.0f64;
        let mut host_x1_value = 0.0f64;
        let mut host_y1_value = 0.0f64;
        let mut host_param_value = [0.0f64; 5];
        if !device_pointer_mode {
            host_d1_value.recv(channel_receiver).unwrap();
            host_d2_value.recv(channel_receiver).unwrap();
            host_x1_value.recv(channel_receiver).unwrap();
            host_y1_value.recv(channel_receiver).unwrap();
        }
        let d1_arg = if device_pointer_mode {
            d1_addr as *mut f64
        } else {
            &raw mut host_d1_value
        };
        let d2_arg = if device_pointer_mode {
            d2_addr as *mut f64
        } else {
            &raw mut host_d2_value
        };
        let x1_arg = if device_pointer_mode {
            x1_addr as *mut f64
        } else {
            &raw mut host_x1_value
        };
        let y1_arg = if device_pointer_mode {
            y1_addr as *const f64
        } else {
            &raw const host_y1_value
        };
        let param_arg = if device_pointer_mode {
            param_addr as *mut f64
        } else {
            host_param_value.as_mut_ptr()
        };
    }
    'server_execution: {
        let result = unsafe { cublasDrotmg_v2(handle, d1_arg, d2_arg, x1_arg, y1_arg, param_arg) };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            host_d1_value.send(channel_sender).unwrap();
            host_d2_value.send(channel_sender).unwrap();
            host_x1_value.send(channel_sender).unwrap();
            send_slice(&host_param_value, channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let mut host_d1_value = 0.0f64;
            let mut host_d2_value = 0.0f64;
            let mut host_x1_value = 0.0f64;
            host_d1_value.recv(channel_receiver).unwrap();
            host_d2_value.recv(channel_receiver).unwrap();
            host_x1_value.recv(channel_receiver).unwrap();
            let host_param_value = recv_slice::<f64, _>(channel_receiver).unwrap();
            assert_eq!(host_param_value.len(), 5);
            unsafe {
                *d1 = host_d1_value;
                *d2 = host_d2_value;
                *x1 = host_x1_value;
                std::ptr::copy_nonoverlapping(host_param_value.as_ptr(), param, 5);
            }
        }
    }
}
