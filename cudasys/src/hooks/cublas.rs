use crate::types::cublas::*;
use codegen::cuda_hook;
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

#[cuda_hook(proc_id = 1119, async_api)]
fn cublasSetMathMode(handle: cublasHandle_t, mode: cublasMath_t) -> cublasStatus_t;

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

#[cuda_hook(proc_id = 1120)]
fn cublasGetMathMode(handle: cublasHandle_t, mode: *mut cublasMath_t) -> cublasStatus_t;

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
