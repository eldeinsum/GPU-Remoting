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
