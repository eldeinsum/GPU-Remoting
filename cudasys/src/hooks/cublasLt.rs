use crate::types::cublasLt::*;
use codegen::{cuda_custom_hook, cuda_hook};
use std::os::raw::*;

#[cuda_hook(proc_id = 1500)]
fn cublasLtCreate(lightHandle: *mut cublasLtHandle_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1501)]
fn cublasLtDestroy(lightHandle: cublasLtHandle_t) -> cublasStatus_t;

#[cuda_custom_hook] // local: returns a client-owned C string
fn cublasLtGetStatusName(status: cublasStatus_t) -> *const c_char;

#[cuda_custom_hook] // local: returns a client-owned C string
fn cublasLtGetStatusString(status: cublasStatus_t) -> *const c_char;

#[cuda_custom_hook] // local: derived from remoted property queries
fn cublasLtGetVersion() -> usize;

#[cuda_custom_hook] // local: derived from remoted runtime version query
fn cublasLtGetCudartVersion() -> usize;

#[cuda_hook(proc_id = 1502)]
fn cublasLtGetProperty(type_: libraryPropertyType, value: *mut c_int) -> cublasStatus_t;

#[cuda_hook(proc_id = 1503)]
fn cublasLtHeuristicsCacheGetCapacity(capacity: *mut usize) -> cublasStatus_t;

#[cuda_hook(proc_id = 1504)]
fn cublasLtHeuristicsCacheSetCapacity(capacity: usize) -> cublasStatus_t;

#[cuda_hook(proc_id = 1511)]
fn cublasLtMatmul(
    lightHandle: cublasLtHandle_t,
    computeDesc: cublasLtMatmulDesc_t,
    #[skip] alpha: *const c_void,
    #[device] A: *const c_void,
    Adesc: cublasLtMatrixLayout_t,
    #[device] B: *const c_void,
    Bdesc: cublasLtMatrixLayout_t,
    #[skip] beta: *const c_void,
    #[device] C: *const c_void,
    Cdesc: cublasLtMatrixLayout_t,
    #[device] D: *mut c_void,
    Ddesc: cublasLtMatrixLayout_t,
    #[host] algo: *const cublasLtMatmulAlgo_t, // FIXME: nullable
    #[device] workspace: *mut c_void,
    workspaceSizeInBytes: usize,
    stream: cudaStream_t,
) -> cublasStatus_t {
    'client_before_send: {
        let desc_state = CUBLAS_CACHE
            .read()
            .unwrap()
            .lt_matmul_descs
            .get(&computeDesc)
            .copied()
            .unwrap_or(CublasLtMatmulDescState {
                pointer_mode: cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST,
                scale_type_size: Some(std::mem::size_of::<f32>()),
            });
        let pointer_mode = desc_state.pointer_mode;
        let scalar_size = desc_state.scale_type_size.unwrap_or(0);
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
        let alpha_host = matches!(
            pointer_mode,
            cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST
        );
        let beta_host = matches!(
            pointer_mode,
            cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST
                | cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST
        );
        let alpha_host_bytes = if alpha_host {
            assert!(scalar_size > 0);
            assert!(!alpha.is_null());
            unsafe { std::slice::from_raw_parts(alpha.cast::<u8>(), scalar_size) }
        } else {
            &[][..]
        };
        let beta_host_bytes = if beta_host {
            assert!(scalar_size > 0);
            assert!(!beta.is_null());
            unsafe { std::slice::from_raw_parts(beta.cast::<u8>(), scalar_size) }
        } else {
            &[][..]
        };
    }
    'client_extra_send: {
        pointer_mode.send(channel_sender).unwrap();
        send_slice(alpha_host_bytes, channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        send_slice(beta_host_bytes, channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut pointer_mode = cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST;
        pointer_mode.recv(channel_receiver).unwrap();
        let alpha_host_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let beta_host_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        #[repr(align(16))]
        struct AlignedScalar([u8; 16]);
        let mut alpha_storage = AlignedScalar([0; 16]);
        let mut beta_storage = AlignedScalar([0; 16]);
        assert!(alpha_host_bytes.len() <= alpha_storage.0.len());
        assert!(beta_host_bytes.len() <= beta_storage.0.len());
        alpha_storage.0[..alpha_host_bytes.len()].copy_from_slice(&alpha_host_bytes);
        beta_storage.0[..beta_host_bytes.len()].copy_from_slice(&beta_host_bytes);
        let alpha_arg = match pointer_mode {
            cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST => alpha_storage.0.as_ptr().cast(),
            cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_DEVICE
            | cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_DEVICE_VECTOR
            | cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO
            | cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST => {
                alpha_addr as *const c_void
            }
        };
        let beta_arg = match pointer_mode {
            cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST
            | cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST => {
                beta_storage.0.as_ptr().cast()
            }
            cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_DEVICE
            | cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_DEVICE_VECTOR
            | cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO => {
                beta_addr as *const c_void
            }
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasLtMatmul(
                lightHandle,
                computeDesc,
                alpha_arg,
                A,
                Adesc,
                B,
                Bdesc,
                beta_arg,
                C,
                Cdesc,
                D,
                Ddesc,
                algo__ptr.cast(),
                workspace,
                workspaceSizeInBytes,
                stream,
            )
        };
    }
}

#[cuda_hook(proc_id = 1516)]
fn cublasLtMatmulAlgoGetHeuristic(
    lightHandle: cublasLtHandle_t,
    operationDesc: cublasLtMatmulDesc_t,
    Adesc: cublasLtMatrixLayout_t,
    Bdesc: cublasLtMatrixLayout_t,
    Cdesc: cublasLtMatrixLayout_t,
    Ddesc: cublasLtMatrixLayout_t,
    preference: cublasLtMatmulPreference_t,
    requestedAlgoCount: c_int,
    #[host(output, len = requestedAlgoCount)]
    heuristicResultsArray: *mut cublasLtMatmulHeuristicResult_t,
    returnAlgoCount: *mut c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1533)]
fn cublasLtMatmulAlgoGetIds(
    lightHandle: cublasLtHandle_t,
    computeType: cublasComputeType_t,
    scaleType: cudaDataType_t,
    Atype: cudaDataType_t,
    Btype: cudaDataType_t,
    Ctype: cudaDataType_t,
    Dtype: cudaDataType_t,
    requestedAlgoCount: c_int,
    #[host(output, len = requestedAlgoCount)] algoIdsArray: *mut c_int,
    returnAlgoCount: *mut c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1534)]
fn cublasLtMatmulAlgoInit(
    lightHandle: cublasLtHandle_t,
    computeType: cublasComputeType_t,
    scaleType: cudaDataType_t,
    Atype: cudaDataType_t,
    Btype: cudaDataType_t,
    Ctype: cudaDataType_t,
    Dtype: cudaDataType_t,
    algoId: c_int,
    algo: *mut cublasLtMatmulAlgo_t,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1536)]
fn cublasLtMatmulAlgoCapGetAttribute(
    #[host] algo: *const cublasLtMatmulAlgo_t,
    attr: cublasLtMatmulAlgoCapAttributes_t,
    #[host(output, len = sizeInBytes)] buf: *mut c_void,
    sizeInBytes: usize,
    sizeWritten: *mut usize,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1537)]
fn cublasLtMatmulAlgoConfigGetAttribute(
    #[host] algo: *const cublasLtMatmulAlgo_t,
    attr: cublasLtMatmulAlgoConfigAttributes_t,
    #[host(output, len = sizeInBytes)] buf: *mut c_void,
    sizeInBytes: usize,
    sizeWritten: *mut usize,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1519)]
fn cublasLtMatmulDescCreate(
    matmulDesc: *mut cublasLtMatmulDesc_t,
    computeType: cublasComputeType_t,
    scaleType: cudaDataType_t,
) -> cublasStatus_t {
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            CUBLAS_CACHE.write().unwrap().lt_matmul_descs.insert(
                *matmulDesc,
                CublasLtMatmulDescState {
                    pointer_mode: cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST,
                    scale_type_size: cublaslt_scale_type_size(scaleType),
                },
            );
        }
    }
}

#[cuda_hook(proc_id = 1521)]
fn cublasLtMatmulDescDestroy(matmulDesc: cublasLtMatmulDesc_t) -> cublasStatus_t {
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            CUBLAS_CACHE
                .write()
                .unwrap()
                .lt_matmul_descs
                .remove(&matmulDesc);
        }
    }
}

#[cuda_hook(proc_id = 1523)]
fn cublasLtMatmulDescSetAttribute(
    matmulDesc: cublasLtMatmulDesc_t,
    attr: cublasLtMatmulDescAttributes_t,
    #[host(len = sizeInBytes)] buf: *const c_void,
    sizeInBytes: usize,
) -> cublasStatus_t {
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            let mut cache = CUBLAS_CACHE.write().unwrap();
            let state =
                cache
                    .lt_matmul_descs
                    .entry(matmulDesc)
                    .or_insert(CublasLtMatmulDescState {
                        pointer_mode: cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST,
                        scale_type_size: Some(std::mem::size_of::<f32>()),
                    });
            match attr {
                cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_POINTER_MODE => {
                    if sizeInBytes >= std::mem::size_of::<u32>() {
                        let value = unsafe { std::ptr::read_unaligned(buf.as_ptr().cast::<u32>()) };
                        if let Some(pointer_mode) = cublaslt_pointer_mode_from_u32(value) {
                            state.pointer_mode = pointer_mode;
                        }
                    }
                }
                cublasLtMatmulDescAttributes_t::CUBLASLT_MATMUL_DESC_SCALE_TYPE => {
                    if sizeInBytes >= std::mem::size_of::<u32>() {
                        let value = unsafe { std::ptr::read_unaligned(buf.as_ptr().cast::<u32>()) };
                        if let Some(scale_type) = cublaslt_scale_type_from_u32(value) {
                            state.scale_type_size = cublaslt_scale_type_size(scale_type);
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

#[cuda_hook(proc_id = 1522)]
fn cublasLtMatmulDescGetAttribute(
    matmulDesc: cublasLtMatmulDesc_t,
    attr: cublasLtMatmulDescAttributes_t,
    #[host(output, len = sizeInBytes)] buf: *mut c_void,
    sizeInBytes: usize,
    sizeWritten: *mut usize,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1524)]
fn cublasLtMatmulPreferenceCreate(pref: *mut cublasLtMatmulPreference_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1526)]
fn cublasLtMatmulPreferenceDestroy(pref: cublasLtMatmulPreference_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1528)]
fn cublasLtMatmulPreferenceSetAttribute(
    pref: cublasLtMatmulPreference_t,
    attr: cublasLtMatmulPreferenceAttributes_t,
    #[host(len = sizeInBytes)] buf: *const c_void,
    sizeInBytes: usize,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1527)]
fn cublasLtMatmulPreferenceGetAttribute(
    pref: cublasLtMatmulPreference_t,
    attr: cublasLtMatmulPreferenceAttributes_t,
    #[host(output, len = sizeInBytes)] buf: *mut c_void,
    sizeInBytes: usize,
    sizeWritten: *mut usize,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1529)]
fn cublasLtMatrixLayoutCreate(
    matLayout: *mut cublasLtMatrixLayout_t,
    type_: cudaDataType,
    rows: u64,
    cols: u64,
    ld: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1531)]
fn cublasLtMatrixLayoutDestroy(matLayout: cublasLtMatrixLayout_t) -> cublasStatus_t;

#[cuda_hook(proc_id = 1530)]
fn cublasLtMatrixLayoutSetAttribute(
    matLayout: cublasLtMatrixLayout_t,
    attr: cublasLtMatrixLayoutAttribute_t,
    #[host(len = sizeInBytes)] buf: *const c_void,
    sizeInBytes: usize,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1532)]
fn cublasLtMatrixLayoutGetAttribute(
    matLayout: cublasLtMatrixLayout_t,
    attr: cublasLtMatrixLayoutAttribute_t,
    #[host(output, len = sizeInBytes)] buf: *mut c_void,
    sizeInBytes: usize,
    sizeWritten: *mut usize,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1542)]
fn cublasLtMatrixTransform(
    lightHandle: cublasLtHandle_t,
    transformDesc: cublasLtMatrixTransformDesc_t,
    #[skip] alpha: *const c_void,
    #[device] A: *const c_void,
    Adesc: cublasLtMatrixLayout_t,
    #[skip] beta: *const c_void,
    #[device] B: *const c_void,
    Bdesc: cublasLtMatrixLayout_t,
    #[device] C: *mut c_void,
    Cdesc: cublasLtMatrixLayout_t,
    stream: cudaStream_t,
) -> cublasStatus_t {
    'client_before_send: {
        let desc_state = CUBLAS_CACHE
            .read()
            .unwrap()
            .lt_transform_descs
            .get(&transformDesc)
            .copied()
            .unwrap_or(CublasLtTransformDescState {
                pointer_mode: cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST,
                scale_type_size: Some(std::mem::size_of::<f32>()),
            });
        let pointer_mode = desc_state.pointer_mode;
        let scalar_size = desc_state.scale_type_size.unwrap_or(0);
        let alpha_addr = alpha as usize;
        let beta_addr = beta as usize;
        let alpha_host = matches!(
            pointer_mode,
            cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST
        );
        let beta_host = matches!(
            pointer_mode,
            cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST
                | cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST
        );
        let alpha_host_bytes = if alpha_host {
            assert!(scalar_size > 0);
            assert!(!alpha.is_null());
            unsafe { std::slice::from_raw_parts(alpha.cast::<u8>(), scalar_size) }
        } else {
            &[][..]
        };
        let beta_host_bytes = if beta_host {
            assert!(scalar_size > 0);
            assert!(!beta.is_null());
            unsafe { std::slice::from_raw_parts(beta.cast::<u8>(), scalar_size) }
        } else {
            &[][..]
        };
    }
    'client_extra_send: {
        pointer_mode.send(channel_sender).unwrap();
        send_slice(alpha_host_bytes, channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        send_slice(beta_host_bytes, channel_sender).unwrap();
        beta_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut pointer_mode = cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST;
        pointer_mode.recv(channel_receiver).unwrap();
        let alpha_host_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        let beta_host_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        #[repr(align(16))]
        struct AlignedScalar([u8; 16]);
        let mut alpha_storage = AlignedScalar([0; 16]);
        let mut beta_storage = AlignedScalar([0; 16]);
        assert!(alpha_host_bytes.len() <= alpha_storage.0.len());
        assert!(beta_host_bytes.len() <= beta_storage.0.len());
        alpha_storage.0[..alpha_host_bytes.len()].copy_from_slice(&alpha_host_bytes);
        beta_storage.0[..beta_host_bytes.len()].copy_from_slice(&beta_host_bytes);
        let alpha_arg = match pointer_mode {
            cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST => alpha_storage.0.as_ptr().cast(),
            cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_DEVICE
            | cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_DEVICE_VECTOR
            | cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO
            | cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST => {
                alpha_addr as *const c_void
            }
        };
        let beta_arg = match pointer_mode {
            cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST
            | cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_HOST => {
                beta_storage.0.as_ptr().cast()
            }
            cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_DEVICE
            | cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_DEVICE_VECTOR
            | cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_ALPHA_DEVICE_VECTOR_BETA_ZERO => {
                beta_addr as *const c_void
            }
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasLtMatrixTransform(
                lightHandle,
                transformDesc,
                alpha_arg,
                A,
                Adesc,
                beta_arg,
                B,
                Bdesc,
                C,
                Cdesc,
                stream,
            )
        };
    }
}

#[cuda_hook(proc_id = 1538)]
fn cublasLtMatrixTransformDescCreate(
    transformDesc: *mut cublasLtMatrixTransformDesc_t,
    scaleType: cudaDataType,
) -> cublasStatus_t {
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            CUBLAS_CACHE.write().unwrap().lt_transform_descs.insert(
                *transformDesc,
                CublasLtTransformDescState {
                    pointer_mode: cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST,
                    scale_type_size: cublaslt_scale_type_size(scaleType),
                },
            );
        }
    }
}

#[cuda_hook(proc_id = 1539)]
fn cublasLtMatrixTransformDescDestroy(
    transformDesc: cublasLtMatrixTransformDesc_t,
) -> cublasStatus_t {
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            CUBLAS_CACHE
                .write()
                .unwrap()
                .lt_transform_descs
                .remove(&transformDesc);
        }
    }
}

#[cuda_hook(proc_id = 1540)]
fn cublasLtMatrixTransformDescSetAttribute(
    transformDesc: cublasLtMatrixTransformDesc_t,
    attr: cublasLtMatrixTransformDescAttributes_t,
    #[host(len = sizeInBytes)] buf: *const c_void,
    sizeInBytes: usize,
) -> cublasStatus_t {
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS {
            let mut cache = CUBLAS_CACHE.write().unwrap();
            let state = cache.lt_transform_descs.entry(transformDesc).or_insert(
                CublasLtTransformDescState {
                    pointer_mode: cublasLtPointerMode_t::CUBLASLT_POINTER_MODE_HOST,
                    scale_type_size: Some(std::mem::size_of::<f32>()),
                },
            );
            match attr {
                cublasLtMatrixTransformDescAttributes_t::CUBLASLT_MATRIX_TRANSFORM_DESC_POINTER_MODE => {
                    if sizeInBytes >= std::mem::size_of::<u32>() {
                        let value = unsafe { std::ptr::read_unaligned(buf.as_ptr().cast::<u32>()) };
                        if let Some(pointer_mode) = cublaslt_pointer_mode_from_u32(value) {
                            state.pointer_mode = pointer_mode;
                        }
                    }
                }
                cublasLtMatrixTransformDescAttributes_t::CUBLASLT_MATRIX_TRANSFORM_DESC_SCALE_TYPE => {
                    if sizeInBytes >= std::mem::size_of::<u32>() {
                        let value = unsafe { std::ptr::read_unaligned(buf.as_ptr().cast::<u32>()) };
                        if let Some(scale_type) = cublaslt_scale_type_from_u32(value) {
                            state.scale_type_size = cublaslt_scale_type_size(scale_type);
                        }
                    }
                }
                _ => {}
            }
        }
    }
}

#[cuda_hook(proc_id = 1541)]
fn cublasLtMatrixTransformDescGetAttribute(
    transformDesc: cublasLtMatrixTransformDesc_t,
    attr: cublasLtMatrixTransformDescAttributes_t,
    #[host(output, len = sizeInBytes)] buf: *mut c_void,
    sizeInBytes: usize,
    sizeWritten: *mut usize,
) -> cublasStatus_t;
