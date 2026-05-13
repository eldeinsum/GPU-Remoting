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

#[cuda_hook(proc_id = 1676)]
fn cublasGetEmulationStrategy(
    handle: cublasHandle_t,
    emulationStrategy: *mut cublasEmulationStrategy_t,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1677)]
fn cublasSetEmulationStrategy(
    handle: cublasHandle_t,
    emulationStrategy: cublasEmulationStrategy_t,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1678)]
fn cublasGetEmulationSpecialValuesSupport(
    handle: cublasHandle_t,
    mask: *mut cudaEmulationSpecialValuesSupport,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1679)]
fn cublasSetEmulationSpecialValuesSupport(
    handle: cublasHandle_t,
    mask: cudaEmulationSpecialValuesSupport,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1680)]
fn cublasGetFixedPointEmulationMantissaControl(
    handle: cublasHandle_t,
    mantissaControl: *mut cudaEmulationMantissaControl,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1681)]
fn cublasSetFixedPointEmulationMantissaControl(
    handle: cublasHandle_t,
    mantissaControl: cudaEmulationMantissaControl,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1682)]
fn cublasGetFixedPointEmulationMaxMantissaBitCount(
    handle: cublasHandle_t,
    maxMantissaBitCount: *mut c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1683)]
fn cublasSetFixedPointEmulationMaxMantissaBitCount(
    handle: cublasHandle_t,
    maxMantissaBitCount: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1684)]
fn cublasGetFixedPointEmulationMantissaBitOffset(
    handle: cublasHandle_t,
    mantissaBitOffset: *mut c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1685)]
fn cublasSetFixedPointEmulationMantissaBitOffset(
    handle: cublasHandle_t,
    mantissaBitOffset: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1686)]
fn cublasGetFixedPointEmulationMantissaBitCountPointer(
    handle: cublasHandle_t,
    mantissaBitCount: *mut *mut c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1687)]
fn cublasSetFixedPointEmulationMantissaBitCountPointer(
    handle: cublasHandle_t,
    #[device] mantissaBitCount: *mut c_int,
) -> cublasStatus_t;

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

#[cuda_hook(proc_id = 1688, async_api)]
fn cublasSetVectorAsync(
    n: c_int,
    elemSize: c_int,
    #[skip] hostPtr: *const c_void,
    incx: c_int,
    #[device] devicePtr: *mut c_void,
    incy: c_int,
    stream: cudaStream_t,
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
            assert!(!hostPtr.is_null());
            assert!(incx != 0);
            for i in 0..n_usize {
                let src = unsafe {
                    hostPtr
                        .cast::<u8>()
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
        let host_arg = if packed.is_empty() {
            std::ptr::null()
        } else {
            packed.as_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            assert_eq!(
                cudasys::cudart::cudaStreamSynchronize(stream.cast()),
                Default::default()
            );
            cublasSetVector(n, elemSize, host_arg, 1, devicePtr, incy)
        };
    }
}

#[cuda_hook(proc_id = 1689, async_api)]
fn cublasSetVectorAsync_64(
    n: i64,
    elemSize: i64,
    #[skip] hostPtr: *const c_void,
    incx: i64,
    #[device] devicePtr: *mut c_void,
    incy: i64,
    stream: cudaStream_t,
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
            assert!(!hostPtr.is_null());
            assert!(incx != 0);
            for i in 0..n_usize {
                let src = unsafe {
                    hostPtr
                        .cast::<u8>()
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
        let host_arg = if packed.is_empty() {
            std::ptr::null()
        } else {
            packed.as_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            assert_eq!(
                cudasys::cudart::cudaStreamSynchronize(stream.cast()),
                Default::default()
            );
            cublasSetVector_64(n, elemSize, host_arg, 1, devicePtr, incy)
        };
    }
}

#[cuda_hook(proc_id = 1690)]
fn cublasGetVectorAsync(
    n: c_int,
    elemSize: c_int,
    #[device] devicePtr: *const c_void,
    incx: c_int,
    #[skip] hostPtr: *mut c_void,
    incy: c_int,
    stream: cudaStream_t,
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
            assert!(!hostPtr.is_null());
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
        let host_arg = if packed.is_empty() {
            std::ptr::null_mut()
        } else {
            packed.as_mut_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            assert_eq!(
                cudasys::cudart::cudaStreamSynchronize(stream.cast()),
                Default::default()
            );
            cublasGetVector(n, elemSize, devicePtr, incx, host_arg, 1)
        };
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
                    hostPtr
                        .cast::<u8>()
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

#[cuda_hook(proc_id = 1691)]
fn cublasGetVectorAsync_64(
    n: i64,
    elemSize: i64,
    #[device] devicePtr: *const c_void,
    incx: i64,
    #[skip] hostPtr: *mut c_void,
    incy: i64,
    stream: cudaStream_t,
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
            assert!(!hostPtr.is_null());
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
        let host_arg = if packed.is_empty() {
            std::ptr::null_mut()
        } else {
            packed.as_mut_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            assert_eq!(
                cudasys::cudart::cudaStreamSynchronize(stream.cast()),
                Default::default()
            );
            cublasGetVector_64(n, elemSize, devicePtr, incx, host_arg, 1)
        };
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
                    hostPtr
                        .cast::<u8>()
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

#[cuda_hook(proc_id = 1695)]
fn cublasInit() -> cublasStatus_t;

#[cuda_hook(proc_id = 1696)]
fn cublasShutdown() -> cublasStatus_t;

#[cuda_hook(proc_id = 1697)]
fn cublasGetError() -> cublasStatus_t;

#[cuda_hook(proc_id = 1698)]
fn cublasAlloc(n: c_int, elemSize: c_int, devicePtr: *mut *mut c_void) -> cublasStatus_t;

#[cuda_hook(proc_id = 1699)]
fn cublasFree(#[device] devicePtr: *mut c_void) -> cublasStatus_t;

#[cuda_hook(proc_id = 1700)]
fn cublasSetKernelStream(stream: cudaStream_t) -> cublasStatus_t;

#[cuda_custom_hook(proc_id = 1701)]
fn cublasLoggerConfigure(
    logIsOn: c_int,
    logToStdOut: c_int,
    logToStdErr: c_int,
    logFileName: *const c_char,
) -> cublasStatus_t;

#[cuda_custom_hook(proc_id = 1702)]
fn cublasSnrm2(n: c_int, x: *const f32, incx: c_int) -> f32;

#[cuda_custom_hook(proc_id = 1703)]
fn cublasDnrm2(n: c_int, x: *const f64, incx: c_int) -> f64;

#[cuda_custom_hook(proc_id = 1704)]
fn cublasScnrm2(n: c_int, x: *const cuComplex, incx: c_int) -> f32;

#[cuda_custom_hook(proc_id = 1705)]
fn cublasDznrm2(n: c_int, x: *const cuDoubleComplex, incx: c_int) -> f64;

#[cuda_custom_hook(proc_id = 1706)]
fn cublasSdot(n: c_int, x: *const f32, incx: c_int, y: *const f32, incy: c_int) -> f32;

#[cuda_custom_hook(proc_id = 1707)]
fn cublasDdot(n: c_int, x: *const f64, incx: c_int, y: *const f64, incy: c_int) -> f64;

#[cuda_custom_hook(proc_id = 1708)]
fn cublasCdotu(
    n: c_int,
    x: *const cuComplex,
    incx: c_int,
    y: *const cuComplex,
    incy: c_int,
) -> cuComplex;

#[cuda_custom_hook(proc_id = 1709)]
fn cublasCdotc(
    n: c_int,
    x: *const cuComplex,
    incx: c_int,
    y: *const cuComplex,
    incy: c_int,
) -> cuComplex;

#[cuda_custom_hook(proc_id = 1710)]
fn cublasZdotu(
    n: c_int,
    x: *const cuDoubleComplex,
    incx: c_int,
    y: *const cuDoubleComplex,
    incy: c_int,
) -> cuDoubleComplex;

#[cuda_custom_hook(proc_id = 1711)]
fn cublasZdotc(
    n: c_int,
    x: *const cuDoubleComplex,
    incx: c_int,
    y: *const cuDoubleComplex,
    incy: c_int,
) -> cuDoubleComplex;

#[cuda_custom_hook(proc_id = 1712)]
fn cublasSscal(n: c_int, alpha: f32, x: *mut f32, incx: c_int);

#[cuda_custom_hook(proc_id = 1713)]
fn cublasDscal(n: c_int, alpha: f64, x: *mut f64, incx: c_int);

#[cuda_custom_hook(proc_id = 1714)]
fn cublasCscal(n: c_int, alpha: cuComplex, x: *mut cuComplex, incx: c_int);

#[cuda_custom_hook(proc_id = 1715)]
fn cublasZscal(n: c_int, alpha: cuDoubleComplex, x: *mut cuDoubleComplex, incx: c_int);

#[cuda_custom_hook(proc_id = 1716)]
fn cublasCsscal(n: c_int, alpha: f32, x: *mut cuComplex, incx: c_int);

#[cuda_custom_hook(proc_id = 1717)]
fn cublasZdscal(n: c_int, alpha: f64, x: *mut cuDoubleComplex, incx: c_int);

#[cuda_custom_hook(proc_id = 1718)]
fn cublasSaxpy(n: c_int, alpha: f32, x: *const f32, incx: c_int, y: *mut f32, incy: c_int);

#[cuda_custom_hook(proc_id = 1719)]
fn cublasDaxpy(n: c_int, alpha: f64, x: *const f64, incx: c_int, y: *mut f64, incy: c_int);

#[cuda_custom_hook(proc_id = 1720)]
fn cublasCaxpy(
    n: c_int,
    alpha: cuComplex,
    x: *const cuComplex,
    incx: c_int,
    y: *mut cuComplex,
    incy: c_int,
);

#[cuda_custom_hook(proc_id = 1721)]
fn cublasZaxpy(
    n: c_int,
    alpha: cuDoubleComplex,
    x: *const cuDoubleComplex,
    incx: c_int,
    y: *mut cuDoubleComplex,
    incy: c_int,
);

#[cuda_custom_hook(proc_id = 1722)]
fn cublasScopy(n: c_int, x: *const f32, incx: c_int, y: *mut f32, incy: c_int);

#[cuda_custom_hook(proc_id = 1723)]
fn cublasDcopy(n: c_int, x: *const f64, incx: c_int, y: *mut f64, incy: c_int);

#[cuda_custom_hook(proc_id = 1724)]
fn cublasCcopy(n: c_int, x: *const cuComplex, incx: c_int, y: *mut cuComplex, incy: c_int);

#[cuda_custom_hook(proc_id = 1725)]
fn cublasZcopy(
    n: c_int,
    x: *const cuDoubleComplex,
    incx: c_int,
    y: *mut cuDoubleComplex,
    incy: c_int,
);

#[cuda_custom_hook(proc_id = 1726)]
fn cublasSswap(n: c_int, x: *mut f32, incx: c_int, y: *mut f32, incy: c_int);

#[cuda_custom_hook(proc_id = 1727)]
fn cublasDswap(n: c_int, x: *mut f64, incx: c_int, y: *mut f64, incy: c_int);

#[cuda_custom_hook(proc_id = 1728)]
fn cublasCswap(n: c_int, x: *mut cuComplex, incx: c_int, y: *mut cuComplex, incy: c_int);

#[cuda_custom_hook(proc_id = 1729)]
fn cublasZswap(
    n: c_int,
    x: *mut cuDoubleComplex,
    incx: c_int,
    y: *mut cuDoubleComplex,
    incy: c_int,
);

#[cuda_custom_hook(proc_id = 1730)]
fn cublasIsamax(n: c_int, x: *const f32, incx: c_int) -> c_int;

#[cuda_custom_hook(proc_id = 1731)]
fn cublasIdamax(n: c_int, x: *const f64, incx: c_int) -> c_int;

#[cuda_custom_hook(proc_id = 1732)]
fn cublasIcamax(n: c_int, x: *const cuComplex, incx: c_int) -> c_int;

#[cuda_custom_hook(proc_id = 1733)]
fn cublasIzamax(n: c_int, x: *const cuDoubleComplex, incx: c_int) -> c_int;

#[cuda_custom_hook(proc_id = 1734)]
fn cublasIsamin(n: c_int, x: *const f32, incx: c_int) -> c_int;

#[cuda_custom_hook(proc_id = 1735)]
fn cublasIdamin(n: c_int, x: *const f64, incx: c_int) -> c_int;

#[cuda_custom_hook(proc_id = 1736)]
fn cublasIcamin(n: c_int, x: *const cuComplex, incx: c_int) -> c_int;

#[cuda_custom_hook(proc_id = 1737)]
fn cublasIzamin(n: c_int, x: *const cuDoubleComplex, incx: c_int) -> c_int;

#[cuda_custom_hook(proc_id = 1738)]
fn cublasSasum(n: c_int, x: *const f32, incx: c_int) -> f32;

#[cuda_custom_hook(proc_id = 1739)]
fn cublasDasum(n: c_int, x: *const f64, incx: c_int) -> f64;

#[cuda_custom_hook(proc_id = 1740)]
fn cublasScasum(n: c_int, x: *const cuComplex, incx: c_int) -> f32;

#[cuda_custom_hook(proc_id = 1741)]
fn cublasDzasum(n: c_int, x: *const cuDoubleComplex, incx: c_int) -> f64;

#[cuda_custom_hook(proc_id = 1742)]
fn cublasSrot(n: c_int, x: *mut f32, incx: c_int, y: *mut f32, incy: c_int, sc: f32, ss: f32);

#[cuda_custom_hook(proc_id = 1743)]
fn cublasDrot(n: c_int, x: *mut f64, incx: c_int, y: *mut f64, incy: c_int, sc: f64, ss: f64);

#[cuda_custom_hook(proc_id = 1744)]
fn cublasCrot(
    n: c_int,
    x: *mut cuComplex,
    incx: c_int,
    y: *mut cuComplex,
    incy: c_int,
    c: f32,
    s: cuComplex,
);

#[cuda_custom_hook(proc_id = 1745)]
fn cublasZrot(
    n: c_int,
    x: *mut cuDoubleComplex,
    incx: c_int,
    y: *mut cuDoubleComplex,
    incy: c_int,
    sc: f64,
    cs: cuDoubleComplex,
);

#[cuda_custom_hook(proc_id = 1746)]
fn cublasCsrot(
    n: c_int,
    x: *mut cuComplex,
    incx: c_int,
    y: *mut cuComplex,
    incy: c_int,
    c: f32,
    s: f32,
);

#[cuda_custom_hook(proc_id = 1747)]
fn cublasZdrot(
    n: c_int,
    x: *mut cuDoubleComplex,
    incx: c_int,
    y: *mut cuDoubleComplex,
    incy: c_int,
    c: f64,
    s: f64,
);

#[cuda_custom_hook(proc_id = 1748)]
fn cublasSgemv(
    trans: c_char,
    m: c_int,
    n: c_int,
    alpha: f32,
    A: *const f32,
    lda: c_int,
    x: *const f32,
    incx: c_int,
    beta: f32,
    y: *mut f32,
    incy: c_int,
);

#[cuda_custom_hook(proc_id = 1749)]
fn cublasDgemv(
    trans: c_char,
    m: c_int,
    n: c_int,
    alpha: f64,
    A: *const f64,
    lda: c_int,
    x: *const f64,
    incx: c_int,
    beta: f64,
    y: *mut f64,
    incy: c_int,
);

#[cuda_custom_hook(proc_id = 1750)]
fn cublasCgemv(
    trans: c_char,
    m: c_int,
    n: c_int,
    alpha: cuComplex,
    A: *const cuComplex,
    lda: c_int,
    x: *const cuComplex,
    incx: c_int,
    beta: cuComplex,
    y: *mut cuComplex,
    incy: c_int,
);

#[cuda_custom_hook(proc_id = 1751)]
fn cublasZgemv(
    trans: c_char,
    m: c_int,
    n: c_int,
    alpha: cuDoubleComplex,
    A: *const cuDoubleComplex,
    lda: c_int,
    x: *const cuDoubleComplex,
    incx: c_int,
    beta: cuDoubleComplex,
    y: *mut cuDoubleComplex,
    incy: c_int,
);

#[cuda_custom_hook(proc_id = 1752)]
fn cublasSger(
    m: c_int,
    n: c_int,
    alpha: f32,
    x: *const f32,
    incx: c_int,
    y: *const f32,
    incy: c_int,
    A: *mut f32,
    lda: c_int,
);

#[cuda_custom_hook(proc_id = 1753)]
fn cublasDger(
    m: c_int,
    n: c_int,
    alpha: f64,
    x: *const f64,
    incx: c_int,
    y: *const f64,
    incy: c_int,
    A: *mut f64,
    lda: c_int,
);

#[cuda_custom_hook(proc_id = 1754)]
fn cublasCgeru(
    m: c_int,
    n: c_int,
    alpha: cuComplex,
    x: *const cuComplex,
    incx: c_int,
    y: *const cuComplex,
    incy: c_int,
    A: *mut cuComplex,
    lda: c_int,
);

#[cuda_custom_hook(proc_id = 1755)]
fn cublasCgerc(
    m: c_int,
    n: c_int,
    alpha: cuComplex,
    x: *const cuComplex,
    incx: c_int,
    y: *const cuComplex,
    incy: c_int,
    A: *mut cuComplex,
    lda: c_int,
);

#[cuda_custom_hook(proc_id = 1756)]
fn cublasZgeru(
    m: c_int,
    n: c_int,
    alpha: cuDoubleComplex,
    x: *const cuDoubleComplex,
    incx: c_int,
    y: *const cuDoubleComplex,
    incy: c_int,
    A: *mut cuDoubleComplex,
    lda: c_int,
);

#[cuda_custom_hook(proc_id = 1757)]
fn cublasZgerc(
    m: c_int,
    n: c_int,
    alpha: cuDoubleComplex,
    x: *const cuDoubleComplex,
    incx: c_int,
    y: *const cuDoubleComplex,
    incy: c_int,
    A: *mut cuDoubleComplex,
    lda: c_int,
);

#[cuda_custom_hook(proc_id = 1758)]
fn cublasStrmv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    A: *const f32,
    lda: c_int,
    x: *mut f32,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1759)]
fn cublasDtrmv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    A: *const f64,
    lda: c_int,
    x: *mut f64,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1760)]
fn cublasCtrmv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    A: *const cuComplex,
    lda: c_int,
    x: *mut cuComplex,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1761)]
fn cublasZtrmv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    A: *const cuDoubleComplex,
    lda: c_int,
    x: *mut cuDoubleComplex,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1762)]
fn cublasStbmv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    k: c_int,
    A: *const f32,
    lda: c_int,
    x: *mut f32,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1763)]
fn cublasDtbmv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    k: c_int,
    A: *const f64,
    lda: c_int,
    x: *mut f64,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1764)]
fn cublasCtbmv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    k: c_int,
    A: *const cuComplex,
    lda: c_int,
    x: *mut cuComplex,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1765)]
fn cublasZtbmv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    k: c_int,
    A: *const cuDoubleComplex,
    lda: c_int,
    x: *mut cuDoubleComplex,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1766)]
fn cublasStpmv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    AP: *const f32,
    x: *mut f32,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1767)]
fn cublasDtpmv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    AP: *const f64,
    x: *mut f64,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1768)]
fn cublasCtpmv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    AP: *const cuComplex,
    x: *mut cuComplex,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1769)]
fn cublasZtpmv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    AP: *const cuDoubleComplex,
    x: *mut cuDoubleComplex,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1770)]
fn cublasStrsv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    A: *const f32,
    lda: c_int,
    x: *mut f32,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1771)]
fn cublasDtrsv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    A: *const f64,
    lda: c_int,
    x: *mut f64,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1772)]
fn cublasCtrsv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    A: *const cuComplex,
    lda: c_int,
    x: *mut cuComplex,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1773)]
fn cublasZtrsv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    A: *const cuDoubleComplex,
    lda: c_int,
    x: *mut cuDoubleComplex,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1774)]
fn cublasStbsv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    k: c_int,
    A: *const f32,
    lda: c_int,
    x: *mut f32,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1775)]
fn cublasDtbsv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    k: c_int,
    A: *const f64,
    lda: c_int,
    x: *mut f64,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1776)]
fn cublasCtbsv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    k: c_int,
    A: *const cuComplex,
    lda: c_int,
    x: *mut cuComplex,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1777)]
fn cublasZtbsv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    k: c_int,
    A: *const cuDoubleComplex,
    lda: c_int,
    x: *mut cuDoubleComplex,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1778)]
fn cublasStpsv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    AP: *const f32,
    x: *mut f32,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1779)]
fn cublasDtpsv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    AP: *const f64,
    x: *mut f64,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1780)]
fn cublasCtpsv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    AP: *const cuComplex,
    x: *mut cuComplex,
    incx: c_int,
);

#[cuda_custom_hook(proc_id = 1781)]
fn cublasZtpsv(
    uplo: c_char,
    trans: c_char,
    diag: c_char,
    n: c_int,
    AP: *const cuDoubleComplex,
    x: *mut cuDoubleComplex,
    incx: c_int,
);

#[cuda_custom_hook] // local: returns a client-owned C string
fn cublasGetStatusName(status: cublasStatus_t) -> *const c_char;

#[cuda_custom_hook] // local: returns a client-owned C string
fn cublasGetStatusString(status: cublasStatus_t) -> *const c_char;

#[cuda_hook(proc_id = 1286, async_api)]
fn cublasSgemv_v2(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: c_int,
    #[device] x: *const f32,
    incx: c_int,
    #[skip] beta: *const f32,
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
            cublasSgemv_v2(
                handle, trans, m, n, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1287, async_api)]
fn cublasSgemv_v2_64(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: i64,
    #[device] x: *const f32,
    incx: i64,
    #[skip] beta: *const f32,
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
            cublasSgemv_v2_64(
                handle, trans, m, n, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1288, async_api)]
fn cublasDgemv_v2(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: c_int,
    #[device] x: *const f64,
    incx: c_int,
    #[skip] beta: *const f64,
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
            cublasDgemv_v2(
                handle, trans, m, n, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1289, async_api)]
fn cublasDgemv_v2_64(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: i64,
    #[device] x: *const f64,
    incx: i64,
    #[skip] beta: *const f64,
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
            cublasDgemv_v2_64(
                handle, trans, m, n, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1290, async_api)]
fn cublasCgemv_v2(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: c_int,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[skip] beta: *const cuComplex,
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
            cublasCgemv_v2(
                handle, trans, m, n, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1291, async_api)]
fn cublasCgemv_v2_64(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: i64,
    #[device] x: *const cuComplex,
    incx: i64,
    #[skip] beta: *const cuComplex,
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
            cublasCgemv_v2_64(
                handle, trans, m, n, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1292, async_api)]
fn cublasZgemv_v2(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[skip] beta: *const cuDoubleComplex,
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
            cublasZgemv_v2(
                handle, trans, m, n, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1293, async_api)]
fn cublasZgemv_v2_64(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: i64,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[skip] beta: *const cuDoubleComplex,
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
            cublasZgemv_v2_64(
                handle, trans, m, n, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1294, async_api)]
fn cublasSgemvBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const f32,
    #[device] Aarray: *const *const f32,
    lda: c_int,
    #[device] xarray: *const *const f32,
    incx: c_int,
    #[skip] beta: *const f32,
    #[device] yarray: *const *mut f32,
    incy: c_int,
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
            cublasSgemvBatched(
                handle, trans, m, n, alpha_arg, Aarray, lda, xarray, incx, beta_arg, yarray, incy,
                batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1295, async_api)]
fn cublasSgemvBatched_64(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const f32,
    #[device] Aarray: *const *const f32,
    lda: i64,
    #[device] xarray: *const *const f32,
    incx: i64,
    #[skip] beta: *const f32,
    #[device] yarray: *const *mut f32,
    incy: i64,
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
            cublasSgemvBatched_64(
                handle, trans, m, n, alpha_arg, Aarray, lda, xarray, incx, beta_arg, yarray, incy,
                batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1296, async_api)]
fn cublasDgemvBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const f64,
    #[device] Aarray: *const *const f64,
    lda: c_int,
    #[device] xarray: *const *const f64,
    incx: c_int,
    #[skip] beta: *const f64,
    #[device] yarray: *const *mut f64,
    incy: c_int,
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
            cublasDgemvBatched(
                handle, trans, m, n, alpha_arg, Aarray, lda, xarray, incx, beta_arg, yarray, incy,
                batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1297, async_api)]
fn cublasDgemvBatched_64(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const f64,
    #[device] Aarray: *const *const f64,
    lda: i64,
    #[device] xarray: *const *const f64,
    incx: i64,
    #[skip] beta: *const f64,
    #[device] yarray: *const *mut f64,
    incy: i64,
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
            cublasDgemvBatched_64(
                handle, trans, m, n, alpha_arg, Aarray, lda, xarray, incx, beta_arg, yarray, incy,
                batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1298, async_api)]
fn cublasCgemvBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] Aarray: *const *const cuComplex,
    lda: c_int,
    #[device] xarray: *const *const cuComplex,
    incx: c_int,
    #[skip] beta: *const cuComplex,
    #[device] yarray: *const *mut cuComplex,
    incy: c_int,
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
            cublasCgemvBatched(
                handle, trans, m, n, alpha_arg, Aarray, lda, xarray, incx, beta_arg, yarray, incy,
                batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1299, async_api)]
fn cublasCgemvBatched_64(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const cuComplex,
    #[device] Aarray: *const *const cuComplex,
    lda: i64,
    #[device] xarray: *const *const cuComplex,
    incx: i64,
    #[skip] beta: *const cuComplex,
    #[device] yarray: *const *mut cuComplex,
    incy: i64,
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
            cublasCgemvBatched_64(
                handle, trans, m, n, alpha_arg, Aarray, lda, xarray, incx, beta_arg, yarray, incy,
                batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1301, async_api)]
fn cublasZgemvBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] Aarray: *const *const cuDoubleComplex,
    lda: c_int,
    #[device] xarray: *const *const cuDoubleComplex,
    incx: c_int,
    #[skip] beta: *const cuDoubleComplex,
    #[device] yarray: *const *mut cuDoubleComplex,
    incy: c_int,
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
            cublasZgemvBatched(
                handle, trans, m, n, alpha_arg, Aarray, lda, xarray, incx, beta_arg, yarray, incy,
                batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1302, async_api)]
fn cublasZgemvBatched_64(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] Aarray: *const *const cuDoubleComplex,
    lda: i64,
    #[device] xarray: *const *const cuDoubleComplex,
    incx: i64,
    #[skip] beta: *const cuDoubleComplex,
    #[device] yarray: *const *mut cuDoubleComplex,
    incy: i64,
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
            cublasZgemvBatched_64(
                handle, trans, m, n, alpha_arg, Aarray, lda, xarray, incx, beta_arg, yarray, incy,
                batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1303, async_api)]
fn cublasSgemvStridedBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: c_int,
    strideA: c_longlong,
    #[device] x: *const f32,
    incx: c_int,
    stridex: c_longlong,
    #[skip] beta: *const f32,
    #[device] y: *mut f32,
    incy: c_int,
    stridey: c_longlong,
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
            cublasSgemvStridedBatched(
                handle, trans, m, n, alpha_arg, A, lda, strideA, x, incx, stridex, beta_arg, y,
                incy, stridey, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1304, async_api)]
fn cublasSgemvStridedBatched_64(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: i64,
    strideA: c_longlong,
    #[device] x: *const f32,
    incx: i64,
    stridex: c_longlong,
    #[skip] beta: *const f32,
    #[device] y: *mut f32,
    incy: i64,
    stridey: c_longlong,
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
            cublasSgemvStridedBatched_64(
                handle, trans, m, n, alpha_arg, A, lda, strideA, x, incx, stridex, beta_arg, y,
                incy, stridey, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1305, async_api)]
fn cublasDgemvStridedBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: c_int,
    strideA: c_longlong,
    #[device] x: *const f64,
    incx: c_int,
    stridex: c_longlong,
    #[skip] beta: *const f64,
    #[device] y: *mut f64,
    incy: c_int,
    stridey: c_longlong,
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
            cublasDgemvStridedBatched(
                handle, trans, m, n, alpha_arg, A, lda, strideA, x, incx, stridex, beta_arg, y,
                incy, stridey, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1306, async_api)]
fn cublasDgemvStridedBatched_64(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: i64,
    strideA: c_longlong,
    #[device] x: *const f64,
    incx: i64,
    stridex: c_longlong,
    #[skip] beta: *const f64,
    #[device] y: *mut f64,
    incy: i64,
    stridey: c_longlong,
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
            cublasDgemvStridedBatched_64(
                handle, trans, m, n, alpha_arg, A, lda, strideA, x, incx, stridex, beta_arg, y,
                incy, stridey, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1307, async_api)]
fn cublasCgemvStridedBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: c_int,
    strideA: c_longlong,
    #[device] x: *const cuComplex,
    incx: c_int,
    stridex: c_longlong,
    #[skip] beta: *const cuComplex,
    #[device] y: *mut cuComplex,
    incy: c_int,
    stridey: c_longlong,
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
            cublasCgemvStridedBatched(
                handle, trans, m, n, alpha_arg, A, lda, strideA, x, incx, stridex, beta_arg, y,
                incy, stridey, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1308, async_api)]
fn cublasCgemvStridedBatched_64(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: i64,
    strideA: c_longlong,
    #[device] x: *const cuComplex,
    incx: i64,
    stridex: c_longlong,
    #[skip] beta: *const cuComplex,
    #[device] y: *mut cuComplex,
    incy: i64,
    stridey: c_longlong,
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
            cublasCgemvStridedBatched_64(
                handle, trans, m, n, alpha_arg, A, lda, strideA, x, incx, stridex, beta_arg, y,
                incy, stridey, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1309, async_api)]
fn cublasZgemvStridedBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
    strideA: c_longlong,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    stridex: c_longlong,
    #[skip] beta: *const cuDoubleComplex,
    #[device] y: *mut cuDoubleComplex,
    incy: c_int,
    stridey: c_longlong,
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
            cublasZgemvStridedBatched(
                handle, trans, m, n, alpha_arg, A, lda, strideA, x, incx, stridex, beta_arg, y,
                incy, stridey, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1310, async_api)]
fn cublasZgemvStridedBatched_64(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: i64,
    strideA: c_longlong,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    stridex: c_longlong,
    #[skip] beta: *const cuDoubleComplex,
    #[device] y: *mut cuDoubleComplex,
    incy: i64,
    stridey: c_longlong,
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
            cublasZgemvStridedBatched_64(
                handle, trans, m, n, alpha_arg, A, lda, strideA, x, incx, stridex, beta_arg, y,
                incy, stridey, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1311, async_api)]
fn cublasSger_v2(
    handle: cublasHandle_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const f32,
    #[device] x: *const f32,
    incx: c_int,
    #[device] y: *const f32,
    incy: c_int,
    #[device] A: *mut f32,
    lda: c_int,
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
        let result = unsafe { cublasSger_v2(handle, m, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1312, async_api)]
fn cublasSger_v2_64(
    handle: cublasHandle_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const f32,
    #[device] x: *const f32,
    incx: i64,
    #[device] y: *const f32,
    incy: i64,
    #[device] A: *mut f32,
    lda: i64,
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
        let result = unsafe { cublasSger_v2_64(handle, m, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1314, async_api)]
fn cublasDger_v2(
    handle: cublasHandle_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const f64,
    #[device] x: *const f64,
    incx: c_int,
    #[device] y: *const f64,
    incy: c_int,
    #[device] A: *mut f64,
    lda: c_int,
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
        let result = unsafe { cublasDger_v2(handle, m, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1315, async_api)]
fn cublasDger_v2_64(
    handle: cublasHandle_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const f64,
    #[device] x: *const f64,
    incx: i64,
    #[device] y: *const f64,
    incy: i64,
    #[device] A: *mut f64,
    lda: i64,
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
        let result = unsafe { cublasDger_v2_64(handle, m, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1316, async_api)]
fn cublasCgeru_v2(
    handle: cublasHandle_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[device] y: *const cuComplex,
    incy: c_int,
    #[device] A: *mut cuComplex,
    lda: c_int,
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
        let result = unsafe { cublasCgeru_v2(handle, m, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1317, async_api)]
fn cublasCgeru_v2_64(
    handle: cublasHandle_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const cuComplex,
    #[device] x: *const cuComplex,
    incx: i64,
    #[device] y: *const cuComplex,
    incy: i64,
    #[device] A: *mut cuComplex,
    lda: i64,
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
        let result =
            unsafe { cublasCgeru_v2_64(handle, m, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1318, async_api)]
fn cublasCgerc_v2(
    handle: cublasHandle_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[device] y: *const cuComplex,
    incy: c_int,
    #[device] A: *mut cuComplex,
    lda: c_int,
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
        let result = unsafe { cublasCgerc_v2(handle, m, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1319, async_api)]
fn cublasCgerc_v2_64(
    handle: cublasHandle_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const cuComplex,
    #[device] x: *const cuComplex,
    incx: i64,
    #[device] y: *const cuComplex,
    incy: i64,
    #[device] A: *mut cuComplex,
    lda: i64,
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
        let result =
            unsafe { cublasCgerc_v2_64(handle, m, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1320, async_api)]
fn cublasZgeru_v2(
    handle: cublasHandle_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[device] y: *const cuDoubleComplex,
    incy: c_int,
    #[device] A: *mut cuDoubleComplex,
    lda: c_int,
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
        let result = unsafe { cublasZgeru_v2(handle, m, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1321, async_api)]
fn cublasZgeru_v2_64(
    handle: cublasHandle_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[device] y: *const cuDoubleComplex,
    incy: i64,
    #[device] A: *mut cuDoubleComplex,
    lda: i64,
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
        let result =
            unsafe { cublasZgeru_v2_64(handle, m, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1322, async_api)]
fn cublasZgerc_v2(
    handle: cublasHandle_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[device] y: *const cuDoubleComplex,
    incy: c_int,
    #[device] A: *mut cuDoubleComplex,
    lda: c_int,
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
        let result = unsafe { cublasZgerc_v2(handle, m, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1323, async_api)]
fn cublasZgerc_v2_64(
    handle: cublasHandle_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[device] y: *const cuDoubleComplex,
    incy: i64,
    #[device] A: *mut cuDoubleComplex,
    lda: i64,
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
        let result =
            unsafe { cublasZgerc_v2_64(handle, m, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1348, async_api)]
fn cublasSsymv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: c_int,
    #[device] x: *const f32,
    incx: c_int,
    #[skip] beta: *const f32,
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
            cublasSsymv_v2(
                handle, uplo, n, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1349, async_api)]
fn cublasSsymv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: i64,
    #[device] x: *const f32,
    incx: i64,
    #[skip] beta: *const f32,
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
            cublasSsymv_v2_64(
                handle, uplo, n, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1350, async_api)]
fn cublasDsymv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: c_int,
    #[device] x: *const f64,
    incx: c_int,
    #[skip] beta: *const f64,
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
            cublasDsymv_v2(
                handle, uplo, n, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1351, async_api)]
fn cublasDsymv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: i64,
    #[device] x: *const f64,
    incx: i64,
    #[skip] beta: *const f64,
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
            cublasDsymv_v2_64(
                handle, uplo, n, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1352, async_api)]
fn cublasCsymv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: c_int,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[skip] beta: *const cuComplex,
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
            cublasCsymv_v2(
                handle, uplo, n, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1353, async_api)]
fn cublasCsymv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: i64,
    #[device] x: *const cuComplex,
    incx: i64,
    #[skip] beta: *const cuComplex,
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
            cublasCsymv_v2_64(
                handle, uplo, n, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1354, async_api)]
fn cublasZsymv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[skip] beta: *const cuDoubleComplex,
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
            cublasZsymv_v2(
                handle, uplo, n, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1355, async_api)]
fn cublasZsymv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: i64,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[skip] beta: *const cuDoubleComplex,
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
            cublasZsymv_v2_64(
                handle, uplo, n, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1356, async_api)]
fn cublasChemv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: c_int,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[skip] beta: *const cuComplex,
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
            cublasChemv_v2(
                handle, uplo, n, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1357, async_api)]
fn cublasChemv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: i64,
    #[device] x: *const cuComplex,
    incx: i64,
    #[skip] beta: *const cuComplex,
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
            cublasChemv_v2_64(
                handle, uplo, n, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1358, async_api)]
fn cublasZhemv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[skip] beta: *const cuDoubleComplex,
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
            cublasZhemv_v2(
                handle, uplo, n, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1359, async_api)]
fn cublasZhemv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: i64,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[skip] beta: *const cuDoubleComplex,
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
            cublasZhemv_v2_64(
                handle, uplo, n, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1360, async_api)]
fn cublasSspmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const f32,
    #[device] AP: *const f32,
    #[device] x: *const f32,
    incx: c_int,
    #[skip] beta: *const f32,
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
        let result =
            unsafe { cublasSspmv_v2(handle, uplo, n, alpha_arg, AP, x, incx, beta_arg, y, incy) };
    }
}

#[cuda_hook(proc_id = 1361, async_api)]
fn cublasSspmv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const f32,
    #[device] AP: *const f32,
    #[device] x: *const f32,
    incx: i64,
    #[skip] beta: *const f32,
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
            cublasSspmv_v2_64(handle, uplo, n, alpha_arg, AP, x, incx, beta_arg, y, incy)
        };
    }
}

#[cuda_hook(proc_id = 1362, async_api)]
fn cublasDspmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const f64,
    #[device] AP: *const f64,
    #[device] x: *const f64,
    incx: c_int,
    #[skip] beta: *const f64,
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
        let result =
            unsafe { cublasDspmv_v2(handle, uplo, n, alpha_arg, AP, x, incx, beta_arg, y, incy) };
    }
}

#[cuda_hook(proc_id = 1363, async_api)]
fn cublasDspmv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const f64,
    #[device] AP: *const f64,
    #[device] x: *const f64,
    incx: i64,
    #[skip] beta: *const f64,
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
            cublasDspmv_v2_64(handle, uplo, n, alpha_arg, AP, x, incx, beta_arg, y, incy)
        };
    }
}

#[cuda_hook(proc_id = 1364, async_api)]
fn cublasChpmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] AP: *const cuComplex,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[skip] beta: *const cuComplex,
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
        let result =
            unsafe { cublasChpmv_v2(handle, uplo, n, alpha_arg, AP, x, incx, beta_arg, y, incy) };
    }
}

#[cuda_hook(proc_id = 1365, async_api)]
fn cublasChpmv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const cuComplex,
    #[device] AP: *const cuComplex,
    #[device] x: *const cuComplex,
    incx: i64,
    #[skip] beta: *const cuComplex,
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
            cublasChpmv_v2_64(handle, uplo, n, alpha_arg, AP, x, incx, beta_arg, y, incy)
        };
    }
}

#[cuda_hook(proc_id = 1366, async_api)]
fn cublasZhpmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] AP: *const cuDoubleComplex,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[skip] beta: *const cuDoubleComplex,
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
        let result =
            unsafe { cublasZhpmv_v2(handle, uplo, n, alpha_arg, AP, x, incx, beta_arg, y, incy) };
    }
}

#[cuda_hook(proc_id = 1367, async_api)]
fn cublasZhpmv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] AP: *const cuDoubleComplex,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[skip] beta: *const cuDoubleComplex,
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
            cublasZhpmv_v2_64(handle, uplo, n, alpha_arg, AP, x, incx, beta_arg, y, incy)
        };
    }
}

#[cuda_hook(proc_id = 1450, async_api)]
fn cublasSsyr_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const f32,
    #[device] x: *const f32,
    incx: c_int,
    #[device] A: *mut f32,
    lda: c_int,
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
            0.0f32
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
        let result = unsafe { cublasSsyr_v2(handle, uplo, n, alpha_arg, x, incx, A, lda) };
    }
}

#[cuda_hook(proc_id = 1451, async_api)]
fn cublasSsyr_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const f32,
    #[device] x: *const f32,
    incx: i64,
    #[device] A: *mut f32,
    lda: i64,
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
            0.0f32
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
        let result = unsafe { cublasSsyr_v2_64(handle, uplo, n, alpha_arg, x, incx, A, lda) };
    }
}

#[cuda_hook(proc_id = 1452, async_api)]
fn cublasDsyr_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const f64,
    #[device] x: *const f64,
    incx: c_int,
    #[device] A: *mut f64,
    lda: c_int,
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
            0.0f64
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
        let result = unsafe { cublasDsyr_v2(handle, uplo, n, alpha_arg, x, incx, A, lda) };
    }
}

#[cuda_hook(proc_id = 1453, async_api)]
fn cublasDsyr_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const f64,
    #[device] x: *const f64,
    incx: i64,
    #[device] A: *mut f64,
    lda: i64,
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
            0.0f64
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
        let result = unsafe { cublasDsyr_v2_64(handle, uplo, n, alpha_arg, x, incx, A, lda) };
    }
}

#[cuda_hook(proc_id = 1454, async_api)]
fn cublasCsyr_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[device] A: *mut cuComplex,
    lda: c_int,
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
        let result = unsafe { cublasCsyr_v2(handle, uplo, n, alpha_arg, x, incx, A, lda) };
    }
}

#[cuda_hook(proc_id = 1455, async_api)]
fn cublasCsyr_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const cuComplex,
    #[device] x: *const cuComplex,
    incx: i64,
    #[device] A: *mut cuComplex,
    lda: i64,
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
        let result = unsafe { cublasCsyr_v2_64(handle, uplo, n, alpha_arg, x, incx, A, lda) };
    }
}

#[cuda_hook(proc_id = 1456, async_api)]
fn cublasZsyr_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[device] A: *mut cuDoubleComplex,
    lda: c_int,
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
        let result = unsafe { cublasZsyr_v2(handle, uplo, n, alpha_arg, x, incx, A, lda) };
    }
}

#[cuda_hook(proc_id = 1457, async_api)]
fn cublasZsyr_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[device] A: *mut cuDoubleComplex,
    lda: i64,
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
        let result = unsafe { cublasZsyr_v2_64(handle, uplo, n, alpha_arg, x, incx, A, lda) };
    }
}

#[cuda_hook(proc_id = 1458, async_api)]
fn cublasCher_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const f32,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[device] A: *mut cuComplex,
    lda: c_int,
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
            0.0f32
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
        let result = unsafe { cublasCher_v2(handle, uplo, n, alpha_arg, x, incx, A, lda) };
    }
}

#[cuda_hook(proc_id = 1459, async_api)]
fn cublasCher_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const f32,
    #[device] x: *const cuComplex,
    incx: i64,
    #[device] A: *mut cuComplex,
    lda: i64,
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
            0.0f32
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
        let result = unsafe { cublasCher_v2_64(handle, uplo, n, alpha_arg, x, incx, A, lda) };
    }
}

#[cuda_hook(proc_id = 1460, async_api)]
fn cublasZher_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const f64,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[device] A: *mut cuDoubleComplex,
    lda: c_int,
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
            0.0f64
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
        let result = unsafe { cublasZher_v2(handle, uplo, n, alpha_arg, x, incx, A, lda) };
    }
}

#[cuda_hook(proc_id = 1461, async_api)]
fn cublasZher_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const f64,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[device] A: *mut cuDoubleComplex,
    lda: i64,
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
            0.0f64
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
        let result = unsafe { cublasZher_v2_64(handle, uplo, n, alpha_arg, x, incx, A, lda) };
    }
}

#[cuda_hook(proc_id = 1462, async_api)]
fn cublasSspr_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const f32,
    #[device] x: *const f32,
    incx: c_int,
    #[device] AP: *mut f32,
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
            0.0f32
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
        let result = unsafe { cublasSspr_v2(handle, uplo, n, alpha_arg, x, incx, AP) };
    }
}

#[cuda_hook(proc_id = 1463, async_api)]
fn cublasSspr_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const f32,
    #[device] x: *const f32,
    incx: i64,
    #[device] AP: *mut f32,
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
            0.0f32
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
        let result = unsafe { cublasSspr_v2_64(handle, uplo, n, alpha_arg, x, incx, AP) };
    }
}

#[cuda_hook(proc_id = 1464, async_api)]
fn cublasDspr_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const f64,
    #[device] x: *const f64,
    incx: c_int,
    #[device] AP: *mut f64,
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
            0.0f64
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
        let result = unsafe { cublasDspr_v2(handle, uplo, n, alpha_arg, x, incx, AP) };
    }
}

#[cuda_hook(proc_id = 1465, async_api)]
fn cublasDspr_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const f64,
    #[device] x: *const f64,
    incx: i64,
    #[device] AP: *mut f64,
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
            0.0f64
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
        let result = unsafe { cublasDspr_v2_64(handle, uplo, n, alpha_arg, x, incx, AP) };
    }
}

#[cuda_hook(proc_id = 1466, async_api)]
fn cublasChpr_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const f32,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[device] AP: *mut cuComplex,
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
            0.0f32
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
        let result = unsafe { cublasChpr_v2(handle, uplo, n, alpha_arg, x, incx, AP) };
    }
}

#[cuda_hook(proc_id = 1467, async_api)]
fn cublasChpr_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const f32,
    #[device] x: *const cuComplex,
    incx: i64,
    #[device] AP: *mut cuComplex,
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
            0.0f32
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
        let result = unsafe { cublasChpr_v2_64(handle, uplo, n, alpha_arg, x, incx, AP) };
    }
}

#[cuda_hook(proc_id = 1468, async_api)]
fn cublasZhpr_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const f64,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[device] AP: *mut cuDoubleComplex,
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
            0.0f64
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
        let result = unsafe { cublasZhpr_v2(handle, uplo, n, alpha_arg, x, incx, AP) };
    }
}

#[cuda_hook(proc_id = 1469, async_api)]
fn cublasZhpr_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const f64,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[device] AP: *mut cuDoubleComplex,
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
            0.0f64
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
        let result = unsafe { cublasZhpr_v2_64(handle, uplo, n, alpha_arg, x, incx, AP) };
    }
}

#[cuda_hook(proc_id = 1470, async_api)]
fn cublasSsyr2_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const f32,
    #[device] x: *const f32,
    incx: c_int,
    #[device] y: *const f32,
    incy: c_int,
    #[device] A: *mut f32,
    lda: c_int,
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
            0.0f32
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
        let result =
            unsafe { cublasSsyr2_v2(handle, uplo, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1471, async_api)]
fn cublasSsyr2_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const f32,
    #[device] x: *const f32,
    incx: i64,
    #[device] y: *const f32,
    incy: i64,
    #[device] A: *mut f32,
    lda: i64,
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
            0.0f32
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
        let result =
            unsafe { cublasSsyr2_v2_64(handle, uplo, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1472, async_api)]
fn cublasDsyr2_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const f64,
    #[device] x: *const f64,
    incx: c_int,
    #[device] y: *const f64,
    incy: c_int,
    #[device] A: *mut f64,
    lda: c_int,
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
            0.0f64
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
        let result =
            unsafe { cublasDsyr2_v2(handle, uplo, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1473, async_api)]
fn cublasDsyr2_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const f64,
    #[device] x: *const f64,
    incx: i64,
    #[device] y: *const f64,
    incy: i64,
    #[device] A: *mut f64,
    lda: i64,
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
            0.0f64
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
        let result =
            unsafe { cublasDsyr2_v2_64(handle, uplo, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1474, async_api)]
fn cublasCsyr2_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[device] y: *const cuComplex,
    incy: c_int,
    #[device] A: *mut cuComplex,
    lda: c_int,
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
        let result =
            unsafe { cublasCsyr2_v2(handle, uplo, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1475, async_api)]
fn cublasCsyr2_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const cuComplex,
    #[device] x: *const cuComplex,
    incx: i64,
    #[device] y: *const cuComplex,
    incy: i64,
    #[device] A: *mut cuComplex,
    lda: i64,
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
        let result =
            unsafe { cublasCsyr2_v2_64(handle, uplo, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1476, async_api)]
fn cublasZsyr2_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[device] y: *const cuDoubleComplex,
    incy: c_int,
    #[device] A: *mut cuDoubleComplex,
    lda: c_int,
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
        let result =
            unsafe { cublasZsyr2_v2(handle, uplo, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1477, async_api)]
fn cublasZsyr2_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[device] y: *const cuDoubleComplex,
    incy: i64,
    #[device] A: *mut cuDoubleComplex,
    lda: i64,
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
        let result =
            unsafe { cublasZsyr2_v2_64(handle, uplo, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1478, async_api)]
fn cublasCher2_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[device] y: *const cuComplex,
    incy: c_int,
    #[device] A: *mut cuComplex,
    lda: c_int,
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
        let result =
            unsafe { cublasCher2_v2(handle, uplo, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1479, async_api)]
fn cublasCher2_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const cuComplex,
    #[device] x: *const cuComplex,
    incx: i64,
    #[device] y: *const cuComplex,
    incy: i64,
    #[device] A: *mut cuComplex,
    lda: i64,
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
        let result =
            unsafe { cublasCher2_v2_64(handle, uplo, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1480, async_api)]
fn cublasZher2_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[device] y: *const cuDoubleComplex,
    incy: c_int,
    #[device] A: *mut cuDoubleComplex,
    lda: c_int,
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
        let result =
            unsafe { cublasZher2_v2(handle, uplo, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1481, async_api)]
fn cublasZher2_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[device] y: *const cuDoubleComplex,
    incy: i64,
    #[device] A: *mut cuDoubleComplex,
    lda: i64,
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
        let result =
            unsafe { cublasZher2_v2_64(handle, uplo, n, alpha_arg, x, incx, y, incy, A, lda) };
    }
}

#[cuda_hook(proc_id = 1482, async_api)]
fn cublasSspr2_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const f32,
    #[device] x: *const f32,
    incx: c_int,
    #[device] y: *const f32,
    incy: c_int,
    #[device] AP: *mut f32,
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
            0.0f32
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
        let result = unsafe { cublasSspr2_v2(handle, uplo, n, alpha_arg, x, incx, y, incy, AP) };
    }
}

#[cuda_hook(proc_id = 1483, async_api)]
fn cublasSspr2_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const f32,
    #[device] x: *const f32,
    incx: i64,
    #[device] y: *const f32,
    incy: i64,
    #[device] AP: *mut f32,
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
            0.0f32
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
        let result = unsafe { cublasSspr2_v2_64(handle, uplo, n, alpha_arg, x, incx, y, incy, AP) };
    }
}

#[cuda_hook(proc_id = 1484, async_api)]
fn cublasDspr2_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const f64,
    #[device] x: *const f64,
    incx: c_int,
    #[device] y: *const f64,
    incy: c_int,
    #[device] AP: *mut f64,
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
            0.0f64
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
        let result = unsafe { cublasDspr2_v2(handle, uplo, n, alpha_arg, x, incx, y, incy, AP) };
    }
}

#[cuda_hook(proc_id = 1485, async_api)]
fn cublasDspr2_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const f64,
    #[device] x: *const f64,
    incx: i64,
    #[device] y: *const f64,
    incy: i64,
    #[device] AP: *mut f64,
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
            0.0f64
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
        let result = unsafe { cublasDspr2_v2_64(handle, uplo, n, alpha_arg, x, incx, y, incy, AP) };
    }
}

#[cuda_hook(proc_id = 1486, async_api)]
fn cublasChpr2_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[device] y: *const cuComplex,
    incy: c_int,
    #[device] AP: *mut cuComplex,
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
        let result = unsafe { cublasChpr2_v2(handle, uplo, n, alpha_arg, x, incx, y, incy, AP) };
    }
}

#[cuda_hook(proc_id = 1487, async_api)]
fn cublasChpr2_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const cuComplex,
    #[device] x: *const cuComplex,
    incx: i64,
    #[device] y: *const cuComplex,
    incy: i64,
    #[device] AP: *mut cuComplex,
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
        let result = unsafe { cublasChpr2_v2_64(handle, uplo, n, alpha_arg, x, incx, y, incy, AP) };
    }
}

#[cuda_hook(proc_id = 1488, async_api)]
fn cublasZhpr2_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[device] y: *const cuDoubleComplex,
    incy: c_int,
    #[device] AP: *mut cuDoubleComplex,
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
        let result = unsafe { cublasZhpr2_v2(handle, uplo, n, alpha_arg, x, incx, y, incy, AP) };
    }
}

#[cuda_hook(proc_id = 1489, async_api)]
fn cublasZhpr2_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[device] y: *const cuDoubleComplex,
    incy: i64,
    #[device] AP: *mut cuDoubleComplex,
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
        let result = unsafe { cublasZhpr2_v2_64(handle, uplo, n, alpha_arg, x, incx, y, incy, AP) };
    }
}

#[cuda_hook(proc_id = 1368, async_api)]
fn cublasSsbmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: c_int,
    #[device] x: *const f32,
    incx: c_int,
    #[skip] beta: *const f32,
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
            cublasSsbmv_v2(
                handle, uplo, n, k, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1369, async_api)]
fn cublasSsbmv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    k: i64,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: i64,
    #[device] x: *const f32,
    incx: i64,
    #[skip] beta: *const f32,
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
            cublasSsbmv_v2_64(
                handle, uplo, n, k, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1370, async_api)]
fn cublasDsbmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: c_int,
    #[device] x: *const f64,
    incx: c_int,
    #[skip] beta: *const f64,
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
            cublasDsbmv_v2(
                handle, uplo, n, k, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1371, async_api)]
fn cublasDsbmv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    k: i64,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: i64,
    #[device] x: *const f64,
    incx: i64,
    #[skip] beta: *const f64,
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
            cublasDsbmv_v2_64(
                handle, uplo, n, k, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1372, async_api)]
fn cublasChbmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: c_int,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[skip] beta: *const cuComplex,
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
            cublasChbmv_v2(
                handle, uplo, n, k, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1373, async_api)]
fn cublasChbmv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    k: i64,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: i64,
    #[device] x: *const cuComplex,
    incx: i64,
    #[skip] beta: *const cuComplex,
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
            cublasChbmv_v2_64(
                handle, uplo, n, k, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1374, async_api)]
fn cublasZhbmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[skip] beta: *const cuDoubleComplex,
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
            cublasZhbmv_v2(
                handle, uplo, n, k, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1375, async_api)]
fn cublasZhbmv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: i64,
    k: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: i64,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[skip] beta: *const cuDoubleComplex,
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
            cublasZhbmv_v2_64(
                handle, uplo, n, k, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1376, async_api)]
fn cublasSgbmv_v2(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    kl: c_int,
    ku: c_int,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: c_int,
    #[device] x: *const f32,
    incx: c_int,
    #[skip] beta: *const f32,
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
            cublasSgbmv_v2(
                handle, trans, m, n, kl, ku, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1377, async_api)]
fn cublasSgbmv_v2_64(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: i64,
    n: i64,
    kl: i64,
    ku: i64,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: i64,
    #[device] x: *const f32,
    incx: i64,
    #[skip] beta: *const f32,
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
            cublasSgbmv_v2_64(
                handle, trans, m, n, kl, ku, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1378, async_api)]
fn cublasDgbmv_v2(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    kl: c_int,
    ku: c_int,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: c_int,
    #[device] x: *const f64,
    incx: c_int,
    #[skip] beta: *const f64,
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
            cublasDgbmv_v2(
                handle, trans, m, n, kl, ku, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1379, async_api)]
fn cublasDgbmv_v2_64(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: i64,
    n: i64,
    kl: i64,
    ku: i64,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: i64,
    #[device] x: *const f64,
    incx: i64,
    #[skip] beta: *const f64,
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
            cublasDgbmv_v2_64(
                handle, trans, m, n, kl, ku, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1380, async_api)]
fn cublasCgbmv_v2(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    kl: c_int,
    ku: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: c_int,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[skip] beta: *const cuComplex,
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
            cublasCgbmv_v2(
                handle, trans, m, n, kl, ku, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1381, async_api)]
fn cublasCgbmv_v2_64(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: i64,
    n: i64,
    kl: i64,
    ku: i64,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: i64,
    #[device] x: *const cuComplex,
    incx: i64,
    #[skip] beta: *const cuComplex,
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
            cublasCgbmv_v2_64(
                handle, trans, m, n, kl, ku, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1382, async_api)]
fn cublasZgbmv_v2(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    kl: c_int,
    ku: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[skip] beta: *const cuDoubleComplex,
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
            cublasZgbmv_v2(
                handle, trans, m, n, kl, ku, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1383, async_api)]
fn cublasZgbmv_v2_64(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: i64,
    n: i64,
    kl: i64,
    ku: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: i64,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[skip] beta: *const cuDoubleComplex,
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
            cublasZgbmv_v2_64(
                handle, trans, m, n, kl, ku, alpha_arg, A, lda, x, incx, beta_arg, y, incy,
            )
        };
    }
}

#[cuda_hook(proc_id = 1384, async_api)]
fn cublasStrmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    #[device] A: *const f32,
    lda: c_int,
    #[device] x: *mut f32,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1385, async_api)]
fn cublasStrmv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    #[device] A: *const f32,
    lda: i64,
    #[device] x: *mut f32,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1386, async_api)]
fn cublasDtrmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    #[device] A: *const f64,
    lda: c_int,
    #[device] x: *mut f64,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1387, async_api)]
fn cublasDtrmv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    #[device] A: *const f64,
    lda: i64,
    #[device] x: *mut f64,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1388, async_api)]
fn cublasCtrmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    #[device] A: *const cuComplex,
    lda: c_int,
    #[device] x: *mut cuComplex,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1389, async_api)]
fn cublasCtrmv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    #[device] A: *const cuComplex,
    lda: i64,
    #[device] x: *mut cuComplex,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1390, async_api)]
fn cublasZtrmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
    #[device] x: *mut cuDoubleComplex,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1391, async_api)]
fn cublasZtrmv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    #[device] A: *const cuDoubleComplex,
    lda: i64,
    #[device] x: *mut cuDoubleComplex,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1392, async_api)]
fn cublasStrsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    #[device] A: *const f32,
    lda: c_int,
    #[device] x: *mut f32,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1393, async_api)]
fn cublasStrsv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    #[device] A: *const f32,
    lda: i64,
    #[device] x: *mut f32,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1394, async_api)]
fn cublasDtrsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    #[device] A: *const f64,
    lda: c_int,
    #[device] x: *mut f64,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1395, async_api)]
fn cublasDtrsv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    #[device] A: *const f64,
    lda: i64,
    #[device] x: *mut f64,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1396, async_api)]
fn cublasCtrsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    #[device] A: *const cuComplex,
    lda: c_int,
    #[device] x: *mut cuComplex,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1397, async_api)]
fn cublasCtrsv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    #[device] A: *const cuComplex,
    lda: i64,
    #[device] x: *mut cuComplex,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1398, async_api)]
fn cublasZtrsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
    #[device] x: *mut cuDoubleComplex,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1399, async_api)]
fn cublasZtrsv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    #[device] A: *const cuDoubleComplex,
    lda: i64,
    #[device] x: *mut cuDoubleComplex,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1400, async_api)]
fn cublasStpmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    #[device] AP: *const f32,
    #[device] x: *mut f32,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1401, async_api)]
fn cublasStpmv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    #[device] AP: *const f32,
    #[device] x: *mut f32,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1402, async_api)]
fn cublasDtpmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    #[device] AP: *const f64,
    #[device] x: *mut f64,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1403, async_api)]
fn cublasDtpmv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    #[device] AP: *const f64,
    #[device] x: *mut f64,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1404, async_api)]
fn cublasCtpmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    #[device] AP: *const cuComplex,
    #[device] x: *mut cuComplex,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1405, async_api)]
fn cublasCtpmv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    #[device] AP: *const cuComplex,
    #[device] x: *mut cuComplex,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1406, async_api)]
fn cublasZtpmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    #[device] AP: *const cuDoubleComplex,
    #[device] x: *mut cuDoubleComplex,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1407, async_api)]
fn cublasZtpmv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    #[device] AP: *const cuDoubleComplex,
    #[device] x: *mut cuDoubleComplex,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1408, async_api)]
fn cublasStpsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    #[device] AP: *const f32,
    #[device] x: *mut f32,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1409, async_api)]
fn cublasStpsv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    #[device] AP: *const f32,
    #[device] x: *mut f32,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1410, async_api)]
fn cublasDtpsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    #[device] AP: *const f64,
    #[device] x: *mut f64,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1411, async_api)]
fn cublasDtpsv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    #[device] AP: *const f64,
    #[device] x: *mut f64,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1412, async_api)]
fn cublasCtpsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    #[device] AP: *const cuComplex,
    #[device] x: *mut cuComplex,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1413, async_api)]
fn cublasCtpsv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    #[device] AP: *const cuComplex,
    #[device] x: *mut cuComplex,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1414, async_api)]
fn cublasZtpsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    #[device] AP: *const cuDoubleComplex,
    #[device] x: *mut cuDoubleComplex,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1415, async_api)]
fn cublasZtpsv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    #[device] AP: *const cuDoubleComplex,
    #[device] x: *mut cuDoubleComplex,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1416, async_api)]
fn cublasStbmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    k: c_int,
    #[device] A: *const f32,
    lda: c_int,
    #[device] x: *mut f32,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1417, async_api)]
fn cublasStbmv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    k: i64,
    #[device] A: *const f32,
    lda: i64,
    #[device] x: *mut f32,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1418, async_api)]
fn cublasDtbmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    k: c_int,
    #[device] A: *const f64,
    lda: c_int,
    #[device] x: *mut f64,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1419, async_api)]
fn cublasDtbmv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    k: i64,
    #[device] A: *const f64,
    lda: i64,
    #[device] x: *mut f64,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1420, async_api)]
fn cublasCtbmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    k: c_int,
    #[device] A: *const cuComplex,
    lda: c_int,
    #[device] x: *mut cuComplex,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1421, async_api)]
fn cublasCtbmv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    k: i64,
    #[device] A: *const cuComplex,
    lda: i64,
    #[device] x: *mut cuComplex,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1422, async_api)]
fn cublasZtbmv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    k: c_int,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
    #[device] x: *mut cuDoubleComplex,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1423, async_api)]
fn cublasZtbmv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    k: i64,
    #[device] A: *const cuDoubleComplex,
    lda: i64,
    #[device] x: *mut cuDoubleComplex,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1424, async_api)]
fn cublasStbsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    k: c_int,
    #[device] A: *const f32,
    lda: c_int,
    #[device] x: *mut f32,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1425, async_api)]
fn cublasStbsv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    k: i64,
    #[device] A: *const f32,
    lda: i64,
    #[device] x: *mut f32,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1426, async_api)]
fn cublasDtbsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    k: c_int,
    #[device] A: *const f64,
    lda: c_int,
    #[device] x: *mut f64,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1427, async_api)]
fn cublasDtbsv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    k: i64,
    #[device] A: *const f64,
    lda: i64,
    #[device] x: *mut f64,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1428, async_api)]
fn cublasCtbsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    k: c_int,
    #[device] A: *const cuComplex,
    lda: c_int,
    #[device] x: *mut cuComplex,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1429, async_api)]
fn cublasCtbsv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    k: i64,
    #[device] A: *const cuComplex,
    lda: i64,
    #[device] x: *mut cuComplex,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1430, async_api)]
fn cublasZtbsv_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: c_int,
    k: c_int,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
    #[device] x: *mut cuDoubleComplex,
    incx: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1431, async_api)]
fn cublasZtbsv_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    n: i64,
    k: i64,
    #[device] A: *const cuDoubleComplex,
    lda: i64,
    #[device] x: *mut cuDoubleComplex,
    incx: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1324, async_api)]
fn cublasSgeam(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: c_int,
    #[skip] beta: *const f32,
    #[device] B: *const f32,
    ldb: c_int,
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
            cublasSgeam(
                handle, transa, transb, m, n, alpha_arg, A, lda, beta_arg, B, ldb, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1325, async_api)]
fn cublasSgeam_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: i64,
    #[skip] beta: *const f32,
    #[device] B: *const f32,
    ldb: i64,
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
            cublasSgeam_64(
                handle, transa, transb, m, n, alpha_arg, A, lda, beta_arg, B, ldb, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1326, async_api)]
fn cublasDgeam(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: c_int,
    #[skip] beta: *const f64,
    #[device] B: *const f64,
    ldb: c_int,
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
            cublasDgeam(
                handle, transa, transb, m, n, alpha_arg, A, lda, beta_arg, B, ldb, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1327, async_api)]
fn cublasDgeam_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: i64,
    #[skip] beta: *const f64,
    #[device] B: *const f64,
    ldb: i64,
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
            cublasDgeam_64(
                handle, transa, transb, m, n, alpha_arg, A, lda, beta_arg, B, ldb, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1328, async_api)]
fn cublasCgeam(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: c_int,
    #[skip] beta: *const cuComplex,
    #[device] B: *const cuComplex,
    ldb: c_int,
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
            cublasCgeam(
                handle, transa, transb, m, n, alpha_arg, A, lda, beta_arg, B, ldb, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1329, async_api)]
fn cublasCgeam_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: i64,
    #[skip] beta: *const cuComplex,
    #[device] B: *const cuComplex,
    ldb: i64,
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
            cublasCgeam_64(
                handle, transa, transb, m, n, alpha_arg, A, lda, beta_arg, B, ldb, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1330, async_api)]
fn cublasZgeam(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
    #[skip] beta: *const cuDoubleComplex,
    #[device] B: *const cuDoubleComplex,
    ldb: c_int,
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
            cublasZgeam(
                handle, transa, transb, m, n, alpha_arg, A, lda, beta_arg, B, ldb, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1331, async_api)]
fn cublasZgeam_64(
    handle: cublasHandle_t,
    transa: cublasOperation_t,
    transb: cublasOperation_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: i64,
    #[skip] beta: *const cuDoubleComplex,
    #[device] B: *const cuDoubleComplex,
    ldb: i64,
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
            cublasZgeam_64(
                handle, transa, transb, m, n, alpha_arg, A, lda, beta_arg, B, ldb, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1332, async_api)]
fn cublasSdgmm(
    handle: cublasHandle_t,
    mode: cublasSideMode_t,
    m: c_int,
    n: c_int,
    #[device] A: *const f32,
    lda: c_int,
    #[device] x: *const f32,
    incx: c_int,
    #[device] C: *mut f32,
    ldc: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1333, async_api)]
fn cublasSdgmm_64(
    handle: cublasHandle_t,
    mode: cublasSideMode_t,
    m: i64,
    n: i64,
    #[device] A: *const f32,
    lda: i64,
    #[device] x: *const f32,
    incx: i64,
    #[device] C: *mut f32,
    ldc: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1334, async_api)]
fn cublasDdgmm(
    handle: cublasHandle_t,
    mode: cublasSideMode_t,
    m: c_int,
    n: c_int,
    #[device] A: *const f64,
    lda: c_int,
    #[device] x: *const f64,
    incx: c_int,
    #[device] C: *mut f64,
    ldc: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1335, async_api)]
fn cublasDdgmm_64(
    handle: cublasHandle_t,
    mode: cublasSideMode_t,
    m: i64,
    n: i64,
    #[device] A: *const f64,
    lda: i64,
    #[device] x: *const f64,
    incx: i64,
    #[device] C: *mut f64,
    ldc: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1336, async_api)]
fn cublasCdgmm(
    handle: cublasHandle_t,
    mode: cublasSideMode_t,
    m: c_int,
    n: c_int,
    #[device] A: *const cuComplex,
    lda: c_int,
    #[device] x: *const cuComplex,
    incx: c_int,
    #[device] C: *mut cuComplex,
    ldc: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1337, async_api)]
fn cublasCdgmm_64(
    handle: cublasHandle_t,
    mode: cublasSideMode_t,
    m: i64,
    n: i64,
    #[device] A: *const cuComplex,
    lda: i64,
    #[device] x: *const cuComplex,
    incx: i64,
    #[device] C: *mut cuComplex,
    ldc: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1338, async_api)]
fn cublasZdgmm(
    handle: cublasHandle_t,
    mode: cublasSideMode_t,
    m: c_int,
    n: c_int,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
    #[device] x: *const cuDoubleComplex,
    incx: c_int,
    #[device] C: *mut cuDoubleComplex,
    ldc: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1339, async_api)]
fn cublasZdgmm_64(
    handle: cublasHandle_t,
    mode: cublasSideMode_t,
    m: i64,
    n: i64,
    #[device] A: *const cuDoubleComplex,
    lda: i64,
    #[device] x: *const cuDoubleComplex,
    incx: i64,
    #[device] C: *mut cuDoubleComplex,
    ldc: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1340, async_api)]
fn cublasStpttr(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[device] AP: *const f32,
    #[device] A: *mut f32,
    lda: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1341, async_api)]
fn cublasDtpttr(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[device] AP: *const f64,
    #[device] A: *mut f64,
    lda: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1342, async_api)]
fn cublasCtpttr(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[device] AP: *const cuComplex,
    #[device] A: *mut cuComplex,
    lda: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1343, async_api)]
fn cublasZtpttr(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[device] AP: *const cuDoubleComplex,
    #[device] A: *mut cuDoubleComplex,
    lda: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1344, async_api)]
fn cublasStrttp(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[device] A: *const f32,
    lda: c_int,
    #[device] AP: *mut f32,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1345, async_api)]
fn cublasDtrttp(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[device] A: *const f64,
    lda: c_int,
    #[device] AP: *mut f64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1346, async_api)]
fn cublasCtrttp(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[device] A: *const cuComplex,
    lda: c_int,
    #[device] AP: *mut cuComplex,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1347, async_api)]
fn cublasZtrttp(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    n: c_int,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
    #[device] AP: *mut cuDoubleComplex,
) -> cublasStatus_t;

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

#[cuda_hook(proc_id = 1554, async_api)]
fn cublasSsymm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: c_int,
    n: c_int,
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
            cublasSsymm_v2(
                handle, side, uplo, m, n, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1555, async_api)]
fn cublasSsymm_v2_64(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: i64,
    n: i64,
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
            cublasSsymm_v2_64(
                handle, side, uplo, m, n, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1556, async_api)]
fn cublasDsymm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: c_int,
    n: c_int,
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
            cublasDsymm_v2(
                handle, side, uplo, m, n, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1557, async_api)]
fn cublasDsymm_v2_64(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: i64,
    n: i64,
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
            cublasDsymm_v2_64(
                handle, side, uplo, m, n, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1558, async_api)]
fn cublasCsymm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: c_int,
    n: c_int,
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
            cublasCsymm_v2(
                handle, side, uplo, m, n, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1559, async_api)]
fn cublasCsymm_v2_64(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: i64,
    n: i64,
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
            cublasCsymm_v2_64(
                handle, side, uplo, m, n, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1560, async_api)]
fn cublasZsymm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: c_int,
    n: c_int,
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
            cublasZsymm_v2(
                handle, side, uplo, m, n, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1561, async_api)]
fn cublasZsymm_v2_64(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: i64,
    n: i64,
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
            cublasZsymm_v2_64(
                handle, side, uplo, m, n, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1562, async_api)]
fn cublasChemm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: c_int,
    n: c_int,
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
            cublasChemm_v2(
                handle, side, uplo, m, n, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1563, async_api)]
fn cublasChemm_v2_64(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: i64,
    n: i64,
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
            cublasChemm_v2_64(
                handle, side, uplo, m, n, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1564, async_api)]
fn cublasZhemm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: c_int,
    n: c_int,
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
            cublasZhemm_v2(
                handle, side, uplo, m, n, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1565, async_api)]
fn cublasZhemm_v2_64(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    m: i64,
    n: i64,
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
            cublasZhemm_v2_64(
                handle, side, uplo, m, n, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1566, async_api)]
fn cublasSsyrk_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: c_int,
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
            0.0f32
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0f32
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
            cublasSsyrk_v2(
                handle, uplo, trans, n, k, alpha_arg, A, lda, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1567, async_api)]
fn cublasSsyrk_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: i64,
    k: i64,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: i64,
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
            0.0f32
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0f32
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
            cublasSsyrk_v2_64(
                handle, uplo, trans, n, k, alpha_arg, A, lda, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1568, async_api)]
fn cublasDsyrk_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: c_int,
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
            0.0f64
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0f64
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
            cublasDsyrk_v2(
                handle, uplo, trans, n, k, alpha_arg, A, lda, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1569, async_api)]
fn cublasDsyrk_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: i64,
    k: i64,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: i64,
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
            0.0f64
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0f64
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
            cublasDsyrk_v2_64(
                handle, uplo, trans, n, k, alpha_arg, A, lda, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1570, async_api)]
fn cublasCsyrk_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: c_int,
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
            cublasCsyrk_v2(
                handle, uplo, trans, n, k, alpha_arg, A, lda, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1571, async_api)]
fn cublasCsyrk_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: i64,
    k: i64,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: i64,
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
            cublasCsyrk_v2_64(
                handle, uplo, trans, n, k, alpha_arg, A, lda, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1572, async_api)]
fn cublasZsyrk_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
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
            cublasZsyrk_v2(
                handle, uplo, trans, n, k, alpha_arg, A, lda, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1573, async_api)]
fn cublasZsyrk_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: i64,
    k: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: i64,
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
            cublasZsyrk_v2_64(
                handle, uplo, trans, n, k, alpha_arg, A, lda, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1574, async_api)]
fn cublasCherk_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const f32,
    #[device] A: *const cuComplex,
    lda: c_int,
    #[skip] beta: *const f32,
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
            0.0f32
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0f32
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
            cublasCherk_v2(
                handle, uplo, trans, n, k, alpha_arg, A, lda, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1575, async_api)]
fn cublasCherk_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: i64,
    k: i64,
    #[skip] alpha: *const f32,
    #[device] A: *const cuComplex,
    lda: i64,
    #[skip] beta: *const f32,
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
            0.0f32
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0f32
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
            cublasCherk_v2_64(
                handle, uplo, trans, n, k, alpha_arg, A, lda, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1576, async_api)]
fn cublasZherk_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const f64,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
    #[skip] beta: *const f64,
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
            0.0f64
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0f64
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
            cublasZherk_v2(
                handle, uplo, trans, n, k, alpha_arg, A, lda, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1577, async_api)]
fn cublasZherk_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: i64,
    k: i64,
    #[skip] alpha: *const f64,
    #[device] A: *const cuDoubleComplex,
    lda: i64,
    #[skip] beta: *const f64,
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
            0.0f64
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0f64
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
            cublasZherk_v2_64(
                handle, uplo, trans, n, k, alpha_arg, A, lda, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1578, async_api)]
fn cublasSsyr2k_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
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
            0.0f32
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0f32
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
            cublasSsyr2k_v2(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1579, async_api)]
fn cublasSsyr2k_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
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
            0.0f32
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0f32
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
            cublasSsyr2k_v2_64(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1580, async_api)]
fn cublasDsyr2k_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
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
            0.0f64
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0f64
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
            cublasDsyr2k_v2(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1581, async_api)]
fn cublasDsyr2k_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
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
            0.0f64
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0f64
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
            cublasDsyr2k_v2_64(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1582, async_api)]
fn cublasCsyr2k_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
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
            cublasCsyr2k_v2(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1583, async_api)]
fn cublasCsyr2k_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
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
            cublasCsyr2k_v2_64(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1584, async_api)]
fn cublasZsyr2k_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
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
            cublasZsyr2k_v2(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1585, async_api)]
fn cublasZsyr2k_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
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
            cublasZsyr2k_v2_64(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1586, async_api)]
fn cublasCher2k_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: c_int,
    #[device] B: *const cuComplex,
    ldb: c_int,
    #[skip] beta: *const f32,
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
            0.0f32
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
        let mut beta_value = 0.0f32;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuComplex
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
            cublasCher2k_v2(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1587, async_api)]
fn cublasCher2k_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: i64,
    k: i64,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: i64,
    #[device] B: *const cuComplex,
    ldb: i64,
    #[skip] beta: *const f32,
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
            0.0f32
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
        let mut beta_value = 0.0f32;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuComplex
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
            cublasCher2k_v2_64(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1588, async_api)]
fn cublasZher2k_v2(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
    #[device] B: *const cuDoubleComplex,
    ldb: c_int,
    #[skip] beta: *const f64,
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
            0.0f64
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
        let mut beta_value = 0.0f64;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuDoubleComplex
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
            cublasZher2k_v2(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1589, async_api)]
fn cublasZher2k_v2_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: i64,
    k: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: i64,
    #[device] B: *const cuDoubleComplex,
    ldb: i64,
    #[skip] beta: *const f64,
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
            0.0f64
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
        let mut beta_value = 0.0f64;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuDoubleComplex
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
            cublasZher2k_v2_64(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1590, async_api)]
fn cublasSsyrkx(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
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
            0.0f32
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0f32
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
            cublasSsyrkx(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1591, async_api)]
fn cublasSsyrkx_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
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
            0.0f32
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0f32
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
            cublasSsyrkx_64(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1592, async_api)]
fn cublasDsyrkx(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
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
            0.0f64
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0f64
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
            cublasDsyrkx(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1593, async_api)]
fn cublasDsyrkx_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
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
            0.0f64
        } else {
            assert!(!alpha.is_null());
            unsafe { *alpha }
        };
        let beta_value = if device_pointer_mode {
            0.0f64
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
            cublasDsyrkx_64(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1594, async_api)]
fn cublasCsyrkx(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
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
            cublasCsyrkx(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1595, async_api)]
fn cublasCsyrkx_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
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
            cublasCsyrkx_64(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1596, async_api)]
fn cublasZsyrkx(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
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
            cublasZsyrkx(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1597, async_api)]
fn cublasZsyrkx_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
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
            cublasZsyrkx_64(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1598, async_api)]
fn cublasCherkx(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: c_int,
    #[device] B: *const cuComplex,
    ldb: c_int,
    #[skip] beta: *const f32,
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
            0.0f32
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
        let mut beta_value = 0.0f32;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuComplex
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
            cublasCherkx(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1599, async_api)]
fn cublasCherkx_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: i64,
    k: i64,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: i64,
    #[device] B: *const cuComplex,
    ldb: i64,
    #[skip] beta: *const f32,
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
            0.0f32
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
        let mut beta_value = 0.0f32;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuComplex
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
            cublasCherkx_64(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1600, async_api)]
fn cublasZherkx(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: c_int,
    k: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
    #[device] B: *const cuDoubleComplex,
    ldb: c_int,
    #[skip] beta: *const f64,
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
            0.0f64
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
        let mut beta_value = 0.0f64;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuDoubleComplex
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
            cublasZherkx(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1601, async_api)]
fn cublasZherkx_64(
    handle: cublasHandle_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    n: i64,
    k: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: i64,
    #[device] B: *const cuDoubleComplex,
    ldb: i64,
    #[skip] beta: *const f64,
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
            0.0f64
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
        let mut beta_value = 0.0f64;
        beta_value.recv(channel_receiver).unwrap();
        let mut beta_addr = 0usize;
        beta_addr.recv(channel_receiver).unwrap();
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const cuDoubleComplex
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
            cublasZherkx_64(
                handle, uplo, trans, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1602, async_api)]
fn cublasStrmm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: c_int,
    #[device] B: *const f32,
    ldb: c_int,
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
            0.0f32
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
        let result = unsafe {
            cublasStrmm_v2(
                handle, side, uplo, trans, diag, m, n, alpha_arg, A, lda, B, ldb, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1603, async_api)]
fn cublasStrmm_v2_64(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: i64,
    #[device] B: *const f32,
    ldb: i64,
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
            0.0f32
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
        let result = unsafe {
            cublasStrmm_v2_64(
                handle, side, uplo, trans, diag, m, n, alpha_arg, A, lda, B, ldb, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1604, async_api)]
fn cublasDtrmm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: c_int,
    #[device] B: *const f64,
    ldb: c_int,
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
            0.0f64
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
        let result = unsafe {
            cublasDtrmm_v2(
                handle, side, uplo, trans, diag, m, n, alpha_arg, A, lda, B, ldb, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1605, async_api)]
fn cublasDtrmm_v2_64(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: i64,
    #[device] B: *const f64,
    ldb: i64,
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
            0.0f64
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
        let result = unsafe {
            cublasDtrmm_v2_64(
                handle, side, uplo, trans, diag, m, n, alpha_arg, A, lda, B, ldb, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1606, async_api)]
fn cublasCtrmm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: c_int,
    #[device] B: *const cuComplex,
    ldb: c_int,
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
        let result = unsafe {
            cublasCtrmm_v2(
                handle, side, uplo, trans, diag, m, n, alpha_arg, A, lda, B, ldb, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1607, async_api)]
fn cublasCtrmm_v2_64(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: i64,
    #[device] B: *const cuComplex,
    ldb: i64,
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
        let result = unsafe {
            cublasCtrmm_v2_64(
                handle, side, uplo, trans, diag, m, n, alpha_arg, A, lda, B, ldb, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1608, async_api)]
fn cublasZtrmm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
    #[device] B: *const cuDoubleComplex,
    ldb: c_int,
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
        let result = unsafe {
            cublasZtrmm_v2(
                handle, side, uplo, trans, diag, m, n, alpha_arg, A, lda, B, ldb, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1609, async_api)]
fn cublasZtrmm_v2_64(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: i64,
    #[device] B: *const cuDoubleComplex,
    ldb: i64,
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
        let result = unsafe {
            cublasZtrmm_v2_64(
                handle, side, uplo, trans, diag, m, n, alpha_arg, A, lda, B, ldb, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1610, async_api)]
fn cublasStrsm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: c_int,
    #[device] B: *mut f32,
    ldb: c_int,
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
            0.0f32
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
        let result = unsafe {
            cublasStrsm_v2(
                handle, side, uplo, trans, diag, m, n, alpha_arg, A, lda, B, ldb,
            )
        };
    }
}

#[cuda_hook(proc_id = 1611, async_api)]
fn cublasStrsm_v2_64(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const f32,
    #[device] A: *const f32,
    lda: i64,
    #[device] B: *mut f32,
    ldb: i64,
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
            0.0f32
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
        let result = unsafe {
            cublasStrsm_v2_64(
                handle, side, uplo, trans, diag, m, n, alpha_arg, A, lda, B, ldb,
            )
        };
    }
}

#[cuda_hook(proc_id = 1612, async_api)]
fn cublasDtrsm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: c_int,
    #[device] B: *mut f64,
    ldb: c_int,
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
            0.0f64
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
        let result = unsafe {
            cublasDtrsm_v2(
                handle, side, uplo, trans, diag, m, n, alpha_arg, A, lda, B, ldb,
            )
        };
    }
}

#[cuda_hook(proc_id = 1613, async_api)]
fn cublasDtrsm_v2_64(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const f64,
    #[device] A: *const f64,
    lda: i64,
    #[device] B: *mut f64,
    ldb: i64,
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
            0.0f64
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
        let result = unsafe {
            cublasDtrsm_v2_64(
                handle, side, uplo, trans, diag, m, n, alpha_arg, A, lda, B, ldb,
            )
        };
    }
}

#[cuda_hook(proc_id = 1614, async_api)]
fn cublasCtrsm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: c_int,
    #[device] B: *mut cuComplex,
    ldb: c_int,
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
        let result = unsafe {
            cublasCtrsm_v2(
                handle, side, uplo, trans, diag, m, n, alpha_arg, A, lda, B, ldb,
            )
        };
    }
}

#[cuda_hook(proc_id = 1615, async_api)]
fn cublasCtrsm_v2_64(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const cuComplex,
    #[device] A: *const cuComplex,
    lda: i64,
    #[device] B: *mut cuComplex,
    ldb: i64,
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
        let result = unsafe {
            cublasCtrsm_v2_64(
                handle, side, uplo, trans, diag, m, n, alpha_arg, A, lda, B, ldb,
            )
        };
    }
}

#[cuda_hook(proc_id = 1616, async_api)]
fn cublasZtrsm_v2(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: c_int,
    #[device] B: *mut cuDoubleComplex,
    ldb: c_int,
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
        let result = unsafe {
            cublasZtrsm_v2(
                handle, side, uplo, trans, diag, m, n, alpha_arg, A, lda, B, ldb,
            )
        };
    }
}

#[cuda_hook(proc_id = 1617, async_api)]
fn cublasZtrsm_v2_64(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] A: *const cuDoubleComplex,
    lda: i64,
    #[device] B: *mut cuDoubleComplex,
    ldb: i64,
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
        let result = unsafe {
            cublasZtrsm_v2_64(
                handle, side, uplo, trans, diag, m, n, alpha_arg, A, lda, B, ldb,
            )
        };
    }
}

#[cuda_hook(proc_id = 1618, async_api)]
fn cublasStrsmBatched(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const f32,
    #[device] Aarray: *const *const f32,
    lda: c_int,
    #[device] Barray: *const *mut f32,
    ldb: c_int,
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
            0.0f32
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
        let result = unsafe {
            cublasStrsmBatched(
                handle, side, uplo, trans, diag, m, n, alpha_arg, Aarray, lda, Barray, ldb,
                batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1619, async_api)]
fn cublasStrsmBatched_64(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const f32,
    #[device] Aarray: *const *const f32,
    lda: i64,
    #[device] Barray: *const *mut f32,
    ldb: i64,
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
            0.0f32
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
        let result = unsafe {
            cublasStrsmBatched_64(
                handle, side, uplo, trans, diag, m, n, alpha_arg, Aarray, lda, Barray, ldb,
                batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1620, async_api)]
fn cublasDtrsmBatched(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const f64,
    #[device] Aarray: *const *const f64,
    lda: c_int,
    #[device] Barray: *const *mut f64,
    ldb: c_int,
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
            0.0f64
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
        let result = unsafe {
            cublasDtrsmBatched(
                handle, side, uplo, trans, diag, m, n, alpha_arg, Aarray, lda, Barray, ldb,
                batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1621, async_api)]
fn cublasDtrsmBatched_64(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const f64,
    #[device] Aarray: *const *const f64,
    lda: i64,
    #[device] Barray: *const *mut f64,
    ldb: i64,
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
            0.0f64
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
        let result = unsafe {
            cublasDtrsmBatched_64(
                handle, side, uplo, trans, diag, m, n, alpha_arg, Aarray, lda, Barray, ldb,
                batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1622, async_api)]
fn cublasCtrsmBatched(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const cuComplex,
    #[device] Aarray: *const *const cuComplex,
    lda: c_int,
    #[device] Barray: *const *mut cuComplex,
    ldb: c_int,
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
        let result = unsafe {
            cublasCtrsmBatched(
                handle, side, uplo, trans, diag, m, n, alpha_arg, Aarray, lda, Barray, ldb,
                batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1623, async_api)]
fn cublasCtrsmBatched_64(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const cuComplex,
    #[device] Aarray: *const *const cuComplex,
    lda: i64,
    #[device] Barray: *const *mut cuComplex,
    ldb: i64,
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
        let result = unsafe {
            cublasCtrsmBatched_64(
                handle, side, uplo, trans, diag, m, n, alpha_arg, Aarray, lda, Barray, ldb,
                batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1624, async_api)]
fn cublasZtrsmBatched(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: c_int,
    n: c_int,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] Aarray: *const *const cuDoubleComplex,
    lda: c_int,
    #[device] Barray: *const *mut cuDoubleComplex,
    ldb: c_int,
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
        let result = unsafe {
            cublasZtrsmBatched(
                handle, side, uplo, trans, diag, m, n, alpha_arg, Aarray, lda, Barray, ldb,
                batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1625, async_api)]
fn cublasZtrsmBatched_64(
    handle: cublasHandle_t,
    side: cublasSideMode_t,
    uplo: cublasFillMode_t,
    trans: cublasOperation_t,
    diag: cublasDiagType_t,
    m: i64,
    n: i64,
    #[skip] alpha: *const cuDoubleComplex,
    #[device] Aarray: *const *const cuDoubleComplex,
    lda: i64,
    #[device] Barray: *const *mut cuDoubleComplex,
    ldb: i64,
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
        let result = unsafe {
            cublasZtrsmBatched_64(
                handle, side, uplo, trans, diag, m, n, alpha_arg, Aarray, lda, Barray, ldb,
                batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1626, async_api)]
fn cublasSmatinvBatched(
    handle: cublasHandle_t,
    n: c_int,
    #[device] A: *const *const f32,
    lda: c_int,
    #[device] Ainv: *const *mut f32,
    lda_inv: c_int,
    #[device] info: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1627, async_api)]
fn cublasDmatinvBatched(
    handle: cublasHandle_t,
    n: c_int,
    #[device] A: *const *const f64,
    lda: c_int,
    #[device] Ainv: *const *mut f64,
    lda_inv: c_int,
    #[device] info: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1628, async_api)]
fn cublasCmatinvBatched(
    handle: cublasHandle_t,
    n: c_int,
    #[device] A: *const *const cuComplex,
    lda: c_int,
    #[device] Ainv: *const *mut cuComplex,
    lda_inv: c_int,
    #[device] info: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1629, async_api)]
fn cublasZmatinvBatched(
    handle: cublasHandle_t,
    n: c_int,
    #[device] A: *const *const cuDoubleComplex,
    lda: c_int,
    #[device] Ainv: *const *mut cuDoubleComplex,
    lda_inv: c_int,
    #[device] info: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1630)]
fn cublasSgeqrfBatched(
    handle: cublasHandle_t,
    m: c_int,
    n: c_int,
    #[device] Aarray: *const *mut f32,
    lda: c_int,
    #[device] TauArray: *const *mut f32,
    info: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1631)]
fn cublasDgeqrfBatched(
    handle: cublasHandle_t,
    m: c_int,
    n: c_int,
    #[device] Aarray: *const *mut f64,
    lda: c_int,
    #[device] TauArray: *const *mut f64,
    info: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1632)]
fn cublasCgeqrfBatched(
    handle: cublasHandle_t,
    m: c_int,
    n: c_int,
    #[device] Aarray: *const *mut cuComplex,
    lda: c_int,
    #[device] TauArray: *const *mut cuComplex,
    info: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1633)]
fn cublasZgeqrfBatched(
    handle: cublasHandle_t,
    m: c_int,
    n: c_int,
    #[device] Aarray: *const *mut cuDoubleComplex,
    lda: c_int,
    #[device] TauArray: *const *mut cuDoubleComplex,
    info: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1634)]
fn cublasSgelsBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    nrhs: c_int,
    #[device] Aarray: *const *mut f32,
    lda: c_int,
    #[device] Carray: *const *mut f32,
    ldc: c_int,
    info: *mut c_int,
    #[device] devInfoArray: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1635)]
fn cublasDgelsBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    nrhs: c_int,
    #[device] Aarray: *const *mut f64,
    lda: c_int,
    #[device] Carray: *const *mut f64,
    ldc: c_int,
    info: *mut c_int,
    #[device] devInfoArray: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1636)]
fn cublasCgelsBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    nrhs: c_int,
    #[device] Aarray: *const *mut cuComplex,
    lda: c_int,
    #[device] Carray: *const *mut cuComplex,
    ldc: c_int,
    info: *mut c_int,
    #[device] devInfoArray: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1637)]
fn cublasZgelsBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    m: c_int,
    n: c_int,
    nrhs: c_int,
    #[device] Aarray: *const *mut cuDoubleComplex,
    lda: c_int,
    #[device] Carray: *const *mut cuDoubleComplex,
    ldc: c_int,
    info: *mut c_int,
    #[device] devInfoArray: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1638, async_api)]
fn cublasSgetrfBatched(
    handle: cublasHandle_t,
    n: c_int,
    #[device] A: *const *mut f32,
    lda: c_int,
    #[device] P: *mut c_int,
    #[device] info: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1639, async_api)]
fn cublasDgetrfBatched(
    handle: cublasHandle_t,
    n: c_int,
    #[device] A: *const *mut f64,
    lda: c_int,
    #[device] P: *mut c_int,
    #[device] info: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1640, async_api)]
fn cublasCgetrfBatched(
    handle: cublasHandle_t,
    n: c_int,
    #[device] A: *const *mut cuComplex,
    lda: c_int,
    #[device] P: *mut c_int,
    #[device] info: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1641, async_api)]
fn cublasZgetrfBatched(
    handle: cublasHandle_t,
    n: c_int,
    #[device] A: *const *mut cuDoubleComplex,
    lda: c_int,
    #[device] P: *mut c_int,
    #[device] info: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1642, async_api)]
fn cublasSgetriBatched(
    handle: cublasHandle_t,
    n: c_int,
    #[device] A: *const *const f32,
    lda: c_int,
    #[device] P: *const c_int,
    #[device] C: *const *mut f32,
    ldc: c_int,
    #[device] info: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1643, async_api)]
fn cublasDgetriBatched(
    handle: cublasHandle_t,
    n: c_int,
    #[device] A: *const *const f64,
    lda: c_int,
    #[device] P: *const c_int,
    #[device] C: *const *mut f64,
    ldc: c_int,
    #[device] info: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1644, async_api)]
fn cublasCgetriBatched(
    handle: cublasHandle_t,
    n: c_int,
    #[device] A: *const *const cuComplex,
    lda: c_int,
    #[device] P: *const c_int,
    #[device] C: *const *mut cuComplex,
    ldc: c_int,
    #[device] info: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1645, async_api)]
fn cublasZgetriBatched(
    handle: cublasHandle_t,
    n: c_int,
    #[device] A: *const *const cuDoubleComplex,
    lda: c_int,
    #[device] P: *const c_int,
    #[device] C: *const *mut cuDoubleComplex,
    ldc: c_int,
    #[device] info: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1646)]
fn cublasSgetrsBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    n: c_int,
    nrhs: c_int,
    #[device] Aarray: *const *const f32,
    lda: c_int,
    #[device] devIpiv: *const c_int,
    #[device] Barray: *const *mut f32,
    ldb: c_int,
    info: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1647)]
fn cublasDgetrsBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    n: c_int,
    nrhs: c_int,
    #[device] Aarray: *const *const f64,
    lda: c_int,
    #[device] devIpiv: *const c_int,
    #[device] Barray: *const *mut f64,
    ldb: c_int,
    info: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1648)]
fn cublasCgetrsBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    n: c_int,
    nrhs: c_int,
    #[device] Aarray: *const *const cuComplex,
    lda: c_int,
    #[device] devIpiv: *const c_int,
    #[device] Barray: *const *mut cuComplex,
    ldb: c_int,
    info: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1649)]
fn cublasZgetrsBatched(
    handle: cublasHandle_t,
    trans: cublasOperation_t,
    n: c_int,
    nrhs: c_int,
    #[device] Aarray: *const *const cuDoubleComplex,
    lda: c_int,
    #[device] devIpiv: *const c_int,
    #[device] Barray: *const *mut cuDoubleComplex,
    ldb: c_int,
    info: *mut c_int,
    batchSize: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1276, async_api)]
fn cublasCgemm3m(
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
            cublasCgemm3m(
                handle, transa, transb, m, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1277, async_api)]
fn cublasCgemm3m_64(
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
            cublasCgemm3m_64(
                handle, transa, transb, m, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1278, async_api)]
fn cublasZgemm3m(
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
            cublasZgemm3m(
                handle, transa, transb, m, n, k, alpha_arg, A, lda, B, ldb, beta_arg, C, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1279, async_api)]
fn cublasZgemm3m_64(
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
            cublasZgemm3m_64(
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

#[cuda_hook(proc_id = 1280, async_api)]
fn cublasCgemm3mStridedBatched(
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
            cublasCgemm3mStridedBatched(
                handle, transa, transb, m, n, k, alpha_arg, A, lda, strideA, B, ldb, strideB,
                beta_arg, C, ldc, strideC, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1281, async_api)]
fn cublasCgemm3mStridedBatched_64(
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
            cublasCgemm3mStridedBatched_64(
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

#[cuda_hook(proc_id = 1282, async_api)]
fn cublasCgemm3mBatched(
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
            cublasCgemm3mBatched(
                handle, transa, transb, m, n, k, alpha_arg, Aarray, lda, Barray, ldb, beta_arg,
                Carray, ldc, batchCount,
            )
        };
    }
}

#[cuda_hook(proc_id = 1283, async_api)]
fn cublasCgemm3mBatched_64(
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
            cublasCgemm3mBatched_64(
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

#[cuda_hook(proc_id = 1284, async_api)]
fn cublasCgemm3mEx(
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
            cublasCgemm3mEx(
                handle, transa, transb, m, n, k, alpha_arg, A, Atype, lda, B, Btype, ldb, beta_arg,
                C, Ctype, ldc,
            )
        };
    }
}

#[cuda_hook(proc_id = 1285, async_api)]
fn cublasCgemm3mEx_64(
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
            cublasCgemm3mEx_64(
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
    #[skip] A: *const c_void,
    lda: c_int,
    #[device] B: *mut c_void,
    ldb: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let rows_usize = if rows > 0 {
            usize::try_from(rows).unwrap()
        } else {
            0
        };
        let cols_usize = if cols > 0 {
            usize::try_from(cols).unwrap()
        } else {
            0
        };
        let elem_size = if elemSize > 0 {
            usize::try_from(elemSize).unwrap()
        } else {
            0
        };
        let lda_isize = isize::try_from(lda).unwrap();
        let elem_size_isize = isize::try_from(elemSize).unwrap();
        let mut packed = vec![0u8; rows_usize * cols_usize * elem_size];
        if !packed.is_empty() {
            assert!(!A.is_null());
            assert!(lda >= rows);
            for col in 0..cols_usize {
                for row in 0..rows_usize {
                    let src = unsafe {
                        A.cast::<u8>().offset(
                            (isize::try_from(col).unwrap() * lda_isize
                                + isize::try_from(row).unwrap())
                                * elem_size_isize,
                        )
                    };
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src,
                            packed
                                .as_mut_ptr()
                                .add((col * rows_usize + row) * elem_size),
                            elem_size,
                        );
                    }
                }
            }
        }
    }
    'client_extra_send: {
        send_slice(&packed, channel_sender).unwrap();
    }
    'server_extra_recv: {
        let packed = recv_slice::<u8, _>(channel_receiver).unwrap();
        let a_arg = if packed.is_empty() {
            std::ptr::null()
        } else {
            packed.as_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe { cublasSetMatrix(rows, cols, elemSize, a_arg, rows, B, ldb) };
    }
}

#[cuda_hook(proc_id = 1116)]
fn cublasSetMatrix_64(
    rows: i64,
    cols: i64,
    elemSize: i64,
    #[skip] A: *const c_void,
    lda: i64,
    #[device] B: *mut c_void,
    ldb: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let rows_usize = if rows > 0 {
            usize::try_from(rows).unwrap()
        } else {
            0
        };
        let cols_usize = if cols > 0 {
            usize::try_from(cols).unwrap()
        } else {
            0
        };
        let elem_size = if elemSize > 0 {
            usize::try_from(elemSize).unwrap()
        } else {
            0
        };
        let lda_isize = isize::try_from(lda).unwrap();
        let elem_size_isize = isize::try_from(elemSize).unwrap();
        let mut packed = vec![0u8; rows_usize * cols_usize * elem_size];
        if !packed.is_empty() {
            assert!(!A.is_null());
            assert!(lda >= rows);
            for col in 0..cols_usize {
                for row in 0..rows_usize {
                    let src = unsafe {
                        A.cast::<u8>().offset(
                            (isize::try_from(col).unwrap() * lda_isize
                                + isize::try_from(row).unwrap())
                                * elem_size_isize,
                        )
                    };
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src,
                            packed
                                .as_mut_ptr()
                                .add((col * rows_usize + row) * elem_size),
                            elem_size,
                        );
                    }
                }
            }
        }
    }
    'client_extra_send: {
        send_slice(&packed, channel_sender).unwrap();
    }
    'server_extra_recv: {
        let packed = recv_slice::<u8, _>(channel_receiver).unwrap();
        let a_arg = if packed.is_empty() {
            std::ptr::null()
        } else {
            packed.as_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe { cublasSetMatrix_64(rows, cols, elemSize, a_arg, rows, B, ldb) };
    }
}

#[cuda_hook(proc_id = 1115, async_api)]
fn cublasSetMatrixAsync(
    rows: c_int,
    cols: c_int,
    elemSize: c_int,
    #[skip] A: *const c_void,
    lda: c_int,
    #[device] B: *mut c_void,
    ldb: c_int,
    stream: cudaStream_t,
) -> cublasStatus_t {
    'client_before_send: {
        let rows_usize = if rows > 0 {
            usize::try_from(rows).unwrap()
        } else {
            0
        };
        let cols_usize = if cols > 0 {
            usize::try_from(cols).unwrap()
        } else {
            0
        };
        let elem_size = if elemSize > 0 {
            usize::try_from(elemSize).unwrap()
        } else {
            0
        };
        let lda_isize = isize::try_from(lda).unwrap();
        let elem_size_isize = isize::try_from(elemSize).unwrap();
        let mut packed = vec![0u8; rows_usize * cols_usize * elem_size];
        if !packed.is_empty() {
            assert!(!A.is_null());
            assert!(lda >= rows);
            for col in 0..cols_usize {
                for row in 0..rows_usize {
                    let src = unsafe {
                        A.cast::<u8>().offset(
                            (isize::try_from(col).unwrap() * lda_isize
                                + isize::try_from(row).unwrap())
                                * elem_size_isize,
                        )
                    };
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src,
                            packed
                                .as_mut_ptr()
                                .add((col * rows_usize + row) * elem_size),
                            elem_size,
                        );
                    }
                }
            }
        }
    }
    'client_extra_send: {
        send_slice(&packed, channel_sender).unwrap();
    }
    'server_extra_recv: {
        let packed = recv_slice::<u8, _>(channel_receiver).unwrap();
        let a_arg = if packed.is_empty() {
            std::ptr::null()
        } else {
            packed.as_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            assert_eq!(
                cudasys::cudart::cudaStreamSynchronize(stream.cast()),
                Default::default()
            );
            cublasSetMatrix(rows, cols, elemSize, a_arg, rows, B, ldb)
        };
    }
}

#[cuda_hook(proc_id = 1692, async_api)]
fn cublasSetMatrixAsync_64(
    rows: i64,
    cols: i64,
    elemSize: i64,
    #[skip] A: *const c_void,
    lda: i64,
    #[device] B: *mut c_void,
    ldb: i64,
    stream: cudaStream_t,
) -> cublasStatus_t {
    'client_before_send: {
        let rows_usize = if rows > 0 {
            usize::try_from(rows).unwrap()
        } else {
            0
        };
        let cols_usize = if cols > 0 {
            usize::try_from(cols).unwrap()
        } else {
            0
        };
        let elem_size = if elemSize > 0 {
            usize::try_from(elemSize).unwrap()
        } else {
            0
        };
        let lda_isize = isize::try_from(lda).unwrap();
        let elem_size_isize = isize::try_from(elemSize).unwrap();
        let mut packed = vec![0u8; rows_usize * cols_usize * elem_size];
        if !packed.is_empty() {
            assert!(!A.is_null());
            assert!(lda >= rows);
            for col in 0..cols_usize {
                for row in 0..rows_usize {
                    let src = unsafe {
                        A.cast::<u8>().offset(
                            (isize::try_from(col).unwrap() * lda_isize
                                + isize::try_from(row).unwrap())
                                * elem_size_isize,
                        )
                    };
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            src,
                            packed
                                .as_mut_ptr()
                                .add((col * rows_usize + row) * elem_size),
                            elem_size,
                        );
                    }
                }
            }
        }
    }
    'client_extra_send: {
        send_slice(&packed, channel_sender).unwrap();
    }
    'server_extra_recv: {
        let packed = recv_slice::<u8, _>(channel_receiver).unwrap();
        let a_arg = if packed.is_empty() {
            std::ptr::null()
        } else {
            packed.as_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            assert_eq!(
                cudasys::cudart::cudaStreamSynchronize(stream.cast()),
                Default::default()
            );
            cublasSetMatrix_64(rows, cols, elemSize, a_arg, rows, B, ldb)
        };
    }
}

#[cuda_hook(proc_id = 1112)]
fn cublasGetMatrix(
    rows: c_int,
    cols: c_int,
    elemSize: c_int,
    #[device] A: *const c_void,
    lda: c_int,
    #[skip] B: *mut c_void,
    ldb: c_int,
) -> cublasStatus_t {
    'client_before_send: {
        let rows_usize = if rows > 0 {
            usize::try_from(rows).unwrap()
        } else {
            0
        };
        let cols_usize = if cols > 0 {
            usize::try_from(cols).unwrap()
        } else {
            0
        };
        let elem_size = if elemSize > 0 {
            usize::try_from(elemSize).unwrap()
        } else {
            0
        };
        let ldb_isize = isize::try_from(ldb).unwrap();
        let elem_size_isize = isize::try_from(elemSize).unwrap();
        if rows_usize * cols_usize * elem_size > 0 {
            assert!(!B.is_null());
            assert!(ldb >= rows);
        }
    }
    'server_extra_recv: {
        let rows_usize = if rows > 0 {
            usize::try_from(rows).unwrap()
        } else {
            0
        };
        let cols_usize = if cols > 0 {
            usize::try_from(cols).unwrap()
        } else {
            0
        };
        let elem_size = if elemSize > 0 {
            usize::try_from(elemSize).unwrap()
        } else {
            0
        };
        let mut packed = vec![0u8; rows_usize * cols_usize * elem_size];
        let b_arg = if packed.is_empty() {
            std::ptr::null_mut()
        } else {
            packed.as_mut_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe { cublasGetMatrix(rows, cols, elemSize, A, lda, b_arg, rows) };
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
            assert_eq!(packed.len(), rows_usize * cols_usize * elem_size);
            for col in 0..cols_usize {
                for row in 0..rows_usize {
                    let dst = unsafe {
                        B.cast::<u8>().offset(
                            (isize::try_from(col).unwrap() * ldb_isize
                                + isize::try_from(row).unwrap())
                                * elem_size_isize,
                        )
                    };
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            packed.as_ptr().add((col * rows_usize + row) * elem_size),
                            dst,
                            elem_size,
                        );
                    }
                }
            }
        }
    }
}

#[cuda_hook(proc_id = 1117)]
fn cublasGetMatrix_64(
    rows: i64,
    cols: i64,
    elemSize: i64,
    #[device] A: *const c_void,
    lda: i64,
    #[skip] B: *mut c_void,
    ldb: i64,
) -> cublasStatus_t {
    'client_before_send: {
        let rows_usize = if rows > 0 {
            usize::try_from(rows).unwrap()
        } else {
            0
        };
        let cols_usize = if cols > 0 {
            usize::try_from(cols).unwrap()
        } else {
            0
        };
        let elem_size = if elemSize > 0 {
            usize::try_from(elemSize).unwrap()
        } else {
            0
        };
        let ldb_isize = isize::try_from(ldb).unwrap();
        let elem_size_isize = isize::try_from(elemSize).unwrap();
        if rows_usize * cols_usize * elem_size > 0 {
            assert!(!B.is_null());
            assert!(ldb >= rows);
        }
    }
    'server_extra_recv: {
        let rows_usize = if rows > 0 {
            usize::try_from(rows).unwrap()
        } else {
            0
        };
        let cols_usize = if cols > 0 {
            usize::try_from(cols).unwrap()
        } else {
            0
        };
        let elem_size = if elemSize > 0 {
            usize::try_from(elemSize).unwrap()
        } else {
            0
        };
        let mut packed = vec![0u8; rows_usize * cols_usize * elem_size];
        let b_arg = if packed.is_empty() {
            std::ptr::null_mut()
        } else {
            packed.as_mut_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe { cublasGetMatrix_64(rows, cols, elemSize, A, lda, b_arg, rows) };
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
            assert_eq!(packed.len(), rows_usize * cols_usize * elem_size);
            for col in 0..cols_usize {
                for row in 0..rows_usize {
                    let dst = unsafe {
                        B.cast::<u8>().offset(
                            (isize::try_from(col).unwrap() * ldb_isize
                                + isize::try_from(row).unwrap())
                                * elem_size_isize,
                        )
                    };
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            packed.as_ptr().add((col * rows_usize + row) * elem_size),
                            dst,
                            elem_size,
                        );
                    }
                }
            }
        }
    }
}

#[cuda_hook(proc_id = 1693)]
fn cublasGetMatrixAsync(
    rows: c_int,
    cols: c_int,
    elemSize: c_int,
    #[device] A: *const c_void,
    lda: c_int,
    #[skip] B: *mut c_void,
    ldb: c_int,
    stream: cudaStream_t,
) -> cublasStatus_t {
    'client_before_send: {
        let rows_usize = if rows > 0 {
            usize::try_from(rows).unwrap()
        } else {
            0
        };
        let cols_usize = if cols > 0 {
            usize::try_from(cols).unwrap()
        } else {
            0
        };
        let elem_size = if elemSize > 0 {
            usize::try_from(elemSize).unwrap()
        } else {
            0
        };
        let ldb_isize = isize::try_from(ldb).unwrap();
        let elem_size_isize = isize::try_from(elemSize).unwrap();
        if rows_usize * cols_usize * elem_size > 0 {
            assert!(!B.is_null());
            assert!(ldb >= rows);
        }
    }
    'server_extra_recv: {
        let rows_usize = if rows > 0 {
            usize::try_from(rows).unwrap()
        } else {
            0
        };
        let cols_usize = if cols > 0 {
            usize::try_from(cols).unwrap()
        } else {
            0
        };
        let elem_size = if elemSize > 0 {
            usize::try_from(elemSize).unwrap()
        } else {
            0
        };
        let mut packed = vec![0u8; rows_usize * cols_usize * elem_size];
        let b_arg = if packed.is_empty() {
            std::ptr::null_mut()
        } else {
            packed.as_mut_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            assert_eq!(
                cudasys::cudart::cudaStreamSynchronize(stream.cast()),
                Default::default()
            );
            cublasGetMatrix(rows, cols, elemSize, A, lda, b_arg, rows)
        };
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
            assert_eq!(packed.len(), rows_usize * cols_usize * elem_size);
            for col in 0..cols_usize {
                for row in 0..rows_usize {
                    let dst = unsafe {
                        B.cast::<u8>().offset(
                            (isize::try_from(col).unwrap() * ldb_isize
                                + isize::try_from(row).unwrap())
                                * elem_size_isize,
                        )
                    };
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            packed.as_ptr().add((col * rows_usize + row) * elem_size),
                            dst,
                            elem_size,
                        );
                    }
                }
            }
        }
    }
}

#[cuda_hook(proc_id = 1694)]
fn cublasGetMatrixAsync_64(
    rows: i64,
    cols: i64,
    elemSize: i64,
    #[device] A: *const c_void,
    lda: i64,
    #[skip] B: *mut c_void,
    ldb: i64,
    stream: cudaStream_t,
) -> cublasStatus_t {
    'client_before_send: {
        let rows_usize = if rows > 0 {
            usize::try_from(rows).unwrap()
        } else {
            0
        };
        let cols_usize = if cols > 0 {
            usize::try_from(cols).unwrap()
        } else {
            0
        };
        let elem_size = if elemSize > 0 {
            usize::try_from(elemSize).unwrap()
        } else {
            0
        };
        let ldb_isize = isize::try_from(ldb).unwrap();
        let elem_size_isize = isize::try_from(elemSize).unwrap();
        if rows_usize * cols_usize * elem_size > 0 {
            assert!(!B.is_null());
            assert!(ldb >= rows);
        }
    }
    'server_extra_recv: {
        let rows_usize = if rows > 0 {
            usize::try_from(rows).unwrap()
        } else {
            0
        };
        let cols_usize = if cols > 0 {
            usize::try_from(cols).unwrap()
        } else {
            0
        };
        let elem_size = if elemSize > 0 {
            usize::try_from(elemSize).unwrap()
        } else {
            0
        };
        let mut packed = vec![0u8; rows_usize * cols_usize * elem_size];
        let b_arg = if packed.is_empty() {
            std::ptr::null_mut()
        } else {
            packed.as_mut_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            assert_eq!(
                cudasys::cudart::cudaStreamSynchronize(stream.cast()),
                Default::default()
            );
            cublasGetMatrix_64(rows, cols, elemSize, A, lda, b_arg, rows)
        };
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
            assert_eq!(packed.len(), rows_usize * cols_usize * elem_size);
            for col in 0..cols_usize {
                for row in 0..rows_usize {
                    let dst = unsafe {
                        B.cast::<u8>().offset(
                            (isize::try_from(col).unwrap() * ldb_isize
                                + isize::try_from(row).unwrap())
                                * elem_size_isize,
                        )
                    };
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            packed.as_ptr().add((col * rows_usize + row) * elem_size),
                            dst,
                            elem_size,
                        );
                    }
                }
            }
        }
    }
}

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

#[cuda_hook(proc_id = 1650, async_api)]
fn cublasCopyEx(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const c_void,
    xType: cudaDataType,
    incx: c_int,
    #[device] y: *mut c_void,
    yType: cudaDataType,
    incy: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1651, async_api)]
fn cublasCopyEx_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const c_void,
    xType: cudaDataType,
    incx: i64,
    #[device] y: *mut c_void,
    yType: cudaDataType,
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

#[cuda_hook(proc_id = 1652, async_api)]
fn cublasSwapEx(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *mut c_void,
    xType: cudaDataType,
    incx: c_int,
    #[device] y: *mut c_void,
    yType: cudaDataType,
    incy: c_int,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1653, async_api)]
fn cublasSwapEx_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *mut c_void,
    xType: cudaDataType,
    incx: i64,
    #[device] y: *mut c_void,
    yType: cudaDataType,
    incy: i64,
) -> cublasStatus_t;

#[cuda_hook(proc_id = 1666)]
fn cublasScalEx(
    handle: cublasHandle_t,
    n: c_int,
    #[skip] alpha: *const c_void,
    alphaType: cudaDataType,
    #[device] x: *mut c_void,
    xType: cudaDataType,
    incx: c_int,
    executionType: cudaDataType,
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
        if !device_pointer_mode && alpha.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let alpha_size = match alphaType {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let alpha_addr = alpha as usize;
        let mut alpha_host_storage = [0u8; 16];
        assert!(alpha_size <= alpha_host_storage.len());
        if !device_pointer_mode {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    alpha.cast::<u8>(),
                    alpha_host_storage.as_mut_ptr(),
                    alpha_size,
                );
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_size.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            send_slice(&alpha_host_storage[..alpha_size], channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_size = 0usize;
        alpha_size.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        #[repr(align(16))]
        struct AlignedAlpha([u8; 16]);
        let mut alpha_host_storage = AlignedAlpha([0; 16]);
        assert!(alpha_size <= alpha_host_storage.0.len());
        if !device_pointer_mode {
            let alpha_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert!(alpha_bytes.len() == alpha_size);
            alpha_host_storage.0[..alpha_size].copy_from_slice(&alpha_bytes);
        }
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const c_void
        } else {
            alpha_host_storage.0.as_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasScalEx(
                handle,
                n,
                alpha_arg,
                alphaType,
                x,
                xType,
                incx,
                executionType,
            )
        };
    }
}

#[cuda_hook(proc_id = 1667)]
fn cublasScalEx_64(
    handle: cublasHandle_t,
    n: i64,
    #[skip] alpha: *const c_void,
    alphaType: cudaDataType,
    #[device] x: *mut c_void,
    xType: cudaDataType,
    incx: i64,
    executionType: cudaDataType,
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
        if !device_pointer_mode && alpha.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let alpha_size = match alphaType {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let alpha_addr = alpha as usize;
        let mut alpha_host_storage = [0u8; 16];
        assert!(alpha_size <= alpha_host_storage.len());
        if !device_pointer_mode {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    alpha.cast::<u8>(),
                    alpha_host_storage.as_mut_ptr(),
                    alpha_size,
                );
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_size.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            send_slice(&alpha_host_storage[..alpha_size], channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_size = 0usize;
        alpha_size.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        #[repr(align(16))]
        struct AlignedAlpha([u8; 16]);
        let mut alpha_host_storage = AlignedAlpha([0; 16]);
        assert!(alpha_size <= alpha_host_storage.0.len());
        if !device_pointer_mode {
            let alpha_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert!(alpha_bytes.len() == alpha_size);
            alpha_host_storage.0[..alpha_size].copy_from_slice(&alpha_bytes);
        }
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const c_void
        } else {
            alpha_host_storage.0.as_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasScalEx_64(
                handle,
                n,
                alpha_arg,
                alphaType,
                x,
                xType,
                incx,
                executionType,
            )
        };
    }
}

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

#[cuda_hook(proc_id = 1668)]
fn cublasAxpyEx(
    handle: cublasHandle_t,
    n: c_int,
    #[skip] alpha: *const c_void,
    alphaType: cudaDataType,
    #[device] x: *const c_void,
    xType: cudaDataType,
    incx: c_int,
    #[device] y: *mut c_void,
    yType: cudaDataType,
    incy: c_int,
    executiontype: cudaDataType,
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
        if !device_pointer_mode && alpha.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let alpha_size = match alphaType {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let alpha_addr = alpha as usize;
        let mut alpha_host_storage = [0u8; 16];
        assert!(alpha_size <= alpha_host_storage.len());
        if !device_pointer_mode {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    alpha.cast::<u8>(),
                    alpha_host_storage.as_mut_ptr(),
                    alpha_size,
                );
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_size.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            send_slice(&alpha_host_storage[..alpha_size], channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_size = 0usize;
        alpha_size.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        #[repr(align(16))]
        struct AlignedAlpha([u8; 16]);
        let mut alpha_host_storage = AlignedAlpha([0; 16]);
        assert!(alpha_size <= alpha_host_storage.0.len());
        if !device_pointer_mode {
            let alpha_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert!(alpha_bytes.len() == alpha_size);
            alpha_host_storage.0[..alpha_size].copy_from_slice(&alpha_bytes);
        }
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const c_void
        } else {
            alpha_host_storage.0.as_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasAxpyEx(
                handle,
                n,
                alpha_arg,
                alphaType,
                x,
                xType,
                incx,
                y,
                yType,
                incy,
                executiontype,
            )
        };
    }
}

#[cuda_hook(proc_id = 1669)]
fn cublasAxpyEx_64(
    handle: cublasHandle_t,
    n: i64,
    #[skip] alpha: *const c_void,
    alphaType: cudaDataType,
    #[device] x: *const c_void,
    xType: cudaDataType,
    incx: i64,
    #[device] y: *mut c_void,
    yType: cudaDataType,
    incy: i64,
    executiontype: cudaDataType,
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
        if !device_pointer_mode && alpha.is_null() {
            return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
        }
        let alpha_size = match alphaType {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let alpha_addr = alpha as usize;
        let mut alpha_host_storage = [0u8; 16];
        assert!(alpha_size <= alpha_host_storage.len());
        if !device_pointer_mode {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    alpha.cast::<u8>(),
                    alpha_host_storage.as_mut_ptr(),
                    alpha_size,
                );
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        alpha_size.send(channel_sender).unwrap();
        alpha_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            send_slice(&alpha_host_storage[..alpha_size], channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut alpha_size = 0usize;
        alpha_size.recv(channel_receiver).unwrap();
        let mut alpha_addr = 0usize;
        alpha_addr.recv(channel_receiver).unwrap();
        #[repr(align(16))]
        struct AlignedAlpha([u8; 16]);
        let mut alpha_host_storage = AlignedAlpha([0; 16]);
        assert!(alpha_size <= alpha_host_storage.0.len());
        if !device_pointer_mode {
            let alpha_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert!(alpha_bytes.len() == alpha_size);
            alpha_host_storage.0[..alpha_size].copy_from_slice(&alpha_bytes);
        }
        let alpha_arg = if device_pointer_mode {
            alpha_addr as *const c_void
        } else {
            alpha_host_storage.0.as_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasAxpyEx_64(
                handle,
                n,
                alpha_arg,
                alphaType,
                x,
                xType,
                incx,
                y,
                yType,
                incy,
                executiontype,
            )
        };
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

#[cuda_hook(proc_id = 1658)]
fn cublasNrm2Ex(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const c_void,
    xType: cudaDataType,
    incx: c_int,
    #[skip] result_ptr: *mut c_void,
    resultType: cudaDataType,
    executionType: cudaDataType,
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
        let result_size = match resultType {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_size.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_size = 0usize;
        result_size.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        #[repr(align(16))]
        struct AlignedResult([u8; 16]);
        let mut host_result_storage = AlignedResult([0; 16]);
        assert!(result_size <= host_result_storage.0.len());
        let result_arg = if device_pointer_mode {
            result_addr as *mut c_void
        } else {
            host_result_storage.0.as_mut_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasNrm2Ex(
                handle,
                n,
                x,
                xType,
                incx,
                result_arg,
                resultType,
                executionType,
            )
        };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            send_slice(&host_result_storage.0[..result_size], channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let host_result_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert!(host_result_bytes.len() == result_size);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    host_result_bytes.as_ptr(),
                    result_ptr.cast::<u8>(),
                    result_size,
                );
            }
        }
    }
}

#[cuda_hook(proc_id = 1659)]
fn cublasNrm2Ex_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const c_void,
    xType: cudaDataType,
    incx: i64,
    #[skip] result_ptr: *mut c_void,
    resultType: cudaDataType,
    executionType: cudaDataType,
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
        let result_size = match resultType {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_size.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_size = 0usize;
        result_size.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        #[repr(align(16))]
        struct AlignedResult([u8; 16]);
        let mut host_result_storage = AlignedResult([0; 16]);
        assert!(result_size <= host_result_storage.0.len());
        let result_arg = if device_pointer_mode {
            result_addr as *mut c_void
        } else {
            host_result_storage.0.as_mut_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasNrm2Ex_64(
                handle,
                n,
                x,
                xType,
                incx,
                result_arg,
                resultType,
                executionType,
            )
        };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            send_slice(&host_result_storage.0[..result_size], channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let host_result_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert!(host_result_bytes.len() == result_size);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    host_result_bytes.as_ptr(),
                    result_ptr.cast::<u8>(),
                    result_size,
                );
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

#[cuda_hook(proc_id = 1660)]
fn cublasAsumEx(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const c_void,
    xType: cudaDataType,
    incx: c_int,
    #[skip] result_ptr: *mut c_void,
    resultType: cudaDataType,
    executionType: cudaDataType,
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
        let result_size = match resultType {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_size.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_size = 0usize;
        result_size.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        #[repr(align(16))]
        struct AlignedResult([u8; 16]);
        let mut host_result_storage = AlignedResult([0; 16]);
        assert!(result_size <= host_result_storage.0.len());
        let result_arg = if device_pointer_mode {
            result_addr as *mut c_void
        } else {
            host_result_storage.0.as_mut_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasAsumEx(
                handle,
                n,
                x,
                xType,
                incx,
                result_arg,
                resultType,
                executionType,
            )
        };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            send_slice(&host_result_storage.0[..result_size], channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let host_result_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert!(host_result_bytes.len() == result_size);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    host_result_bytes.as_ptr(),
                    result_ptr.cast::<u8>(),
                    result_size,
                );
            }
        }
    }
}

#[cuda_hook(proc_id = 1661)]
fn cublasAsumEx_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const c_void,
    xType: cudaDataType,
    incx: i64,
    #[skip] result_ptr: *mut c_void,
    resultType: cudaDataType,
    executionType: cudaDataType,
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
        let result_size = match resultType {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_size.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_size = 0usize;
        result_size.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        #[repr(align(16))]
        struct AlignedResult([u8; 16]);
        let mut host_result_storage = AlignedResult([0; 16]);
        assert!(result_size <= host_result_storage.0.len());
        let result_arg = if device_pointer_mode {
            result_addr as *mut c_void
        } else {
            host_result_storage.0.as_mut_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasAsumEx_64(
                handle,
                n,
                x,
                xType,
                incx,
                result_arg,
                resultType,
                executionType,
            )
        };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            send_slice(&host_result_storage.0[..result_size], channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let host_result_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert!(host_result_bytes.len() == result_size);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    host_result_bytes.as_ptr(),
                    result_ptr.cast::<u8>(),
                    result_size,
                );
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

#[cuda_hook(proc_id = 1662)]
fn cublasDotEx(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const c_void,
    xType: cudaDataType,
    incx: c_int,
    #[device] y: *const c_void,
    yType: cudaDataType,
    incy: c_int,
    #[skip] result_ptr: *mut c_void,
    resultType: cudaDataType,
    executionType: cudaDataType,
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
        let result_size = match resultType {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_size.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_size = 0usize;
        result_size.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        #[repr(align(16))]
        struct AlignedResult([u8; 16]);
        let mut host_result_storage = AlignedResult([0; 16]);
        assert!(result_size <= host_result_storage.0.len());
        let result_arg = if device_pointer_mode {
            result_addr as *mut c_void
        } else {
            host_result_storage.0.as_mut_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasDotEx(
                handle,
                n,
                x,
                xType,
                incx,
                y,
                yType,
                incy,
                result_arg,
                resultType,
                executionType,
            )
        };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            send_slice(&host_result_storage.0[..result_size], channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let host_result_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert!(host_result_bytes.len() == result_size);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    host_result_bytes.as_ptr(),
                    result_ptr.cast::<u8>(),
                    result_size,
                );
            }
        }
    }
}

#[cuda_hook(proc_id = 1663)]
fn cublasDotEx_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const c_void,
    xType: cudaDataType,
    incx: i64,
    #[device] y: *const c_void,
    yType: cudaDataType,
    incy: i64,
    #[skip] result_ptr: *mut c_void,
    resultType: cudaDataType,
    executionType: cudaDataType,
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
        let result_size = match resultType {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_size.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_size = 0usize;
        result_size.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        #[repr(align(16))]
        struct AlignedResult([u8; 16]);
        let mut host_result_storage = AlignedResult([0; 16]);
        assert!(result_size <= host_result_storage.0.len());
        let result_arg = if device_pointer_mode {
            result_addr as *mut c_void
        } else {
            host_result_storage.0.as_mut_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasDotEx_64(
                handle,
                n,
                x,
                xType,
                incx,
                y,
                yType,
                incy,
                result_arg,
                resultType,
                executionType,
            )
        };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            send_slice(&host_result_storage.0[..result_size], channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let host_result_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert!(host_result_bytes.len() == result_size);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    host_result_bytes.as_ptr(),
                    result_ptr.cast::<u8>(),
                    result_size,
                );
            }
        }
    }
}

#[cuda_hook(proc_id = 1664)]
fn cublasDotcEx(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const c_void,
    xType: cudaDataType,
    incx: c_int,
    #[device] y: *const c_void,
    yType: cudaDataType,
    incy: c_int,
    #[skip] result_ptr: *mut c_void,
    resultType: cudaDataType,
    executionType: cudaDataType,
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
        let result_size = match resultType {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_size.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_size = 0usize;
        result_size.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        #[repr(align(16))]
        struct AlignedResult([u8; 16]);
        let mut host_result_storage = AlignedResult([0; 16]);
        assert!(result_size <= host_result_storage.0.len());
        let result_arg = if device_pointer_mode {
            result_addr as *mut c_void
        } else {
            host_result_storage.0.as_mut_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasDotcEx(
                handle,
                n,
                x,
                xType,
                incx,
                y,
                yType,
                incy,
                result_arg,
                resultType,
                executionType,
            )
        };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            send_slice(&host_result_storage.0[..result_size], channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let host_result_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert!(host_result_bytes.len() == result_size);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    host_result_bytes.as_ptr(),
                    result_ptr.cast::<u8>(),
                    result_size,
                );
            }
        }
    }
}

#[cuda_hook(proc_id = 1665)]
fn cublasDotcEx_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const c_void,
    xType: cudaDataType,
    incx: i64,
    #[device] y: *const c_void,
    yType: cudaDataType,
    incy: i64,
    #[skip] result_ptr: *mut c_void,
    resultType: cudaDataType,
    executionType: cudaDataType,
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
        let result_size = match resultType {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let result_addr = result_ptr as usize;
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        result_size.send(channel_sender).unwrap();
        result_addr.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut result_size = 0usize;
        result_size.recv(channel_receiver).unwrap();
        let mut result_addr = 0usize;
        result_addr.recv(channel_receiver).unwrap();
        #[repr(align(16))]
        struct AlignedResult([u8; 16]);
        let mut host_result_storage = AlignedResult([0; 16]);
        assert!(result_size <= host_result_storage.0.len());
        let result_arg = if device_pointer_mode {
            result_addr as *mut c_void
        } else {
            host_result_storage.0.as_mut_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasDotcEx_64(
                handle,
                n,
                x,
                xType,
                incx,
                y,
                yType,
                incy,
                result_arg,
                resultType,
                executionType,
            )
        };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            send_slice(&host_result_storage.0[..result_size], channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let host_result_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert!(host_result_bytes.len() == result_size);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    host_result_bytes.as_ptr(),
                    result_ptr.cast::<u8>(),
                    result_size,
                );
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

#[cuda_hook(proc_id = 1654)]
fn cublasIamaxEx(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const c_void,
    xType: cudaDataType,
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
        let result = unsafe { cublasIamaxEx(handle, n, x, xType, incx, result_arg) };
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

#[cuda_hook(proc_id = 1655)]
fn cublasIamaxEx_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const c_void,
    xType: cudaDataType,
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
        let result = unsafe { cublasIamaxEx_64(handle, n, x, xType, incx, result_arg) };
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

#[cuda_hook(proc_id = 1656)]
fn cublasIaminEx(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *const c_void,
    xType: cudaDataType,
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
        let result = unsafe { cublasIaminEx(handle, n, x, xType, incx, result_arg) };
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

#[cuda_hook(proc_id = 1657)]
fn cublasIaminEx_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *const c_void,
    xType: cudaDataType,
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
        let result = unsafe { cublasIaminEx_64(handle, n, x, xType, incx, result_arg) };
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

#[cuda_hook(proc_id = 1670)]
fn cublasRotEx(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *mut c_void,
    xType: cudaDataType,
    incx: c_int,
    #[device] y: *mut c_void,
    yType: cudaDataType,
    incy: c_int,
    #[skip] c: *const c_void,
    #[skip] s: *const c_void,
    csType: cudaDataType,
    executiontype: cudaDataType,
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
        let cs_size = match csType {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let c_addr = c as usize;
        let s_addr = s as usize;
        let mut host_c_storage = [0u8; 16];
        let mut host_s_storage = [0u8; 16];
        assert!(cs_size <= host_c_storage.len());
        if !device_pointer_mode {
            unsafe {
                std::ptr::copy_nonoverlapping(c.cast::<u8>(), host_c_storage.as_mut_ptr(), cs_size);
                std::ptr::copy_nonoverlapping(s.cast::<u8>(), host_s_storage.as_mut_ptr(), cs_size);
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        cs_size.send(channel_sender).unwrap();
        c_addr.send(channel_sender).unwrap();
        s_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            send_slice(&host_c_storage[..cs_size], channel_sender).unwrap();
            send_slice(&host_s_storage[..cs_size], channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut cs_size = 0usize;
        cs_size.recv(channel_receiver).unwrap();
        let mut c_addr = 0usize;
        c_addr.recv(channel_receiver).unwrap();
        let mut s_addr = 0usize;
        s_addr.recv(channel_receiver).unwrap();
        #[repr(align(16))]
        struct AlignedScalar([u8; 16]);
        let mut host_c_storage = AlignedScalar([0; 16]);
        let mut host_s_storage = AlignedScalar([0; 16]);
        assert!(cs_size <= host_c_storage.0.len());
        if !device_pointer_mode {
            let c_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            let s_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert!(c_bytes.len() == cs_size);
            assert!(s_bytes.len() == cs_size);
            host_c_storage.0[..cs_size].copy_from_slice(&c_bytes);
            host_s_storage.0[..cs_size].copy_from_slice(&s_bytes);
        }
        let c_arg = if device_pointer_mode {
            c_addr as *const c_void
        } else {
            host_c_storage.0.as_ptr().cast::<c_void>()
        };
        let s_arg = if device_pointer_mode {
            s_addr as *const c_void
        } else {
            host_s_storage.0.as_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasRotEx(
                handle,
                n,
                x,
                xType,
                incx,
                y,
                yType,
                incy,
                c_arg,
                s_arg,
                csType,
                executiontype,
            )
        };
    }
}

#[cuda_hook(proc_id = 1671)]
fn cublasRotEx_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *mut c_void,
    xType: cudaDataType,
    incx: i64,
    #[device] y: *mut c_void,
    yType: cudaDataType,
    incy: i64,
    #[skip] c: *const c_void,
    #[skip] s: *const c_void,
    csType: cudaDataType,
    executiontype: cudaDataType,
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
        let cs_size = match csType {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let c_addr = c as usize;
        let s_addr = s as usize;
        let mut host_c_storage = [0u8; 16];
        let mut host_s_storage = [0u8; 16];
        assert!(cs_size <= host_c_storage.len());
        if !device_pointer_mode {
            unsafe {
                std::ptr::copy_nonoverlapping(c.cast::<u8>(), host_c_storage.as_mut_ptr(), cs_size);
                std::ptr::copy_nonoverlapping(s.cast::<u8>(), host_s_storage.as_mut_ptr(), cs_size);
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        cs_size.send(channel_sender).unwrap();
        c_addr.send(channel_sender).unwrap();
        s_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            send_slice(&host_c_storage[..cs_size], channel_sender).unwrap();
            send_slice(&host_s_storage[..cs_size], channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut cs_size = 0usize;
        cs_size.recv(channel_receiver).unwrap();
        let mut c_addr = 0usize;
        c_addr.recv(channel_receiver).unwrap();
        let mut s_addr = 0usize;
        s_addr.recv(channel_receiver).unwrap();
        #[repr(align(16))]
        struct AlignedScalar([u8; 16]);
        let mut host_c_storage = AlignedScalar([0; 16]);
        let mut host_s_storage = AlignedScalar([0; 16]);
        assert!(cs_size <= host_c_storage.0.len());
        if !device_pointer_mode {
            let c_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            let s_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert!(c_bytes.len() == cs_size);
            assert!(s_bytes.len() == cs_size);
            host_c_storage.0[..cs_size].copy_from_slice(&c_bytes);
            host_s_storage.0[..cs_size].copy_from_slice(&s_bytes);
        }
        let c_arg = if device_pointer_mode {
            c_addr as *const c_void
        } else {
            host_c_storage.0.as_ptr().cast::<c_void>()
        };
        let s_arg = if device_pointer_mode {
            s_addr as *const c_void
        } else {
            host_s_storage.0.as_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasRotEx_64(
                handle,
                n,
                x,
                xType,
                incx,
                y,
                yType,
                incy,
                c_arg,
                s_arg,
                csType,
                executiontype,
            )
        };
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

#[cuda_hook(proc_id = 1674)]
fn cublasRotgEx(
    handle: cublasHandle_t,
    #[skip] a: *mut c_void,
    #[skip] b: *mut c_void,
    abType: cudaDataType,
    #[skip] c: *mut c_void,
    #[skip] s: *mut c_void,
    csType: cudaDataType,
    executiontype: cudaDataType,
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
        let ab_size = match abType {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let cs_size = match csType {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let a_addr = a as usize;
        let b_addr = b as usize;
        let c_addr = c as usize;
        let s_addr = s as usize;
        let mut host_a_storage = [0u8; 16];
        let mut host_b_storage = [0u8; 16];
        assert!(ab_size <= host_a_storage.len());
        assert!(cs_size <= host_a_storage.len());
        if !device_pointer_mode {
            unsafe {
                std::ptr::copy_nonoverlapping(a.cast::<u8>(), host_a_storage.as_mut_ptr(), ab_size);
                std::ptr::copy_nonoverlapping(b.cast::<u8>(), host_b_storage.as_mut_ptr(), ab_size);
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        ab_size.send(channel_sender).unwrap();
        cs_size.send(channel_sender).unwrap();
        a_addr.send(channel_sender).unwrap();
        b_addr.send(channel_sender).unwrap();
        c_addr.send(channel_sender).unwrap();
        s_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            send_slice(&host_a_storage[..ab_size], channel_sender).unwrap();
            send_slice(&host_b_storage[..ab_size], channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut ab_size = 0usize;
        ab_size.recv(channel_receiver).unwrap();
        let mut cs_size = 0usize;
        cs_size.recv(channel_receiver).unwrap();
        let mut a_addr = 0usize;
        a_addr.recv(channel_receiver).unwrap();
        let mut b_addr = 0usize;
        b_addr.recv(channel_receiver).unwrap();
        let mut c_addr = 0usize;
        c_addr.recv(channel_receiver).unwrap();
        let mut s_addr = 0usize;
        s_addr.recv(channel_receiver).unwrap();
        #[repr(align(16))]
        struct AlignedScalar([u8; 16]);
        let mut host_a_storage = AlignedScalar([0; 16]);
        let mut host_b_storage = AlignedScalar([0; 16]);
        let mut host_c_storage = AlignedScalar([0; 16]);
        let mut host_s_storage = AlignedScalar([0; 16]);
        assert!(ab_size <= host_a_storage.0.len());
        assert!(cs_size <= host_c_storage.0.len());
        if !device_pointer_mode {
            let a_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            let b_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert!(a_bytes.len() == ab_size);
            assert!(b_bytes.len() == ab_size);
            host_a_storage.0[..ab_size].copy_from_slice(&a_bytes);
            host_b_storage.0[..ab_size].copy_from_slice(&b_bytes);
        }
        let a_arg = if device_pointer_mode {
            a_addr as *mut c_void
        } else {
            host_a_storage.0.as_mut_ptr().cast::<c_void>()
        };
        let b_arg = if device_pointer_mode {
            b_addr as *mut c_void
        } else {
            host_b_storage.0.as_mut_ptr().cast::<c_void>()
        };
        let c_arg = if device_pointer_mode {
            c_addr as *mut c_void
        } else {
            host_c_storage.0.as_mut_ptr().cast::<c_void>()
        };
        let s_arg = if device_pointer_mode {
            s_addr as *mut c_void
        } else {
            host_s_storage.0.as_mut_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasRotgEx(
                handle,
                a_arg,
                b_arg,
                abType,
                c_arg,
                s_arg,
                csType,
                executiontype,
            )
        };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            send_slice(&host_a_storage.0[..ab_size], channel_sender).unwrap();
            send_slice(&host_b_storage.0[..ab_size], channel_sender).unwrap();
            send_slice(&host_c_storage.0[..cs_size], channel_sender).unwrap();
            send_slice(&host_s_storage.0[..cs_size], channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let host_a_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            let host_b_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            let host_c_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            let host_s_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert!(host_a_bytes.len() == ab_size);
            assert!(host_b_bytes.len() == ab_size);
            assert!(host_c_bytes.len() == cs_size);
            assert!(host_s_bytes.len() == cs_size);
            unsafe {
                std::ptr::copy_nonoverlapping(host_a_bytes.as_ptr(), a.cast::<u8>(), ab_size);
                std::ptr::copy_nonoverlapping(host_b_bytes.as_ptr(), b.cast::<u8>(), ab_size);
                std::ptr::copy_nonoverlapping(host_c_bytes.as_ptr(), c.cast::<u8>(), cs_size);
                std::ptr::copy_nonoverlapping(host_s_bytes.as_ptr(), s.cast::<u8>(), cs_size);
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

#[cuda_hook(proc_id = 1672)]
fn cublasRotmEx(
    handle: cublasHandle_t,
    n: c_int,
    #[device] x: *mut c_void,
    xType: cudaDataType,
    incx: c_int,
    #[device] y: *mut c_void,
    yType: cudaDataType,
    incy: c_int,
    #[skip] param: *const c_void,
    paramType: cudaDataType,
    executiontype: cudaDataType,
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
        let param_elem_size = match paramType {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let param_size = param_elem_size * 5;
        let param_addr = param as usize;
        let mut host_param_storage = [0u8; 80];
        assert!(param_size <= host_param_storage.len());
        if !device_pointer_mode {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    param.cast::<u8>(),
                    host_param_storage.as_mut_ptr(),
                    param_size,
                );
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        param_size.send(channel_sender).unwrap();
        param_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            send_slice(&host_param_storage[..param_size], channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut param_size = 0usize;
        param_size.recv(channel_receiver).unwrap();
        let mut param_addr = 0usize;
        param_addr.recv(channel_receiver).unwrap();
        #[repr(align(16))]
        struct AlignedParam([u8; 80]);
        let mut host_param_storage = AlignedParam([0; 80]);
        assert!(param_size <= host_param_storage.0.len());
        if !device_pointer_mode {
            let param_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert!(param_bytes.len() == param_size);
            host_param_storage.0[..param_size].copy_from_slice(&param_bytes);
        }
        let param_arg = if device_pointer_mode {
            param_addr as *const c_void
        } else {
            host_param_storage.0.as_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasRotmEx(
                handle,
                n,
                x,
                xType,
                incx,
                y,
                yType,
                incy,
                param_arg,
                paramType,
                executiontype,
            )
        };
    }
}

#[cuda_hook(proc_id = 1673)]
fn cublasRotmEx_64(
    handle: cublasHandle_t,
    n: i64,
    #[device] x: *mut c_void,
    xType: cudaDataType,
    incx: i64,
    #[device] y: *mut c_void,
    yType: cudaDataType,
    incy: i64,
    #[skip] param: *const c_void,
    paramType: cudaDataType,
    executiontype: cudaDataType,
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
        let param_elem_size = match paramType {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let param_size = param_elem_size * 5;
        let param_addr = param as usize;
        let mut host_param_storage = [0u8; 80];
        assert!(param_size <= host_param_storage.len());
        if !device_pointer_mode {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    param.cast::<u8>(),
                    host_param_storage.as_mut_ptr(),
                    param_size,
                );
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        param_size.send(channel_sender).unwrap();
        param_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            send_slice(&host_param_storage[..param_size], channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut param_size = 0usize;
        param_size.recv(channel_receiver).unwrap();
        let mut param_addr = 0usize;
        param_addr.recv(channel_receiver).unwrap();
        #[repr(align(16))]
        struct AlignedParam([u8; 80]);
        let mut host_param_storage = AlignedParam([0; 80]);
        assert!(param_size <= host_param_storage.0.len());
        if !device_pointer_mode {
            let param_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert!(param_bytes.len() == param_size);
            host_param_storage.0[..param_size].copy_from_slice(&param_bytes);
        }
        let param_arg = if device_pointer_mode {
            param_addr as *const c_void
        } else {
            host_param_storage.0.as_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasRotmEx_64(
                handle,
                n,
                x,
                xType,
                incx,
                y,
                yType,
                incy,
                param_arg,
                paramType,
                executiontype,
            )
        };
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

#[cuda_hook(proc_id = 1675)]
fn cublasRotmgEx(
    handle: cublasHandle_t,
    #[skip] d1: *mut c_void,
    d1Type: cudaDataType,
    #[skip] d2: *mut c_void,
    d2Type: cudaDataType,
    #[skip] x1: *mut c_void,
    x1Type: cudaDataType,
    #[skip] y1: *const c_void,
    y1Type: cudaDataType,
    #[skip] param: *mut c_void,
    paramType: cudaDataType,
    executiontype: cudaDataType,
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
        let d1_size = match d1Type {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let d2_size = match d2Type {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let x1_size = match x1Type {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let y1_size = match y1Type {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let param_elem_size = match paramType {
            cudaDataType::CUDA_R_16F
            | cudaDataType::CUDA_R_16BF
            | cudaDataType::CUDA_R_16I
            | cudaDataType::CUDA_R_16U => 2usize,
            cudaDataType::CUDA_C_16F
            | cudaDataType::CUDA_C_16BF
            | cudaDataType::CUDA_C_16I
            | cudaDataType::CUDA_C_16U
            | cudaDataType::CUDA_R_32F
            | cudaDataType::CUDA_R_32I
            | cudaDataType::CUDA_R_32U => 4usize,
            cudaDataType::CUDA_C_32F
            | cudaDataType::CUDA_C_32I
            | cudaDataType::CUDA_C_32U
            | cudaDataType::CUDA_R_64F
            | cudaDataType::CUDA_R_64I
            | cudaDataType::CUDA_R_64U => 8usize,
            cudaDataType::CUDA_C_64F | cudaDataType::CUDA_C_64I | cudaDataType::CUDA_C_64U => {
                16usize
            }
            cudaDataType::CUDA_R_4I
            | cudaDataType::CUDA_R_4U
            | cudaDataType::CUDA_C_4I
            | cudaDataType::CUDA_C_4U
            | cudaDataType::CUDA_R_8I
            | cudaDataType::CUDA_R_8U
            | cudaDataType::CUDA_R_8F_E4M3
            | cudaDataType::CUDA_R_8F_E5M2
            | cudaDataType::CUDA_R_8F_UE8M0
            | cudaDataType::CUDA_R_6F_E2M3
            | cudaDataType::CUDA_R_6F_E3M2
            | cudaDataType::CUDA_R_4F_E2M1 => 1usize,
            cudaDataType::CUDA_C_8I | cudaDataType::CUDA_C_8U => 2usize,
        };
        let param_size = param_elem_size * 5;
        let d1_addr = d1 as usize;
        let d2_addr = d2 as usize;
        let x1_addr = x1 as usize;
        let y1_addr = y1 as usize;
        let param_addr = param as usize;
        let mut host_d1_storage = [0u8; 16];
        let mut host_d2_storage = [0u8; 16];
        let mut host_x1_storage = [0u8; 16];
        let mut host_y1_storage = [0u8; 16];
        assert!(d1_size <= host_d1_storage.len());
        assert!(d2_size <= host_d2_storage.len());
        assert!(x1_size <= host_x1_storage.len());
        assert!(y1_size <= host_y1_storage.len());
        assert!(param_size <= 80);
        if !device_pointer_mode {
            unsafe {
                std::ptr::copy_nonoverlapping(
                    d1.cast::<u8>(),
                    host_d1_storage.as_mut_ptr(),
                    d1_size,
                );
                std::ptr::copy_nonoverlapping(
                    d2.cast::<u8>(),
                    host_d2_storage.as_mut_ptr(),
                    d2_size,
                );
                std::ptr::copy_nonoverlapping(
                    x1.cast::<u8>(),
                    host_x1_storage.as_mut_ptr(),
                    x1_size,
                );
                std::ptr::copy_nonoverlapping(
                    y1.cast::<u8>(),
                    host_y1_storage.as_mut_ptr(),
                    y1_size,
                );
            }
        }
    }
    'client_extra_send: {
        device_pointer_mode.send(channel_sender).unwrap();
        d1_size.send(channel_sender).unwrap();
        d2_size.send(channel_sender).unwrap();
        x1_size.send(channel_sender).unwrap();
        y1_size.send(channel_sender).unwrap();
        param_size.send(channel_sender).unwrap();
        d1_addr.send(channel_sender).unwrap();
        d2_addr.send(channel_sender).unwrap();
        x1_addr.send(channel_sender).unwrap();
        y1_addr.send(channel_sender).unwrap();
        param_addr.send(channel_sender).unwrap();
        if !device_pointer_mode {
            send_slice(&host_d1_storage[..d1_size], channel_sender).unwrap();
            send_slice(&host_d2_storage[..d2_size], channel_sender).unwrap();
            send_slice(&host_x1_storage[..x1_size], channel_sender).unwrap();
            send_slice(&host_y1_storage[..y1_size], channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut device_pointer_mode = false;
        device_pointer_mode.recv(channel_receiver).unwrap();
        let mut d1_size = 0usize;
        d1_size.recv(channel_receiver).unwrap();
        let mut d2_size = 0usize;
        d2_size.recv(channel_receiver).unwrap();
        let mut x1_size = 0usize;
        x1_size.recv(channel_receiver).unwrap();
        let mut y1_size = 0usize;
        y1_size.recv(channel_receiver).unwrap();
        let mut param_size = 0usize;
        param_size.recv(channel_receiver).unwrap();
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
        #[repr(align(16))]
        struct AlignedScalar([u8; 16]);
        #[repr(align(16))]
        struct AlignedParam([u8; 80]);
        let mut host_d1_storage = AlignedScalar([0; 16]);
        let mut host_d2_storage = AlignedScalar([0; 16]);
        let mut host_x1_storage = AlignedScalar([0; 16]);
        let mut host_y1_storage = AlignedScalar([0; 16]);
        let mut host_param_storage = AlignedParam([0; 80]);
        assert!(d1_size <= host_d1_storage.0.len());
        assert!(d2_size <= host_d2_storage.0.len());
        assert!(x1_size <= host_x1_storage.0.len());
        assert!(y1_size <= host_y1_storage.0.len());
        assert!(param_size <= host_param_storage.0.len());
        if !device_pointer_mode {
            let d1_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            let d2_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            let x1_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            let y1_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert!(d1_bytes.len() == d1_size);
            assert!(d2_bytes.len() == d2_size);
            assert!(x1_bytes.len() == x1_size);
            assert!(y1_bytes.len() == y1_size);
            host_d1_storage.0[..d1_size].copy_from_slice(&d1_bytes);
            host_d2_storage.0[..d2_size].copy_from_slice(&d2_bytes);
            host_x1_storage.0[..x1_size].copy_from_slice(&x1_bytes);
            host_y1_storage.0[..y1_size].copy_from_slice(&y1_bytes);
        }
        let d1_arg = if device_pointer_mode {
            d1_addr as *mut c_void
        } else {
            host_d1_storage.0.as_mut_ptr().cast::<c_void>()
        };
        let d2_arg = if device_pointer_mode {
            d2_addr as *mut c_void
        } else {
            host_d2_storage.0.as_mut_ptr().cast::<c_void>()
        };
        let x1_arg = if device_pointer_mode {
            x1_addr as *mut c_void
        } else {
            host_x1_storage.0.as_mut_ptr().cast::<c_void>()
        };
        let y1_arg = if device_pointer_mode {
            y1_addr as *const c_void
        } else {
            host_y1_storage.0.as_ptr().cast::<c_void>()
        };
        let param_arg = if device_pointer_mode {
            param_addr as *mut c_void
        } else {
            host_param_storage.0.as_mut_ptr().cast::<c_void>()
        };
    }
    'server_execution: {
        let result = unsafe {
            cublasRotmgEx(
                handle,
                d1_arg,
                d1Type,
                d2_arg,
                d2Type,
                x1_arg,
                x1Type,
                y1_arg,
                y1Type,
                param_arg,
                paramType,
                executiontype,
            )
        };
    }
    'server_after_send: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            send_slice(&host_d1_storage.0[..d1_size], channel_sender).unwrap();
            send_slice(&host_d2_storage.0[..d2_size], channel_sender).unwrap();
            send_slice(&host_x1_storage.0[..x1_size], channel_sender).unwrap();
            send_slice(&host_param_storage.0[..param_size], channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == cublasStatus_t::CUBLAS_STATUS_SUCCESS && !device_pointer_mode {
            let host_d1_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            let host_d2_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            let host_x1_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            let host_param_bytes = recv_slice::<u8, _>(channel_receiver).unwrap();
            assert!(host_d1_bytes.len() == d1_size);
            assert!(host_d2_bytes.len() == d2_size);
            assert!(host_x1_bytes.len() == x1_size);
            assert!(host_param_bytes.len() == param_size);
            unsafe {
                std::ptr::copy_nonoverlapping(host_d1_bytes.as_ptr(), d1.cast::<u8>(), d1_size);
                std::ptr::copy_nonoverlapping(host_d2_bytes.as_ptr(), d2.cast::<u8>(), d2_size);
                std::ptr::copy_nonoverlapping(host_x1_bytes.as_ptr(), x1.cast::<u8>(), x1_size);
                std::ptr::copy_nonoverlapping(
                    host_param_bytes.as_ptr(),
                    param.cast::<u8>(),
                    param_size,
                );
            }
        }
    }
}
