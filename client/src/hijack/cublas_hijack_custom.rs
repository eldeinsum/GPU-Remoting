use cudasys::types::cublas::*;
use cudasys::types::cudart::cudaError_t;
use network::type_impl::send_slice;
use network::{Channel, CommChannel, Transportable};
use std::collections::BTreeMap;
use std::ffi::{CStr, CString};
use std::mem::MaybeUninit;
use std::os::raw::*;
use std::sync::{Mutex, OnceLock};

use crate::{ClientThread, CLIENT_THREAD};

fn recv_value<T: Copy, C: CommChannel>(channel: &C) -> T {
    let mut value = MaybeUninit::<T>::uninit();
    value.recv(channel).unwrap();
    unsafe { value.assume_init() }
}

fn legacy_call<R, F>(proc_id: i32, send_args: F) -> R
where
    R: Copy,
    MaybeUninit<R>: Transportable,
    F: FnOnce(&Channel),
{
    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        let ClientThread {
            channel_sender,
            channel_receiver,
            ..
        } = client;

        proc_id.send(channel_sender).unwrap();
        send_args(channel_sender);
        channel_sender.flush_out().unwrap();

        let result = recv_value::<R, _>(channel_receiver);
        channel_receiver.recv_ts().unwrap();
        result
    })
}

fn legacy_void_call<F>(proc_id: i32, send_args: F)
where
    F: FnOnce(&Channel),
{
    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        let ClientThread {
            channel_sender,
            channel_receiver,
            ..
        } = client;

        proc_id.send(channel_sender).unwrap();
        send_args(channel_sender);
        channel_sender.flush_out().unwrap();
        channel_receiver.recv_ts().unwrap();
    });
}

macro_rules! legacy_ret {
    ($name:ident, $proc_id:expr, ($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty) => {
        #[no_mangle]
        pub extern "C" fn $name($($arg: $ty),*) -> $ret {
            legacy_call($proc_id, |channel| {
                $($arg.send(channel).unwrap();)*
            })
        }
    };
}

macro_rules! legacy_void {
    ($name:ident, $proc_id:expr, ($($arg:ident: $ty:ty),* $(,)?)) => {
        #[no_mangle]
        pub extern "C" fn $name($($arg: $ty),*) {
            legacy_void_call($proc_id, |channel| {
                $($arg.send(channel).unwrap();)*
            })
        }
    };
}

legacy_ret!(cublasSnrm2, 1702, (n: c_int, x: *const f32, incx: c_int) -> f32);
legacy_ret!(cublasDnrm2, 1703, (n: c_int, x: *const f64, incx: c_int) -> f64);
legacy_ret!(cublasScnrm2, 1704, (n: c_int, x: *const cuComplex, incx: c_int) -> f32);
legacy_ret!(cublasDznrm2, 1705, (n: c_int, x: *const cuDoubleComplex, incx: c_int) -> f64);
legacy_ret!(cublasSdot, 1706, (n: c_int, x: *const f32, incx: c_int, y: *const f32, incy: c_int) -> f32);
legacy_ret!(cublasDdot, 1707, (n: c_int, x: *const f64, incx: c_int, y: *const f64, incy: c_int) -> f64);
legacy_ret!(cublasCdotu, 1708, (n: c_int, x: *const cuComplex, incx: c_int, y: *const cuComplex, incy: c_int) -> cuComplex);
legacy_ret!(cublasCdotc, 1709, (n: c_int, x: *const cuComplex, incx: c_int, y: *const cuComplex, incy: c_int) -> cuComplex);
legacy_ret!(cublasZdotu, 1710, (n: c_int, x: *const cuDoubleComplex, incx: c_int, y: *const cuDoubleComplex, incy: c_int) -> cuDoubleComplex);
legacy_ret!(cublasZdotc, 1711, (n: c_int, x: *const cuDoubleComplex, incx: c_int, y: *const cuDoubleComplex, incy: c_int) -> cuDoubleComplex);

legacy_void!(cublasSscal, 1712, (n: c_int, alpha: f32, x: *mut f32, incx: c_int));
legacy_void!(cublasDscal, 1713, (n: c_int, alpha: f64, x: *mut f64, incx: c_int));
legacy_void!(cublasCscal, 1714, (n: c_int, alpha: cuComplex, x: *mut cuComplex, incx: c_int));
legacy_void!(cublasZscal, 1715, (n: c_int, alpha: cuDoubleComplex, x: *mut cuDoubleComplex, incx: c_int));
legacy_void!(cublasCsscal, 1716, (n: c_int, alpha: f32, x: *mut cuComplex, incx: c_int));
legacy_void!(cublasZdscal, 1717, (n: c_int, alpha: f64, x: *mut cuDoubleComplex, incx: c_int));
legacy_void!(cublasSaxpy, 1718, (n: c_int, alpha: f32, x: *const f32, incx: c_int, y: *mut f32, incy: c_int));
legacy_void!(cublasDaxpy, 1719, (n: c_int, alpha: f64, x: *const f64, incx: c_int, y: *mut f64, incy: c_int));
legacy_void!(cublasCaxpy, 1720, (n: c_int, alpha: cuComplex, x: *const cuComplex, incx: c_int, y: *mut cuComplex, incy: c_int));
legacy_void!(cublasZaxpy, 1721, (n: c_int, alpha: cuDoubleComplex, x: *const cuDoubleComplex, incx: c_int, y: *mut cuDoubleComplex, incy: c_int));
legacy_void!(cublasScopy, 1722, (n: c_int, x: *const f32, incx: c_int, y: *mut f32, incy: c_int));
legacy_void!(cublasDcopy, 1723, (n: c_int, x: *const f64, incx: c_int, y: *mut f64, incy: c_int));
legacy_void!(cublasCcopy, 1724, (n: c_int, x: *const cuComplex, incx: c_int, y: *mut cuComplex, incy: c_int));
legacy_void!(cublasZcopy, 1725, (n: c_int, x: *const cuDoubleComplex, incx: c_int, y: *mut cuDoubleComplex, incy: c_int));
legacy_void!(cublasSswap, 1726, (n: c_int, x: *mut f32, incx: c_int, y: *mut f32, incy: c_int));
legacy_void!(cublasDswap, 1727, (n: c_int, x: *mut f64, incx: c_int, y: *mut f64, incy: c_int));
legacy_void!(cublasCswap, 1728, (n: c_int, x: *mut cuComplex, incx: c_int, y: *mut cuComplex, incy: c_int));
legacy_void!(cublasZswap, 1729, (n: c_int, x: *mut cuDoubleComplex, incx: c_int, y: *mut cuDoubleComplex, incy: c_int));

legacy_ret!(cublasIsamax, 1730, (n: c_int, x: *const f32, incx: c_int) -> c_int);
legacy_ret!(cublasIdamax, 1731, (n: c_int, x: *const f64, incx: c_int) -> c_int);
legacy_ret!(cublasIcamax, 1732, (n: c_int, x: *const cuComplex, incx: c_int) -> c_int);
legacy_ret!(cublasIzamax, 1733, (n: c_int, x: *const cuDoubleComplex, incx: c_int) -> c_int);
legacy_ret!(cublasIsamin, 1734, (n: c_int, x: *const f32, incx: c_int) -> c_int);
legacy_ret!(cublasIdamin, 1735, (n: c_int, x: *const f64, incx: c_int) -> c_int);
legacy_ret!(cublasIcamin, 1736, (n: c_int, x: *const cuComplex, incx: c_int) -> c_int);
legacy_ret!(cublasIzamin, 1737, (n: c_int, x: *const cuDoubleComplex, incx: c_int) -> c_int);
legacy_ret!(cublasSasum, 1738, (n: c_int, x: *const f32, incx: c_int) -> f32);
legacy_ret!(cublasDasum, 1739, (n: c_int, x: *const f64, incx: c_int) -> f64);
legacy_ret!(cublasScasum, 1740, (n: c_int, x: *const cuComplex, incx: c_int) -> f32);
legacy_ret!(cublasDzasum, 1741, (n: c_int, x: *const cuDoubleComplex, incx: c_int) -> f64);

legacy_void!(cublasSrot, 1742, (n: c_int, x: *mut f32, incx: c_int, y: *mut f32, incy: c_int, sc: f32, ss: f32));
legacy_void!(cublasDrot, 1743, (n: c_int, x: *mut f64, incx: c_int, y: *mut f64, incy: c_int, sc: f64, ss: f64));
legacy_void!(cublasCrot, 1744, (n: c_int, x: *mut cuComplex, incx: c_int, y: *mut cuComplex, incy: c_int, c: f32, s: cuComplex));
legacy_void!(cublasZrot, 1745, (n: c_int, x: *mut cuDoubleComplex, incx: c_int, y: *mut cuDoubleComplex, incy: c_int, sc: f64, cs: cuDoubleComplex));
legacy_void!(cublasCsrot, 1746, (n: c_int, x: *mut cuComplex, incx: c_int, y: *mut cuComplex, incy: c_int, c: f32, s: f32));
legacy_void!(cublasZdrot, 1747, (n: c_int, x: *mut cuDoubleComplex, incx: c_int, y: *mut cuDoubleComplex, incy: c_int, c: f64, s: f64));
legacy_void!(cublasSgemv, 1748, (trans: c_char, m: c_int, n: c_int, alpha: f32, A: *const f32, lda: c_int, x: *const f32, incx: c_int, beta: f32, y: *mut f32, incy: c_int));
legacy_void!(cublasDgemv, 1749, (trans: c_char, m: c_int, n: c_int, alpha: f64, A: *const f64, lda: c_int, x: *const f64, incx: c_int, beta: f64, y: *mut f64, incy: c_int));
legacy_void!(cublasCgemv, 1750, (trans: c_char, m: c_int, n: c_int, alpha: cuComplex, A: *const cuComplex, lda: c_int, x: *const cuComplex, incx: c_int, beta: cuComplex, y: *mut cuComplex, incy: c_int));
legacy_void!(cublasZgemv, 1751, (trans: c_char, m: c_int, n: c_int, alpha: cuDoubleComplex, A: *const cuDoubleComplex, lda: c_int, x: *const cuDoubleComplex, incx: c_int, beta: cuDoubleComplex, y: *mut cuDoubleComplex, incy: c_int));
legacy_void!(cublasSger, 1752, (m: c_int, n: c_int, alpha: f32, x: *const f32, incx: c_int, y: *const f32, incy: c_int, A: *mut f32, lda: c_int));
legacy_void!(cublasDger, 1753, (m: c_int, n: c_int, alpha: f64, x: *const f64, incx: c_int, y: *const f64, incy: c_int, A: *mut f64, lda: c_int));
legacy_void!(cublasCgeru, 1754, (m: c_int, n: c_int, alpha: cuComplex, x: *const cuComplex, incx: c_int, y: *const cuComplex, incy: c_int, A: *mut cuComplex, lda: c_int));
legacy_void!(cublasCgerc, 1755, (m: c_int, n: c_int, alpha: cuComplex, x: *const cuComplex, incx: c_int, y: *const cuComplex, incy: c_int, A: *mut cuComplex, lda: c_int));
legacy_void!(cublasZgeru, 1756, (m: c_int, n: c_int, alpha: cuDoubleComplex, x: *const cuDoubleComplex, incx: c_int, y: *const cuDoubleComplex, incy: c_int, A: *mut cuDoubleComplex, lda: c_int));
legacy_void!(cublasZgerc, 1757, (m: c_int, n: c_int, alpha: cuDoubleComplex, x: *const cuDoubleComplex, incx: c_int, y: *const cuDoubleComplex, incy: c_int, A: *mut cuDoubleComplex, lda: c_int));

#[no_mangle]
pub extern "C" fn cublasLoggerConfigure(
    logIsOn: c_int,
    logToStdOut: c_int,
    logToStdErr: c_int,
    logFileName: *const c_char,
) -> cublasStatus_t {
    legacy_call(1701, |channel| {
        logIsOn.send(channel).unwrap();
        logToStdOut.send(channel).unwrap();
        logToStdErr.send(channel).unwrap();
        let has_log_file = !logFileName.is_null();
        has_log_file.send(channel).unwrap();
        if has_log_file {
            let bytes = unsafe { CStr::from_ptr(logFileName) }.to_bytes_with_nul();
            send_slice(bytes, channel).unwrap();
        }
    })
}

fn status_name(status: cublasStatus_t) -> &'static str {
    match status {
        cublasStatus_t::CUBLAS_STATUS_SUCCESS => "CUBLAS_STATUS_SUCCESS",
        cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => "CUBLAS_STATUS_NOT_INITIALIZED",
        cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED => "CUBLAS_STATUS_ALLOC_FAILED",
        cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE => "CUBLAS_STATUS_INVALID_VALUE",
        cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH => "CUBLAS_STATUS_ARCH_MISMATCH",
        cublasStatus_t::CUBLAS_STATUS_MAPPING_ERROR => "CUBLAS_STATUS_MAPPING_ERROR",
        cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => "CUBLAS_STATUS_EXECUTION_FAILED",
        cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR => "CUBLAS_STATUS_INTERNAL_ERROR",
        cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED => "CUBLAS_STATUS_NOT_SUPPORTED",
        cublasStatus_t::CUBLAS_STATUS_LICENSE_ERROR => "CUBLAS_STATUS_LICENSE_ERROR",
    }
}

fn status_description(status: cublasStatus_t) -> &'static str {
    match status {
        cublasStatus_t::CUBLAS_STATUS_SUCCESS => "success",
        cublasStatus_t::CUBLAS_STATUS_NOT_INITIALIZED => "the cuBLAS library was not initialized",
        cublasStatus_t::CUBLAS_STATUS_ALLOC_FAILED => "resource allocation failed",
        cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE => {
            "an unsupported value or parameter was passed"
        }
        cublasStatus_t::CUBLAS_STATUS_ARCH_MISMATCH => "the device architecture is not supported",
        cublasStatus_t::CUBLAS_STATUS_MAPPING_ERROR => "access to GPU memory space failed",
        cublasStatus_t::CUBLAS_STATUS_EXECUTION_FAILED => "the GPU program failed to execute",
        cublasStatus_t::CUBLAS_STATUS_INTERNAL_ERROR => "an internal cuBLAS operation failed",
        cublasStatus_t::CUBLAS_STATUS_NOT_SUPPORTED => "the requested operation is not supported",
        cublasStatus_t::CUBLAS_STATUS_LICENSE_ERROR => "license check failed",
    }
}

fn cached_status_text(status: cublasStatus_t, description: bool) -> *const c_char {
    static STATUS_TEXTS: OnceLock<Mutex<BTreeMap<(c_int, bool), CString>>> = OnceLock::new();
    let code = status as c_int;
    let mut texts = STATUS_TEXTS
        .get_or_init(|| Mutex::new(BTreeMap::new()))
        .lock()
        .unwrap();
    let text = texts.entry((code, description)).or_insert_with(|| {
        CString::new(if description {
            status_description(status)
        } else {
            status_name(status)
        })
        .unwrap()
    });
    text.as_ptr()
}

#[no_mangle]
pub extern "C" fn cublasGetStatusName(status: cublasStatus_t) -> *const c_char {
    cached_status_text(status, false)
}

#[no_mangle]
pub extern "C" fn cublasGetStatusString(status: cublasStatus_t) -> *const c_char {
    cached_status_text(status, true)
}

#[no_mangle]
pub extern "C" fn cublasGetCudartVersion() -> usize {
    let mut version = 0;
    let result = super::cudart_hijack::cudaRuntimeGetVersion(&mut version);
    if result == cudaError_t::cudaSuccess {
        version as usize
    } else {
        0
    }
}
