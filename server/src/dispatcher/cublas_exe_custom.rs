#![expect(non_snake_case)]

use crate::ServerWorker;
use cudasys::cublas::*;
use network::type_impl::recv_slice;
use network::{CommChannel, Transportable};
use std::ffi::CString;
use std::mem::MaybeUninit;
use std::os::raw::{c_char, c_int};

fn recv_value<T: Copy, C: CommChannel>(channel: &C) -> T {
    let mut value = MaybeUninit::<T>::uninit();
    value.recv(channel).unwrap();
    unsafe { value.assume_init() }
}

macro_rules! legacy_ret_exe {
    ($exe:ident, $real:ident, ($($arg:ident: $ty:ty),* $(,)?) -> $ret:ty) => {
        pub fn $exe<C: CommChannel>(server: &mut ServerWorker<C>) {
            $(let $arg: $ty = recv_value(&server.channel_receiver);)*
            server.channel_receiver.recv_ts().unwrap();
            let result: $ret = unsafe { $real($($arg),*) };
            result.send(&server.channel_sender).unwrap();
            server.channel_sender.flush_out().unwrap();
        }
    };
}

macro_rules! legacy_void_exe {
    ($exe:ident, $real:ident, ($($arg:ident: $ty:ty),* $(,)?)) => {
        pub fn $exe<C: CommChannel>(server: &mut ServerWorker<C>) {
            $(let $arg: $ty = recv_value(&server.channel_receiver);)*
            server.channel_receiver.recv_ts().unwrap();
            unsafe { $real($($arg),*) };
            server.channel_sender.flush_out().unwrap();
        }
    };
}

pub fn cublasLoggerConfigureExe<C: CommChannel>(server: &mut ServerWorker<C>) {
    let logIsOn: c_int = recv_value(&server.channel_receiver);
    let logToStdOut: c_int = recv_value(&server.channel_receiver);
    let logToStdErr: c_int = recv_value(&server.channel_receiver);
    let has_log_file: bool = recv_value(&server.channel_receiver);
    let log_file = if has_log_file {
        let bytes = recv_slice::<u8, _>(&server.channel_receiver).unwrap();
        Some(CString::from_vec_with_nul(bytes.into_vec()).unwrap())
    } else {
        None
    };
    server.channel_receiver.recv_ts().unwrap();

    let logFileName = log_file
        .as_ref()
        .map_or(std::ptr::null::<c_char>(), |name| name.as_ptr());
    let result = unsafe { cublasLoggerConfigure(logIsOn, logToStdOut, logToStdErr, logFileName) };
    result.send(&server.channel_sender).unwrap();
    server.channel_sender.flush_out().unwrap();
}

legacy_ret_exe!(cublasSnrm2Exe, cublasSnrm2, (n: c_int, x: *const f32, incx: c_int) -> f32);
legacy_ret_exe!(cublasDnrm2Exe, cublasDnrm2, (n: c_int, x: *const f64, incx: c_int) -> f64);
legacy_ret_exe!(cublasScnrm2Exe, cublasScnrm2, (n: c_int, x: *const cuComplex, incx: c_int) -> f32);
legacy_ret_exe!(cublasDznrm2Exe, cublasDznrm2, (n: c_int, x: *const cuDoubleComplex, incx: c_int) -> f64);
legacy_ret_exe!(cublasSdotExe, cublasSdot, (n: c_int, x: *const f32, incx: c_int, y: *const f32, incy: c_int) -> f32);
legacy_ret_exe!(cublasDdotExe, cublasDdot, (n: c_int, x: *const f64, incx: c_int, y: *const f64, incy: c_int) -> f64);
legacy_ret_exe!(cublasCdotuExe, cublasCdotu, (n: c_int, x: *const cuComplex, incx: c_int, y: *const cuComplex, incy: c_int) -> cuComplex);
legacy_ret_exe!(cublasCdotcExe, cublasCdotc, (n: c_int, x: *const cuComplex, incx: c_int, y: *const cuComplex, incy: c_int) -> cuComplex);
legacy_ret_exe!(cublasZdotuExe, cublasZdotu, (n: c_int, x: *const cuDoubleComplex, incx: c_int, y: *const cuDoubleComplex, incy: c_int) -> cuDoubleComplex);
legacy_ret_exe!(cublasZdotcExe, cublasZdotc, (n: c_int, x: *const cuDoubleComplex, incx: c_int, y: *const cuDoubleComplex, incy: c_int) -> cuDoubleComplex);

legacy_void_exe!(cublasSscalExe, cublasSscal, (n: c_int, alpha: f32, x: *mut f32, incx: c_int));
legacy_void_exe!(cublasDscalExe, cublasDscal, (n: c_int, alpha: f64, x: *mut f64, incx: c_int));
legacy_void_exe!(cublasCscalExe, cublasCscal, (n: c_int, alpha: cuComplex, x: *mut cuComplex, incx: c_int));
legacy_void_exe!(cublasZscalExe, cublasZscal, (n: c_int, alpha: cuDoubleComplex, x: *mut cuDoubleComplex, incx: c_int));
legacy_void_exe!(cublasCsscalExe, cublasCsscal, (n: c_int, alpha: f32, x: *mut cuComplex, incx: c_int));
legacy_void_exe!(cublasZdscalExe, cublasZdscal, (n: c_int, alpha: f64, x: *mut cuDoubleComplex, incx: c_int));
legacy_void_exe!(cublasSaxpyExe, cublasSaxpy, (n: c_int, alpha: f32, x: *const f32, incx: c_int, y: *mut f32, incy: c_int));
legacy_void_exe!(cublasDaxpyExe, cublasDaxpy, (n: c_int, alpha: f64, x: *const f64, incx: c_int, y: *mut f64, incy: c_int));
legacy_void_exe!(cublasCaxpyExe, cublasCaxpy, (n: c_int, alpha: cuComplex, x: *const cuComplex, incx: c_int, y: *mut cuComplex, incy: c_int));
legacy_void_exe!(cublasZaxpyExe, cublasZaxpy, (n: c_int, alpha: cuDoubleComplex, x: *const cuDoubleComplex, incx: c_int, y: *mut cuDoubleComplex, incy: c_int));
legacy_void_exe!(cublasScopyExe, cublasScopy, (n: c_int, x: *const f32, incx: c_int, y: *mut f32, incy: c_int));
legacy_void_exe!(cublasDcopyExe, cublasDcopy, (n: c_int, x: *const f64, incx: c_int, y: *mut f64, incy: c_int));
legacy_void_exe!(cublasCcopyExe, cublasCcopy, (n: c_int, x: *const cuComplex, incx: c_int, y: *mut cuComplex, incy: c_int));
legacy_void_exe!(cublasZcopyExe, cublasZcopy, (n: c_int, x: *const cuDoubleComplex, incx: c_int, y: *mut cuDoubleComplex, incy: c_int));
legacy_void_exe!(cublasSswapExe, cublasSswap, (n: c_int, x: *mut f32, incx: c_int, y: *mut f32, incy: c_int));
legacy_void_exe!(cublasDswapExe, cublasDswap, (n: c_int, x: *mut f64, incx: c_int, y: *mut f64, incy: c_int));
legacy_void_exe!(cublasCswapExe, cublasCswap, (n: c_int, x: *mut cuComplex, incx: c_int, y: *mut cuComplex, incy: c_int));
legacy_void_exe!(cublasZswapExe, cublasZswap, (n: c_int, x: *mut cuDoubleComplex, incx: c_int, y: *mut cuDoubleComplex, incy: c_int));

legacy_ret_exe!(cublasIsamaxExe, cublasIsamax, (n: c_int, x: *const f32, incx: c_int) -> c_int);
legacy_ret_exe!(cublasIdamaxExe, cublasIdamax, (n: c_int, x: *const f64, incx: c_int) -> c_int);
legacy_ret_exe!(cublasIcamaxExe, cublasIcamax, (n: c_int, x: *const cuComplex, incx: c_int) -> c_int);
legacy_ret_exe!(cublasIzamaxExe, cublasIzamax, (n: c_int, x: *const cuDoubleComplex, incx: c_int) -> c_int);
legacy_ret_exe!(cublasIsaminExe, cublasIsamin, (n: c_int, x: *const f32, incx: c_int) -> c_int);
legacy_ret_exe!(cublasIdaminExe, cublasIdamin, (n: c_int, x: *const f64, incx: c_int) -> c_int);
legacy_ret_exe!(cublasIcaminExe, cublasIcamin, (n: c_int, x: *const cuComplex, incx: c_int) -> c_int);
legacy_ret_exe!(cublasIzaminExe, cublasIzamin, (n: c_int, x: *const cuDoubleComplex, incx: c_int) -> c_int);
legacy_ret_exe!(cublasSasumExe, cublasSasum, (n: c_int, x: *const f32, incx: c_int) -> f32);
legacy_ret_exe!(cublasDasumExe, cublasDasum, (n: c_int, x: *const f64, incx: c_int) -> f64);
legacy_ret_exe!(cublasScasumExe, cublasScasum, (n: c_int, x: *const cuComplex, incx: c_int) -> f32);
legacy_ret_exe!(cublasDzasumExe, cublasDzasum, (n: c_int, x: *const cuDoubleComplex, incx: c_int) -> f64);

legacy_void_exe!(cublasSrotExe, cublasSrot, (n: c_int, x: *mut f32, incx: c_int, y: *mut f32, incy: c_int, sc: f32, ss: f32));
legacy_void_exe!(cublasDrotExe, cublasDrot, (n: c_int, x: *mut f64, incx: c_int, y: *mut f64, incy: c_int, sc: f64, ss: f64));
legacy_void_exe!(cublasCrotExe, cublasCrot, (n: c_int, x: *mut cuComplex, incx: c_int, y: *mut cuComplex, incy: c_int, c: f32, s: cuComplex));
legacy_void_exe!(cublasZrotExe, cublasZrot, (n: c_int, x: *mut cuDoubleComplex, incx: c_int, y: *mut cuDoubleComplex, incy: c_int, sc: f64, cs: cuDoubleComplex));
legacy_void_exe!(cublasCsrotExe, cublasCsrot, (n: c_int, x: *mut cuComplex, incx: c_int, y: *mut cuComplex, incy: c_int, c: f32, s: f32));
legacy_void_exe!(cublasZdrotExe, cublasZdrot, (n: c_int, x: *mut cuDoubleComplex, incx: c_int, y: *mut cuDoubleComplex, incy: c_int, c: f64, s: f64));
legacy_void_exe!(cublasSgemvExe, cublasSgemv, (trans: c_char, m: c_int, n: c_int, alpha: f32, A: *const f32, lda: c_int, x: *const f32, incx: c_int, beta: f32, y: *mut f32, incy: c_int));
legacy_void_exe!(cublasDgemvExe, cublasDgemv, (trans: c_char, m: c_int, n: c_int, alpha: f64, A: *const f64, lda: c_int, x: *const f64, incx: c_int, beta: f64, y: *mut f64, incy: c_int));
legacy_void_exe!(cublasCgemvExe, cublasCgemv, (trans: c_char, m: c_int, n: c_int, alpha: cuComplex, A: *const cuComplex, lda: c_int, x: *const cuComplex, incx: c_int, beta: cuComplex, y: *mut cuComplex, incy: c_int));
legacy_void_exe!(cublasZgemvExe, cublasZgemv, (trans: c_char, m: c_int, n: c_int, alpha: cuDoubleComplex, A: *const cuDoubleComplex, lda: c_int, x: *const cuDoubleComplex, incx: c_int, beta: cuDoubleComplex, y: *mut cuDoubleComplex, incy: c_int));
legacy_void_exe!(cublasSgerExe, cublasSger, (m: c_int, n: c_int, alpha: f32, x: *const f32, incx: c_int, y: *const f32, incy: c_int, A: *mut f32, lda: c_int));
legacy_void_exe!(cublasDgerExe, cublasDger, (m: c_int, n: c_int, alpha: f64, x: *const f64, incx: c_int, y: *const f64, incy: c_int, A: *mut f64, lda: c_int));
legacy_void_exe!(cublasCgeruExe, cublasCgeru, (m: c_int, n: c_int, alpha: cuComplex, x: *const cuComplex, incx: c_int, y: *const cuComplex, incy: c_int, A: *mut cuComplex, lda: c_int));
legacy_void_exe!(cublasCgercExe, cublasCgerc, (m: c_int, n: c_int, alpha: cuComplex, x: *const cuComplex, incx: c_int, y: *const cuComplex, incy: c_int, A: *mut cuComplex, lda: c_int));
legacy_void_exe!(cublasZgeruExe, cublasZgeru, (m: c_int, n: c_int, alpha: cuDoubleComplex, x: *const cuDoubleComplex, incx: c_int, y: *const cuDoubleComplex, incy: c_int, A: *mut cuDoubleComplex, lda: c_int));
legacy_void_exe!(cublasZgercExe, cublasZgerc, (m: c_int, n: c_int, alpha: cuDoubleComplex, x: *const cuDoubleComplex, incx: c_int, y: *const cuDoubleComplex, incy: c_int, A: *mut cuDoubleComplex, lda: c_int));
