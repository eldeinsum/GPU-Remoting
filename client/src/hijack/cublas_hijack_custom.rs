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

fn send_host_array5<T>(ptr: *const T, channel: &Channel)
where
    T: Copy + Transportable,
{
    for i in 0..5 {
        unsafe { *ptr.add(i) }.send(channel).unwrap();
    }
}

fn recv_host_array5<T>(channel: &Channel, ptr: *mut T)
where
    T: Copy,
    MaybeUninit<T>: Transportable,
{
    for i in 0..5 {
        let value = recv_value::<T, _>(channel);
        unsafe {
            *ptr.add(i) = value;
        }
    }
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

#[no_mangle]
pub extern "C" fn cublasSrotg(sa: *mut f32, sb: *mut f32, sc: *mut f32, ss: *mut f32) {
    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        let ClientThread {
            channel_sender,
            channel_receiver,
            ..
        } = client;

        1906.send(channel_sender).unwrap();
        unsafe {
            (*sa).send(channel_sender).unwrap();
            (*sb).send(channel_sender).unwrap();
            (*sc).send(channel_sender).unwrap();
            (*ss).send(channel_sender).unwrap();
        }
        channel_sender.flush_out().unwrap();

        let new_sa = recv_value::<f32, _>(channel_receiver);
        let new_sb = recv_value::<f32, _>(channel_receiver);
        let new_sc = recv_value::<f32, _>(channel_receiver);
        let new_ss = recv_value::<f32, _>(channel_receiver);
        channel_receiver.recv_ts().unwrap();
        unsafe {
            *sa = new_sa;
            *sb = new_sb;
            *sc = new_sc;
            *ss = new_ss;
        }
    });
}

#[no_mangle]
pub extern "C" fn cublasDrotg(sa: *mut f64, sb: *mut f64, sc: *mut f64, ss: *mut f64) {
    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        let ClientThread {
            channel_sender,
            channel_receiver,
            ..
        } = client;

        1907.send(channel_sender).unwrap();
        unsafe {
            (*sa).send(channel_sender).unwrap();
            (*sb).send(channel_sender).unwrap();
            (*sc).send(channel_sender).unwrap();
            (*ss).send(channel_sender).unwrap();
        }
        channel_sender.flush_out().unwrap();

        let new_sa = recv_value::<f64, _>(channel_receiver);
        let new_sb = recv_value::<f64, _>(channel_receiver);
        let new_sc = recv_value::<f64, _>(channel_receiver);
        let new_ss = recv_value::<f64, _>(channel_receiver);
        channel_receiver.recv_ts().unwrap();
        unsafe {
            *sa = new_sa;
            *sb = new_sb;
            *sc = new_sc;
            *ss = new_ss;
        }
    });
}

#[no_mangle]
pub extern "C" fn cublasCrotg(ca: *mut cuComplex, cb: cuComplex, sc: *mut f32, cs: *mut cuComplex) {
    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        let ClientThread {
            channel_sender,
            channel_receiver,
            ..
        } = client;

        1908.send(channel_sender).unwrap();
        unsafe {
            (*ca).send(channel_sender).unwrap();
            cb.send(channel_sender).unwrap();
            (*sc).send(channel_sender).unwrap();
            (*cs).send(channel_sender).unwrap();
        }
        channel_sender.flush_out().unwrap();

        let new_ca = recv_value::<cuComplex, _>(channel_receiver);
        let new_sc = recv_value::<f32, _>(channel_receiver);
        let new_cs = recv_value::<cuComplex, _>(channel_receiver);
        channel_receiver.recv_ts().unwrap();
        unsafe {
            *ca = new_ca;
            *sc = new_sc;
            *cs = new_cs;
        }
    });
}

#[no_mangle]
pub extern "C" fn cublasZrotg(
    ca: *mut cuDoubleComplex,
    cb: cuDoubleComplex,
    sc: *mut f64,
    cs: *mut cuDoubleComplex,
) {
    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        let ClientThread {
            channel_sender,
            channel_receiver,
            ..
        } = client;

        1909.send(channel_sender).unwrap();
        unsafe {
            (*ca).send(channel_sender).unwrap();
            cb.send(channel_sender).unwrap();
            (*sc).send(channel_sender).unwrap();
            (*cs).send(channel_sender).unwrap();
        }
        channel_sender.flush_out().unwrap();

        let new_ca = recv_value::<cuDoubleComplex, _>(channel_receiver);
        let new_sc = recv_value::<f64, _>(channel_receiver);
        let new_cs = recv_value::<cuDoubleComplex, _>(channel_receiver);
        channel_receiver.recv_ts().unwrap();
        unsafe {
            *ca = new_ca;
            *sc = new_sc;
            *cs = new_cs;
        }
    });
}

#[no_mangle]
pub extern "C" fn cublasSrotm(
    n: c_int,
    x: *mut f32,
    incx: c_int,
    y: *mut f32,
    incy: c_int,
    sparam: *const f32,
) {
    legacy_void_call(1910, |channel| {
        n.send(channel).unwrap();
        x.send(channel).unwrap();
        incx.send(channel).unwrap();
        y.send(channel).unwrap();
        incy.send(channel).unwrap();
        send_host_array5(sparam, channel);
    });
}

#[no_mangle]
pub extern "C" fn cublasDrotm(
    n: c_int,
    x: *mut f64,
    incx: c_int,
    y: *mut f64,
    incy: c_int,
    sparam: *const f64,
) {
    legacy_void_call(1911, |channel| {
        n.send(channel).unwrap();
        x.send(channel).unwrap();
        incx.send(channel).unwrap();
        y.send(channel).unwrap();
        incy.send(channel).unwrap();
        send_host_array5(sparam, channel);
    });
}

#[no_mangle]
pub extern "C" fn cublasSrotmg(
    sd1: *mut f32,
    sd2: *mut f32,
    sx1: *mut f32,
    sy1: *const f32,
    sparam: *mut f32,
) {
    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        let ClientThread {
            channel_sender,
            channel_receiver,
            ..
        } = client;

        1912.send(channel_sender).unwrap();
        unsafe {
            (*sd1).send(channel_sender).unwrap();
            (*sd2).send(channel_sender).unwrap();
            (*sx1).send(channel_sender).unwrap();
            (*sy1).send(channel_sender).unwrap();
        }
        send_host_array5(sparam, channel_sender);
        channel_sender.flush_out().unwrap();

        let new_sd1 = recv_value::<f32, _>(channel_receiver);
        let new_sd2 = recv_value::<f32, _>(channel_receiver);
        let new_sx1 = recv_value::<f32, _>(channel_receiver);
        recv_host_array5(channel_receiver, sparam);
        channel_receiver.recv_ts().unwrap();
        unsafe {
            *sd1 = new_sd1;
            *sd2 = new_sd2;
            *sx1 = new_sx1;
        }
    });
}

#[no_mangle]
pub extern "C" fn cublasDrotmg(
    sd1: *mut f64,
    sd2: *mut f64,
    sx1: *mut f64,
    sy1: *const f64,
    sparam: *mut f64,
) {
    CLIENT_THREAD.with_borrow_mut(|client| {
        client.ensure_current_process();
        let ClientThread {
            channel_sender,
            channel_receiver,
            ..
        } = client;

        1913.send(channel_sender).unwrap();
        unsafe {
            (*sd1).send(channel_sender).unwrap();
            (*sd2).send(channel_sender).unwrap();
            (*sx1).send(channel_sender).unwrap();
            (*sy1).send(channel_sender).unwrap();
        }
        send_host_array5(sparam, channel_sender);
        channel_sender.flush_out().unwrap();

        let new_sd1 = recv_value::<f64, _>(channel_receiver);
        let new_sd2 = recv_value::<f64, _>(channel_receiver);
        let new_sx1 = recv_value::<f64, _>(channel_receiver);
        recv_host_array5(channel_receiver, sparam);
        channel_receiver.recv_ts().unwrap();
        unsafe {
            *sd1 = new_sd1;
            *sd2 = new_sd2;
            *sx1 = new_sx1;
        }
    });
}

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
legacy_void!(cublasStrmv, 1758, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, A: *const f32, lda: c_int, x: *mut f32, incx: c_int));
legacy_void!(cublasDtrmv, 1759, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, A: *const f64, lda: c_int, x: *mut f64, incx: c_int));
legacy_void!(cublasCtrmv, 1760, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, A: *const cuComplex, lda: c_int, x: *mut cuComplex, incx: c_int));
legacy_void!(cublasZtrmv, 1761, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, A: *const cuDoubleComplex, lda: c_int, x: *mut cuDoubleComplex, incx: c_int));
legacy_void!(cublasStbmv, 1762, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, k: c_int, A: *const f32, lda: c_int, x: *mut f32, incx: c_int));
legacy_void!(cublasDtbmv, 1763, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, k: c_int, A: *const f64, lda: c_int, x: *mut f64, incx: c_int));
legacy_void!(cublasCtbmv, 1764, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, k: c_int, A: *const cuComplex, lda: c_int, x: *mut cuComplex, incx: c_int));
legacy_void!(cublasZtbmv, 1765, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, k: c_int, A: *const cuDoubleComplex, lda: c_int, x: *mut cuDoubleComplex, incx: c_int));
legacy_void!(cublasStpmv, 1766, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, AP: *const f32, x: *mut f32, incx: c_int));
legacy_void!(cublasDtpmv, 1767, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, AP: *const f64, x: *mut f64, incx: c_int));
legacy_void!(cublasCtpmv, 1768, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, AP: *const cuComplex, x: *mut cuComplex, incx: c_int));
legacy_void!(cublasZtpmv, 1769, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, AP: *const cuDoubleComplex, x: *mut cuDoubleComplex, incx: c_int));
legacy_void!(cublasStrsv, 1770, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, A: *const f32, lda: c_int, x: *mut f32, incx: c_int));
legacy_void!(cublasDtrsv, 1771, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, A: *const f64, lda: c_int, x: *mut f64, incx: c_int));
legacy_void!(cublasCtrsv, 1772, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, A: *const cuComplex, lda: c_int, x: *mut cuComplex, incx: c_int));
legacy_void!(cublasZtrsv, 1773, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, A: *const cuDoubleComplex, lda: c_int, x: *mut cuDoubleComplex, incx: c_int));
legacy_void!(cublasStbsv, 1774, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, k: c_int, A: *const f32, lda: c_int, x: *mut f32, incx: c_int));
legacy_void!(cublasDtbsv, 1775, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, k: c_int, A: *const f64, lda: c_int, x: *mut f64, incx: c_int));
legacy_void!(cublasCtbsv, 1776, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, k: c_int, A: *const cuComplex, lda: c_int, x: *mut cuComplex, incx: c_int));
legacy_void!(cublasZtbsv, 1777, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, k: c_int, A: *const cuDoubleComplex, lda: c_int, x: *mut cuDoubleComplex, incx: c_int));
legacy_void!(cublasStpsv, 1778, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, AP: *const f32, x: *mut f32, incx: c_int));
legacy_void!(cublasDtpsv, 1779, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, AP: *const f64, x: *mut f64, incx: c_int));
legacy_void!(cublasCtpsv, 1780, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, AP: *const cuComplex, x: *mut cuComplex, incx: c_int));
legacy_void!(cublasZtpsv, 1781, (uplo: c_char, trans: c_char, diag: c_char, n: c_int, AP: *const cuDoubleComplex, x: *mut cuDoubleComplex, incx: c_int));
legacy_void!(cublasSgbmv, 1827, (trans: c_char, m: c_int, n: c_int, kl: c_int, ku: c_int, alpha: f32, A: *const f32, lda: c_int, x: *const f32, incx: c_int, beta: f32, y: *mut f32, incy: c_int));
legacy_void!(cublasDgbmv, 1828, (trans: c_char, m: c_int, n: c_int, kl: c_int, ku: c_int, alpha: f64, A: *const f64, lda: c_int, x: *const f64, incx: c_int, beta: f64, y: *mut f64, incy: c_int));
legacy_void!(cublasCgbmv, 1829, (trans: c_char, m: c_int, n: c_int, kl: c_int, ku: c_int, alpha: cuComplex, A: *const cuComplex, lda: c_int, x: *const cuComplex, incx: c_int, beta: cuComplex, y: *mut cuComplex, incy: c_int));
legacy_void!(cublasZgbmv, 1830, (trans: c_char, m: c_int, n: c_int, kl: c_int, ku: c_int, alpha: cuDoubleComplex, A: *const cuDoubleComplex, lda: c_int, x: *const cuDoubleComplex, incx: c_int, beta: cuDoubleComplex, y: *mut cuDoubleComplex, incy: c_int));
legacy_void!(cublasSsymv, 1831, (uplo: c_char, n: c_int, alpha: f32, A: *const f32, lda: c_int, x: *const f32, incx: c_int, beta: f32, y: *mut f32, incy: c_int));
legacy_void!(cublasDsymv, 1832, (uplo: c_char, n: c_int, alpha: f64, A: *const f64, lda: c_int, x: *const f64, incx: c_int, beta: f64, y: *mut f64, incy: c_int));
legacy_void!(cublasChemv, 1833, (uplo: c_char, n: c_int, alpha: cuComplex, A: *const cuComplex, lda: c_int, x: *const cuComplex, incx: c_int, beta: cuComplex, y: *mut cuComplex, incy: c_int));
legacy_void!(cublasZhemv, 1834, (uplo: c_char, n: c_int, alpha: cuDoubleComplex, A: *const cuDoubleComplex, lda: c_int, x: *const cuDoubleComplex, incx: c_int, beta: cuDoubleComplex, y: *mut cuDoubleComplex, incy: c_int));
legacy_void!(cublasSsbmv, 1835, (uplo: c_char, n: c_int, k: c_int, alpha: f32, A: *const f32, lda: c_int, x: *const f32, incx: c_int, beta: f32, y: *mut f32, incy: c_int));
legacy_void!(cublasDsbmv, 1836, (uplo: c_char, n: c_int, k: c_int, alpha: f64, A: *const f64, lda: c_int, x: *const f64, incx: c_int, beta: f64, y: *mut f64, incy: c_int));
legacy_void!(cublasChbmv, 1837, (uplo: c_char, n: c_int, k: c_int, alpha: cuComplex, A: *const cuComplex, lda: c_int, x: *const cuComplex, incx: c_int, beta: cuComplex, y: *mut cuComplex, incy: c_int));
legacy_void!(cublasZhbmv, 1838, (uplo: c_char, n: c_int, k: c_int, alpha: cuDoubleComplex, A: *const cuDoubleComplex, lda: c_int, x: *const cuDoubleComplex, incx: c_int, beta: cuDoubleComplex, y: *mut cuDoubleComplex, incy: c_int));
legacy_void!(cublasSspmv, 1839, (uplo: c_char, n: c_int, alpha: f32, AP: *const f32, x: *const f32, incx: c_int, beta: f32, y: *mut f32, incy: c_int));
legacy_void!(cublasDspmv, 1840, (uplo: c_char, n: c_int, alpha: f64, AP: *const f64, x: *const f64, incx: c_int, beta: f64, y: *mut f64, incy: c_int));
legacy_void!(cublasChpmv, 1841, (uplo: c_char, n: c_int, alpha: cuComplex, AP: *const cuComplex, x: *const cuComplex, incx: c_int, beta: cuComplex, y: *mut cuComplex, incy: c_int));
legacy_void!(cublasZhpmv, 1842, (uplo: c_char, n: c_int, alpha: cuDoubleComplex, AP: *const cuDoubleComplex, x: *const cuDoubleComplex, incx: c_int, beta: cuDoubleComplex, y: *mut cuDoubleComplex, incy: c_int));
legacy_void!(cublasSsyr, 1843, (uplo: c_char, n: c_int, alpha: f32, x: *const f32, incx: c_int, A: *mut f32, lda: c_int));
legacy_void!(cublasDsyr, 1844, (uplo: c_char, n: c_int, alpha: f64, x: *const f64, incx: c_int, A: *mut f64, lda: c_int));
legacy_void!(cublasCher, 1845, (uplo: c_char, n: c_int, alpha: f32, x: *const cuComplex, incx: c_int, A: *mut cuComplex, lda: c_int));
legacy_void!(cublasZher, 1846, (uplo: c_char, n: c_int, alpha: f64, x: *const cuDoubleComplex, incx: c_int, A: *mut cuDoubleComplex, lda: c_int));
legacy_void!(cublasSspr, 1847, (uplo: c_char, n: c_int, alpha: f32, x: *const f32, incx: c_int, AP: *mut f32));
legacy_void!(cublasDspr, 1848, (uplo: c_char, n: c_int, alpha: f64, x: *const f64, incx: c_int, AP: *mut f64));
legacy_void!(cublasChpr, 1849, (uplo: c_char, n: c_int, alpha: f32, x: *const cuComplex, incx: c_int, AP: *mut cuComplex));
legacy_void!(cublasZhpr, 1850, (uplo: c_char, n: c_int, alpha: f64, x: *const cuDoubleComplex, incx: c_int, AP: *mut cuDoubleComplex));
legacy_void!(cublasSsyr2, 1851, (uplo: c_char, n: c_int, alpha: f32, x: *const f32, incx: c_int, y: *const f32, incy: c_int, A: *mut f32, lda: c_int));
legacy_void!(cublasDsyr2, 1852, (uplo: c_char, n: c_int, alpha: f64, x: *const f64, incx: c_int, y: *const f64, incy: c_int, A: *mut f64, lda: c_int));
legacy_void!(cublasCher2, 1853, (uplo: c_char, n: c_int, alpha: cuComplex, x: *const cuComplex, incx: c_int, y: *const cuComplex, incy: c_int, A: *mut cuComplex, lda: c_int));
legacy_void!(cublasZher2, 1854, (uplo: c_char, n: c_int, alpha: cuDoubleComplex, x: *const cuDoubleComplex, incx: c_int, y: *const cuDoubleComplex, incy: c_int, A: *mut cuDoubleComplex, lda: c_int));
legacy_void!(cublasSspr2, 1855, (uplo: c_char, n: c_int, alpha: f32, x: *const f32, incx: c_int, y: *const f32, incy: c_int, AP: *mut f32));
legacy_void!(cublasDspr2, 1856, (uplo: c_char, n: c_int, alpha: f64, x: *const f64, incx: c_int, y: *const f64, incy: c_int, AP: *mut f64));
legacy_void!(cublasChpr2, 1857, (uplo: c_char, n: c_int, alpha: cuComplex, x: *const cuComplex, incx: c_int, y: *const cuComplex, incy: c_int, AP: *mut cuComplex));
legacy_void!(cublasZhpr2, 1858, (uplo: c_char, n: c_int, alpha: cuDoubleComplex, x: *const cuDoubleComplex, incx: c_int, y: *const cuDoubleComplex, incy: c_int, AP: *mut cuDoubleComplex));
legacy_void!(cublasSgemm, 1876, (transa: c_char, transb: c_char, m: c_int, n: c_int, k: c_int, alpha: f32, A: *const f32, lda: c_int, B: *const f32, ldb: c_int, beta: f32, C: *mut f32, ldc: c_int));
legacy_void!(cublasDgemm, 1877, (transa: c_char, transb: c_char, m: c_int, n: c_int, k: c_int, alpha: f64, A: *const f64, lda: c_int, B: *const f64, ldb: c_int, beta: f64, C: *mut f64, ldc: c_int));
legacy_void!(cublasCgemm, 1878, (transa: c_char, transb: c_char, m: c_int, n: c_int, k: c_int, alpha: cuComplex, A: *const cuComplex, lda: c_int, B: *const cuComplex, ldb: c_int, beta: cuComplex, C: *mut cuComplex, ldc: c_int));
legacy_void!(cublasZgemm, 1879, (transa: c_char, transb: c_char, m: c_int, n: c_int, k: c_int, alpha: cuDoubleComplex, A: *const cuDoubleComplex, lda: c_int, B: *const cuDoubleComplex, ldb: c_int, beta: cuDoubleComplex, C: *mut cuDoubleComplex, ldc: c_int));
legacy_void!(cublasSsymm, 1880, (side: c_char, uplo: c_char, m: c_int, n: c_int, alpha: f32, A: *const f32, lda: c_int, B: *const f32, ldb: c_int, beta: f32, C: *mut f32, ldc: c_int));
legacy_void!(cublasDsymm, 1881, (side: c_char, uplo: c_char, m: c_int, n: c_int, alpha: f64, A: *const f64, lda: c_int, B: *const f64, ldb: c_int, beta: f64, C: *mut f64, ldc: c_int));
legacy_void!(cublasCsymm, 1882, (side: c_char, uplo: c_char, m: c_int, n: c_int, alpha: cuComplex, A: *const cuComplex, lda: c_int, B: *const cuComplex, ldb: c_int, beta: cuComplex, C: *mut cuComplex, ldc: c_int));
legacy_void!(cublasZsymm, 1883, (side: c_char, uplo: c_char, m: c_int, n: c_int, alpha: cuDoubleComplex, A: *const cuDoubleComplex, lda: c_int, B: *const cuDoubleComplex, ldb: c_int, beta: cuDoubleComplex, C: *mut cuDoubleComplex, ldc: c_int));
legacy_void!(cublasChemm, 1884, (side: c_char, uplo: c_char, m: c_int, n: c_int, alpha: cuComplex, A: *const cuComplex, lda: c_int, B: *const cuComplex, ldb: c_int, beta: cuComplex, C: *mut cuComplex, ldc: c_int));
legacy_void!(cublasZhemm, 1885, (side: c_char, uplo: c_char, m: c_int, n: c_int, alpha: cuDoubleComplex, A: *const cuDoubleComplex, lda: c_int, B: *const cuDoubleComplex, ldb: c_int, beta: cuDoubleComplex, C: *mut cuDoubleComplex, ldc: c_int));
legacy_void!(cublasSsyrk, 1886, (uplo: c_char, trans: c_char, n: c_int, k: c_int, alpha: f32, A: *const f32, lda: c_int, beta: f32, C: *mut f32, ldc: c_int));
legacy_void!(cublasDsyrk, 1887, (uplo: c_char, trans: c_char, n: c_int, k: c_int, alpha: f64, A: *const f64, lda: c_int, beta: f64, C: *mut f64, ldc: c_int));
legacy_void!(cublasCsyrk, 1888, (uplo: c_char, trans: c_char, n: c_int, k: c_int, alpha: cuComplex, A: *const cuComplex, lda: c_int, beta: cuComplex, C: *mut cuComplex, ldc: c_int));
legacy_void!(cublasZsyrk, 1889, (uplo: c_char, trans: c_char, n: c_int, k: c_int, alpha: cuDoubleComplex, A: *const cuDoubleComplex, lda: c_int, beta: cuDoubleComplex, C: *mut cuDoubleComplex, ldc: c_int));
legacy_void!(cublasCherk, 1890, (uplo: c_char, trans: c_char, n: c_int, k: c_int, alpha: f32, A: *const cuComplex, lda: c_int, beta: f32, C: *mut cuComplex, ldc: c_int));
legacy_void!(cublasZherk, 1891, (uplo: c_char, trans: c_char, n: c_int, k: c_int, alpha: f64, A: *const cuDoubleComplex, lda: c_int, beta: f64, C: *mut cuDoubleComplex, ldc: c_int));
legacy_void!(cublasSsyr2k, 1892, (uplo: c_char, trans: c_char, n: c_int, k: c_int, alpha: f32, A: *const f32, lda: c_int, B: *const f32, ldb: c_int, beta: f32, C: *mut f32, ldc: c_int));
legacy_void!(cublasDsyr2k, 1893, (uplo: c_char, trans: c_char, n: c_int, k: c_int, alpha: f64, A: *const f64, lda: c_int, B: *const f64, ldb: c_int, beta: f64, C: *mut f64, ldc: c_int));
legacy_void!(cublasCsyr2k, 1894, (uplo: c_char, trans: c_char, n: c_int, k: c_int, alpha: cuComplex, A: *const cuComplex, lda: c_int, B: *const cuComplex, ldb: c_int, beta: cuComplex, C: *mut cuComplex, ldc: c_int));
legacy_void!(cublasZsyr2k, 1895, (uplo: c_char, trans: c_char, n: c_int, k: c_int, alpha: cuDoubleComplex, A: *const cuDoubleComplex, lda: c_int, B: *const cuDoubleComplex, ldb: c_int, beta: cuDoubleComplex, C: *mut cuDoubleComplex, ldc: c_int));
legacy_void!(cublasCher2k, 1896, (uplo: c_char, trans: c_char, n: c_int, k: c_int, alpha: cuComplex, A: *const cuComplex, lda: c_int, B: *const cuComplex, ldb: c_int, beta: f32, C: *mut cuComplex, ldc: c_int));
legacy_void!(cublasZher2k, 1897, (uplo: c_char, trans: c_char, n: c_int, k: c_int, alpha: cuDoubleComplex, A: *const cuDoubleComplex, lda: c_int, B: *const cuDoubleComplex, ldb: c_int, beta: f64, C: *mut cuDoubleComplex, ldc: c_int));
legacy_void!(cublasStrmm, 1898, (side: c_char, uplo: c_char, transa: c_char, diag: c_char, m: c_int, n: c_int, alpha: f32, A: *const f32, lda: c_int, B: *mut f32, ldb: c_int));
legacy_void!(cublasDtrmm, 1899, (side: c_char, uplo: c_char, transa: c_char, diag: c_char, m: c_int, n: c_int, alpha: f64, A: *const f64, lda: c_int, B: *mut f64, ldb: c_int));
legacy_void!(cublasCtrmm, 1900, (side: c_char, uplo: c_char, transa: c_char, diag: c_char, m: c_int, n: c_int, alpha: cuComplex, A: *const cuComplex, lda: c_int, B: *mut cuComplex, ldb: c_int));
legacy_void!(cublasZtrmm, 1901, (side: c_char, uplo: c_char, transa: c_char, diag: c_char, m: c_int, n: c_int, alpha: cuDoubleComplex, A: *const cuDoubleComplex, lda: c_int, B: *mut cuDoubleComplex, ldb: c_int));
legacy_void!(cublasStrsm, 1902, (side: c_char, uplo: c_char, transa: c_char, diag: c_char, m: c_int, n: c_int, alpha: f32, A: *const f32, lda: c_int, B: *mut f32, ldb: c_int));
legacy_void!(cublasDtrsm, 1903, (side: c_char, uplo: c_char, transa: c_char, diag: c_char, m: c_int, n: c_int, alpha: f64, A: *const f64, lda: c_int, B: *mut f64, ldb: c_int));
legacy_void!(cublasCtrsm, 1904, (side: c_char, uplo: c_char, transa: c_char, diag: c_char, m: c_int, n: c_int, alpha: cuComplex, A: *const cuComplex, lda: c_int, B: *mut cuComplex, ldb: c_int));
legacy_void!(cublasZtrsm, 1905, (side: c_char, uplo: c_char, transa: c_char, diag: c_char, m: c_int, n: c_int, alpha: cuDoubleComplex, A: *const cuDoubleComplex, lda: c_int, B: *mut cuDoubleComplex, ldb: c_int));

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

fn cublas_logger_callback_state() -> &'static Mutex<cublasLogCallback> {
    static STATE: OnceLock<Mutex<cublasLogCallback>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(None))
}

#[no_mangle]
pub extern "C" fn cublasSetLoggerCallback(userCallback: cublasLogCallback) -> cublasStatus_t {
    *cublas_logger_callback_state().lock().unwrap() = userCallback;
    cublasStatus_t::CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub extern "C" fn cublasGetLoggerCallback(userCallback: *mut cublasLogCallback) -> cublasStatus_t {
    if userCallback.is_null() {
        return cublasStatus_t::CUBLAS_STATUS_INVALID_VALUE;
    }
    unsafe {
        *userCallback = *cublas_logger_callback_state().lock().unwrap();
    }
    cublasStatus_t::CUBLAS_STATUS_SUCCESS
}

#[no_mangle]
pub extern "C" fn cublasXerbla(_srName: *const c_char, _info: c_int) {}

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
