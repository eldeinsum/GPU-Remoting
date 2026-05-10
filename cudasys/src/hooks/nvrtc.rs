use crate::types::nvrtc::*;
use codegen::{cuda_custom_hook, cuda_hook};
use std::os::raw::*;

#[cuda_hook(proc_id = 3000)]
fn nvrtcVersion(major: *mut c_int, minor: *mut c_int) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcAddNameExpression(prog: nvrtcProgram, name_expression: *const c_char) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcCompileProgram(
    prog: nvrtcProgram,
    numOptions: c_int,
    options: *const *const c_char,
) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcCreateProgram(
    prog: *mut nvrtcProgram,
    src: *const c_char,
    name: *const c_char,
    numHeaders: c_int,
    headers: *const *const c_char,
    includeNames: *const *const c_char,
) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcDestroyProgram(prog: *mut nvrtcProgram) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcGetCUBIN(prog: nvrtcProgram, cubin: *mut c_char) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcGetCUBINSize(prog: nvrtcProgram, cubinSizeRet: *mut usize) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcGetErrorString(result: nvrtcResult) -> *const c_char;

#[cuda_custom_hook] // local
fn nvrtcGetLTOIR(prog: nvrtcProgram, LTOIR: *mut c_char) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcGetLTOIRSize(prog: nvrtcProgram, LTOIRSizeRet: *mut usize) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcGetLoweredName(
    prog: nvrtcProgram,
    name_expression: *const c_char,
    lowered_name: *mut *const c_char,
) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcGetNumSupportedArchs(numArchs: *mut c_int) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcGetOptiXIR(prog: nvrtcProgram, optixir: *mut c_char) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcGetOptiXIRSize(prog: nvrtcProgram, optixirSizeRet: *mut usize) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcGetPCHCreateStatus(prog: nvrtcProgram) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcGetPCHHeapSize(ret: *mut usize) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcGetPCHHeapSizeRequired(prog: nvrtcProgram, size: *mut usize) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcGetPTX(prog: nvrtcProgram, ptx: *mut c_char) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcGetPTXSize(prog: nvrtcProgram, ptxSizeRet: *mut usize) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcGetProgramLog(prog: nvrtcProgram, log: *mut c_char) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcGetProgramLogSize(prog: nvrtcProgram, logSizeRet: *mut usize) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcGetSupportedArchs(supportedArchs: *mut c_int) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcGetTileIR(prog: nvrtcProgram, TileIR: *mut c_char) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcGetTileIRSize(prog: nvrtcProgram, TileIRSizeRet: *mut usize) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcSetFlowCallback(
    prog: nvrtcProgram,
    callback: Option<unsafe extern "C" fn(arg1: *mut c_void, arg2: *mut c_void) -> c_int>,
    payload: *mut c_void,
) -> nvrtcResult;

#[cuda_custom_hook] // local
fn nvrtcSetPCHHeapSize(size: usize) -> nvrtcResult;
