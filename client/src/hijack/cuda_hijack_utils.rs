pub fn pack_kernel_args(
    arg_ptrs: *mut *mut std::ffi::c_void,
    info: &[crate::elf::KernelParamInfo],
) -> Box<[u8]> {
    let Some(last) = info.last() else {
        return Default::default();
    };
    let mut result = vec![0u8; (last.offset + last.size()) as usize];
    for (param, arg_ptr) in info
        .iter()
        .zip(unsafe { std::slice::from_raw_parts(arg_ptrs, info.len()) })
    {
        unsafe {
            std::ptr::copy_nonoverlapping(
                arg_ptr.cast(),
                result.as_mut_ptr().wrapping_add(param.offset as usize),
                param.size() as usize,
            );
        }
        match param.size() {
            8 if arg_ptr.cast::<u64>().is_aligned() => {
                let arg = unsafe { *arg_ptr.cast::<u64>() };
                log::trace!(target: "cuLaunchKernel", "arg = {arg:#x}");
            }
            4 if arg_ptr.cast::<i32>().is_aligned() => {
                let arg = unsafe { *arg_ptr.cast::<i32>() };
                log::trace!(target: "cuLaunchKernel", "arg = {arg}");
            }
            size => log::trace!(target: "cuLaunchKernel", "arg<{size}> = {:?}", unsafe {
                std::slice::from_raw_parts(arg_ptr.cast::<u8>(), param.size() as usize)
            }),
        }
    }
    result.into_boxed_slice()
}

pub fn pack_kernel_args_with_offsets(
    arg_ptrs: *mut *mut std::ffi::c_void,
    info: &[crate::elf::KernelParamInfo],
) -> (Box<[u8]>, Box<[u32]>) {
    let args = pack_kernel_args(arg_ptrs, info);
    let offsets = info
        .iter()
        .map(|param| u32::from(param.offset))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    (args, offsets)
}
