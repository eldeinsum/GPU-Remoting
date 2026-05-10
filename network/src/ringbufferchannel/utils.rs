use std::alloc::{alloc, dealloc, Layout};
use std::ptr::NonNull;

pub fn allocate_cache_line_aligned(size: usize, cache_line_size: usize) -> NonNull<u8> {
    let layout = Layout::from_size_align(size, cache_line_size).expect("Failed to create layout");

    unsafe {
        let ptr = alloc(layout);
        if ptr.is_null() {
            panic!("Allocation failed");
        }
        NonNull::new_unchecked(ptr)
    }
}

pub fn deallocate(ptr: NonNull<u8>, size: usize, cache_line_size: usize) {
    let layout = Layout::from_size_align(size, cache_line_size).expect("Failed to create layout");

    unsafe {
        dealloc(ptr.as_ptr(), layout);
    }
}

pub fn is_cache_line_aligned<T>(ptr: *const T) -> bool {
    let alignment = 64; // Typical cache line size in bytes
    (ptr as usize).is_multiple_of(alignment)
}
