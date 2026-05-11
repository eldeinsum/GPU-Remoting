use std::collections::BTreeMap;
use std::ffi::c_void;
use std::os::raw::c_uint;
use std::sync::{Mutex, OnceLock};

#[derive(Copy, Clone)]
struct HostRange {
    size: usize,
    flags: c_uint,
    owned: bool,
}

pub enum HostMemoryError {
    InvalidValue,
    MemoryAllocation,
}

fn ranges() -> &'static Mutex<BTreeMap<usize, HostRange>> {
    static RANGES: OnceLock<Mutex<BTreeMap<usize, HostRange>>> = OnceLock::new();
    RANGES.get_or_init(|| Mutex::new(BTreeMap::new()))
}

fn find_range(map: &BTreeMap<usize, HostRange>, ptr: *mut c_void) -> Option<HostRange> {
    let addr = ptr as usize;
    let (&base, range) = map.range(..=addr).next_back()?;
    let end = base.checked_add(range.size)?;
    (addr < end).then_some(*range)
}

fn overlaps_range(map: &BTreeMap<usize, HostRange>, base: usize, size: usize) -> bool {
    let Some(end) = base.checked_add(size) else {
        return true;
    };
    if let Some((&prev_base, prev_range)) = map.range(..=base).next_back() {
        let Some(prev_end) = prev_base.checked_add(prev_range.size) else {
            return true;
        };
        if prev_end > base {
            return true;
        }
    }
    map.range(base..)
        .next()
        .is_some_and(|(&next_base, _)| next_base < end)
}

pub fn allocate(
    ptr_out: *mut *mut c_void,
    size: usize,
    flags: c_uint,
) -> Result<(), HostMemoryError> {
    if ptr_out.is_null() {
        return Err(HostMemoryError::InvalidValue);
    }

    if size == 0 {
        unsafe {
            *ptr_out = std::ptr::null_mut();
        }
        return Ok(());
    }

    let ptr = unsafe { libc::malloc(size) };
    if ptr.is_null() {
        return Err(HostMemoryError::MemoryAllocation);
    }

    ranges().lock().unwrap().insert(
        ptr as usize,
        HostRange {
            size,
            flags,
            owned: true,
        },
    );
    unsafe {
        *ptr_out = ptr;
    }
    Ok(())
}

pub fn free(ptr: *mut c_void) -> Result<(), HostMemoryError> {
    if ptr.is_null() {
        return Ok(());
    }

    let mut ranges = ranges().lock().unwrap();
    let range = ranges
        .remove(&(ptr as usize))
        .ok_or(HostMemoryError::InvalidValue)?;
    if !range.owned {
        ranges.insert(ptr as usize, range);
        return Err(HostMemoryError::InvalidValue);
    }
    drop(ranges);

    unsafe {
        libc::free(ptr);
    }
    Ok(())
}

pub fn register(ptr: *mut c_void, size: usize, flags: c_uint) -> Result<(), HostMemoryError> {
    if ptr.is_null() || size == 0 {
        return Err(HostMemoryError::InvalidValue);
    }

    let mut ranges = ranges().lock().unwrap();
    let base = ptr as usize;
    if overlaps_range(&ranges, base, size) {
        return Err(HostMemoryError::InvalidValue);
    }
    ranges.insert(
        base,
        HostRange {
            size,
            flags,
            owned: false,
        },
    );
    Ok(())
}

pub fn unregister(ptr: *mut c_void) -> Result<(), HostMemoryError> {
    if ptr.is_null() {
        return Err(HostMemoryError::InvalidValue);
    }

    let mut ranges = ranges().lock().unwrap();
    let range = ranges
        .remove(&(ptr as usize))
        .ok_or(HostMemoryError::InvalidValue)?;
    if range.owned {
        ranges.insert(ptr as usize, range);
        return Err(HostMemoryError::InvalidValue);
    }
    Ok(())
}

pub fn get_flags(flags_out: *mut c_uint, ptr: *mut c_void) -> Result<(), HostMemoryError> {
    if flags_out.is_null() || ptr.is_null() {
        return Err(HostMemoryError::InvalidValue);
    }

    let ranges = ranges().lock().unwrap();
    let flags = find_range(&ranges, ptr)
        .ok_or(HostMemoryError::InvalidValue)?
        .flags;
    unsafe {
        *flags_out = flags;
    }
    Ok(())
}
