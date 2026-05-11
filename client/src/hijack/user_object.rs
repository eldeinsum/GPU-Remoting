use cudasys::types::cuda::{CUgraph, CUhostFn, CUresult, CUuserObject, CUuserObject_st};
use std::collections::BTreeMap;
use std::ffi::c_void;
use std::sync::{Mutex, OnceLock};

const USER_OBJECT_NO_DESTRUCTOR_SYNC: u32 = 1;
const GRAPH_USER_OBJECT_MOVE: u32 = 1;
const MAX_REFCOUNT: u32 = i32::MAX as u32;

struct UserObject {
    ptr: usize,
    destroy: unsafe extern "C" fn(*mut c_void),
    caller_refs: u64,
    graph_refs: u64,
}

struct Store {
    next_handle: usize,
    objects: BTreeMap<usize, UserObject>,
    graph_refs: BTreeMap<(usize, usize), u64>,
}

fn store() -> &'static Mutex<Store> {
    static STORE: OnceLock<Mutex<Store>> = OnceLock::new();
    STORE.get_or_init(|| {
        Mutex::new(Store {
            next_handle: 1,
            objects: BTreeMap::new(),
            graph_refs: BTreeMap::new(),
        })
    })
}

fn destroy_if_unreferenced(store: &mut Store, handle: usize) -> Option<UserObject> {
    let object = store.objects.get(&handle)?;
    (object.caller_refs == 0 && object.graph_refs == 0)
        .then(|| store.objects.remove(&handle))
        .flatten()
}

fn invoke_destroy(object: UserObject) {
    unsafe {
        (object.destroy)(object.ptr as *mut c_void);
    }
}

fn invalid_refcount(count: u32) -> bool {
    count == 0 || count > MAX_REFCOUNT
}

pub fn create(
    object_out: *mut CUuserObject,
    ptr: *mut c_void,
    destroy: CUhostFn,
    initial_refcount: u32,
    flags: u32,
) -> CUresult {
    let Some(destroy) = destroy else {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    };
    if object_out.is_null()
        || invalid_refcount(initial_refcount)
        || flags != USER_OBJECT_NO_DESTRUCTOR_SYNC
    {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    let mut store = store().lock().unwrap();
    let handle = store.next_handle;
    let Some(next_handle) = store.next_handle.checked_add(1) else {
        return CUresult::CUDA_ERROR_OUT_OF_MEMORY;
    };
    store.next_handle = next_handle;
    store.objects.insert(
        handle,
        UserObject {
            ptr: ptr as usize,
            destroy,
            caller_refs: initial_refcount.into(),
            graph_refs: 0,
        },
    );
    unsafe {
        *object_out = handle as *mut CUuserObject_st;
    }
    CUresult::CUDA_SUCCESS
}

pub fn retain(object: CUuserObject, count: u32) -> CUresult {
    if object.is_null() || invalid_refcount(count) {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    let mut store = store().lock().unwrap();
    let Some(state) = store.objects.get_mut(&(object as usize)) else {
        return CUresult::CUDA_ERROR_INVALID_HANDLE;
    };
    let Some(refs) = state.caller_refs.checked_add(count.into()) else {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    };
    state.caller_refs = refs;
    CUresult::CUDA_SUCCESS
}

pub fn release(object: CUuserObject, count: u32) -> CUresult {
    if object.is_null() || invalid_refcount(count) {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    let mut store = store().lock().unwrap();
    let handle = object as usize;
    let Some(state) = store.objects.get_mut(&handle) else {
        return CUresult::CUDA_ERROR_INVALID_HANDLE;
    };
    if state.caller_refs < count.into() {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    state.caller_refs -= u64::from(count);
    let destroy = destroy_if_unreferenced(&mut store, handle);
    drop(store);

    if let Some(object) = destroy {
        invoke_destroy(object);
    }
    CUresult::CUDA_SUCCESS
}

pub fn graph_retain(graph: CUgraph, object: CUuserObject, count: u32, flags: u32) -> CUresult {
    if graph.is_null()
        || object.is_null()
        || invalid_refcount(count)
        || (flags & !GRAPH_USER_OBJECT_MOVE) != 0
    {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    let mut store = store().lock().unwrap();
    let graph_handle = graph as usize;
    let object_handle = object as usize;
    let key = (graph_handle, object_handle);
    let count = u64::from(count);
    let Some(next_graph_entry_refs) = store
        .graph_refs
        .get(&key)
        .copied()
        .unwrap_or(0)
        .checked_add(count)
    else {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    };
    let Some(state) = store.objects.get_mut(&object_handle) else {
        return CUresult::CUDA_ERROR_INVALID_HANDLE;
    };
    if flags & GRAPH_USER_OBJECT_MOVE != 0 {
        if state.caller_refs < count {
            return CUresult::CUDA_ERROR_INVALID_VALUE;
        }
        state.caller_refs -= count;
    }
    let Some(graph_refs) = state.graph_refs.checked_add(count) else {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    };
    state.graph_refs = graph_refs;
    store.graph_refs.insert(key, next_graph_entry_refs);
    CUresult::CUDA_SUCCESS
}

pub fn graph_release(graph: CUgraph, object: CUuserObject, count: u32) -> CUresult {
    if graph.is_null() || object.is_null() || invalid_refcount(count) {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    let mut store = store().lock().unwrap();
    let graph_handle = graph as usize;
    let object_handle = object as usize;
    let count = u64::from(count);
    let key = (graph_handle, object_handle);
    let Some(graph_refs) = store.graph_refs.get(&key).copied() else {
        return CUresult::CUDA_ERROR_INVALID_HANDLE;
    };
    if graph_refs < count {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }
    let Some(state_graph_refs) = store
        .objects
        .get(&object_handle)
        .map(|state| state.graph_refs)
    else {
        return CUresult::CUDA_ERROR_INVALID_HANDLE;
    };
    if state_graph_refs < count {
        return CUresult::CUDA_ERROR_INVALID_VALUE;
    }

    if graph_refs == count {
        store.graph_refs.remove(&key);
    } else {
        store.graph_refs.insert(key, graph_refs - count);
    }
    let state = store.objects.get_mut(&object_handle).unwrap();
    state.graph_refs -= count;
    let destroy = destroy_if_unreferenced(&mut store, object_handle);
    drop(store);

    if let Some(object) = destroy {
        invoke_destroy(object);
    }
    CUresult::CUDA_SUCCESS
}

pub fn graph_destroy(graph: CUgraph) {
    if graph.is_null() {
        return;
    }

    let mut store = store().lock().unwrap();
    let graph_handle = graph as usize;
    let refs = store
        .graph_refs
        .iter()
        .filter_map(|(&(entry_graph, object_handle), &count)| {
            (entry_graph == graph_handle).then_some((object_handle, count))
        })
        .collect::<Vec<_>>();

    let mut destroys = Vec::new();
    for (object_handle, count) in refs {
        store.graph_refs.remove(&(graph_handle, object_handle));
        let Some(state) = store.objects.get_mut(&object_handle) else {
            continue;
        };
        state.graph_refs = state.graph_refs.saturating_sub(count);
        if let Some(object) = destroy_if_unreferenced(&mut store, object_handle) {
            destroys.push(object);
        }
    }
    drop(store);

    for object in destroys {
        invoke_destroy(object);
    }
}

pub fn graph_clone(original_graph: CUgraph, cloned_graph: CUgraph) {
    if original_graph.is_null() || cloned_graph.is_null() {
        return;
    }

    let mut store = store().lock().unwrap();
    let original_graph = original_graph as usize;
    let cloned_graph = cloned_graph as usize;
    let refs = store
        .graph_refs
        .iter()
        .filter_map(|(&(entry_graph, object_handle), &count)| {
            (entry_graph == original_graph).then_some((object_handle, count))
        })
        .collect::<Vec<_>>();

    for (object_handle, count) in refs {
        let key = (cloned_graph, object_handle);
        let Some(next_graph_entry_refs) = store
            .graph_refs
            .get(&key)
            .copied()
            .unwrap_or(0)
            .checked_add(count)
        else {
            log::error!(target: "user_object", "user object graph clone refcount overflow");
            continue;
        };
        let Some(state) = store.objects.get_mut(&object_handle) else {
            continue;
        };
        let Some(next_object_refs) = state.graph_refs.checked_add(count) else {
            log::error!(target: "user_object", "user object total graph refcount overflow");
            continue;
        };
        state.graph_refs = next_object_refs;
        store.graph_refs.insert(key, next_graph_entry_refs);
    }
}
