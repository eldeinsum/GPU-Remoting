#![expect(non_snake_case)]
#![expect(clippy::let_unit_value, clippy::type_complexity)]

include!("mod_passthrough.rs");

use std::collections::BTreeSet;

struct State {
    depth: usize,
    functions: BTreeSet<&'static str>,
}

#[thread_local]
static mut __STATE: State = State {
    depth: 0,
    functions: BTreeSet::new(),
};

fn begin(name: &'static str) {
    unsafe {
        #[expect(static_mut_refs)]
        if __STATE.depth == 0 && __STATE.functions.insert(name) {
            eprintln!("[begin] {name}");
        }
        __STATE.depth += 1;
    }
}

fn end(_name: &'static str) {
    unsafe {
        __STATE.depth -= 1;
    }
}
