fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=../cudasys/src/hooks");
    println!("cargo:rerun-if-changed=../cudasys/src/bindings/funcs");

    hookgen::generate_impls(
        "../cudasys/src/hooks/{}.rs",
        "../cudasys/src/bindings/funcs",
        "./src/dispatcher",
        "_exe",
        None,
        (cudasys::cuda::CUDA_VERSION / 1000) as u8,
    );
}
