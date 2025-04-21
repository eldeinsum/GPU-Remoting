#![expect(non_camel_case_types, non_upper_case_globals)]

macro_rules! success_return_value {
    ($ty:ident::$variant:ident) => {
        const _: () = {
            assert!($ty::$variant as u32 == 0);
            const fn test_num_derive<T: crate::FromPrimitive>() {}
            test_num_derive::<$ty>(); // see `DeriveCallback` in `build.rs`
        };
        impl Default for $ty {
            #[inline(always)]
            fn default() -> Self {
                Self::$variant
            }
        }
    };
}

macro_rules! impl_is_error {
    ($ty:ident) => {
        impl $ty {
            #[inline(always)]
            pub fn is_error(self) -> bool {
                self != Self::default()
            }
        }
    };
}

pub mod cuda {
    include!("bindings/types/cuda.rs");

    const _: () = assert!(CUDA_VERSION >= 11030);

    success_return_value!(CUresult::CUDA_SUCCESS);
    impl_is_error!(CUresult);

    impl CUpointer_attribute {
        pub fn data_size(&self) -> usize {
            match self {
                Self::CU_POINTER_ATTRIBUTE_DEVICE_POINTER => size_of::<*mut CUdeviceptr>(),
                Self::CU_POINTER_ATTRIBUTE_RANGE_START_ADDR => size_of::<usize>(),
                _ => panic!("unsupported pointer attribute {self:?}"),
            }
        }
    }
}

pub mod cudart {
    include!("bindings/types/cudart.rs");

    const _: () = assert!(CUDART_VERSION >= 11030);

    success_return_value!(cudaError_t::cudaSuccess);

    impl cudaError_t {
        pub fn is_error(self) -> bool {
            !matches!(
                self,
                Self::cudaSuccess
                    | Self::cudaErrorNotReady
                    | Self::cudaErrorPeerAccessAlreadyEnabled
            )
        }
    }
}

pub mod nvml {
    include!("bindings/types/nvml.rs");

    success_return_value!(nvmlReturn_t::NVML_SUCCESS);
    impl_is_error!(nvmlReturn_t);
}

pub mod cudnn {
    include!("bindings/types/cudnn.rs");

    success_return_value!(cudnnStatus_t::CUDNN_STATUS_SUCCESS);
    impl_is_error!(cudnnStatus_t);
}

pub mod cublas {
    include!("bindings/types/cublas.rs");

    success_return_value!(cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    impl_is_error!(cublasStatus_t);
}

pub mod cublasLt {
    include!("bindings/types/cublasLt.rs");

    success_return_value!(cublasStatus_t::CUBLAS_STATUS_SUCCESS);
    impl_is_error!(cublasStatus_t);
}

pub mod nvrtc {
    include!("bindings/types/nvrtc.rs");

    success_return_value!(nvrtcResult::NVRTC_SUCCESS);
    impl_is_error!(nvrtcResult);
}

pub mod nccl {
    include!("bindings/types/nccl.rs");

    const _: () = assert!(NCCL_VERSION_CODE >= 21602, "run `apt install libnccl2 libnccl-dev`");

    success_return_value!(ncclResult_t::ncclSuccess);
    impl_is_error!(ncclResult_t);
}
