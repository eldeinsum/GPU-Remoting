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
                Self::CU_POINTER_ATTRIBUTE_CONTEXT => size_of::<CUcontext>(),
                Self::CU_POINTER_ATTRIBUTE_MEMORY_TYPE => size_of::<CUmemorytype>(),
                Self::CU_POINTER_ATTRIBUTE_DEVICE_POINTER => size_of::<CUdeviceptr>(),
                Self::CU_POINTER_ATTRIBUTE_HOST_POINTER => size_of::<*mut std::ffi::c_void>(),
                Self::CU_POINTER_ATTRIBUTE_P2P_TOKENS => {
                    size_of::<CUDA_POINTER_ATTRIBUTE_P2P_TOKENS>()
                }
                Self::CU_POINTER_ATTRIBUTE_SYNC_MEMOPS => size_of::<u32>(),
                Self::CU_POINTER_ATTRIBUTE_BUFFER_ID => size_of::<u64>(),
                Self::CU_POINTER_ATTRIBUTE_IS_MANAGED
                | Self::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL
                | Self::CU_POINTER_ATTRIBUTE_IS_LEGACY_CUDA_IPC_CAPABLE
                | Self::CU_POINTER_ATTRIBUTE_MAPPED
                | Self::CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE
                | Self::CU_POINTER_ATTRIBUTE_IS_HW_DECOMPRESS_CAPABLE => size_of::<i32>(),
                Self::CU_POINTER_ATTRIBUTE_RANGE_START_ADDR
                | Self::CU_POINTER_ATTRIBUTE_MAPPING_BASE_ADDR => size_of::<CUdeviceptr>(),
                Self::CU_POINTER_ATTRIBUTE_RANGE_SIZE | Self::CU_POINTER_ATTRIBUTE_MAPPING_SIZE => {
                    size_of::<usize>()
                }
                Self::CU_POINTER_ATTRIBUTE_ALLOWED_HANDLE_TYPES => {
                    size_of::<CUmemAllocationHandleType>()
                }
                Self::CU_POINTER_ATTRIBUTE_ACCESS_FLAGS => {
                    size_of::<CUDA_POINTER_ATTRIBUTE_ACCESS_FLAGS>()
                }
                Self::CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE => size_of::<CUmemoryPool>(),
                Self::CU_POINTER_ATTRIBUTE_MEMORY_BLOCK_ID => size_of::<u64>(),
            }
        }
    }

    impl CUmemPool_attribute {
        pub fn data_size(&self) -> usize {
            match self {
                Self::CU_MEMPOOL_ATTR_REUSE_FOLLOW_EVENT_DEPENDENCIES
                | Self::CU_MEMPOOL_ATTR_REUSE_ALLOW_OPPORTUNISTIC
                | Self::CU_MEMPOOL_ATTR_REUSE_ALLOW_INTERNAL_DEPENDENCIES
                | Self::CU_MEMPOOL_ATTR_LOCATION_ID
                | Self::CU_MEMPOOL_ATTR_HW_DECOMPRESS_ENABLED => size_of::<i32>(),
                Self::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD
                | Self::CU_MEMPOOL_ATTR_RESERVED_MEM_CURRENT
                | Self::CU_MEMPOOL_ATTR_RESERVED_MEM_HIGH
                | Self::CU_MEMPOOL_ATTR_USED_MEM_CURRENT
                | Self::CU_MEMPOOL_ATTR_USED_MEM_HIGH
                | Self::CU_MEMPOOL_ATTR_MAX_POOL_SIZE => size_of::<u64>(),
                Self::CU_MEMPOOL_ATTR_ALLOCATION_TYPE => size_of::<CUmemAllocationType>(),
                Self::CU_MEMPOOL_ATTR_EXPORT_HANDLE_TYPES => size_of::<CUmemAllocationHandleType>(),
                Self::CU_MEMPOOL_ATTR_LOCATION_TYPE => size_of::<CUmemLocationType>(),
            }
        }
    }

    impl CUgraphMem_attribute {
        pub fn data_size(&self) -> usize {
            size_of::<u64>()
        }
    }

    impl std::fmt::Debug for CUDA_MEM_ALLOC_NODE_PARAMS_v1_st {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let location_id = unsafe { self.poolProps.location.__bindgen_anon_1.id };
            f.debug_struct("CUDA_MEM_ALLOC_NODE_PARAMS_v1_st")
                .field("poolProps.allocType", &self.poolProps.allocType)
                .field("poolProps.handleTypes", &self.poolProps.handleTypes)
                .field("poolProps.location.type_", &self.poolProps.location.type_)
                .field("poolProps.location.id", &location_id)
                .field(
                    "poolProps.win32SecurityAttributes",
                    &self.poolProps.win32SecurityAttributes,
                )
                .field("poolProps.maxSize", &self.poolProps.maxSize)
                .field("poolProps.usage", &self.poolProps.usage)
                .field("accessDescs", &self.accessDescs)
                .field("accessDescCount", &self.accessDescCount)
                .field("bytesize", &self.bytesize)
                .field("dptr", &self.dptr)
                .finish()
        }
    }

    impl std::fmt::Debug for CUDA_MEM_ALLOC_NODE_PARAMS_v2_st {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let location_id = unsafe { self.poolProps.location.__bindgen_anon_1.id };
            f.debug_struct("CUDA_MEM_ALLOC_NODE_PARAMS_v2_st")
                .field("poolProps.allocType", &self.poolProps.allocType)
                .field("poolProps.handleTypes", &self.poolProps.handleTypes)
                .field("poolProps.location.type_", &self.poolProps.location.type_)
                .field("poolProps.location.id", &location_id)
                .field(
                    "poolProps.win32SecurityAttributes",
                    &self.poolProps.win32SecurityAttributes,
                )
                .field("poolProps.maxSize", &self.poolProps.maxSize)
                .field("poolProps.usage", &self.poolProps.usage)
                .field("accessDescs", &self.accessDescs)
                .field("accessDescCount", &self.accessDescCount)
                .field("bytesize", &self.bytesize)
                .field("dptr", &self.dptr)
                .finish()
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

    impl cudaMemPoolAttr {
        pub fn data_size(&self) -> usize {
            match self {
                Self::cudaMemPoolReuseFollowEventDependencies
                | Self::cudaMemPoolReuseAllowOpportunistic
                | Self::cudaMemPoolReuseAllowInternalDependencies
                | Self::cudaMemPoolAttrLocationId
                | Self::cudaMemPoolAttrHwDecompressEnabled => size_of::<i32>(),
                Self::cudaMemPoolAttrReleaseThreshold
                | Self::cudaMemPoolAttrReservedMemCurrent
                | Self::cudaMemPoolAttrReservedMemHigh
                | Self::cudaMemPoolAttrUsedMemCurrent
                | Self::cudaMemPoolAttrUsedMemHigh
                | Self::cudaMemPoolAttrMaxPoolSize => size_of::<u64>(),
                Self::cudaMemPoolAttrAllocationType => size_of::<cudaMemAllocationType>(),
                Self::cudaMemPoolAttrExportHandleTypes => size_of::<cudaMemAllocationHandleType>(),
                Self::cudaMemPoolAttrLocationType => size_of::<cudaMemLocationType>(),
            }
        }
    }

    impl cudaGraphMemAttributeType {
        pub fn data_size(&self) -> usize {
            size_of::<u64>()
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

    impl cudnnBackendAttributeType_t {
        pub fn data_size(&self) -> i64 {
            let result = match self {
                Self::CUDNN_TYPE_HANDLE => size_of::<cudnnHandle_t>(),
                Self::CUDNN_TYPE_DATA_TYPE => size_of::<cudnnDataType_t>(),
                // Self::CUDNN_TYPE_BOOLEAN => size_of::<bool>(),
                Self::CUDNN_TYPE_INT64 => size_of::<i64>(),
                Self::CUDNN_TYPE_FLOAT => size_of::<f32>(),
                // Self::CUDNN_TYPE_DOUBLE => size_of::<f64>(),
                Self::CUDNN_TYPE_VOID_PTR => size_of::<*mut std::ffi::c_void>(),
                Self::CUDNN_TYPE_CONVOLUTION_MODE => size_of::<cudnnConvolutionMode_t>(),
                Self::CUDNN_TYPE_HEUR_MODE => size_of::<cudnnBackendHeurMode_t>(),
                Self::CUDNN_TYPE_KNOB_TYPE => size_of::<cudnnBackendKnobType_t>(),
                // Self::CUDNN_TYPE_NAN_PROPOGATION => size_of::<cudnnNanPropagation_t>(),
                Self::CUDNN_TYPE_NUMERICAL_NOTE => size_of::<cudnnBackendNumericalNote_t>(),
                // Self::CUDNN_TYPE_LAYOUT_TYPE => size_of::<cudnnBackendLayoutType_t>(),
                // Self::CUDNN_TYPE_ATTRIB_NAME => size_of::<cudnnBackendAttributeName_t>(),
                // Self::CUDNN_TYPE_POINTWISE_MODE => size_of::<cudnnPointwiseMode_t>(),
                Self::CUDNN_TYPE_BACKEND_DESCRIPTOR => size_of::<cudnnBackendDescriptor_t>(),
                // Self::CUDNN_TYPE_GENSTATS_MODE => size_of::<cudnnGenStatsMode_t>(),
                // Self::CUDNN_TYPE_BN_FINALIZE_STATS_MODE => size_of::<cudnnBnFinalizeStatsMode_t>(),
                // Self::CUDNN_TYPE_REDUCTION_OPERATOR_TYPE => size_of::<cudnnReduceTensorOp_t>(),
                // Self::CUDNN_TYPE_BEHAVIOR_NOTE => size_of::<cudnnBackendBehaviorNote_t>(),
                // Self::CUDNN_TYPE_TENSOR_REORDERING_MODE => size_of::<cudnnBackendTensorReordering_t>(),
                // Self::CUDNN_TYPE_RESAMPLE_MODE => size_of::<cudnnResampleMode_t>(),
                // Self::CUDNN_TYPE_PADDING_MODE => size_of::<cudnnPaddingMode_t>(),
                // Self::CUDNN_TYPE_INT32 => size_of::<i32>(),
                // Self::CUDNN_TYPE_CHAR => size_of::<std::ffi::c_char>(),
                // Self::CUDNN_TYPE_SIGNAL_MODE => size_of::<cudnnSignalMode_t>(),
                // Self::CUDNN_TYPE_FRACTION => size_of::<cudnnFraction_t>(),
                // Self::CUDNN_TYPE_NORM_MODE => size_of::<cudnnBackendNormMode_t>(),
                // Self::CUDNN_TYPE_NORM_FWD_PHASE => size_of::<cudnnBackendNormFwdPhase_t>(),
                // Self::CUDNN_TYPE_RNG_DISTRIBUTION => size_of::<cudnnRngDistribution_t>(),
                _ => panic!("unsupported attribute type {self:?}"),
            };
            result as i64
        }
    }
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

    const _: () = assert!(
        NCCL_VERSION_CODE >= 21602,
        "run `apt install libnccl2 libnccl-dev`"
    );

    success_return_value!(ncclResult_t::ncclSuccess);
    impl_is_error!(ncclResult_t);
}
