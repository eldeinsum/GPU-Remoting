use crate::types::cudnn::*;
use codegen::{cuda_custom_hook, cuda_hook};
use std::os::raw::*;

/// FIXME: void pointer hacking
type HackedAssumeDouble = f64;

#[cuda_custom_hook] // local: derived from remoted property queries
fn cudnnGetVersion() -> usize;

#[cuda_custom_hook(proc_id = 1801)] // remoted: non-status return
fn cudnnGetMaxDeviceVersion() -> usize;

#[cuda_custom_hook] // local: derived from remoted runtime version query
fn cudnnGetCudartVersion() -> usize;

#[cuda_hook(proc_id = 1800)]
fn cudnnGetProperty(type_: libraryPropertyType, value: *mut c_int) -> cudnnStatus_t;

#[cuda_custom_hook] // local: returns a client-owned C string
fn cudnnGetErrorString(status: cudnnStatus_t) -> *const c_char;

#[cuda_custom_hook] // local: writes a client-owned diagnostic string
fn cudnnGetLastErrorString(message: *mut c_char, max_size: usize);

#[cuda_hook(proc_id = 1803)]
fn cudnnGraphVersionCheck() -> cudnnStatus_t;

#[cuda_hook(proc_id = 1806)]
fn cudnnAdvVersionCheck() -> cudnnStatus_t;

#[cuda_hook(proc_id = 1807)]
fn cudnnCnnVersionCheck() -> cudnnStatus_t;

#[cuda_hook(proc_id = 1809)]
fn cudnnOpsVersionCheck() -> cudnnStatus_t;

#[cuda_hook(proc_id = 1810)]
fn cudnnSubquadraticOpsVersionCheck() -> cudnnStatus_t;

#[cuda_hook(proc_id = 1804)]
fn cudnnCreate(handle: *mut cudnnHandle_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1811)]
fn cudnnGetStream(handle: cudnnHandle_t, streamId: *mut cudaStream_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1814)]
fn cudnnCreateTensorDescriptor(tensorDesc: *mut cudnnTensorDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1873)]
fn cudnnSetTensor4dDescriptor(
    tensorDesc: cudnnTensorDescriptor_t,
    format: cudnnTensorFormat_t,
    dataType: cudnnDataType_t,
    n: c_int,
    c: c_int,
    h: c_int,
    w: c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1812)]
fn cudnnSetTensor4dDescriptorEx(
    tensorDesc: cudnnTensorDescriptor_t,
    dataType: cudnnDataType_t,
    n: c_int,
    c: c_int,
    h: c_int,
    w: c_int,
    nStride: c_int,
    cStride: c_int,
    hStride: c_int,
    wStride: c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1813)]
fn cudnnGetTensor4dDescriptor(
    tensorDesc: cudnnTensorDescriptor_t,
    dataType: *mut cudnnDataType_t,
    n: *mut c_int,
    c: *mut c_int,
    h: *mut c_int,
    w: *mut c_int,
    nStride: *mut c_int,
    cStride: *mut c_int,
    hStride: *mut c_int,
    wStride: *mut c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1805)]
fn cudnnCreateActivationDescriptor(
    activationDesc: *mut cudnnActivationDescriptor_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1861)]
fn cudnnSetActivationDescriptor(
    activationDesc: cudnnActivationDescriptor_t,
    mode: cudnnActivationMode_t,
    reluNanOpt: cudnnNanPropagation_t,
    coef: f64,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1862)]
fn cudnnGetActivationDescriptor(
    activationDesc: cudnnActivationDescriptor_t,
    mode: *mut cudnnActivationMode_t,
    reluNanOpt: *mut cudnnNanPropagation_t,
    coef: *mut f64,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1863)]
fn cudnnSetActivationDescriptorSwishBeta(
    activationDesc: cudnnActivationDescriptor_t,
    swish_beta: f64,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1866)]
fn cudnnGetActivationDescriptorSwishBeta(
    activationDesc: cudnnActivationDescriptor_t,
    swish_beta: *mut f64,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1867)]
fn cudnnDestroyActivationDescriptor(activationDesc: cudnnActivationDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1816)]
fn cudnnDestroy(handle: cudnnHandle_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2126)]
fn cudnnSetConvolution2dDescriptor(
    convDesc: cudnnConvolutionDescriptor_t,
    pad_h: c_int,
    pad_w: c_int,
    u: c_int,
    v: c_int,
    dilation_h: c_int,
    dilation_w: c_int,
    mode: cudnnConvolutionMode_t,
    computeType: cudnnDataType_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2131)]
fn cudnnGetConvolution2dDescriptor(
    convDesc: cudnnConvolutionDescriptor_t,
    pad_h: *mut c_int,
    pad_w: *mut c_int,
    u: *mut c_int,
    v: *mut c_int,
    dilation_h: *mut c_int,
    dilation_w: *mut c_int,
    mode: *mut cudnnConvolutionMode_t,
    computeType: *mut cudnnDataType_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1871, async_api)]
fn cudnnSetStream(handle: cudnnHandle_t, streamId: cudaStream_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1875, async_api)]
fn cudnnSetTensorNdDescriptor(
    tensorDesc: cudnnTensorDescriptor_t,
    dataType: cudnnDataType_t,
    nbDims: c_int,
    #[host(len = nbDims)] dimA: *const c_int,
    #[host(len = nbDims)] strideA: *const c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1815)]
fn cudnnSetTensorNdDescriptorEx(
    tensorDesc: cudnnTensorDescriptor_t,
    format: cudnnTensorFormat_t,
    dataType: cudnnDataType_t,
    nbDims: c_int,
    #[host(len = nbDims)] dimA: *const c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1817)]
fn cudnnGetTensorNdDescriptor(
    tensorDesc: cudnnTensorDescriptor_t,
    nbDimsRequested: c_int,
    dataType: *mut cudnnDataType_t,
    nbDims: *mut c_int,
    #[host(output, len = nbDims, cap = nbDimsRequested)] dimA: *mut c_int,
    #[host(output, len = nbDims, cap = nbDimsRequested)] strideA: *mut c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1818)]
fn cudnnGetTensorSizeInBytes(
    tensorDesc: cudnnTensorDescriptor_t,
    size: *mut usize,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1826, async_api)]
fn cudnnDestroyTensorDescriptor(tensorDesc: cudnnTensorDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1823)]
fn cudnnCreateTensorTransformDescriptor(
    transformDesc: *mut cudnnTensorTransformDescriptor_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1824)]
fn cudnnSetTensorTransformDescriptor(
    transformDesc: cudnnTensorTransformDescriptor_t,
    nbDims: u32,
    destFormat: cudnnTensorFormat_t,
    #[host(len = nbDims)] padBeforeA: *const i32,
    #[host(len = nbDims)] padAfterA: *const i32,
    #[host(len = nbDims)] foldA: *const u32,
    direction: cudnnFoldingDirection_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1825)]
fn cudnnGetTensorTransformDescriptor(
    transformDesc: cudnnTensorTransformDescriptor_t,
    nbDimsRequested: u32,
    destFormat: *mut cudnnTensorFormat_t,
    #[host(output, len = nbDimsRequested)] padBeforeA: *mut i32,
    #[host(output, len = nbDimsRequested)] padAfterA: *mut i32,
    #[host(output, len = nbDimsRequested)] foldA: *mut u32,
    direction: *mut cudnnFoldingDirection_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1859, async_api)]
fn cudnnDestroyTensorTransformDescriptor(
    transformDesc: cudnnTensorTransformDescriptor_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1860)]
fn cudnnInitTransformDest(
    transformDesc: cudnnTensorTransformDescriptor_t,
    srcDesc: cudnnTensorDescriptor_t,
    destDesc: cudnnTensorDescriptor_t,
    destSizeInBytes: *mut usize,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1808)]
fn cudnnCreateFilterDescriptor(filterDesc: *mut cudnnFilterDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1820, async_api)]
fn cudnnDestroyFilterDescriptor(filterDesc: cudnnFilterDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1865, async_api)]
fn cudnnSetFilterNdDescriptor(
    filterDesc: cudnnFilterDescriptor_t,
    dataType: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    nbDims: c_int,
    #[host(len = nbDims)] filterDimA: *const c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1819)]
fn cudnnGetFilter4dDescriptor(
    filterDesc: cudnnFilterDescriptor_t,
    dataType: *mut cudnnDataType_t,
    format: *mut cudnnTensorFormat_t,
    k: *mut c_int,
    c: *mut c_int,
    h: *mut c_int,
    w: *mut c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1821)]
fn cudnnGetFilterNdDescriptor(
    filterDesc: cudnnFilterDescriptor_t,
    nbDimsRequested: c_int,
    dataType: *mut cudnnDataType_t,
    format: *mut cudnnTensorFormat_t,
    nbDims: *mut c_int,
    #[host(output, len = nbDims, cap = nbDimsRequested)] filterDimA: *mut c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1822)]
fn cudnnGetFilterSizeInBytes(
    filterDesc: cudnnFilterDescriptor_t,
    size: *mut usize,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2300)]
fn cudnnCreateOpTensorDescriptor(opTensorDesc: *mut cudnnOpTensorDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2301)]
fn cudnnSetOpTensorDescriptor(
    opTensorDesc: cudnnOpTensorDescriptor_t,
    opTensorOp: cudnnOpTensorOp_t,
    opTensorCompType: cudnnDataType_t,
    opTensorNanOpt: cudnnNanPropagation_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2302)]
fn cudnnGetOpTensorDescriptor(
    opTensorDesc: cudnnOpTensorDescriptor_t,
    opTensorOp: *mut cudnnOpTensorOp_t,
    opTensorCompType: *mut cudnnDataType_t,
    opTensorNanOpt: *mut cudnnNanPropagation_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2303, async_api)]
fn cudnnDestroyOpTensorDescriptor(opTensorDesc: cudnnOpTensorDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2304)]
fn cudnnCreateReduceTensorDescriptor(
    reduceTensorDesc: *mut cudnnReduceTensorDescriptor_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2305)]
fn cudnnSetReduceTensorDescriptor(
    reduceTensorDesc: cudnnReduceTensorDescriptor_t,
    reduceTensorOp: cudnnReduceTensorOp_t,
    reduceTensorCompType: cudnnDataType_t,
    reduceTensorNanOpt: cudnnNanPropagation_t,
    reduceTensorIndices: cudnnReduceTensorIndices_t,
    reduceTensorIndicesType: cudnnIndicesType_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2306)]
fn cudnnGetReduceTensorDescriptor(
    reduceTensorDesc: cudnnReduceTensorDescriptor_t,
    reduceTensorOp: *mut cudnnReduceTensorOp_t,
    reduceTensorCompType: *mut cudnnDataType_t,
    reduceTensorNanOpt: *mut cudnnNanPropagation_t,
    reduceTensorIndices: *mut cudnnReduceTensorIndices_t,
    reduceTensorIndicesType: *mut cudnnIndicesType_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2307, async_api)]
fn cudnnDestroyReduceTensorDescriptor(
    reduceTensorDesc: cudnnReduceTensorDescriptor_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2308)]
fn cudnnCreatePoolingDescriptor(poolingDesc: *mut cudnnPoolingDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2309)]
fn cudnnSetPooling2dDescriptor(
    poolingDesc: cudnnPoolingDescriptor_t,
    mode: cudnnPoolingMode_t,
    maxpoolingNanOpt: cudnnNanPropagation_t,
    windowHeight: c_int,
    windowWidth: c_int,
    verticalPadding: c_int,
    horizontalPadding: c_int,
    verticalStride: c_int,
    horizontalStride: c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2310)]
fn cudnnGetPooling2dDescriptor(
    poolingDesc: cudnnPoolingDescriptor_t,
    mode: *mut cudnnPoolingMode_t,
    maxpoolingNanOpt: *mut cudnnNanPropagation_t,
    windowHeight: *mut c_int,
    windowWidth: *mut c_int,
    verticalPadding: *mut c_int,
    horizontalPadding: *mut c_int,
    verticalStride: *mut c_int,
    horizontalStride: *mut c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2311)]
fn cudnnSetPoolingNdDescriptor(
    poolingDesc: cudnnPoolingDescriptor_t,
    mode: cudnnPoolingMode_t,
    maxpoolingNanOpt: cudnnNanPropagation_t,
    nbDims: c_int,
    #[host(len = nbDims)] windowDimA: *const c_int,
    #[host(len = nbDims)] paddingA: *const c_int,
    #[host(len = nbDims)] strideA: *const c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2312)]
fn cudnnGetPoolingNdDescriptor(
    poolingDesc: cudnnPoolingDescriptor_t,
    nbDimsRequested: c_int,
    mode: *mut cudnnPoolingMode_t,
    maxpoolingNanOpt: *mut cudnnNanPropagation_t,
    nbDims: *mut c_int,
    #[host(output, len = nbDims, cap = nbDimsRequested)] windowDimA: *mut c_int,
    #[host(output, len = nbDims, cap = nbDimsRequested)] paddingA: *mut c_int,
    #[host(output, len = nbDims, cap = nbDimsRequested)] strideA: *mut c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2313)]
fn cudnnGetPoolingNdForwardOutputDim(
    poolingDesc: cudnnPoolingDescriptor_t,
    inputTensorDesc: cudnnTensorDescriptor_t,
    nbDims: c_int,
    #[host(output, len = nbDims)] outputTensorDimA: *mut c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2314)]
fn cudnnGetPooling2dForwardOutputDim(
    poolingDesc: cudnnPoolingDescriptor_t,
    inputTensorDesc: cudnnTensorDescriptor_t,
    n: *mut c_int,
    c: *mut c_int,
    h: *mut c_int,
    w: *mut c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2315, async_api)]
fn cudnnDestroyPoolingDescriptor(poolingDesc: cudnnPoolingDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2316)]
fn cudnnCreateLRNDescriptor(normDesc: *mut cudnnLRNDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2317)]
fn cudnnSetLRNDescriptor(
    normDesc: cudnnLRNDescriptor_t,
    lrnN: c_uint,
    lrnAlpha: f64,
    lrnBeta: f64,
    lrnK: f64,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2318)]
fn cudnnGetLRNDescriptor(
    normDesc: cudnnLRNDescriptor_t,
    lrnN: *mut c_uint,
    lrnAlpha: *mut f64,
    lrnBeta: *mut f64,
    lrnK: *mut f64,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2319, async_api)]
fn cudnnDestroyLRNDescriptor(lrnDesc: cudnnLRNDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2104)]
fn cudnnCreateConvolutionDescriptor(convDesc: *mut cudnnConvolutionDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2105, async_api)]
fn cudnnDestroyConvolutionDescriptor(convDesc: cudnnConvolutionDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2129, async_api)]
fn cudnnSetConvolutionNdDescriptor(
    convDesc: cudnnConvolutionDescriptor_t,
    arrayLength: c_int,
    #[host(len = arrayLength)] padA: *const c_int,
    #[host(len = arrayLength)] filterStrideA: *const c_int,
    #[host(len = arrayLength)] dilationA: *const c_int,
    mode: cudnnConvolutionMode_t,
    computeType: cudnnDataType_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2132)]
fn cudnnGetConvolutionNdDescriptor(
    convDesc: cudnnConvolutionDescriptor_t,
    arrayLengthRequested: c_int,
    arrayLength: *mut c_int,
    #[host(output, len = arrayLength, cap = arrayLengthRequested)] padA: *mut c_int,
    #[host(output, len = arrayLength, cap = arrayLengthRequested)] strideA: *mut c_int,
    #[host(output, len = arrayLength, cap = arrayLengthRequested)] dilationA: *mut c_int,
    mode: *mut cudnnConvolutionMode_t,
    computeType: *mut cudnnDataType_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2127, async_api)]
fn cudnnSetConvolutionGroupCount(
    convDesc: cudnnConvolutionDescriptor_t,
    groupCount: c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2133)]
fn cudnnGetConvolutionGroupCount(
    convDesc: cudnnConvolutionDescriptor_t,
    groupCount: *mut c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2128, async_api)]
fn cudnnSetConvolutionMathType(
    convDesc: cudnnConvolutionDescriptor_t,
    mathType: cudnnMathType_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2134)]
fn cudnnGetConvolutionMathType(
    convDesc: cudnnConvolutionDescriptor_t,
    mathType: *mut cudnnMathType_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2130)]
fn cudnnSetConvolutionReorderType(
    convDesc: cudnnConvolutionDescriptor_t,
    reorderType: cudnnReorderType_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2135)]
fn cudnnGetConvolutionReorderType(
    convDesc: cudnnConvolutionDescriptor_t,
    reorderType: *mut cudnnReorderType_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2136)]
fn cudnnGetConvolution2dForwardOutputDim(
    convDesc: cudnnConvolutionDescriptor_t,
    inputTensorDesc: cudnnTensorDescriptor_t,
    filterDesc: cudnnFilterDescriptor_t,
    n: *mut c_int,
    c: *mut c_int,
    h: *mut c_int,
    w: *mut c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2137)]
fn cudnnGetConvolutionForwardAlgorithmMaxCount(
    handle: cudnnHandle_t,
    count: *mut c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2116)]
fn cudnnGetConvolutionForwardAlgorithm_v7(
    handle: cudnnHandle_t,
    srcDesc: cudnnTensorDescriptor_t,
    filterDesc: cudnnFilterDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    destDesc: cudnnTensorDescriptor_t,
    requestedAlgoCount: c_int,
    returnedAlgoCount: *mut c_int,
    #[host(output, len = returnedAlgoCount, cap = requestedAlgoCount)]
    perfResults: *mut cudnnConvolutionFwdAlgoPerf_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2103, async_api)]
fn cudnnConvolutionForward(
    handle: cudnnHandle_t,
    #[host] alpha: *const HackedAssumeDouble,
    xDesc: cudnnTensorDescriptor_t,
    #[device] x: *const c_void,
    wDesc: cudnnFilterDescriptor_t,
    #[device] w: *const c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    #[device] workSpace: *mut c_void,
    workSpaceSizeInBytes: usize,
    #[host] beta: *const HackedAssumeDouble,
    yDesc: cudnnTensorDescriptor_t,
    #[device] y: *mut c_void,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2008)]
fn cudnnGetBatchNormalizationForwardTrainingExWorkspaceSize(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    xDesc: cudnnTensorDescriptor_t,
    zDesc: cudnnTensorDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    activationDesc: cudnnActivationDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2010)]
fn cudnnDeriveBNTensorDescriptor(
    derivedBnDesc: cudnnTensorDescriptor_t,
    xDesc: cudnnTensorDescriptor_t,
    mode: cudnnBatchNormMode_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2011)]
fn cudnnDeriveNormTensorDescriptor(
    derivedNormScaleBiasDesc: cudnnTensorDescriptor_t,
    derivedNormMeanVarDesc: cudnnTensorDescriptor_t,
    xDesc: cudnnTensorDescriptor_t,
    mode: cudnnNormMode_t,
    groupCnt: c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2009)]
fn cudnnGetBatchNormalizationTrainingExReserveSpaceSize(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    activationDesc: cudnnActivationDescriptor_t,
    xDesc: cudnnTensorDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2004, async_api)]
fn cudnnBatchNormalizationForwardTrainingEx(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    #[host] alpha: *const HackedAssumeDouble,
    #[host] beta: *const HackedAssumeDouble,
    xDesc: cudnnTensorDescriptor_t,
    #[device] xData: *const c_void,
    zDesc: cudnnTensorDescriptor_t,
    #[device] zData: *const c_void,
    yDesc: cudnnTensorDescriptor_t,
    #[device] yData: *mut c_void,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    #[device] bnScale: *const c_void,
    #[device] bnBias: *const c_void,
    exponentialAverageFactor: f64,
    #[device] resultRunningMean: *mut c_void,
    #[device] resultRunningVariance: *mut c_void,
    epsilon: f64,
    #[device] resultSaveMean: *mut c_void,
    #[device] resultSaveInvVariance: *mut c_void,
    activationDesc: cudnnActivationDescriptor_t,
    #[device] workspace: *mut c_void,
    workSpaceSizeInBytes: usize,
    #[device] reserveSpace: *mut c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2007)]
fn cudnnGetBatchNormalizationBackwardExWorkspaceSize(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    xDesc: cudnnTensorDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    dyDesc: cudnnTensorDescriptor_t,
    dzDesc: cudnnTensorDescriptor_t,
    dxDesc: cudnnTensorDescriptor_t,
    dBnScaleBiasDesc: cudnnTensorDescriptor_t,
    activationDesc: cudnnActivationDescriptor_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2002, async_api)]
fn cudnnBatchNormalizationBackwardEx(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    bnOps: cudnnBatchNormOps_t,
    #[host] alphaDataDiff: *const HackedAssumeDouble,
    #[host] betaDataDiff: *const HackedAssumeDouble,
    #[host] alphaParamDiff: *const HackedAssumeDouble,
    #[host] betaParamDiff: *const HackedAssumeDouble,
    xDesc: cudnnTensorDescriptor_t,
    #[device] xData: *const c_void,
    yDesc: cudnnTensorDescriptor_t,
    #[device] yData: *const c_void,
    dyDesc: cudnnTensorDescriptor_t,
    #[device] dyData: *const c_void,
    dzDesc: cudnnTensorDescriptor_t,
    #[device] dzData: *mut c_void,
    dxDesc: cudnnTensorDescriptor_t,
    #[device] dxData: *mut c_void,
    dBnScaleBiasDesc: cudnnTensorDescriptor_t,
    #[device] bnScaleData: *const c_void,
    #[device] bnBiasData: *const c_void,
    #[device] dBnScaleData: *mut c_void,
    #[device] dBnBiasData: *mut c_void,
    epsilon: f64,
    #[device] savedMean: *const c_void,
    #[device] savedInvVariance: *const c_void,
    activationDesc: cudnnActivationDescriptor_t,
    #[device] workSpace: *mut c_void,
    workSpaceSizeInBytes: usize,
    #[device] reserveSpace: *mut c_void,
    reserveSpaceSizeInBytes: usize,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2113)]
fn cudnnGetConvolutionBackwardDataAlgorithm_v7(
    handle: cudnnHandle_t,
    filterDesc: cudnnFilterDescriptor_t,
    diffDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    gradDesc: cudnnTensorDescriptor_t,
    requestedAlgoCount: c_int,
    returnedAlgoCount: *mut c_int,
    #[host(output, len = returnedAlgoCount, cap = requestedAlgoCount)]
    perfResults: *mut cudnnConvolutionBwdDataAlgoPerf_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2114)]
fn cudnnGetConvolutionBackwardDataAlgorithmMaxCount(
    handle: cudnnHandle_t,
    count: *mut c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2115)]
fn cudnnGetConvolutionBackwardDataWorkspaceSize(
    handle: cudnnHandle_t,
    wDesc: cudnnFilterDescriptor_t,
    dyDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    dxDesc: cudnnTensorDescriptor_t,
    algo: cudnnConvolutionBwdDataAlgo_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2101, async_api)]
fn cudnnConvolutionBackwardData(
    handle: cudnnHandle_t,
    #[host] alpha: *const HackedAssumeDouble,
    wDesc: cudnnFilterDescriptor_t,
    #[device] w: *const c_void,
    dyDesc: cudnnTensorDescriptor_t,
    #[device] dy: *const c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionBwdDataAlgo_t,
    #[device] workSpace: *mut c_void,
    workSpaceSizeInBytes: usize,
    #[host] beta: *const HackedAssumeDouble,
    dxDesc: cudnnTensorDescriptor_t,
    #[device] dx: *mut c_void,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2213)]
fn cudnnGetConvolutionBackwardFilterAlgorithm_v7(
    handle: cudnnHandle_t,
    srcDesc: cudnnTensorDescriptor_t,
    diffDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    gradDesc: cudnnFilterDescriptor_t,
    requestedAlgoCount: c_int,
    returnedAlgoCount: *mut c_int,
    #[host(output, len = returnedAlgoCount, cap = requestedAlgoCount)]
    perfResults: *mut cudnnConvolutionBwdFilterAlgoPerf_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2214)]
fn cudnnGetConvolutionBackwardFilterAlgorithmMaxCount(
    handle: cudnnHandle_t,
    count: *mut c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2215)]
fn cudnnGetConvolutionBackwardFilterWorkspaceSize(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    dyDesc: cudnnTensorDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    gradDesc: cudnnFilterDescriptor_t,
    algo: cudnnConvolutionBwdFilterAlgo_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2202, async_api)]
fn cudnnConvolutionBackwardFilter(
    handle: cudnnHandle_t,
    #[host] alpha: *const HackedAssumeDouble,
    xDesc: cudnnTensorDescriptor_t,
    #[device] x: *const c_void,
    dyDesc: cudnnTensorDescriptor_t,
    #[device] dy: *const c_void,
    convDesc: cudnnConvolutionDescriptor_t,
    algo: cudnnConvolutionBwdFilterAlgo_t,
    #[device] workSpace: *mut c_void,
    workSpaceSizeInBytes: usize,
    #[host] beta: *const HackedAssumeDouble,
    dwDesc: cudnnFilterDescriptor_t,
    #[device] dw: *mut c_void,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1802, async_api)]
fn cudnnBatchNormalizationForwardInference(
    handle: cudnnHandle_t,
    mode: cudnnBatchNormMode_t,
    #[host] alpha: *const HackedAssumeDouble,
    #[host] beta: *const HackedAssumeDouble,
    xDesc: cudnnTensorDescriptor_t,
    #[device] x: *const c_void,
    yDesc: cudnnTensorDescriptor_t,
    #[device] y: *mut c_void,
    bnScaleBiasMeanVarDesc: cudnnTensorDescriptor_t,
    #[device] bnScale: *const c_void,
    #[device] bnBias: *const c_void,
    #[device] estimatedMean: *const c_void,
    #[device] estimatedVariance: *const c_void,
    epsilon: f64,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 1864)]
fn cudnnSetFilter4dDescriptor(
    filterDesc: cudnnFilterDescriptor_t,
    dataType: cudnnDataType_t,
    format: cudnnTensorFormat_t,
    k: c_int,
    c: c_int,
    h: c_int,
    w: c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2121)]
fn cudnnGetConvolutionNdForwardOutputDim(
    convDesc: cudnnConvolutionDescriptor_t,
    inputTensorDesc: cudnnTensorDescriptor_t,
    filterDesc: cudnnFilterDescriptor_t,
    nbDims: c_int,
    #[host(output, len = nbDims)] tensorOuputDimA: *mut c_int,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2117)]
fn cudnnGetConvolutionForwardWorkspaceSize(
    handle: cudnnHandle_t,
    xDesc: cudnnTensorDescriptor_t,
    wDesc: cudnnFilterDescriptor_t,
    convDesc: cudnnConvolutionDescriptor_t,
    yDesc: cudnnTensorDescriptor_t,
    algo: cudnnConvolutionFwdAlgo_t,
    sizeInBytes: *mut usize,
) -> cudnnStatus_t;

// TODO: shadow_desc
#[cuda_hook(proc_id = 2500)]
fn cudnnBackendCreateDescriptor(
    descriptorType: cudnnBackendDescriptorType_t,
    descriptor: *mut cudnnBackendDescriptor_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2501)]
fn cudnnBackendDestroyDescriptor(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2502)]
fn cudnnBackendExecute(
    handle: cudnnHandle_t,
    executionPlan: cudnnBackendDescriptor_t,
    variantPack: cudnnBackendDescriptor_t,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2503)]
fn cudnnBackendFinalize(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2504)]
fn cudnnBackendInitialize(descriptor: cudnnBackendDescriptor_t) -> cudnnStatus_t;

#[cuda_custom_hook] // calls one of the following internal APIs
fn cudnnBackendGetAttribute(
    descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    attributeType: cudnnBackendAttributeType_t,
    requestedElementCount: i64,
    elementCount: *mut i64,
    arrayOfElements: *mut c_void,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 992504, parent = cudnnBackendGetAttribute)]
fn cudnnBackendGetAttributeCount(
    descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    attributeType: cudnnBackendAttributeType_t,
    requestedElementCount: i64,
    elementCount: *mut i64,
    #[device] arrayOfElements: *mut c_void, // null
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 992505, parent = cudnnBackendGetAttribute)]
fn cudnnBackendGetAttributeData(
    descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    attributeType: cudnnBackendAttributeType_t, // not CUDNN_TYPE_BACKEND_DESCRIPTOR
    requestedElementCount: i64,
    elementCount: *mut i64,
    // `len` and `cap` are required because some callers provide buffers shorter than `requestedElementCount`
    // so that using `len = requestedElementCount` alone will lead to memory corruption
    // https://github.com/NVIDIA/cudnn-frontend/blob/v1.11.0/include/cudnn_frontend_ExecutionPlan.h#L165-L177
    #[host(
        output,
        len = attributeType.data_size() * elementCount.to_owned(),
        cap = attributeType.data_size() * requestedElementCount,
    )]
    arrayOfElements: *mut c_void,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 992506, parent = cudnnBackendGetAttribute)]
fn cudnnBackendGetAttributeDescriptors(
    descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    attributeType: cudnnBackendAttributeType_t, // CUDNN_TYPE_BACKEND_DESCRIPTOR
    requestedElementCount: i64,
    elementCount: *mut i64,
    #[host(input, len = attributeType.data_size() * requestedElementCount)]
    arrayOfElements: *mut c_void,
) -> cudnnStatus_t;

#[cuda_hook(proc_id = 2506)]
fn cudnnBackendSetAttribute(
    descriptor: cudnnBackendDescriptor_t,
    attributeName: cudnnBackendAttributeName_t,
    attributeType: cudnnBackendAttributeType_t,
    elementCount: i64,
    #[host(len = attributeType.data_size() * elementCount)] arrayOfElements: *const c_void,
) -> cudnnStatus_t;
