use crate::types::nvml::*;
use codegen::cuda_hook;
use std::os::raw::*;

#[cuda_hook(proc_id = 991000, async_api = false)]
fn nvmlInit_v2() -> nvmlReturn_t {
    'server_execution: {
        let result = super::nvml_exe_custom::server_nvml_init_v2();
    }
}

#[cuda_hook(proc_id = 991001)]
fn nvmlDeviceGetCount_v2(deviceCount: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991002, async_api = false)]
fn nvmlInitWithFlags(flags: c_uint) -> nvmlReturn_t {
    'server_execution: {
        let result = super::nvml_exe_custom::server_nvml_init_with_flags(flags);
    }
}

#[cuda_hook(proc_id = 991003, async_api = false)]
fn nvmlShutdown() -> nvmlReturn_t {
    'server_execution: {
        let result = super::nvml_exe_custom::server_nvml_shutdown();
    }
}

#[cuda_hook(proc_id = 991004)]
fn nvmlDeviceGetHandleByIndex_v2(index: c_uint, device: *mut nvmlDevice_t) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991005)]
fn nvmlDeviceGetName(
    device: nvmlDevice_t,
    #[host(output, len = length)] name: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991006)]
fn nvmlDeviceGetCudaComputeCapability(
    device: nvmlDevice_t,
    major: *mut c_int,
    minor: *mut c_int,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991007)]
fn nvmlDeviceGetP2PStatus(
    device1: nvmlDevice_t,
    device2: nvmlDevice_t,
    p2pIndex: nvmlGpuP2PCapsIndex_t,
    p2pStatus: *mut nvmlGpuP2PStatus_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991008)]
fn nvmlDeviceGetMemoryInfo(device: nvmlDevice_t, memory: *mut nvmlMemory_t) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991009)]
fn nvmlSystemGetDriverVersion(
    #[host(output, len = length)] version: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991010)]
fn nvmlSystemGetNVMLVersion(
    #[host(output, len = length)] version: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991011)]
fn nvmlSystemGetCudaDriverVersion(cudaDriverVersion: *mut c_int) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991012)]
fn nvmlSystemGetCudaDriverVersion_v2(cudaDriverVersion: *mut c_int) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991013)]
fn nvmlDeviceGetHandleByUUID(uuid: *const c_char, device: *mut nvmlDevice_t) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991014)]
fn nvmlDeviceGetHandleByPciBusId_v2(
    pciBusId: *const c_char,
    device: *mut nvmlDevice_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991015)]
fn nvmlDeviceGetUUID(
    device: nvmlDevice_t,
    #[host(output, len = length)] uuid: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991016)]
fn nvmlDeviceGetSerial(
    device: nvmlDevice_t,
    #[host(output, len = length)] serial: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991017)]
fn nvmlDeviceGetPciInfo_v3(device: nvmlDevice_t, pci: *mut nvmlPciInfo_t) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991018)]
fn nvmlDeviceGetBrand(device: nvmlDevice_t, type_: *mut nvmlBrandType_t) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991019)]
fn nvmlDeviceGetIndex(device: nvmlDevice_t, index: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991020)]
fn nvmlDeviceGetMinorNumber(device: nvmlDevice_t, minorNumber: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991021)]
fn nvmlDeviceGetMemoryInfo_v2(device: nvmlDevice_t, memory: *mut nvmlMemory_v2_t) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991022)]
fn nvmlDeviceGetComputeMode(device: nvmlDevice_t, mode: *mut nvmlComputeMode_t) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991023)]
fn nvmlDeviceGetPersistenceMode(device: nvmlDevice_t, mode: *mut nvmlEnableState_t)
    -> nvmlReturn_t;

#[cuda_hook(proc_id = 991024)]
fn nvmlDeviceGetArchitecture(
    device: nvmlDevice_t,
    arch: *mut nvmlDeviceArchitecture_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991025)]
fn nvmlDeviceGetUtilizationRates(
    device: nvmlDevice_t,
    utilization: *mut nvmlUtilization_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991026)]
fn nvmlDeviceGetTemperature(
    device: nvmlDevice_t,
    sensorType: nvmlTemperatureSensors_t,
    temp: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991027)]
fn nvmlDeviceGetPerformanceState(device: nvmlDevice_t, pState: *mut nvmlPstates_t) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991028)]
fn nvmlDeviceGetPowerUsage(device: nvmlDevice_t, power: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991029)]
fn nvmlDeviceGetTotalEnergyConsumption(
    device: nvmlDevice_t,
    energy: *mut c_ulonglong,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991030)]
fn nvmlDeviceGetPowerManagementMode(
    device: nvmlDevice_t,
    mode: *mut nvmlEnableState_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991031)]
fn nvmlDeviceGetPowerManagementLimit(device: nvmlDevice_t, limit: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991032)]
fn nvmlDeviceGetPowerManagementDefaultLimit(
    device: nvmlDevice_t,
    defaultLimit: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991033)]
fn nvmlDeviceGetPowerManagementLimitConstraints(
    device: nvmlDevice_t,
    minLimit: *mut c_uint,
    maxLimit: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991034)]
fn nvmlDeviceGetEnforcedPowerLimit(device: nvmlDevice_t, limit: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991035)]
fn nvmlDeviceGetClockInfo(
    device: nvmlDevice_t,
    type_: nvmlClockType_t,
    clock: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991036)]
fn nvmlDeviceGetMaxClockInfo(
    device: nvmlDevice_t,
    type_: nvmlClockType_t,
    clock: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991037)]
fn nvmlDeviceGetApplicationsClock(
    device: nvmlDevice_t,
    clockType: nvmlClockType_t,
    clockMHz: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991038)]
fn nvmlDeviceGetDefaultApplicationsClock(
    device: nvmlDevice_t,
    clockType: nvmlClockType_t,
    clockMHz: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991039)]
fn nvmlDeviceGetCurrPcieLinkGeneration(
    device: nvmlDevice_t,
    currLinkGen: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991040)]
fn nvmlDeviceGetCurrPcieLinkWidth(device: nvmlDevice_t, currLinkWidth: *mut c_uint)
    -> nvmlReturn_t;

#[cuda_hook(proc_id = 991041)]
fn nvmlDeviceGetMaxPcieLinkGeneration(
    device: nvmlDevice_t,
    maxLinkGen: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991042)]
fn nvmlDeviceGetMaxPcieLinkWidth(device: nvmlDevice_t, maxLinkWidth: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991043)]
fn nvmlDeviceGetPcieThroughput(
    device: nvmlDevice_t,
    counter: nvmlPcieUtilCounter_t,
    value: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991044)]
fn nvmlDeviceGetPcieReplayCounter(device: nvmlDevice_t, value: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991045)]
fn nvmlDeviceGetFanSpeed(device: nvmlDevice_t, speed: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991046)]
fn nvmlDeviceGetVbiosVersion(
    device: nvmlDevice_t,
    #[host(output, len = length)] version: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991047)]
fn nvmlDeviceGetBoardPartNumber(
    device: nvmlDevice_t,
    #[host(output, len = length)] partNumber: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991048)]
fn nvmlDeviceGetBAR1MemoryInfo(
    device: nvmlDevice_t,
    bar1Memory: *mut nvmlBAR1Memory_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991049)]
fn nvmlDeviceGetBoardId(device: nvmlDevice_t, boardId: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991050)]
fn nvmlDeviceGetMultiGpuBoard(device: nvmlDevice_t, multiGpuBool: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991051)]
fn nvmlDeviceGetMigMode(
    device: nvmlDevice_t,
    currentMode: *mut c_uint,
    pendingMode: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991052)]
fn nvmlUnitGetCount(unitCount: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991053)]
fn nvmlDeviceGetAttributes_v2(
    device: nvmlDevice_t,
    attributes: *mut nvmlDeviceAttributes_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991054)]
fn nvmlDeviceGetModuleId(device: nvmlDevice_t, moduleId: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991055)]
fn nvmlDeviceGetNumaNodeId(device: nvmlDevice_t, node: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991056)]
fn nvmlDeviceGetUnrepairableMemoryFlag_v1(
    device: nvmlDevice_t,
    unrepairableMemoryStatus: *mut nvmlUnrepairableMemoryStatus_v1_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991057)]
fn nvmlDeviceGetTopologyCommonAncestor(
    device1: nvmlDevice_t,
    device2: nvmlDevice_t,
    pathInfo: *mut nvmlGpuTopologyLevel_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991058)]
fn nvmlDeviceGetInforomVersion(
    device: nvmlDevice_t,
    object: nvmlInforomObject_t,
    #[host(output, len = length)] version: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991059)]
fn nvmlDeviceGetInforomImageVersion(
    device: nvmlDevice_t,
    #[host(output, len = length)] version: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991060)]
fn nvmlDeviceGetInforomConfigurationChecksum(
    device: nvmlDevice_t,
    checksum: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991061)]
fn nvmlDeviceGetLastBBXFlushTime(
    device: nvmlDevice_t,
    timestamp: *mut c_ulonglong,
    durationUs: *mut c_ulong,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991062)]
fn nvmlDeviceGetDisplayMode(device: nvmlDevice_t, display: *mut nvmlEnableState_t) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991063)]
fn nvmlDeviceGetDisplayActive(
    device: nvmlDevice_t,
    isActive: *mut nvmlEnableState_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991064)]
fn nvmlDeviceGetGpuMaxPcieLinkGeneration(
    device: nvmlDevice_t,
    maxLinkGenDevice: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991065)]
fn nvmlDeviceGetGpcClkVfOffset(device: nvmlDevice_t, offset: *mut c_int) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991066)]
fn nvmlDeviceGetClock(
    device: nvmlDevice_t,
    clockType: nvmlClockType_t,
    clockId: nvmlClockId_t,
    clockMHz: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991067)]
fn nvmlDeviceGetMaxCustomerBoostClock(
    device: nvmlDevice_t,
    clockType: nvmlClockType_t,
    clockMHz: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991068)]
fn nvmlDeviceGetAutoBoostedClocksEnabled(
    device: nvmlDevice_t,
    isEnabled: *mut nvmlEnableState_t,
    defaultIsEnabled: *mut nvmlEnableState_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991069)]
fn nvmlDeviceGetFanSpeed_v2(device: nvmlDevice_t, fan: c_uint, speed: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991070)]
fn nvmlDeviceGetTargetFanSpeed(
    device: nvmlDevice_t,
    fan: c_uint,
    targetSpeed: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991071)]
fn nvmlDeviceGetMinMaxFanSpeed(
    device: nvmlDevice_t,
    minSpeed: *mut c_uint,
    maxSpeed: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991072)]
fn nvmlDeviceGetFanControlPolicy_v2(
    device: nvmlDevice_t,
    fan: c_uint,
    policy: *mut nvmlFanControlPolicy_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991073)]
fn nvmlDeviceGetNumFans(device: nvmlDevice_t, numFans: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991074)]
fn nvmlDeviceGetTemperatureThreshold(
    device: nvmlDevice_t,
    thresholdType: nvmlTemperatureThresholds_t,
    temp: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991075)]
fn nvmlDeviceGetCurrentClocksEventReasons(
    device: nvmlDevice_t,
    clocksEventReasons: *mut c_ulonglong,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991076)]
fn nvmlDeviceGetCurrentClocksThrottleReasons(
    device: nvmlDevice_t,
    clocksThrottleReasons: *mut c_ulonglong,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991077)]
fn nvmlDeviceGetSupportedClocksEventReasons(
    device: nvmlDevice_t,
    supportedClocksEventReasons: *mut c_ulonglong,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991078)]
fn nvmlDeviceGetSupportedClocksThrottleReasons(
    device: nvmlDevice_t,
    supportedClocksThrottleReasons: *mut c_ulonglong,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991079)]
fn nvmlDeviceGetPowerState(device: nvmlDevice_t, pState: *mut nvmlPstates_t) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991080)]
fn nvmlDeviceGetDynamicPstatesInfo(
    device: nvmlDevice_t,
    pDynamicPstatesInfo: *mut nvmlGpuDynamicPstatesInfo_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991081)]
fn nvmlDeviceGetMemClkVfOffset(device: nvmlDevice_t, offset: *mut c_int) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991082)]
fn nvmlDeviceGetMinMaxClockOfPState(
    device: nvmlDevice_t,
    type_: nvmlClockType_t,
    pstate: nvmlPstates_t,
    minClockMHz: *mut c_uint,
    maxClockMHz: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991083)]
fn nvmlDeviceGetSupportedPerformanceStates(
    device: nvmlDevice_t,
    #[host(
        output,
        len = (size as usize) / std::mem::size_of::<nvmlPstates_t>()
    )]
    pstates: *mut nvmlPstates_t,
    size: c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991084)]
fn nvmlDeviceGetGpcClkMinMaxVfOffset(
    device: nvmlDevice_t,
    minOffset: *mut c_int,
    maxOffset: *mut c_int,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991085)]
fn nvmlDeviceGetMemClkMinMaxVfOffset(
    device: nvmlDevice_t,
    minOffset: *mut c_int,
    maxOffset: *mut c_int,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991086)]
fn nvmlDeviceGetGpuOperationMode(
    device: nvmlDevice_t,
    current: *mut nvmlGpuOperationMode_t,
    pending: *mut nvmlGpuOperationMode_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991087)]
fn nvmlDeviceGetEccMode(
    device: nvmlDevice_t,
    current: *mut nvmlEnableState_t,
    pending: *mut nvmlEnableState_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991088)]
fn nvmlDeviceGetDefaultEccMode(
    device: nvmlDevice_t,
    defaultMode: *mut nvmlEnableState_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991089)]
fn nvmlDeviceGetTotalEccErrors(
    device: nvmlDevice_t,
    errorType: nvmlMemoryErrorType_t,
    counterType: nvmlEccCounterType_t,
    eccCounts: *mut c_ulonglong,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991090)]
fn nvmlDeviceGetDetailedEccErrors(
    device: nvmlDevice_t,
    errorType: nvmlMemoryErrorType_t,
    counterType: nvmlEccCounterType_t,
    eccCounts: *mut nvmlEccErrorCounts_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991091)]
fn nvmlDeviceGetMemoryErrorCounter(
    device: nvmlDevice_t,
    errorType: nvmlMemoryErrorType_t,
    counterType: nvmlEccCounterType_t,
    locationType: nvmlMemoryLocation_t,
    count: *mut c_ulonglong,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991092)]
fn nvmlDeviceGetEncoderUtilization(
    device: nvmlDevice_t,
    utilization: *mut c_uint,
    samplingPeriodUs: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991093)]
fn nvmlDeviceGetEncoderCapacity(
    device: nvmlDevice_t,
    encoderQueryType: nvmlEncoderType_t,
    encoderCapacity: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991094)]
fn nvmlDeviceGetEncoderStats(
    device: nvmlDevice_t,
    sessionCount: *mut c_uint,
    averageFps: *mut c_uint,
    averageLatency: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991095)]
fn nvmlDeviceGetDecoderUtilization(
    device: nvmlDevice_t,
    utilization: *mut c_uint,
    samplingPeriodUs: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991096)]
fn nvmlDeviceGetJpgUtilization(
    device: nvmlDevice_t,
    utilization: *mut c_uint,
    samplingPeriodUs: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991097)]
fn nvmlDeviceGetOfaUtilization(
    device: nvmlDevice_t,
    utilization: *mut c_uint,
    samplingPeriodUs: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991098)]
fn nvmlDeviceGetFBCStats(device: nvmlDevice_t, fbcStats: *mut nvmlFBCStats_t) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991099)]
fn nvmlDeviceGetDriverModel_v2(
    device: nvmlDevice_t,
    current: *mut nvmlDriverModel_t,
    pending: *mut nvmlDriverModel_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991100)]
fn nvmlDeviceGetBridgeChipInfo(
    device: nvmlDevice_t,
    bridgeHierarchy: *mut nvmlBridgeChipHierarchy_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991101)]
fn nvmlDeviceOnSameBoard(
    device1: nvmlDevice_t,
    device2: nvmlDevice_t,
    onSameBoard: *mut c_int,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991102)]
fn nvmlDeviceGetAPIRestriction(
    device: nvmlDevice_t,
    apiType: nvmlRestrictedAPI_t,
    isRestricted: *mut nvmlEnableState_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991103)]
fn nvmlDeviceGetViolationStatus(
    device: nvmlDevice_t,
    perfPolicyType: nvmlPerfPolicyType_t,
    violTime: *mut nvmlViolationTime_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991104)]
fn nvmlDeviceGetIrqNum(device: nvmlDevice_t, irqNum: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991105)]
fn nvmlDeviceGetNumGpuCores(device: nvmlDevice_t, numCores: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991106)]
fn nvmlDeviceGetPowerSource(
    device: nvmlDevice_t,
    powerSource: *mut nvmlPowerSource_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991107)]
fn nvmlDeviceGetMemoryBusWidth(device: nvmlDevice_t, busWidth: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991108)]
fn nvmlDeviceGetPcieLinkMaxSpeed(device: nvmlDevice_t, maxSpeed: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991109)]
fn nvmlDeviceGetPcieSpeed(device: nvmlDevice_t, pcieSpeed: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991110)]
fn nvmlDeviceGetAdaptiveClockInfoStatus(
    device: nvmlDevice_t,
    adaptiveClockStatus: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991111)]
fn nvmlDeviceGetBusType(device: nvmlDevice_t, type_: *mut nvmlBusType_t) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991112)]
fn nvmlDeviceGetGpuFabricInfo(
    device: nvmlDevice_t,
    gpuFabricInfo: *mut nvmlGpuFabricInfo_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991113)]
fn nvmlDeviceGetGspFirmwareVersion(
    device: nvmlDevice_t,
    #[host(output, len = NVML_GSP_FIRMWARE_VERSION_BUF_SIZE as usize)] version: *mut c_char,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991114)]
fn nvmlDeviceGetGspFirmwareMode(
    device: nvmlDevice_t,
    isEnabled: *mut c_uint,
    defaultMode: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991115)]
fn nvmlDeviceGetAccountingMode(device: nvmlDevice_t, mode: *mut nvmlEnableState_t) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991116)]
fn nvmlDeviceGetAccountingBufferSize(device: nvmlDevice_t, bufferSize: *mut c_uint)
    -> nvmlReturn_t;

#[cuda_hook(proc_id = 991117)]
fn nvmlDeviceGetRetiredPagesPendingStatus(
    device: nvmlDevice_t,
    isPending: *mut nvmlEnableState_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991118)]
fn nvmlDeviceGetRowRemapperHistogram(
    device: nvmlDevice_t,
    values: *mut nvmlRowRemapperHistogramValues_t,
) -> nvmlReturn_t;
