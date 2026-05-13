use crate::types::nvml::*;
use codegen::{cuda_custom_hook, cuda_hook};
use std::os::raw::*;

#[cuda_custom_hook] // local: returns a client-owned C string
fn nvmlErrorString(result: nvmlReturn_t) -> *const c_char;

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

#[cuda_hook(proc_id = 991119)]
fn nvmlSystemGetDriverBranch(
    branchInfo: *mut nvmlSystemDriverBranchInfo_t,
    length: c_uint,
) -> nvmlReturn_t {
    'client_extra_send: {
        assert!(!branchInfo.is_null());
        unsafe { std::ptr::read_unaligned(branchInfo) }
            .send(channel_sender)
            .unwrap();
    }
    'server_extra_recv: {
        let mut branchInfo_in = std::mem::MaybeUninit::<nvmlSystemDriverBranchInfo_t>::uninit();
        branchInfo_in.recv(channel_receiver).unwrap();
        let branchInfo_in = unsafe { branchInfo_in.assume_init() };
    }
    'server_before_execution: {
        branchInfo.write(branchInfo_in);
    }
}

#[cuda_hook(proc_id = 991120)]
fn nvmlDeviceGetC2cModeInfoV(
    device: nvmlDevice_t,
    c2cModeInfo: *mut nvmlC2cModeInfo_v1_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991121)]
fn nvmlDeviceGetAddressingMode(
    device: nvmlDevice_t,
    mode: *mut nvmlDeviceAddressingMode_t,
) -> nvmlReturn_t {
    'client_extra_send: {
        assert!(!mode.is_null());
        unsafe { std::ptr::read_unaligned(mode) }
            .send(channel_sender)
            .unwrap();
    }
    'server_extra_recv: {
        let mut mode_in = std::mem::MaybeUninit::<nvmlDeviceAddressingMode_t>::uninit();
        mode_in.recv(channel_receiver).unwrap();
        let mode_in = unsafe { mode_in.assume_init() };
    }
    'server_before_execution: {
        mode.write(mode_in);
    }
}

#[cuda_hook(proc_id = 991122)]
fn nvmlDeviceGetRepairStatus(
    device: nvmlDevice_t,
    repairStatus: *mut nvmlRepairStatus_t,
) -> nvmlReturn_t {
    'client_extra_send: {
        assert!(!repairStatus.is_null());
        unsafe { std::ptr::read_unaligned(repairStatus) }
            .send(channel_sender)
            .unwrap();
    }
    'server_extra_recv: {
        let mut repairStatus_in = std::mem::MaybeUninit::<nvmlRepairStatus_t>::uninit();
        repairStatus_in.recv(channel_receiver).unwrap();
        let repairStatus_in = unsafe { repairStatus_in.assume_init() };
    }
    'server_before_execution: {
        repairStatus.write(repairStatus_in);
    }
}

#[cuda_hook(proc_id = 991123)]
fn nvmlDeviceGetPciInfoExt(device: nvmlDevice_t, pci: *mut nvmlPciInfoExt_t) -> nvmlReturn_t {
    'client_extra_send: {
        assert!(!pci.is_null());
        unsafe { std::ptr::read_unaligned(pci) }
            .send(channel_sender)
            .unwrap();
    }
    'server_extra_recv: {
        let mut pci_in = std::mem::MaybeUninit::<nvmlPciInfoExt_t>::uninit();
        pci_in.recv(channel_receiver).unwrap();
        let pci_in = unsafe { pci_in.assume_init() };
    }
    'server_before_execution: {
        pci.write(pci_in);
    }
}

#[cuda_hook(proc_id = 991124)]
fn nvmlDeviceGetFanSpeedRPM(
    device: nvmlDevice_t,
    fanSpeed: *mut nvmlFanSpeedInfo_t,
) -> nvmlReturn_t {
    'client_extra_send: {
        assert!(!fanSpeed.is_null());
        unsafe { std::ptr::read_unaligned(fanSpeed) }
            .send(channel_sender)
            .unwrap();
    }
    'server_extra_recv: {
        let mut fanSpeed_in = std::mem::MaybeUninit::<nvmlFanSpeedInfo_t>::uninit();
        fanSpeed_in.recv(channel_receiver).unwrap();
        let fanSpeed_in = unsafe { fanSpeed_in.assume_init() };
    }
    'server_before_execution: {
        fanSpeed.write(fanSpeed_in);
    }
}

#[cuda_hook(proc_id = 991125)]
fn nvmlDeviceGetCoolerInfo(
    device: nvmlDevice_t,
    coolerInfo: *mut nvmlCoolerInfo_t,
) -> nvmlReturn_t {
    'client_extra_send: {
        assert!(!coolerInfo.is_null());
        unsafe { std::ptr::read_unaligned(coolerInfo) }
            .send(channel_sender)
            .unwrap();
    }
    'server_extra_recv: {
        let mut coolerInfo_in = std::mem::MaybeUninit::<nvmlCoolerInfo_t>::uninit();
        coolerInfo_in.recv(channel_receiver).unwrap();
        let coolerInfo_in = unsafe { coolerInfo_in.assume_init() };
    }
    'server_before_execution: {
        coolerInfo.write(coolerInfo_in);
    }
}

#[cuda_hook(proc_id = 991126)]
fn nvmlDeviceGetTemperatureV(
    device: nvmlDevice_t,
    temperature: *mut nvmlTemperature_t,
) -> nvmlReturn_t {
    'client_extra_send: {
        assert!(!temperature.is_null());
        unsafe { std::ptr::read_unaligned(temperature) }
            .send(channel_sender)
            .unwrap();
    }
    'server_extra_recv: {
        let mut temperature_in = std::mem::MaybeUninit::<nvmlTemperature_t>::uninit();
        temperature_in.recv(channel_receiver).unwrap();
        let temperature_in = unsafe { temperature_in.assume_init() };
    }
    'server_before_execution: {
        temperature.write(temperature_in);
    }
}

#[cuda_hook(proc_id = 991127)]
fn nvmlDeviceGetMarginTemperature(
    device: nvmlDevice_t,
    marginTempInfo: *mut nvmlMarginTemperature_t,
) -> nvmlReturn_t {
    'client_extra_send: {
        assert!(!marginTempInfo.is_null());
        unsafe { std::ptr::read_unaligned(marginTempInfo) }
            .send(channel_sender)
            .unwrap();
    }
    'server_extra_recv: {
        let mut marginTempInfo_in = std::mem::MaybeUninit::<nvmlMarginTemperature_t>::uninit();
        marginTempInfo_in.recv(channel_receiver).unwrap();
        let marginTempInfo_in = unsafe { marginTempInfo_in.assume_init() };
    }
    'server_before_execution: {
        marginTempInfo.write(marginTempInfo_in);
    }
}

#[cuda_hook(proc_id = 991128)]
fn nvmlDeviceGetClockOffsets(device: nvmlDevice_t, info: *mut nvmlClockOffset_t) -> nvmlReturn_t {
    'client_extra_send: {
        assert!(!info.is_null());
        unsafe { std::ptr::read_unaligned(info) }
            .send(channel_sender)
            .unwrap();
    }
    'server_extra_recv: {
        let mut info_in = std::mem::MaybeUninit::<nvmlClockOffset_t>::uninit();
        info_in.recv(channel_receiver).unwrap();
        let info_in = unsafe { info_in.assume_init() };
    }
    'server_before_execution: {
        info.write(info_in);
    }
}

#[cuda_hook(proc_id = 991129)]
fn nvmlDeviceGetPerformanceModes(
    device: nvmlDevice_t,
    perfModes: *mut nvmlDevicePerfModes_t,
) -> nvmlReturn_t {
    'client_extra_send: {
        assert!(!perfModes.is_null());
        unsafe { std::ptr::read_unaligned(perfModes) }
            .send(channel_sender)
            .unwrap();
    }
    'server_extra_recv: {
        let mut perfModes_in = std::mem::MaybeUninit::<nvmlDevicePerfModes_t>::uninit();
        perfModes_in.recv(channel_receiver).unwrap();
        let perfModes_in = unsafe { perfModes_in.assume_init() };
    }
    'server_before_execution: {
        perfModes.write(perfModes_in);
    }
}

#[cuda_hook(proc_id = 991130)]
fn nvmlDeviceGetCurrentClockFreqs(
    device: nvmlDevice_t,
    currentClockFreqs: *mut nvmlDeviceCurrentClockFreqs_t,
) -> nvmlReturn_t {
    'client_extra_send: {
        assert!(!currentClockFreqs.is_null());
        unsafe { std::ptr::read_unaligned(currentClockFreqs) }
            .send(channel_sender)
            .unwrap();
    }
    'server_extra_recv: {
        let mut currentClockFreqs_in =
            std::mem::MaybeUninit::<nvmlDeviceCurrentClockFreqs_t>::uninit();
        currentClockFreqs_in.recv(channel_receiver).unwrap();
        let currentClockFreqs_in = unsafe { currentClockFreqs_in.assume_init() };
    }
    'server_before_execution: {
        currentClockFreqs.write(currentClockFreqs_in);
    }
}

#[cuda_hook(proc_id = 991131)]
fn nvmlDeviceGetPowerMizerMode_v1(
    device: nvmlDevice_t,
    powerMizerMode: *mut nvmlDevicePowerMizerModes_v1_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991132)]
fn nvmlDeviceGetGpuFabricInfoV(
    device: nvmlDevice_t,
    gpuFabricInfo: *mut nvmlGpuFabricInfoV_t,
) -> nvmlReturn_t {
    'client_extra_send: {
        assert!(!gpuFabricInfo.is_null());
        unsafe { std::ptr::read_unaligned(gpuFabricInfo) }
            .send(channel_sender)
            .unwrap();
    }
    'server_extra_recv: {
        let mut gpuFabricInfo_in = std::mem::MaybeUninit::<nvmlGpuFabricInfoV_t>::uninit();
        gpuFabricInfo_in.recv(channel_receiver).unwrap();
        let gpuFabricInfo_in = unsafe { gpuFabricInfo_in.assume_init() };
    }
    'server_before_execution: {
        gpuFabricInfo.write(gpuFabricInfo_in);
    }
}

#[cuda_hook(proc_id = 991133)]
fn nvmlSystemGetProcessName(
    pid: c_uint,
    #[host(output, len = length)] name: *mut c_char,
    length: c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991134)]
fn nvmlDeviceGetTopologyNearestGpus(
    device: nvmlDevice_t,
    level: nvmlGpuTopologyLevel_t,
    #[skip] count: *mut c_uint,
    #[skip] deviceArray: *mut nvmlDevice_t,
) -> nvmlReturn_t {
    'client_before_send: {
        let count_is_null = count.is_null();
        let deviceArray_is_null = deviceArray.is_null();
        let count_in = if count_is_null {
            0
        } else {
            unsafe { std::ptr::read_unaligned(count) }
        };
    }
    'client_extra_send: {
        count_is_null.send(channel_sender).unwrap();
        deviceArray_is_null.send(channel_sender).unwrap();
        count_in.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut count_is_null = false;
        count_is_null.recv(channel_receiver).unwrap();
        let mut deviceArray_is_null = false;
        deviceArray_is_null.recv(channel_receiver).unwrap();
        let mut count_in = 0u32;
        count_in.recv(channel_receiver).unwrap();
    }
    'server_execution: {
        let mut count_storage = count_in;
        let mut deviceArray_storage =
            Vec::<std::mem::MaybeUninit<nvmlDevice_t>>::with_capacity(count_in as usize);
        let count_ptr = if count_is_null {
            std::ptr::null_mut()
        } else {
            &mut count_storage
        };
        let deviceArray_ptr = if deviceArray_is_null {
            std::ptr::null_mut()
        } else {
            deviceArray_storage.as_mut_ptr().cast::<nvmlDevice_t>()
        };
        let result =
            unsafe { nvmlDeviceGetTopologyNearestGpus(device, level, count_ptr, deviceArray_ptr) };
    }
    'server_after_send: {
        if !count_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            count_storage.send(channel_sender).unwrap();
            if result == nvmlReturn_t::NVML_SUCCESS && !deviceArray_is_null && count_storage > 0 {
                assert!((count_storage as usize) <= count_in as usize);
                let deviceArray_out = unsafe {
                    std::slice::from_raw_parts(
                        deviceArray_storage.as_ptr().cast::<nvmlDevice_t>(),
                        count_storage as usize,
                    )
                };
                send_slice(deviceArray_out, channel_sender).unwrap();
            }
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if !count_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            let mut count_out = 0u32;
            count_out.recv(channel_receiver).unwrap();
            unsafe {
                std::ptr::write(count, count_out);
            }
            if result == nvmlReturn_t::NVML_SUCCESS && !deviceArray_is_null && count_out > 0 {
                assert!((count_out as usize) <= count_in as usize);
                let deviceArray_out =
                    unsafe { std::slice::from_raw_parts_mut(deviceArray, count_out as usize) };
                recv_slice_to(deviceArray_out, channel_receiver).unwrap();
            }
        }
    }
}

#[cuda_hook(proc_id = 991135)]
fn nvmlDeviceGetSupportedMemoryClocks(
    device: nvmlDevice_t,
    #[skip] count: *mut c_uint,
    #[skip] clocksMHz: *mut c_uint,
) -> nvmlReturn_t {
    'client_before_send: {
        let count_is_null = count.is_null();
        let clocksMHz_is_null = clocksMHz.is_null();
        let count_in = if count_is_null {
            0
        } else {
            unsafe { std::ptr::read_unaligned(count) }
        };
    }
    'client_extra_send: {
        count_is_null.send(channel_sender).unwrap();
        clocksMHz_is_null.send(channel_sender).unwrap();
        count_in.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut count_is_null = false;
        count_is_null.recv(channel_receiver).unwrap();
        let mut clocksMHz_is_null = false;
        clocksMHz_is_null.recv(channel_receiver).unwrap();
        let mut count_in = 0u32;
        count_in.recv(channel_receiver).unwrap();
    }
    'server_execution: {
        let mut count_storage = count_in;
        let mut clocksMHz_storage =
            Vec::<std::mem::MaybeUninit<c_uint>>::with_capacity(count_in as usize);
        let count_ptr = if count_is_null {
            std::ptr::null_mut()
        } else {
            &mut count_storage
        };
        let clocksMHz_ptr = if clocksMHz_is_null {
            std::ptr::null_mut()
        } else {
            clocksMHz_storage.as_mut_ptr().cast::<c_uint>()
        };
        let result =
            unsafe { nvmlDeviceGetSupportedMemoryClocks(device, count_ptr, clocksMHz_ptr) };
    }
    'server_after_send: {
        if !count_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            count_storage.send(channel_sender).unwrap();
            if result == nvmlReturn_t::NVML_SUCCESS && !clocksMHz_is_null && count_storage > 0 {
                assert!((count_storage as usize) <= count_in as usize);
                let clocksMHz_out = unsafe {
                    std::slice::from_raw_parts(
                        clocksMHz_storage.as_ptr().cast::<c_uint>(),
                        count_storage as usize,
                    )
                };
                send_slice(clocksMHz_out, channel_sender).unwrap();
            }
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if !count_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            let mut count_out = 0u32;
            count_out.recv(channel_receiver).unwrap();
            unsafe {
                std::ptr::write(count, count_out);
            }
            if result == nvmlReturn_t::NVML_SUCCESS && !clocksMHz_is_null && count_out > 0 {
                assert!((count_out as usize) <= count_in as usize);
                let clocksMHz_out =
                    unsafe { std::slice::from_raw_parts_mut(clocksMHz, count_out as usize) };
                recv_slice_to(clocksMHz_out, channel_receiver).unwrap();
            }
        }
    }
}

#[cuda_hook(proc_id = 991136)]
fn nvmlDeviceGetSupportedGraphicsClocks(
    device: nvmlDevice_t,
    memoryClockMHz: c_uint,
    #[skip] count: *mut c_uint,
    #[skip] clocksMHz: *mut c_uint,
) -> nvmlReturn_t {
    'client_before_send: {
        let count_is_null = count.is_null();
        let clocksMHz_is_null = clocksMHz.is_null();
        let count_in = if count_is_null {
            0
        } else {
            unsafe { std::ptr::read_unaligned(count) }
        };
    }
    'client_extra_send: {
        count_is_null.send(channel_sender).unwrap();
        clocksMHz_is_null.send(channel_sender).unwrap();
        count_in.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut count_is_null = false;
        count_is_null.recv(channel_receiver).unwrap();
        let mut clocksMHz_is_null = false;
        clocksMHz_is_null.recv(channel_receiver).unwrap();
        let mut count_in = 0u32;
        count_in.recv(channel_receiver).unwrap();
    }
    'server_execution: {
        let mut count_storage = count_in;
        let mut clocksMHz_storage =
            Vec::<std::mem::MaybeUninit<c_uint>>::with_capacity(count_in as usize);
        let count_ptr = if count_is_null {
            std::ptr::null_mut()
        } else {
            &mut count_storage
        };
        let clocksMHz_ptr = if clocksMHz_is_null {
            std::ptr::null_mut()
        } else {
            clocksMHz_storage.as_mut_ptr().cast::<c_uint>()
        };
        let result = unsafe {
            nvmlDeviceGetSupportedGraphicsClocks(device, memoryClockMHz, count_ptr, clocksMHz_ptr)
        };
    }
    'server_after_send: {
        if !count_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            count_storage.send(channel_sender).unwrap();
            if result == nvmlReturn_t::NVML_SUCCESS && !clocksMHz_is_null && count_storage > 0 {
                assert!((count_storage as usize) <= count_in as usize);
                let clocksMHz_out = unsafe {
                    std::slice::from_raw_parts(
                        clocksMHz_storage.as_ptr().cast::<c_uint>(),
                        count_storage as usize,
                    )
                };
                send_slice(clocksMHz_out, channel_sender).unwrap();
            }
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if !count_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            let mut count_out = 0u32;
            count_out.recv(channel_receiver).unwrap();
            unsafe {
                std::ptr::write(count, count_out);
            }
            if result == nvmlReturn_t::NVML_SUCCESS && !clocksMHz_is_null && count_out > 0 {
                assert!((count_out as usize) <= count_in as usize);
                let clocksMHz_out =
                    unsafe { std::slice::from_raw_parts_mut(clocksMHz, count_out as usize) };
                recv_slice_to(clocksMHz_out, channel_receiver).unwrap();
            }
        }
    }
}

#[cuda_hook(proc_id = 991137)]
fn nvmlDeviceGetComputeRunningProcesses_v3(
    device: nvmlDevice_t,
    #[skip] infoCount: *mut c_uint,
    #[skip] infos: *mut nvmlProcessInfo_t,
) -> nvmlReturn_t {
    'client_before_send: {
        let infoCount_is_null = infoCount.is_null();
        let infos_is_null = infos.is_null();
        let infoCount_in = if infoCount_is_null {
            0
        } else {
            unsafe { std::ptr::read_unaligned(infoCount) }
        };
    }
    'client_extra_send: {
        infoCount_is_null.send(channel_sender).unwrap();
        infos_is_null.send(channel_sender).unwrap();
        infoCount_in.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut infoCount_is_null = false;
        infoCount_is_null.recv(channel_receiver).unwrap();
        let mut infos_is_null = false;
        infos_is_null.recv(channel_receiver).unwrap();
        let mut infoCount_in = 0u32;
        infoCount_in.recv(channel_receiver).unwrap();
    }
    'server_execution: {
        let mut infoCount_storage = infoCount_in;
        let mut infos_storage =
            Vec::<std::mem::MaybeUninit<nvmlProcessInfo_t>>::with_capacity(infoCount_in as usize);
        let infoCount_ptr = if infoCount_is_null {
            std::ptr::null_mut()
        } else {
            &mut infoCount_storage
        };
        let infos_ptr = if infos_is_null {
            std::ptr::null_mut()
        } else {
            infos_storage.as_mut_ptr().cast::<nvmlProcessInfo_t>()
        };
        let result =
            unsafe { nvmlDeviceGetComputeRunningProcesses_v3(device, infoCount_ptr, infos_ptr) };
    }
    'server_after_send: {
        if !infoCount_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            infoCount_storage.send(channel_sender).unwrap();
            if result == nvmlReturn_t::NVML_SUCCESS && !infos_is_null && infoCount_storage > 0 {
                assert!((infoCount_storage as usize) <= infoCount_in as usize);
                let infos_out = unsafe {
                    std::slice::from_raw_parts(
                        infos_storage.as_ptr().cast::<nvmlProcessInfo_t>(),
                        infoCount_storage as usize,
                    )
                };
                send_slice(infos_out, channel_sender).unwrap();
            }
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if !infoCount_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            let mut infoCount_out = 0u32;
            infoCount_out.recv(channel_receiver).unwrap();
            unsafe {
                std::ptr::write(infoCount, infoCount_out);
            }
            if result == nvmlReturn_t::NVML_SUCCESS && !infos_is_null && infoCount_out > 0 {
                assert!((infoCount_out as usize) <= infoCount_in as usize);
                let infos_out =
                    unsafe { std::slice::from_raw_parts_mut(infos, infoCount_out as usize) };
                recv_slice_to(infos_out, channel_receiver).unwrap();
            }
        }
    }
}

#[cuda_hook(proc_id = 991138)]
fn nvmlDeviceGetGraphicsRunningProcesses_v3(
    device: nvmlDevice_t,
    #[skip] infoCount: *mut c_uint,
    #[skip] infos: *mut nvmlProcessInfo_t,
) -> nvmlReturn_t {
    'client_before_send: {
        let infoCount_is_null = infoCount.is_null();
        let infos_is_null = infos.is_null();
        let infoCount_in = if infoCount_is_null {
            0
        } else {
            unsafe { std::ptr::read_unaligned(infoCount) }
        };
    }
    'client_extra_send: {
        infoCount_is_null.send(channel_sender).unwrap();
        infos_is_null.send(channel_sender).unwrap();
        infoCount_in.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut infoCount_is_null = false;
        infoCount_is_null.recv(channel_receiver).unwrap();
        let mut infos_is_null = false;
        infos_is_null.recv(channel_receiver).unwrap();
        let mut infoCount_in = 0u32;
        infoCount_in.recv(channel_receiver).unwrap();
    }
    'server_execution: {
        let mut infoCount_storage = infoCount_in;
        let mut infos_storage =
            Vec::<std::mem::MaybeUninit<nvmlProcessInfo_t>>::with_capacity(infoCount_in as usize);
        let infoCount_ptr = if infoCount_is_null {
            std::ptr::null_mut()
        } else {
            &mut infoCount_storage
        };
        let infos_ptr = if infos_is_null {
            std::ptr::null_mut()
        } else {
            infos_storage.as_mut_ptr().cast::<nvmlProcessInfo_t>()
        };
        let result =
            unsafe { nvmlDeviceGetGraphicsRunningProcesses_v3(device, infoCount_ptr, infos_ptr) };
    }
    'server_after_send: {
        if !infoCount_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            infoCount_storage.send(channel_sender).unwrap();
            if result == nvmlReturn_t::NVML_SUCCESS && !infos_is_null && infoCount_storage > 0 {
                assert!((infoCount_storage as usize) <= infoCount_in as usize);
                let infos_out = unsafe {
                    std::slice::from_raw_parts(
                        infos_storage.as_ptr().cast::<nvmlProcessInfo_t>(),
                        infoCount_storage as usize,
                    )
                };
                send_slice(infos_out, channel_sender).unwrap();
            }
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if !infoCount_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            let mut infoCount_out = 0u32;
            infoCount_out.recv(channel_receiver).unwrap();
            unsafe {
                std::ptr::write(infoCount, infoCount_out);
            }
            if result == nvmlReturn_t::NVML_SUCCESS && !infos_is_null && infoCount_out > 0 {
                assert!((infoCount_out as usize) <= infoCount_in as usize);
                let infos_out =
                    unsafe { std::slice::from_raw_parts_mut(infos, infoCount_out as usize) };
                recv_slice_to(infos_out, channel_receiver).unwrap();
            }
        }
    }
}

#[cuda_hook(proc_id = 991139)]
fn nvmlDeviceGetMPSComputeRunningProcesses_v3(
    device: nvmlDevice_t,
    #[skip] infoCount: *mut c_uint,
    #[skip] infos: *mut nvmlProcessInfo_t,
) -> nvmlReturn_t {
    'client_before_send: {
        let infoCount_is_null = infoCount.is_null();
        let infos_is_null = infos.is_null();
        let infoCount_in = if infoCount_is_null {
            0
        } else {
            unsafe { std::ptr::read_unaligned(infoCount) }
        };
    }
    'client_extra_send: {
        infoCount_is_null.send(channel_sender).unwrap();
        infos_is_null.send(channel_sender).unwrap();
        infoCount_in.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut infoCount_is_null = false;
        infoCount_is_null.recv(channel_receiver).unwrap();
        let mut infos_is_null = false;
        infos_is_null.recv(channel_receiver).unwrap();
        let mut infoCount_in = 0u32;
        infoCount_in.recv(channel_receiver).unwrap();
    }
    'server_execution: {
        let mut infoCount_storage = infoCount_in;
        let mut infos_storage =
            Vec::<std::mem::MaybeUninit<nvmlProcessInfo_t>>::with_capacity(infoCount_in as usize);
        let infoCount_ptr = if infoCount_is_null {
            std::ptr::null_mut()
        } else {
            &mut infoCount_storage
        };
        let infos_ptr = if infos_is_null {
            std::ptr::null_mut()
        } else {
            infos_storage.as_mut_ptr().cast::<nvmlProcessInfo_t>()
        };
        let result =
            unsafe { nvmlDeviceGetMPSComputeRunningProcesses_v3(device, infoCount_ptr, infos_ptr) };
    }
    'server_after_send: {
        if !infoCount_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            infoCount_storage.send(channel_sender).unwrap();
            if result == nvmlReturn_t::NVML_SUCCESS && !infos_is_null && infoCount_storage > 0 {
                assert!((infoCount_storage as usize) <= infoCount_in as usize);
                let infos_out = unsafe {
                    std::slice::from_raw_parts(
                        infos_storage.as_ptr().cast::<nvmlProcessInfo_t>(),
                        infoCount_storage as usize,
                    )
                };
                send_slice(infos_out, channel_sender).unwrap();
            }
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if !infoCount_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            let mut infoCount_out = 0u32;
            infoCount_out.recv(channel_receiver).unwrap();
            unsafe {
                std::ptr::write(infoCount, infoCount_out);
            }
            if result == nvmlReturn_t::NVML_SUCCESS && !infos_is_null && infoCount_out > 0 {
                assert!((infoCount_out as usize) <= infoCount_in as usize);
                let infos_out =
                    unsafe { std::slice::from_raw_parts_mut(infos, infoCount_out as usize) };
                recv_slice_to(infos_out, channel_receiver).unwrap();
            }
        }
    }
}

#[cuda_hook(proc_id = 991140)]
fn nvmlDeviceGetAccountingPids(
    device: nvmlDevice_t,
    #[skip] count: *mut c_uint,
    #[skip] pids: *mut c_uint,
) -> nvmlReturn_t {
    'client_before_send: {
        let count_is_null = count.is_null();
        let pids_is_null = pids.is_null();
        let count_in = if count_is_null {
            0
        } else {
            unsafe { std::ptr::read_unaligned(count) }
        };
    }
    'client_extra_send: {
        count_is_null.send(channel_sender).unwrap();
        pids_is_null.send(channel_sender).unwrap();
        count_in.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut count_is_null = false;
        count_is_null.recv(channel_receiver).unwrap();
        let mut pids_is_null = false;
        pids_is_null.recv(channel_receiver).unwrap();
        let mut count_in = 0u32;
        count_in.recv(channel_receiver).unwrap();
    }
    'server_execution: {
        let mut count_storage = count_in;
        let mut pids_storage =
            Vec::<std::mem::MaybeUninit<c_uint>>::with_capacity(count_in as usize);
        let count_ptr = if count_is_null {
            std::ptr::null_mut()
        } else {
            &mut count_storage
        };
        let pids_ptr = if pids_is_null {
            std::ptr::null_mut()
        } else {
            pids_storage.as_mut_ptr().cast::<c_uint>()
        };
        let result = unsafe { nvmlDeviceGetAccountingPids(device, count_ptr, pids_ptr) };
    }
    'server_after_send: {
        if !count_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            count_storage.send(channel_sender).unwrap();
            if result == nvmlReturn_t::NVML_SUCCESS && !pids_is_null && count_storage > 0 {
                assert!((count_storage as usize) <= count_in as usize);
                let pids_out = unsafe {
                    std::slice::from_raw_parts(
                        pids_storage.as_ptr().cast::<c_uint>(),
                        count_storage as usize,
                    )
                };
                send_slice(pids_out, channel_sender).unwrap();
            }
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if !count_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            let mut count_out = 0u32;
            count_out.recv(channel_receiver).unwrap();
            unsafe {
                std::ptr::write(count, count_out);
            }
            if result == nvmlReturn_t::NVML_SUCCESS && !pids_is_null && count_out > 0 {
                assert!((count_out as usize) <= count_in as usize);
                let pids_out = unsafe { std::slice::from_raw_parts_mut(pids, count_out as usize) };
                recv_slice_to(pids_out, channel_receiver).unwrap();
            }
        }
    }
}

#[cuda_hook(proc_id = 991141)]
fn nvmlDeviceGetAccountingStats(
    device: nvmlDevice_t,
    pid: c_uint,
    stats: *mut nvmlAccountingStats_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991142)]
fn nvmlDeviceGetRetiredPages(
    device: nvmlDevice_t,
    cause: nvmlPageRetirementCause_t,
    #[skip] pageCount: *mut c_uint,
    #[skip] addresses: *mut c_ulonglong,
) -> nvmlReturn_t {
    'client_before_send: {
        let pageCount_is_null = pageCount.is_null();
        let addresses_is_null = addresses.is_null();
        let pageCount_in = if pageCount_is_null {
            0
        } else {
            unsafe { std::ptr::read_unaligned(pageCount) }
        };
    }
    'client_extra_send: {
        pageCount_is_null.send(channel_sender).unwrap();
        addresses_is_null.send(channel_sender).unwrap();
        pageCount_in.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut pageCount_is_null = false;
        pageCount_is_null.recv(channel_receiver).unwrap();
        let mut addresses_is_null = false;
        addresses_is_null.recv(channel_receiver).unwrap();
        let mut pageCount_in = 0u32;
        pageCount_in.recv(channel_receiver).unwrap();
    }
    'server_execution: {
        let mut pageCount_storage = pageCount_in;
        let mut addresses_storage =
            Vec::<std::mem::MaybeUninit<c_ulonglong>>::with_capacity(pageCount_in as usize);
        let pageCount_ptr = if pageCount_is_null {
            std::ptr::null_mut()
        } else {
            &mut pageCount_storage
        };
        let addresses_ptr = if addresses_is_null {
            std::ptr::null_mut()
        } else {
            addresses_storage.as_mut_ptr().cast::<c_ulonglong>()
        };
        let result =
            unsafe { nvmlDeviceGetRetiredPages(device, cause, pageCount_ptr, addresses_ptr) };
    }
    'server_after_send: {
        if !pageCount_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            pageCount_storage.send(channel_sender).unwrap();
            if result == nvmlReturn_t::NVML_SUCCESS && !addresses_is_null && pageCount_storage > 0 {
                assert!((pageCount_storage as usize) <= pageCount_in as usize);
                let addresses_out = unsafe {
                    std::slice::from_raw_parts(
                        addresses_storage.as_ptr().cast::<c_ulonglong>(),
                        pageCount_storage as usize,
                    )
                };
                send_slice(addresses_out, channel_sender).unwrap();
            }
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if !pageCount_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            let mut pageCount_out = 0u32;
            pageCount_out.recv(channel_receiver).unwrap();
            unsafe {
                std::ptr::write(pageCount, pageCount_out);
            }
            if result == nvmlReturn_t::NVML_SUCCESS && !addresses_is_null && pageCount_out > 0 {
                assert!((pageCount_out as usize) <= pageCount_in as usize);
                let addresses_out =
                    unsafe { std::slice::from_raw_parts_mut(addresses, pageCount_out as usize) };
                recv_slice_to(addresses_out, channel_receiver).unwrap();
            }
        }
    }
}

#[cuda_hook(proc_id = 991143)]
fn nvmlDeviceGetRetiredPages_v2(
    device: nvmlDevice_t,
    cause: nvmlPageRetirementCause_t,
    #[skip] pageCount: *mut c_uint,
    #[skip] addresses: *mut c_ulonglong,
    #[skip] timestamps: *mut c_ulonglong,
) -> nvmlReturn_t {
    'client_before_send: {
        let pageCount_is_null = pageCount.is_null();
        let addresses_is_null = addresses.is_null();
        let timestamps_is_null = timestamps.is_null();
        let pageCount_in = if pageCount_is_null {
            0
        } else {
            unsafe { std::ptr::read_unaligned(pageCount) }
        };
    }
    'client_extra_send: {
        pageCount_is_null.send(channel_sender).unwrap();
        addresses_is_null.send(channel_sender).unwrap();
        timestamps_is_null.send(channel_sender).unwrap();
        pageCount_in.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut pageCount_is_null = false;
        pageCount_is_null.recv(channel_receiver).unwrap();
        let mut addresses_is_null = false;
        addresses_is_null.recv(channel_receiver).unwrap();
        let mut timestamps_is_null = false;
        timestamps_is_null.recv(channel_receiver).unwrap();
        let mut pageCount_in = 0u32;
        pageCount_in.recv(channel_receiver).unwrap();
    }
    'server_execution: {
        let mut pageCount_storage = pageCount_in;
        let mut addresses_storage =
            Vec::<std::mem::MaybeUninit<c_ulonglong>>::with_capacity(pageCount_in as usize);
        let mut timestamps_storage =
            Vec::<std::mem::MaybeUninit<c_ulonglong>>::with_capacity(pageCount_in as usize);
        let pageCount_ptr = if pageCount_is_null {
            std::ptr::null_mut()
        } else {
            &mut pageCount_storage
        };
        let addresses_ptr = if addresses_is_null {
            std::ptr::null_mut()
        } else {
            addresses_storage.as_mut_ptr().cast::<c_ulonglong>()
        };
        let timestamps_ptr = if timestamps_is_null {
            std::ptr::null_mut()
        } else {
            timestamps_storage.as_mut_ptr().cast::<c_ulonglong>()
        };
        let result = unsafe {
            nvmlDeviceGetRetiredPages_v2(
                device,
                cause,
                pageCount_ptr,
                addresses_ptr,
                timestamps_ptr,
            )
        };
    }
    'server_after_send: {
        if !pageCount_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            pageCount_storage.send(channel_sender).unwrap();
            if result == nvmlReturn_t::NVML_SUCCESS && pageCount_storage > 0 {
                assert!((pageCount_storage as usize) <= pageCount_in as usize);
                if !addresses_is_null {
                    let addresses_out = unsafe {
                        std::slice::from_raw_parts(
                            addresses_storage.as_ptr().cast::<c_ulonglong>(),
                            pageCount_storage as usize,
                        )
                    };
                    send_slice(addresses_out, channel_sender).unwrap();
                }
                if !timestamps_is_null {
                    let timestamps_out = unsafe {
                        std::slice::from_raw_parts(
                            timestamps_storage.as_ptr().cast::<c_ulonglong>(),
                            pageCount_storage as usize,
                        )
                    };
                    send_slice(timestamps_out, channel_sender).unwrap();
                }
            }
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if !pageCount_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            let mut pageCount_out = 0u32;
            pageCount_out.recv(channel_receiver).unwrap();
            unsafe {
                std::ptr::write(pageCount, pageCount_out);
            }
            if result == nvmlReturn_t::NVML_SUCCESS && pageCount_out > 0 {
                assert!((pageCount_out as usize) <= pageCount_in as usize);
                if !addresses_is_null {
                    let addresses_out = unsafe {
                        std::slice::from_raw_parts_mut(addresses, pageCount_out as usize)
                    };
                    recv_slice_to(addresses_out, channel_receiver).unwrap();
                }
                if !timestamps_is_null {
                    let timestamps_out = unsafe {
                        std::slice::from_raw_parts_mut(timestamps, pageCount_out as usize)
                    };
                    recv_slice_to(timestamps_out, channel_receiver).unwrap();
                }
            }
        }
    }
}

#[cuda_hook(proc_id = 991144)]
fn nvmlDeviceGetRemappedRows(
    device: nvmlDevice_t,
    corrRows: *mut c_uint,
    uncRows: *mut c_uint,
    isPending: *mut c_uint,
    failureOccurred: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991145)]
fn nvmlDeviceGetRemappedRows_v2(
    device: nvmlDevice_t,
    info: *mut nvmlRemappedRowsInfo_v2_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991146)]
fn nvmlDeviceGetProcessUtilization(
    device: nvmlDevice_t,
    #[skip] utilization: *mut nvmlProcessUtilizationSample_t,
    #[skip] processSamplesCount: *mut c_uint,
    lastSeenTimeStamp: c_ulonglong,
) -> nvmlReturn_t {
    'client_before_send: {
        let utilization_is_null = utilization.is_null();
        let processSamplesCount_is_null = processSamplesCount.is_null();
        let processSamplesCount_in = if processSamplesCount_is_null {
            0
        } else {
            unsafe { std::ptr::read_unaligned(processSamplesCount) }
        };
    }
    'client_extra_send: {
        utilization_is_null.send(channel_sender).unwrap();
        processSamplesCount_is_null.send(channel_sender).unwrap();
        processSamplesCount_in.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut utilization_is_null = false;
        utilization_is_null.recv(channel_receiver).unwrap();
        let mut processSamplesCount_is_null = false;
        processSamplesCount_is_null.recv(channel_receiver).unwrap();
        let mut processSamplesCount_in = 0u32;
        processSamplesCount_in.recv(channel_receiver).unwrap();
    }
    'server_execution: {
        let mut processSamplesCount_storage = processSamplesCount_in;
        let mut utilization_storage =
            Vec::<std::mem::MaybeUninit<nvmlProcessUtilizationSample_t>>::with_capacity(
                processSamplesCount_in as usize,
            );
        let utilization_ptr = if utilization_is_null {
            std::ptr::null_mut()
        } else {
            utilization_storage
                .as_mut_ptr()
                .cast::<nvmlProcessUtilizationSample_t>()
        };
        let processSamplesCount_ptr = if processSamplesCount_is_null {
            std::ptr::null_mut()
        } else {
            &mut processSamplesCount_storage
        };
        let result = unsafe {
            nvmlDeviceGetProcessUtilization(
                device,
                utilization_ptr,
                processSamplesCount_ptr,
                lastSeenTimeStamp,
            )
        };
    }
    'server_after_send: {
        if !processSamplesCount_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            processSamplesCount_storage.send(channel_sender).unwrap();
            if result == nvmlReturn_t::NVML_SUCCESS
                && !utilization_is_null
                && processSamplesCount_storage > 0
            {
                assert!((processSamplesCount_storage as usize) <= processSamplesCount_in as usize);
                let utilization_out = unsafe {
                    std::slice::from_raw_parts(
                        utilization_storage
                            .as_ptr()
                            .cast::<nvmlProcessUtilizationSample_t>(),
                        processSamplesCount_storage as usize,
                    )
                };
                send_slice(utilization_out, channel_sender).unwrap();
            }
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if !processSamplesCount_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            let mut processSamplesCount_out = 0u32;
            processSamplesCount_out.recv(channel_receiver).unwrap();
            unsafe {
                std::ptr::write(processSamplesCount, processSamplesCount_out);
            }
            if result == nvmlReturn_t::NVML_SUCCESS
                && !utilization_is_null
                && processSamplesCount_out > 0
            {
                assert!((processSamplesCount_out as usize) <= processSamplesCount_in as usize);
                let utilization_out = unsafe {
                    std::slice::from_raw_parts_mut(utilization, processSamplesCount_out as usize)
                };
                recv_slice_to(utilization_out, channel_receiver).unwrap();
            }
        }
    }
}

#[cuda_hook(proc_id = 991147)]
fn nvmlDeviceGetSamples(
    device: nvmlDevice_t,
    type_: nvmlSamplingType_t,
    lastSeenTimeStamp: c_ulonglong,
    #[skip] sampleValType: *mut nvmlValueType_t,
    #[skip] sampleCount: *mut c_uint,
    #[skip] samples: *mut nvmlSample_t,
) -> nvmlReturn_t {
    'client_before_send: {
        let sampleValType_is_null = sampleValType.is_null();
        let sampleCount_is_null = sampleCount.is_null();
        let samples_is_null = samples.is_null();
        let sampleCount_in = if sampleCount_is_null {
            0
        } else {
            unsafe { std::ptr::read_unaligned(sampleCount) }
        };
    }
    'client_extra_send: {
        sampleValType_is_null.send(channel_sender).unwrap();
        sampleCount_is_null.send(channel_sender).unwrap();
        samples_is_null.send(channel_sender).unwrap();
        sampleCount_in.send(channel_sender).unwrap();
    }
    'server_extra_recv: {
        let mut sampleValType_is_null = false;
        sampleValType_is_null.recv(channel_receiver).unwrap();
        let mut sampleCount_is_null = false;
        sampleCount_is_null.recv(channel_receiver).unwrap();
        let mut samples_is_null = false;
        samples_is_null.recv(channel_receiver).unwrap();
        let mut sampleCount_in = 0u32;
        sampleCount_in.recv(channel_receiver).unwrap();
    }
    'server_execution: {
        let mut sampleValType_storage = std::mem::MaybeUninit::<nvmlValueType_t>::uninit();
        let mut sampleCount_storage = sampleCount_in;
        let mut samples_storage =
            Vec::<std::mem::MaybeUninit<nvmlSample_t>>::with_capacity(sampleCount_in as usize);
        let sampleValType_ptr = if sampleValType_is_null {
            std::ptr::null_mut()
        } else {
            sampleValType_storage.as_mut_ptr()
        };
        let sampleCount_ptr = if sampleCount_is_null {
            std::ptr::null_mut()
        } else {
            &mut sampleCount_storage
        };
        let samples_ptr = if samples_is_null {
            std::ptr::null_mut()
        } else {
            samples_storage.as_mut_ptr().cast::<nvmlSample_t>()
        };
        let result = unsafe {
            nvmlDeviceGetSamples(
                device,
                type_,
                lastSeenTimeStamp,
                sampleValType_ptr,
                sampleCount_ptr,
                samples_ptr,
            )
        };
    }
    'server_after_send: {
        if result == nvmlReturn_t::NVML_SUCCESS && !sampleValType_is_null {
            let sampleValType_out = unsafe { sampleValType_storage.assume_init() };
            sampleValType_out.send(channel_sender).unwrap();
        }
        if !sampleCount_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            sampleCount_storage.send(channel_sender).unwrap();
        }
        if result == nvmlReturn_t::NVML_SUCCESS && !samples_is_null && sampleCount_storage > 0 {
            assert!((sampleCount_storage as usize) <= sampleCount_in as usize);
            let samples_out = unsafe {
                std::slice::from_raw_parts(
                    samples_storage.as_ptr().cast::<nvmlSample_t>(),
                    sampleCount_storage as usize,
                )
            };
            send_slice(samples_out, channel_sender).unwrap();
        }
        if result == nvmlReturn_t::NVML_SUCCESS
            || result == nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
        {
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == nvmlReturn_t::NVML_SUCCESS && !sampleValType_is_null {
            let mut sampleValType_out = std::mem::MaybeUninit::<nvmlValueType_t>::uninit();
            sampleValType_out.recv(channel_receiver).unwrap();
            unsafe {
                std::ptr::write(sampleValType, sampleValType_out.assume_init());
            }
        }
        if !sampleCount_is_null
            && matches!(
                result,
                nvmlReturn_t::NVML_SUCCESS | nvmlReturn_t::NVML_ERROR_INSUFFICIENT_SIZE
            )
        {
            let mut sampleCount_out = 0u32;
            sampleCount_out.recv(channel_receiver).unwrap();
            unsafe {
                std::ptr::write(sampleCount, sampleCount_out);
            }
            if result == nvmlReturn_t::NVML_SUCCESS && !samples_is_null && sampleCount_out > 0 {
                assert!((sampleCount_out as usize) <= sampleCount_in as usize);
                let samples_out =
                    unsafe { std::slice::from_raw_parts_mut(samples, sampleCount_out as usize) };
                recv_slice_to(samples_out, channel_receiver).unwrap();
            }
        }
    }
}

#[cuda_hook(proc_id = 991148)]
fn nvmlDeviceGetFieldValues(
    device: nvmlDevice_t,
    valuesCount: c_int,
    #[skip] values: *mut nvmlFieldValue_t,
) -> nvmlReturn_t {
    'client_before_send: {
        let values_is_null = values.is_null();
    }
    'client_extra_send: {
        values_is_null.send(channel_sender).unwrap();
        if !values_is_null && valuesCount > 0 {
            let values_in = unsafe { std::slice::from_raw_parts(values, valuesCount as usize) };
            send_slice(values_in, channel_sender).unwrap();
        }
    }
    'server_extra_recv: {
        let mut values_is_null = false;
        values_is_null.recv(channel_receiver).unwrap();
        let values_in = if !values_is_null && valuesCount > 0 {
            recv_slice::<nvmlFieldValue_t, _>(channel_receiver).unwrap()
        } else {
            Vec::<nvmlFieldValue_t>::new().into_boxed_slice()
        };
    }
    'server_execution: {
        let mut values_storage = values_in.into_vec();
        let values_ptr = if values_is_null {
            std::ptr::null_mut()
        } else {
            values_storage.as_mut_ptr()
        };
        let result = unsafe { nvmlDeviceGetFieldValues(device, valuesCount, values_ptr) };
    }
    'server_after_send: {
        if result == nvmlReturn_t::NVML_SUCCESS && !values_is_null && valuesCount > 0 {
            send_slice(&values_storage[..valuesCount as usize], channel_sender).unwrap();
            channel_sender.flush_out().unwrap();
        }
    }
    'client_after_recv: {
        if result == nvmlReturn_t::NVML_SUCCESS && !values_is_null && valuesCount > 0 {
            let values_out =
                unsafe { std::slice::from_raw_parts_mut(values, valuesCount as usize) };
            recv_slice_to(values_out, channel_receiver).unwrap();
        }
    }
}

#[cuda_hook(proc_id = 991149)]
fn nvmlDeviceGetSupportedEventTypes(
    device: nvmlDevice_t,
    eventTypes: *mut c_ulonglong,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991150)]
fn nvmlDeviceGetMemoryAffinity(
    device: nvmlDevice_t,
    nodeSetSize: c_uint,
    #[host(output, len = nodeSetSize)] nodeSet: *mut c_ulong,
    scope: nvmlAffinityScope_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991151)]
fn nvmlDeviceGetCpuAffinityWithinScope(
    device: nvmlDevice_t,
    cpuSetSize: c_uint,
    #[host(output, len = cpuSetSize)] cpuSet: *mut c_ulong,
    scope: nvmlAffinityScope_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991152)]
fn nvmlDeviceGetCpuAffinity(
    device: nvmlDevice_t,
    cpuSetSize: c_uint,
    #[host(output, len = cpuSetSize)] cpuSet: *mut c_ulong,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991153)]
fn nvmlDeviceGetThermalSettings(
    device: nvmlDevice_t,
    sensorIndex: c_uint,
    pThermalSettings: *mut nvmlGpuThermalSettings_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991154)]
fn nvmlDeviceGetDramEncryptionMode(
    device: nvmlDevice_t,
    current: *mut nvmlDramEncryptionInfo_t,
    pending: *mut nvmlDramEncryptionInfo_t,
) -> nvmlReturn_t {
    'client_extra_send: {
        assert!(!current.is_null());
        assert!(!pending.is_null());
        unsafe { std::ptr::read_unaligned(current) }
            .send(channel_sender)
            .unwrap();
        unsafe { std::ptr::read_unaligned(pending) }
            .send(channel_sender)
            .unwrap();
    }
    'server_extra_recv: {
        let mut current_in = std::mem::MaybeUninit::<nvmlDramEncryptionInfo_t>::uninit();
        current_in.recv(channel_receiver).unwrap();
        let current_in = unsafe { current_in.assume_init() };
        let mut pending_in = std::mem::MaybeUninit::<nvmlDramEncryptionInfo_t>::uninit();
        pending_in.recv(channel_receiver).unwrap();
        let pending_in = unsafe { pending_in.assume_init() };
    }
    'server_before_execution: {
        current.write(current_in);
        pending.write(pending_in);
    }
}

#[cuda_hook(proc_id = 991155)]
fn nvmlDeviceGetSramEccErrorStatus(
    device: nvmlDevice_t,
    status: *mut nvmlEccSramErrorStatus_t,
) -> nvmlReturn_t {
    'client_extra_send: {
        assert!(!status.is_null());
        unsafe { std::ptr::read_unaligned(status) }
            .send(channel_sender)
            .unwrap();
    }
    'server_extra_recv: {
        let mut status_in = std::mem::MaybeUninit::<nvmlEccSramErrorStatus_t>::uninit();
        status_in.recv(channel_receiver).unwrap();
        let status_in = unsafe { status_in.assume_init() };
    }
    'server_before_execution: {
        status.write(status_in);
    }
}

#[cuda_hook(proc_id = 991156)]
fn nvmlDeviceGetClkMonStatus(device: nvmlDevice_t, status: *mut nvmlClkMonStatus_t)
    -> nvmlReturn_t;

#[cuda_hook(proc_id = 991157)]
fn nvmlDeviceGetPlatformInfo(
    device: nvmlDevice_t,
    platformInfo: *mut nvmlPlatformInfo_t,
) -> nvmlReturn_t {
    'client_extra_send: {
        assert!(!platformInfo.is_null());
        unsafe { std::ptr::read_unaligned(platformInfo) }
            .send(channel_sender)
            .unwrap();
    }
    'server_extra_recv: {
        let mut platformInfo_in = std::mem::MaybeUninit::<nvmlPlatformInfo_t>::uninit();
        platformInfo_in.recv(channel_receiver).unwrap();
        let platformInfo_in = unsafe { platformInfo_in.assume_init() };
    }
    'server_before_execution: {
        platformInfo.write(platformInfo_in);
    }
}

#[cuda_hook(proc_id = 991158)]
fn nvmlDeviceGetPdi(device: nvmlDevice_t, pdi: *mut nvmlPdi_t) -> nvmlReturn_t {
    'client_extra_send: {
        assert!(!pdi.is_null());
        unsafe { std::ptr::read_unaligned(pdi) }
            .send(channel_sender)
            .unwrap();
    }
    'server_extra_recv: {
        let mut pdi_in = std::mem::MaybeUninit::<nvmlPdi_t>::uninit();
        pdi_in.recv(channel_receiver).unwrap();
        let pdi_in = unsafe { pdi_in.assume_init() };
    }
    'server_before_execution: {
        pdi.write(pdi_in);
    }
}

#[cuda_hook(proc_id = 991159)]
fn nvmlDeviceGetHostname_v1(device: nvmlDevice_t, hostname: *mut nvmlHostname_v1_t)
    -> nvmlReturn_t;

#[cuda_hook(proc_id = 991160)]
fn nvmlDeviceGetNvLinkState(
    device: nvmlDevice_t,
    link: c_uint,
    isActive: *mut nvmlEnableState_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991161)]
fn nvmlDeviceGetNvLinkVersion(
    device: nvmlDevice_t,
    link: c_uint,
    version: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991162)]
fn nvmlDeviceGetNvLinkCapability(
    device: nvmlDevice_t,
    link: c_uint,
    capability: nvmlNvLinkCapability_t,
    capResult: *mut c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991163)]
fn nvmlDeviceGetNvLinkRemotePciInfo_v2(
    device: nvmlDevice_t,
    link: c_uint,
    pci: *mut nvmlPciInfo_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991164)]
fn nvmlDeviceGetNvLinkErrorCounter(
    device: nvmlDevice_t,
    link: c_uint,
    counter: nvmlNvLinkErrorCounter_t,
    counterValue: *mut c_ulonglong,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991165)]
fn nvmlDeviceGetNvLinkUtilizationControl(
    device: nvmlDevice_t,
    link: c_uint,
    counter: c_uint,
    control: *mut nvmlNvLinkUtilizationControl_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991166)]
fn nvmlDeviceGetNvLinkUtilizationCounter(
    device: nvmlDevice_t,
    link: c_uint,
    counter: c_uint,
    rxcounter: *mut c_ulonglong,
    txcounter: *mut c_ulonglong,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991167)]
fn nvmlDeviceGetNvLinkRemoteDeviceType(
    device: nvmlDevice_t,
    link: c_uint,
    pNvLinkDeviceType: *mut nvmlIntNvLinkDeviceType_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991168)]
fn nvmlSystemGetNvlinkBwMode(nvlinkBwMode: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991169)]
fn nvmlDeviceGetNvlinkSupportedBwModes(
    device: nvmlDevice_t,
    supportedBwMode: *mut nvmlNvlinkSupportedBwModes_t,
) -> nvmlReturn_t {
    'client_extra_send: {
        assert!(!supportedBwMode.is_null());
        unsafe { std::ptr::read_unaligned(supportedBwMode) }
            .send(channel_sender)
            .unwrap();
    }
    'server_extra_recv: {
        let mut supportedBwMode_in =
            std::mem::MaybeUninit::<nvmlNvlinkSupportedBwModes_t>::uninit();
        supportedBwMode_in.recv(channel_receiver).unwrap();
        let supportedBwMode_in = unsafe { supportedBwMode_in.assume_init() };
    }
    'server_before_execution: {
        supportedBwMode.write(supportedBwMode_in);
    }
}

#[cuda_hook(proc_id = 991170)]
fn nvmlDeviceGetNvlinkBwMode(
    device: nvmlDevice_t,
    getBwMode: *mut nvmlNvlinkGetBwMode_t,
) -> nvmlReturn_t {
    'client_extra_send: {
        assert!(!getBwMode.is_null());
        unsafe { std::ptr::read_unaligned(getBwMode) }
            .send(channel_sender)
            .unwrap();
    }
    'server_extra_recv: {
        let mut getBwMode_in = std::mem::MaybeUninit::<nvmlNvlinkGetBwMode_t>::uninit();
        getBwMode_in.recv(channel_receiver).unwrap();
        let getBwMode_in = unsafe { getBwMode_in.assume_init() };
    }
    'server_before_execution: {
        getBwMode.write(getBwMode_in);
    }
}

#[cuda_hook(proc_id = 991171)]
fn nvmlDeviceGetNvLinkInfo(device: nvmlDevice_t, info: *mut nvmlNvLinkInfo_t) -> nvmlReturn_t {
    'client_extra_send: {
        assert!(!info.is_null());
        unsafe { std::ptr::read_unaligned(info) }
            .send(channel_sender)
            .unwrap();
    }
    'server_extra_recv: {
        let mut info_in = std::mem::MaybeUninit::<nvmlNvLinkInfo_t>::uninit();
        info_in.recv(channel_receiver).unwrap();
        let info_in = unsafe { info_in.assume_init() };
    }
    'server_before_execution: {
        info.write(info_in);
    }
}

#[cuda_hook(proc_id = 991172)]
fn nvmlDeviceGetVirtualizationMode(
    device: nvmlDevice_t,
    pVirtualMode: *mut nvmlGpuVirtualizationMode_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991173)]
fn nvmlDeviceGetHostVgpuMode(
    device: nvmlDevice_t,
    pHostVgpuMode: *mut nvmlHostVgpuMode_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991174)]
fn nvmlDeviceIsMigDeviceHandle(device: nvmlDevice_t, isMigDevice: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991175)]
fn nvmlDeviceGetGpuInstanceId(device: nvmlDevice_t, id: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991176)]
fn nvmlDeviceGetComputeInstanceId(device: nvmlDevice_t, id: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991177)]
fn nvmlDeviceGetMaxMigDeviceCount(device: nvmlDevice_t, count: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991178)]
fn nvmlDeviceGetMigDeviceHandleByIndex(
    device: nvmlDevice_t,
    index: c_uint,
    migDevice: *mut nvmlDevice_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991179)]
fn nvmlDeviceGetDeviceHandleFromMigDeviceHandle(
    migDevice: nvmlDevice_t,
    device: *mut nvmlDevice_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991180)]
fn nvmlDeviceGetCapabilities(
    device: nvmlDevice_t,
    caps: *mut nvmlDeviceCapabilities_t,
) -> nvmlReturn_t {
    'client_extra_send: {
        assert!(!caps.is_null());
        unsafe { std::ptr::read_unaligned(caps) }
            .send(channel_sender)
            .unwrap();
    }
    'server_extra_recv: {
        let mut caps_in = std::mem::MaybeUninit::<nvmlDeviceCapabilities_t>::uninit();
        caps_in.recv(channel_receiver).unwrap();
        let caps_in = unsafe { caps_in.assume_init() };
    }
    'server_before_execution: {
        caps.write(caps_in);
    }
}

#[cuda_hook(proc_id = 991181)]
fn nvmlEventSetCreate(set: *mut nvmlEventSet_t) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991182)]
fn nvmlDeviceRegisterEvents(
    device: nvmlDevice_t,
    eventTypes: c_ulonglong,
    set: nvmlEventSet_t,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991183)]
fn nvmlEventSetWait_v2(
    set: nvmlEventSet_t,
    data: *mut nvmlEventData_t,
    timeoutms: c_uint,
) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991184)]
fn nvmlEventSetFree(set: nvmlEventSet_t) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991185)]
fn nvmlGetExcludedDeviceCount(deviceCount: *mut c_uint) -> nvmlReturn_t;

#[cuda_hook(proc_id = 991186)]
fn nvmlGetExcludedDeviceInfoByIndex(
    index: c_uint,
    info: *mut nvmlExcludedDeviceInfo_t,
) -> nvmlReturn_t;
