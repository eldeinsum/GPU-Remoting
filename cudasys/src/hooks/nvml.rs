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
