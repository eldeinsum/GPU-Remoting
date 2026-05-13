#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <nvml.h>
#include <unistd.h>
#include <vector>

template <typename T>
static T lookup_nvml_symbol(const char *name, bool report_missing)
{
    void *symbol = dlsym(RTLD_DEFAULT, name);
    if (symbol == nullptr)
    {
        if (report_missing)
        {
            std::cout << "Failed to find NVML symbol " << name << ": "
                      << dlerror() << std::endl;
        }
        return nullptr;
    }
    return reinterpret_cast<T>(symbol);
}

static int check_success(nvmlReturn_t result, const char *label)
{
    if (result == NVML_SUCCESS)
    {
        return 0;
    }
    std::cout << label << " failed: " << nvmlErrorString(result) << std::endl;
    std::cout << label << " status code: " << static_cast<int>(result) << std::endl;
    return 1;
}

static int check_optional(nvmlReturn_t result, const char *label)
{
    if (result == NVML_SUCCESS || result == NVML_ERROR_NOT_SUPPORTED ||
        result == NVML_ERROR_FUNCTION_NOT_FOUND || result == NVML_ERROR_NO_PERMISSION ||
        result == NVML_ERROR_NOT_READY)
    {
        return 0;
    }
    std::cout << label << " returned unexpected status: "
              << nvmlErrorString(result) << std::endl;
    std::cout << label << " status code: " << static_cast<int>(result) << std::endl;
    return 1;
}

static int check_optional_indexed(nvmlReturn_t result, const char *label)
{
    if (result == NVML_ERROR_INVALID_ARGUMENT)
    {
        return 0;
    }
    return check_optional(result, label);
}

static int check_optional_list(nvmlReturn_t result, const char *label)
{
    if (result == NVML_SUCCESS || result == NVML_ERROR_INSUFFICIENT_SIZE ||
        result == NVML_ERROR_NOT_SUPPORTED || result == NVML_ERROR_FUNCTION_NOT_FOUND ||
        result == NVML_ERROR_NO_PERMISSION || result == NVML_ERROR_NOT_READY ||
        result == NVML_ERROR_NOT_FOUND)
    {
        return 0;
    }
    std::cout << label << " returned unexpected status: "
              << nvmlErrorString(result) << std::endl;
    std::cout << label << " status code: " << static_cast<int>(result) << std::endl;
    return 1;
}

static int check_nonempty(const char *value, const char *label)
{
    if (value[0] != '\0')
    {
        return 0;
    }
    std::cout << label << " returned an empty string" << std::endl;
    return 1;
}

static int check_process_list(nvmlReturn_t (*get_processes)(nvmlDevice_t, unsigned int *,
                                                            nvmlProcessInfo_t *),
                              nvmlDevice_t device, const char *label)
{
    unsigned int required = 0;
    nvmlProcessInfo_t dummy = {};
    nvmlReturn_t result = get_processes(device, &required, &dummy);
    if (check_optional_list(result, label))
    {
        return 1;
    }

    unsigned int capacity = required > 0 ? required : 64;
    std::vector<nvmlProcessInfo_t> infos(capacity);
    unsigned int count = capacity;
    result = get_processes(device, &count, infos.data());
    if (check_optional_list(result, label))
    {
        return 1;
    }
    if (result == NVML_SUCCESS && count > capacity)
    {
        std::cout << label << " returned " << count
                  << " entries for capacity " << capacity << std::endl;
        return 1;
    }

    if (result == NVML_SUCCESS)
    {
        for (unsigned int i = 0; i < count; ++i)
        {
            char process_name[128] = {};
            nvmlReturn_t name_result =
                nvmlSystemGetProcessName(infos[i].pid, process_name,
                                         sizeof(process_name));
            if (check_optional(name_result, "nvmlSystemGetProcessName"))
            {
                return 1;
            }
        }
    }
    return 0;
}

int main()
{
    // const int iterations = 1;
    nvmlReturn_t result;
    const char *preload = std::getenv("LD_PRELOAD");
    bool remoted = preload != nullptr && std::strstr(preload, "libclient.so") != nullptr;
    auto legacy_init = lookup_nvml_symbol<nvmlReturn_t (*)()>("nvmlInit", remoted);
    auto legacy_get_count =
        lookup_nvml_symbol<nvmlReturn_t (*)(unsigned int *)>("nvmlDeviceGetCount",
                                                             remoted);
    auto legacy_get_handle =
        lookup_nvml_symbol<nvmlReturn_t (*)(unsigned int, nvmlDevice_t *)>(
            "nvmlDeviceGetHandleByIndex", remoted);
    bool legacy_available = legacy_init != nullptr && legacy_get_count != nullptr &&
                            legacy_get_handle != nullptr;
    if (!legacy_available && remoted)
    {
        return 1;
    }

    if (legacy_available)
    {
        result = legacy_init();
        if (NVML_SUCCESS != result)
        {
            std::cout << "Failed to initialize NVML through nvmlInit: "
                      << nvmlErrorString(result) << std::endl;
            return 1;
        }
    }
    result = nvmlInit_v2();
    if (check_success(result, "nvmlInit_v2"))
    {
        return 1;
    }
    // result = nvmlInitWithFlags(NVML_INIT_FLAG_NO_ATTACH);
    // if (NVML_SUCCESS != result)
    // {
    //     std::cout << "Failed to initialize NVML with NO_ATTACH: " << nvmlErrorString(result) << std::endl;
    //     return 1;
    // }

    char driver_version[128] = {};
    result = nvmlSystemGetDriverVersion(driver_version, sizeof(driver_version));
    if (check_success(result, "nvmlSystemGetDriverVersion") ||
        check_nonempty(driver_version, "nvmlSystemGetDriverVersion"))
    {
        return 1;
    }

    char nvml_version[128] = {};
    result = nvmlSystemGetNVMLVersion(nvml_version, sizeof(nvml_version));
    if (check_success(result, "nvmlSystemGetNVMLVersion") ||
        check_nonempty(nvml_version, "nvmlSystemGetNVMLVersion"))
    {
        return 1;
    }

    int cuda_driver_version = 0;
    result = nvmlSystemGetCudaDriverVersion(&cuda_driver_version);
    if (check_success(result, "nvmlSystemGetCudaDriverVersion") ||
        cuda_driver_version <= 0)
    {
        std::cout << "Unexpected CUDA driver version: " << cuda_driver_version
                  << std::endl;
        return 1;
    }
    result = nvmlSystemGetCudaDriverVersion_v2(&cuda_driver_version);
    if (check_success(result, "nvmlSystemGetCudaDriverVersion_v2") ||
        cuda_driver_version <= 0)
    {
        std::cout << "Unexpected CUDA driver version v2: "
                  << cuda_driver_version << std::endl;
        return 1;
    }

    unsigned int device_count;
    if (legacy_available)
    {
        result = legacy_get_count(&device_count);
        if (NVML_SUCCESS != result)
        {
            std::cout << "Failed to get legacy device count: "
                      << nvmlErrorString(result) << std::endl;
            return 1;
        }
    }
    unsigned int device_count_v2;
    result = nvmlDeviceGetCount_v2(&device_count_v2);
    if (NVML_SUCCESS != result)
    {
        std::cout << "Failed to get device count: " << nvmlErrorString(result) << std::endl;
        return 1;
    }
    if (legacy_available && device_count != device_count_v2)
    {
        std::cout << "Legacy and v2 NVML device counts differ: " << device_count
                  << " vs " << device_count_v2 << std::endl;
        return 1;
    }
    if (!legacy_available)
    {
        device_count = device_count_v2;
    }
    std::cout << "Found " << device_count << " devices" << std::endl;
    if (device_count == 0)
    {
        std::cout << "Expected at least one NVML device" << std::endl;
        return 1;
    }

    nvmlDevice_t device;
    if (legacy_available)
    {
        result = legacy_get_handle(0, &device);
        if (NVML_SUCCESS != result)
        {
            std::cout << "Failed to get legacy NVML device handle: "
                      << nvmlErrorString(result) << std::endl;
            return 1;
        }
    }
    nvmlDevice_t device_v2;
    result = nvmlDeviceGetHandleByIndex_v2(0, &device_v2);
    if (NVML_SUCCESS != result)
    {
        std::cout << "Failed to get NVML v2 device handle: "
                  << nvmlErrorString(result) << std::endl;
        return 1;
    }
    if (!legacy_available)
    {
        device = device_v2;
    }

    cudaError_t cuda_result = cudaFree(nullptr);
    if (cuda_result != cudaSuccess)
    {
        std::cout << "Failed to initialize CUDA context: "
                  << cudaGetErrorString(cuda_result) << std::endl;
        return 1;
    }

    char current_process_name[128] = {};
    result = nvmlSystemGetProcessName(getpid(), current_process_name,
                                      sizeof(current_process_name));
    if (check_success(result, "nvmlSystemGetProcessName") ||
        check_nonempty(current_process_name, "nvmlSystemGetProcessName"))
    {
        return 1;
    }

    char device_name[128] = {};
    result = nvmlDeviceGetName(device, device_name, sizeof(device_name));
    if (check_success(result, "nvmlDeviceGetName") ||
        check_nonempty(device_name, "nvmlDeviceGetName"))
    {
        return 1;
    }

    char uuid[NVML_DEVICE_UUID_ASCII_LEN] = {};
    result = nvmlDeviceGetUUID(device, uuid, sizeof(uuid));
    if (check_success(result, "nvmlDeviceGetUUID") ||
        check_nonempty(uuid, "nvmlDeviceGetUUID"))
    {
        return 1;
    }

    nvmlDevice_t by_uuid = nullptr;
    result = nvmlDeviceGetHandleByUUID(uuid, &by_uuid);
    if (check_success(result, "nvmlDeviceGetHandleByUUID"))
    {
        return 1;
    }

    char serial[128] = {};
    result = nvmlDeviceGetSerial(device, serial, sizeof(serial));
    if (check_optional(result, "nvmlDeviceGetSerial"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS && check_nonempty(serial, "nvmlDeviceGetSerial"))
    {
        return 1;
    }

    nvmlPciInfo_t pci = {};
    result = nvmlDeviceGetPciInfo_v3(device, &pci);
    if (check_success(result, "nvmlDeviceGetPciInfo_v3") ||
        check_nonempty(pci.busId, "nvmlDeviceGetPciInfo_v3"))
    {
        return 1;
    }

    nvmlDevice_t by_pci = nullptr;
    result = nvmlDeviceGetHandleByPciBusId_v2(pci.busId, &by_pci);
    if (check_success(result, "nvmlDeviceGetHandleByPciBusId_v2"))
    {
        return 1;
    }

    nvmlBrandType_t brand;
    result = nvmlDeviceGetBrand(device, &brand);
    if (check_success(result, "nvmlDeviceGetBrand"))
    {
        return 1;
    }

    unsigned int device_index = 0;
    result = nvmlDeviceGetIndex(device, &device_index);
    if (check_success(result, "nvmlDeviceGetIndex") || device_index >= device_count)
    {
        std::cout << "Unexpected NVML device index: " << device_index
                  << std::endl;
        return 1;
    }

    unsigned int minor_number = 0;
    result = nvmlDeviceGetMinorNumber(device, &minor_number);
    if (check_success(result, "nvmlDeviceGetMinorNumber"))
    {
        return 1;
    }

    nvmlMemory_t memory;
    result = nvmlDeviceGetMemoryInfo(device, &memory);
    if (check_success(result, "nvmlDeviceGetMemoryInfo"))
    {
        return 1;
    }
    if (memory.total == 0 || memory.free > memory.total || memory.used > memory.total)
    {
        std::cout << "Unexpected NVML memory info: total=" << memory.total
                  << " free=" << memory.free << " used=" << memory.used << std::endl;
        return 1;
    }
    std::cout << "Memory total " << memory.total << " free " << memory.free
              << " used " << memory.used << std::endl;

    nvmlMemory_v2_t memory_v2 = {};
    result = nvmlDeviceGetMemoryInfo_v2(device, &memory_v2);
    if (check_optional(result, "nvmlDeviceGetMemoryInfo_v2"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS && memory_v2.total == 0)
    {
        std::cout << "Unexpected NVML memory v2 total: " << memory_v2.total
                  << std::endl;
        return 1;
    }

    int compute_major = 0;
    int compute_minor = 0;
    result = nvmlDeviceGetCudaComputeCapability(device, &compute_major, &compute_minor);
    if (check_success(result, "nvmlDeviceGetCudaComputeCapability") ||
        compute_major <= 0 || compute_minor < 0)
    {
        std::cout << "Unexpected CUDA compute capability: " << compute_major
                  << "." << compute_minor << std::endl;
        return 1;
    }

    nvmlComputeMode_t compute_mode;
    result = nvmlDeviceGetComputeMode(device, &compute_mode);
    if (check_success(result, "nvmlDeviceGetComputeMode"))
    {
        return 1;
    }

    nvmlEnableState_t enable_state;
    result = nvmlDeviceGetPersistenceMode(device, &enable_state);
    if (check_success(result, "nvmlDeviceGetPersistenceMode"))
    {
        return 1;
    }

    nvmlDeviceArchitecture_t architecture;
    result = nvmlDeviceGetArchitecture(device, &architecture);
    if (check_success(result, "nvmlDeviceGetArchitecture"))
    {
        return 1;
    }

    nvmlUtilization_t utilization;
    result = nvmlDeviceGetUtilizationRates(device, &utilization);
    if (check_success(result, "nvmlDeviceGetUtilizationRates") ||
        utilization.gpu > 100 || utilization.memory > 100)
    {
        std::cout << "Unexpected utilization: gpu=" << utilization.gpu
                  << " memory=" << utilization.memory << std::endl;
        return 1;
    }

    unsigned int value = 0;
    nvmlPstates_t pstate;
    unsigned long long energy = 0;
    nvmlReturn_t optional_results[] = {
        nvmlDeviceGetTemperature(device, NVML_TEMPERATURE_GPU, &value),
        nvmlDeviceGetPerformanceState(device, &pstate),
        nvmlDeviceGetPowerUsage(device, &value),
        nvmlDeviceGetTotalEnergyConsumption(device, &energy),
        nvmlDeviceGetPowerManagementMode(device, &enable_state),
        nvmlDeviceGetPowerManagementLimit(device, &value),
        nvmlDeviceGetPowerManagementDefaultLimit(device, &value),
        nvmlDeviceGetEnforcedPowerLimit(device, &value),
        nvmlDeviceGetClockInfo(device, NVML_CLOCK_GRAPHICS, &value),
        nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &value),
        nvmlDeviceGetClockInfo(device, NVML_CLOCK_MEM, &value),
        nvmlDeviceGetMaxClockInfo(device, NVML_CLOCK_GRAPHICS, &value),
        nvmlDeviceGetApplicationsClock(device, NVML_CLOCK_GRAPHICS, &value),
        nvmlDeviceGetDefaultApplicationsClock(device, NVML_CLOCK_GRAPHICS, &value),
        nvmlDeviceGetCurrPcieLinkGeneration(device, &value),
        nvmlDeviceGetCurrPcieLinkWidth(device, &value),
        nvmlDeviceGetMaxPcieLinkGeneration(device, &value),
        nvmlDeviceGetMaxPcieLinkWidth(device, &value),
        nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_TX_BYTES, &value),
        nvmlDeviceGetPcieThroughput(device, NVML_PCIE_UTIL_RX_BYTES, &value),
        nvmlDeviceGetPcieReplayCounter(device, &value),
        nvmlDeviceGetFanSpeed(device, &value),
    };
    const char *optional_labels[] = {
        "nvmlDeviceGetTemperature",
        "nvmlDeviceGetPerformanceState",
        "nvmlDeviceGetPowerUsage",
        "nvmlDeviceGetTotalEnergyConsumption",
        "nvmlDeviceGetPowerManagementMode",
        "nvmlDeviceGetPowerManagementLimit",
        "nvmlDeviceGetPowerManagementDefaultLimit",
        "nvmlDeviceGetEnforcedPowerLimit",
        "nvmlDeviceGetClockInfo graphics",
        "nvmlDeviceGetClockInfo sm",
        "nvmlDeviceGetClockInfo memory",
        "nvmlDeviceGetMaxClockInfo",
        "nvmlDeviceGetApplicationsClock",
        "nvmlDeviceGetDefaultApplicationsClock",
        "nvmlDeviceGetCurrPcieLinkGeneration",
        "nvmlDeviceGetCurrPcieLinkWidth",
        "nvmlDeviceGetMaxPcieLinkGeneration",
        "nvmlDeviceGetMaxPcieLinkWidth",
        "nvmlDeviceGetPcieThroughput tx",
        "nvmlDeviceGetPcieThroughput rx",
        "nvmlDeviceGetPcieReplayCounter",
        "nvmlDeviceGetFanSpeed",
    };
    static_assert(sizeof(optional_results) / sizeof(optional_results[0]) ==
                  sizeof(optional_labels) / sizeof(optional_labels[0]));
    for (size_t i = 0; i < sizeof(optional_results) / sizeof(optional_results[0]); ++i)
    {
        if (check_optional(optional_results[i], optional_labels[i]))
        {
            return 1;
        }
    }

    unsigned int unit_count = 0;
    result = nvmlUnitGetCount(&unit_count);
    if (check_optional(result, "nvmlUnitGetCount"))
    {
        return 1;
    }

    nvmlDeviceAttributes_t attributes = {};
    result = nvmlDeviceGetAttributes_v2(device, &attributes);
    if (check_optional(result, "nvmlDeviceGetAttributes_v2"))
    {
        return 1;
    }

    nvmlUnrepairableMemoryStatus_v1_t unrepairable_memory = {};
    nvmlGpuTopologyLevel_t topology_level;
    unsigned long long timestamp = 0;
    unsigned long duration_us = 0;
    int signed_value = 0;
    nvmlEnableState_t default_enable_state;
    nvmlGpuDynamicPstatesInfo_t dynamic_pstates = {};
    int min_offset = 0;
    int max_offset = 0;
    unsigned int min_limit = 0;
    unsigned int max_limit = 0;
    nvmlPstates_t supported_pstates[NVML_MAX_GPU_PERF_PSTATES] = {};
    nvmlGpuOperationMode_t current_gom;
    nvmlGpuOperationMode_t pending_gom;
    unsigned long long counter = 0;
    nvmlEccErrorCounts_t ecc_counts = {};
    nvmlFBCStats_t fbc_stats = {};
    nvmlDriverModel_t current_driver_model;
    nvmlDriverModel_t pending_driver_model;
    nvmlBridgeChipHierarchy_t bridge_hierarchy = {};
    int same_board = 0;
    nvmlViolationTime_t violation_time = {};
    nvmlPowerSource_t power_source = 0;
    nvmlBusType_t bus_type = 0;
    nvmlGpuFabricInfo_t fabric_info = {};
    nvmlRowRemapperHistogramValues_t row_remapper = {};
    unsigned int corrected_rows = 0;
    unsigned int uncorrected_rows = 0;
    unsigned int remap_pending = 0;
    unsigned int remap_failure = 0;
    nvmlRemappedRowsInfo_v2_t remapped_rows_v2 = {};
    unsigned long long supported_event_types = 0;
    nvmlReturn_t more_optional_results[] = {
        nvmlDeviceGetModuleId(device, &value),
        nvmlDeviceGetNumaNodeId(device, &value),
        nvmlDeviceGetUnrepairableMemoryFlag_v1(device, &unrepairable_memory),
        nvmlDeviceGetTopologyCommonAncestor(device, device, &topology_level),
        nvmlDeviceGetInforomConfigurationChecksum(device, &value),
        nvmlDeviceGetLastBBXFlushTime(device, &timestamp, &duration_us),
        nvmlDeviceGetDisplayMode(device, &enable_state),
        nvmlDeviceGetDisplayActive(device, &enable_state),
        nvmlDeviceGetGpuMaxPcieLinkGeneration(device, &value),
        nvmlDeviceGetGpcClkVfOffset(device, &signed_value),
        nvmlDeviceGetClock(device, NVML_CLOCK_GRAPHICS, NVML_CLOCK_ID_CURRENT, &value),
        nvmlDeviceGetMaxCustomerBoostClock(device, NVML_CLOCK_GRAPHICS, &value),
        nvmlDeviceGetAutoBoostedClocksEnabled(device, &enable_state,
                                              &default_enable_state),
        nvmlDeviceGetFanSpeed_v2(device, 0, &value),
        nvmlDeviceGetTargetFanSpeed(device, 0, &value),
        nvmlDeviceGetMinMaxFanSpeed(device, &min_limit, &max_limit),
        nvmlDeviceGetFanControlPolicy_v2(device, 0, &value),
        nvmlDeviceGetNumFans(device, &value),
        nvmlDeviceGetTemperatureThreshold(device, NVML_TEMPERATURE_THRESHOLD_SHUTDOWN,
                                          &value),
        nvmlDeviceGetCurrentClocksEventReasons(device, &counter),
        nvmlDeviceGetCurrentClocksThrottleReasons(device, &counter),
        nvmlDeviceGetSupportedClocksEventReasons(device, &counter),
        nvmlDeviceGetSupportedClocksThrottleReasons(device, &counter),
        nvmlDeviceGetPowerState(device, &pstate),
        nvmlDeviceGetDynamicPstatesInfo(device, &dynamic_pstates),
        nvmlDeviceGetMemClkVfOffset(device, &signed_value),
        nvmlDeviceGetMinMaxClockOfPState(device, NVML_CLOCK_GRAPHICS, NVML_PSTATE_0,
                                         &min_limit, &max_limit),
        nvmlDeviceGetSupportedPerformanceStates(device, supported_pstates,
                                                sizeof(supported_pstates)),
        nvmlDeviceGetGpcClkMinMaxVfOffset(device, &min_offset, &max_offset),
        nvmlDeviceGetMemClkMinMaxVfOffset(device, &min_offset, &max_offset),
        nvmlDeviceGetGpuOperationMode(device, &current_gom, &pending_gom),
        nvmlDeviceGetEccMode(device, &enable_state, &default_enable_state),
        nvmlDeviceGetDefaultEccMode(device, &enable_state),
        nvmlDeviceGetTotalEccErrors(device, NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                    NVML_VOLATILE_ECC, &counter),
        nvmlDeviceGetDetailedEccErrors(device, NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                       NVML_VOLATILE_ECC, &ecc_counts),
        nvmlDeviceGetMemoryErrorCounter(device, NVML_MEMORY_ERROR_TYPE_CORRECTED,
                                        NVML_VOLATILE_ECC,
                                        NVML_MEMORY_LOCATION_L2_CACHE, &counter),
        nvmlDeviceGetEncoderUtilization(device, &value, &device_index),
        nvmlDeviceGetEncoderCapacity(device, NVML_ENCODER_QUERY_H264, &value),
        nvmlDeviceGetEncoderStats(device, &value, &device_index, &minor_number),
        nvmlDeviceGetDecoderUtilization(device, &value, &device_index),
        nvmlDeviceGetJpgUtilization(device, &value, &device_index),
        nvmlDeviceGetOfaUtilization(device, &value, &device_index),
        nvmlDeviceGetFBCStats(device, &fbc_stats),
        nvmlDeviceGetDriverModel_v2(device, &current_driver_model,
                                    &pending_driver_model),
        nvmlDeviceGetBridgeChipInfo(device, &bridge_hierarchy),
        nvmlDeviceGetAPIRestriction(device, NVML_RESTRICTED_API_SET_APPLICATION_CLOCKS,
                                    &enable_state),
        nvmlDeviceGetViolationStatus(device, NVML_PERF_POLICY_POWER, &violation_time),
        nvmlDeviceGetIrqNum(device, &value),
        nvmlDeviceGetNumGpuCores(device, &value),
        nvmlDeviceGetPowerSource(device, &power_source),
        nvmlDeviceGetMemoryBusWidth(device, &value),
        nvmlDeviceGetPcieLinkMaxSpeed(device, &value),
        nvmlDeviceGetPcieSpeed(device, &value),
        nvmlDeviceGetAdaptiveClockInfoStatus(device, &value),
        nvmlDeviceGetBusType(device, &bus_type),
        nvmlDeviceGetGpuFabricInfo(device, &fabric_info),
        nvmlDeviceGetGspFirmwareMode(device, &value, &device_index),
        nvmlDeviceGetAccountingMode(device, &enable_state),
        nvmlDeviceGetAccountingBufferSize(device, &value),
        nvmlDeviceGetRetiredPagesPendingStatus(device, &enable_state),
        nvmlDeviceGetRowRemapperHistogram(device, &row_remapper),
        nvmlDeviceGetRemappedRows(device, &corrected_rows, &uncorrected_rows,
                                  &remap_pending, &remap_failure),
        nvmlDeviceGetRemappedRows_v2(device, &remapped_rows_v2),
        nvmlDeviceGetSupportedEventTypes(device, &supported_event_types),
    };
    const char *more_optional_labels[] = {
        "nvmlDeviceGetModuleId",
        "nvmlDeviceGetNumaNodeId",
        "nvmlDeviceGetUnrepairableMemoryFlag_v1",
        "nvmlDeviceGetTopologyCommonAncestor",
        "nvmlDeviceGetInforomConfigurationChecksum",
        "nvmlDeviceGetLastBBXFlushTime",
        "nvmlDeviceGetDisplayMode",
        "nvmlDeviceGetDisplayActive",
        "nvmlDeviceGetGpuMaxPcieLinkGeneration",
        "nvmlDeviceGetGpcClkVfOffset",
        "nvmlDeviceGetClock",
        "nvmlDeviceGetMaxCustomerBoostClock",
        "nvmlDeviceGetAutoBoostedClocksEnabled",
        "nvmlDeviceGetFanSpeed_v2",
        "nvmlDeviceGetTargetFanSpeed",
        "nvmlDeviceGetMinMaxFanSpeed",
        "nvmlDeviceGetFanControlPolicy_v2",
        "nvmlDeviceGetNumFans",
        "nvmlDeviceGetTemperatureThreshold",
        "nvmlDeviceGetCurrentClocksEventReasons",
        "nvmlDeviceGetCurrentClocksThrottleReasons",
        "nvmlDeviceGetSupportedClocksEventReasons",
        "nvmlDeviceGetSupportedClocksThrottleReasons",
        "nvmlDeviceGetPowerState",
        "nvmlDeviceGetDynamicPstatesInfo",
        "nvmlDeviceGetMemClkVfOffset",
        "nvmlDeviceGetMinMaxClockOfPState",
        "nvmlDeviceGetSupportedPerformanceStates",
        "nvmlDeviceGetGpcClkMinMaxVfOffset",
        "nvmlDeviceGetMemClkMinMaxVfOffset",
        "nvmlDeviceGetGpuOperationMode",
        "nvmlDeviceGetEccMode",
        "nvmlDeviceGetDefaultEccMode",
        "nvmlDeviceGetTotalEccErrors",
        "nvmlDeviceGetDetailedEccErrors",
        "nvmlDeviceGetMemoryErrorCounter",
        "nvmlDeviceGetEncoderUtilization",
        "nvmlDeviceGetEncoderCapacity",
        "nvmlDeviceGetEncoderStats",
        "nvmlDeviceGetDecoderUtilization",
        "nvmlDeviceGetJpgUtilization",
        "nvmlDeviceGetOfaUtilization",
        "nvmlDeviceGetFBCStats",
        "nvmlDeviceGetDriverModel_v2",
        "nvmlDeviceGetBridgeChipInfo",
        "nvmlDeviceGetAPIRestriction",
        "nvmlDeviceGetViolationStatus",
        "nvmlDeviceGetIrqNum",
        "nvmlDeviceGetNumGpuCores",
        "nvmlDeviceGetPowerSource",
        "nvmlDeviceGetMemoryBusWidth",
        "nvmlDeviceGetPcieLinkMaxSpeed",
        "nvmlDeviceGetPcieSpeed",
        "nvmlDeviceGetAdaptiveClockInfoStatus",
        "nvmlDeviceGetBusType",
        "nvmlDeviceGetGpuFabricInfo",
        "nvmlDeviceGetGspFirmwareMode",
        "nvmlDeviceGetAccountingMode",
        "nvmlDeviceGetAccountingBufferSize",
        "nvmlDeviceGetRetiredPagesPendingStatus",
        "nvmlDeviceGetRowRemapperHistogram",
        "nvmlDeviceGetRemappedRows",
        "nvmlDeviceGetRemappedRows_v2",
        "nvmlDeviceGetSupportedEventTypes",
    };
    static_assert(sizeof(more_optional_results) / sizeof(more_optional_results[0]) ==
                  sizeof(more_optional_labels) / sizeof(more_optional_labels[0]));
    for (size_t i = 0;
         i < sizeof(more_optional_results) / sizeof(more_optional_results[0]); ++i)
    {
        if (check_optional(more_optional_results[i], more_optional_labels[i]))
        {
            return 1;
        }
    }

    result = nvmlDeviceOnSameBoard(device, device, &same_board);
    if (check_optional(result, "nvmlDeviceOnSameBoard"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS && same_board == 0)
    {
        std::cout << "Expected device to be on the same board as itself" << std::endl;
        return 1;
    }

    char inforom_version[NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE] = {};
    result = nvmlDeviceGetInforomVersion(device, NVML_INFOROM_OEM,
                                         inforom_version,
                                         sizeof(inforom_version));
    if (check_optional(result, "nvmlDeviceGetInforomVersion"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS &&
        check_nonempty(inforom_version, "nvmlDeviceGetInforomVersion"))
    {
        return 1;
    }

    char inforom_image_version[NVML_DEVICE_INFOROM_VERSION_BUFFER_SIZE] = {};
    result = nvmlDeviceGetInforomImageVersion(device, inforom_image_version,
                                              sizeof(inforom_image_version));
    if (check_optional(result, "nvmlDeviceGetInforomImageVersion"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS &&
        check_nonempty(inforom_image_version, "nvmlDeviceGetInforomImageVersion"))
    {
        return 1;
    }

    char gsp_firmware_version[NVML_GSP_FIRMWARE_VERSION_BUF_SIZE] = {};
    result = nvmlDeviceGetGspFirmwareVersion(device, gsp_firmware_version);
    if (check_optional(result, "nvmlDeviceGetGspFirmwareVersion"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS &&
        check_nonempty(gsp_firmware_version, "nvmlDeviceGetGspFirmwareVersion"))
    {
        return 1;
    }

    nvmlSystemDriverBranchInfo_t driver_branch = {};
    driver_branch.version = nvmlSystemDriverBranchInfo_v1;
    result = nvmlSystemGetDriverBranch(&driver_branch, sizeof(driver_branch));
    if (check_optional(result, "nvmlSystemGetDriverBranch"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS &&
        check_nonempty(driver_branch.branch, "nvmlSystemGetDriverBranch"))
    {
        return 1;
    }

    nvmlC2cModeInfo_v1_t c2c_mode = {};
    result = nvmlDeviceGetC2cModeInfoV(device, &c2c_mode);
    if (check_optional(result, "nvmlDeviceGetC2cModeInfoV"))
    {
        return 1;
    }

    nvmlDeviceAddressingMode_t addressing_mode = {};
    addressing_mode.version = nvmlDeviceAddressingMode_v1;
    result = nvmlDeviceGetAddressingMode(device, &addressing_mode);
    if (check_optional(result, "nvmlDeviceGetAddressingMode"))
    {
        return 1;
    }

    nvmlRepairStatus_t repair_status = {};
    repair_status.version = nvmlRepairStatus_v1;
    result = nvmlDeviceGetRepairStatus(device, &repair_status);
    if (check_optional(result, "nvmlDeviceGetRepairStatus"))
    {
        return 1;
    }

    nvmlPciInfoExt_t pci_ext = {};
    pci_ext.version = nvmlPciInfoExt_v1;
    result = nvmlDeviceGetPciInfoExt(device, &pci_ext);
    if (check_optional(result, "nvmlDeviceGetPciInfoExt"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS && check_nonempty(pci_ext.busId, "nvmlDeviceGetPciInfoExt"))
    {
        return 1;
    }

    nvmlFanSpeedInfo_t fan_speed_rpm = {};
    fan_speed_rpm.version = nvmlFanSpeedInfo_v1;
    fan_speed_rpm.fan = 0;
    result = nvmlDeviceGetFanSpeedRPM(device, &fan_speed_rpm);
    if (check_optional(result, "nvmlDeviceGetFanSpeedRPM"))
    {
        return 1;
    }

    nvmlCoolerInfo_t cooler_info = {};
    cooler_info.version = nvmlCoolerInfo_v1;
    cooler_info.index = 0;
    result = nvmlDeviceGetCoolerInfo(device, &cooler_info);
    if (check_optional(result, "nvmlDeviceGetCoolerInfo"))
    {
        return 1;
    }

    nvmlTemperature_t temperature_v = {};
    temperature_v.version = nvmlTemperature_v1;
    temperature_v.sensorType = NVML_TEMPERATURE_GPU;
    result = nvmlDeviceGetTemperatureV(device, &temperature_v);
    if (check_optional(result, "nvmlDeviceGetTemperatureV"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS && temperature_v.temperature <= 0)
    {
        std::cout << "Unexpected temperature: " << temperature_v.temperature
                  << std::endl;
        return 1;
    }

    nvmlMarginTemperature_t margin_temperature = {};
    margin_temperature.version = nvmlMarginTemperature_v1;
    result = nvmlDeviceGetMarginTemperature(device, &margin_temperature);
    if (check_optional(result, "nvmlDeviceGetMarginTemperature"))
    {
        return 1;
    }

    nvmlClockOffset_t clock_offsets = {};
    clock_offsets.version = nvmlClockOffset_v1;
    clock_offsets.type = NVML_CLOCK_GRAPHICS;
    clock_offsets.pstate = NVML_PSTATE_0;
    result = nvmlDeviceGetClockOffsets(device, &clock_offsets);
    if (check_optional(result, "nvmlDeviceGetClockOffsets"))
    {
        return 1;
    }

    nvmlDevicePerfModes_t performance_modes = {};
    performance_modes.version = nvmlDevicePerfModes_v1;
    result = nvmlDeviceGetPerformanceModes(device, &performance_modes);
    if (check_optional(result, "nvmlDeviceGetPerformanceModes"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS &&
        check_nonempty(performance_modes.str, "nvmlDeviceGetPerformanceModes"))
    {
        return 1;
    }

    nvmlDeviceCurrentClockFreqs_t current_clock_freqs = {};
    current_clock_freqs.version = nvmlDeviceCurrentClockFreqs_v1;
    result = nvmlDeviceGetCurrentClockFreqs(device, &current_clock_freqs);
    if (check_optional(result, "nvmlDeviceGetCurrentClockFreqs"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS &&
        check_nonempty(current_clock_freqs.str, "nvmlDeviceGetCurrentClockFreqs"))
    {
        return 1;
    }

    nvmlDevicePowerMizerModes_v1_t power_mizer = {};
    result = nvmlDeviceGetPowerMizerMode_v1(device, &power_mizer);
    if (check_optional(result, "nvmlDeviceGetPowerMizerMode_v1"))
    {
        return 1;
    }

    nvmlGpuFabricInfoV_t fabric_info_v = {};
    fabric_info_v.version = nvmlGpuFabricInfo_v3;
    result = nvmlDeviceGetGpuFabricInfoV(device, &fabric_info_v);
    if (check_optional(result, "nvmlDeviceGetGpuFabricInfoV"))
    {
        return 1;
    }

    if (check_process_list(nvmlDeviceGetComputeRunningProcesses_v3, device,
                           "nvmlDeviceGetComputeRunningProcesses_v3") ||
        check_process_list(nvmlDeviceGetGraphicsRunningProcesses_v3, device,
                           "nvmlDeviceGetGraphicsRunningProcesses_v3") ||
        check_process_list(nvmlDeviceGetMPSComputeRunningProcesses_v3, device,
                           "nvmlDeviceGetMPSComputeRunningProcesses_v3"))
    {
        return 1;
    }

    unsigned int topology_capacity = device_count > 0 ? device_count : 1;
    std::vector<nvmlDevice_t> nearest_devices(topology_capacity);
    unsigned int topology_count = topology_capacity;
    result = nvmlDeviceGetTopologyNearestGpus(device, NVML_TOPOLOGY_SYSTEM,
                                              &topology_count,
                                              nearest_devices.data());
    if (result != NVML_ERROR_UNKNOWN &&
        result != NVML_ERROR_LIB_RM_VERSION_MISMATCH &&
        check_optional_list(result, "nvmlDeviceGetTopologyNearestGpus"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS && topology_count > topology_capacity)
    {
        std::cout << "Topology query returned " << topology_count
                  << " entries for capacity " << topology_capacity << std::endl;
        return 1;
    }

    unsigned int memory_clock_count = 0;
    unsigned int clock_dummy = 0;
    result = nvmlDeviceGetSupportedMemoryClocks(device, &memory_clock_count,
                                                &clock_dummy);
    if (check_optional_list(result, "nvmlDeviceGetSupportedMemoryClocks"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS || result == NVML_ERROR_INSUFFICIENT_SIZE)
    {
        unsigned int memory_clock_capacity =
            memory_clock_count > 0 ? memory_clock_count : 64;
        std::vector<unsigned int> memory_clocks(memory_clock_capacity);
        memory_clock_count = memory_clock_capacity;
        result = nvmlDeviceGetSupportedMemoryClocks(device, &memory_clock_count,
                                                    memory_clocks.data());
        if (check_optional_list(result, "nvmlDeviceGetSupportedMemoryClocks"))
        {
            return 1;
        }
        if (result == NVML_SUCCESS && memory_clock_count > memory_clock_capacity)
        {
            std::cout << "Memory clock query returned " << memory_clock_count
                      << " entries for capacity " << memory_clock_capacity
                      << std::endl;
            return 1;
        }
        if (result == NVML_SUCCESS && memory_clock_count > 0)
        {
            unsigned int graphics_clock_count = 0;
            result = nvmlDeviceGetSupportedGraphicsClocks(
                device, memory_clocks[0], &graphics_clock_count, &clock_dummy);
            if (check_optional_list(result, "nvmlDeviceGetSupportedGraphicsClocks"))
            {
                return 1;
            }
            if (result == NVML_SUCCESS || result == NVML_ERROR_INSUFFICIENT_SIZE)
            {
                unsigned int graphics_clock_capacity =
                    graphics_clock_count > 0 ? graphics_clock_count : 128;
                std::vector<unsigned int> graphics_clocks(graphics_clock_capacity);
                graphics_clock_count = graphics_clock_capacity;
                result = nvmlDeviceGetSupportedGraphicsClocks(
                    device, memory_clocks[0], &graphics_clock_count,
                    graphics_clocks.data());
                if (check_optional_list(result, "nvmlDeviceGetSupportedGraphicsClocks"))
                {
                    return 1;
                }
                if (result == NVML_SUCCESS &&
                    graphics_clock_count > graphics_clock_capacity)
                {
                    std::cout << "Graphics clock query returned "
                              << graphics_clock_count << " entries for capacity "
                              << graphics_clock_capacity << std::endl;
                    return 1;
                }
            }
        }
    }

    unsigned int accounting_pid_count = 0;
    unsigned int pid_dummy = 0;
    result = nvmlDeviceGetAccountingPids(device, &accounting_pid_count, &pid_dummy);
    if (check_optional_list(result, "nvmlDeviceGetAccountingPids"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS || result == NVML_ERROR_INSUFFICIENT_SIZE)
    {
        unsigned int accounting_pid_capacity =
            accounting_pid_count > 0 ? accounting_pid_count : 64;
        std::vector<unsigned int> accounting_pids(accounting_pid_capacity);
        accounting_pid_count = accounting_pid_capacity;
        result = nvmlDeviceGetAccountingPids(device, &accounting_pid_count,
                                             accounting_pids.data());
        if (check_optional_list(result, "nvmlDeviceGetAccountingPids"))
        {
            return 1;
        }
        if (result == NVML_SUCCESS)
        {
            if (accounting_pid_count > accounting_pid_capacity)
            {
                std::cout << "Accounting PID query returned "
                          << accounting_pid_count << " entries for capacity "
                          << accounting_pid_capacity << std::endl;
                return 1;
            }
            for (unsigned int i = 0; i < accounting_pid_count; ++i)
            {
                nvmlAccountingStats_t accounting_stats = {};
                result = nvmlDeviceGetAccountingStats(device, accounting_pids[i],
                                                      &accounting_stats);
                if (check_optional(result, "nvmlDeviceGetAccountingStats"))
                {
                    return 1;
                }
            }
        }
    }

    unsigned int retired_page_count = 0;
    unsigned long long retired_address_dummy = 0;
    result = nvmlDeviceGetRetiredPages(
        device, NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS,
        &retired_page_count, &retired_address_dummy);
    if (check_optional_list(result, "nvmlDeviceGetRetiredPages"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS || result == NVML_ERROR_INSUFFICIENT_SIZE)
    {
        unsigned int retired_page_capacity =
            retired_page_count > 0 ? retired_page_count : 16;
        std::vector<unsigned long long> retired_addresses(retired_page_capacity);
        retired_page_count = retired_page_capacity;
        result = nvmlDeviceGetRetiredPages(
            device, NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS,
            &retired_page_count, retired_addresses.data());
        if (check_optional_list(result, "nvmlDeviceGetRetiredPages"))
        {
            return 1;
        }
        if (result == NVML_SUCCESS && retired_page_count > retired_page_capacity)
        {
            std::cout << "Retired page query returned " << retired_page_count
                      << " entries for capacity " << retired_page_capacity
                      << std::endl;
            return 1;
        }
    }

    retired_page_count = 0;
    unsigned long long retired_timestamp_dummy = 0;
    result = nvmlDeviceGetRetiredPages_v2(
        device, NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS,
        &retired_page_count, &retired_address_dummy, &retired_timestamp_dummy);
    if (check_optional_list(result, "nvmlDeviceGetRetiredPages_v2"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS || result == NVML_ERROR_INSUFFICIENT_SIZE)
    {
        unsigned int retired_page_capacity =
            retired_page_count > 0 ? retired_page_count : 16;
        std::vector<unsigned long long> retired_addresses(retired_page_capacity);
        std::vector<unsigned long long> retired_timestamps(retired_page_capacity);
        retired_page_count = retired_page_capacity;
        result = nvmlDeviceGetRetiredPages_v2(
            device, NVML_PAGE_RETIREMENT_CAUSE_MULTIPLE_SINGLE_BIT_ECC_ERRORS,
            &retired_page_count, retired_addresses.data(),
            retired_timestamps.data());
        if (check_optional_list(result, "nvmlDeviceGetRetiredPages_v2"))
        {
            return 1;
        }
        if (result == NVML_SUCCESS && retired_page_count > retired_page_capacity)
        {
            std::cout << "Retired page v2 query returned " << retired_page_count
                      << " entries for capacity " << retired_page_capacity
                      << std::endl;
            return 1;
        }
    }

    std::vector<nvmlProcessUtilizationSample_t> process_utilization(64);
    unsigned int process_sample_count = process_utilization.size();
    result = nvmlDeviceGetProcessUtilization(device, process_utilization.data(),
                                             &process_sample_count, 0);
    if (check_optional_list(result, "nvmlDeviceGetProcessUtilization"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS && process_sample_count > process_utilization.size())
    {
        std::cout << "Process utilization query returned " << process_sample_count
                  << " entries for capacity " << process_utilization.size()
                  << std::endl;
        return 1;
    }

    nvmlValueType_t sample_value_type;
    std::vector<nvmlSample_t> samples(64);
    unsigned int sample_count = samples.size();
    result = nvmlDeviceGetSamples(device, NVML_GPU_UTILIZATION_SAMPLES, 0,
                                  &sample_value_type, &sample_count,
                                  samples.data());
    if (check_optional_list(result, "nvmlDeviceGetSamples"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS && sample_count > samples.size())
    {
        std::cout << "Samples query returned " << sample_count
                  << " entries for capacity " << samples.size() << std::endl;
        return 1;
    }

    nvmlFieldValue_t field_values[2] = {};
    field_values[0].fieldId = NVML_FI_DEV_MEMORY_TEMP;
    field_values[1].fieldId = NVML_FI_DEV_TOTAL_ENERGY_CONSUMPTION;
    result = nvmlDeviceGetFieldValues(device, 2, field_values);
    if (check_optional(result, "nvmlDeviceGetFieldValues"))
    {
        return 1;
    }

    result = nvmlDeviceGetPowerManagementLimitConstraints(device, &min_limit, &max_limit);
    if (check_optional(result, "nvmlDeviceGetPowerManagementLimitConstraints"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS && min_limit > max_limit)
    {
        std::cout << "Unexpected power limit range: " << min_limit
                  << " > " << max_limit << std::endl;
        return 1;
    }

    char vbios_version[128] = {};
    result = nvmlDeviceGetVbiosVersion(device, vbios_version, sizeof(vbios_version));
    if (check_optional(result, "nvmlDeviceGetVbiosVersion"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS && check_nonempty(vbios_version, "nvmlDeviceGetVbiosVersion"))
    {
        return 1;
    }

    char board_part_number[128] = {};
    result = nvmlDeviceGetBoardPartNumber(device, board_part_number, sizeof(board_part_number));
    if (check_optional(result, "nvmlDeviceGetBoardPartNumber"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS &&
        check_nonempty(board_part_number, "nvmlDeviceGetBoardPartNumber"))
    {
        return 1;
    }

    nvmlBAR1Memory_t bar1 = {};
    result = nvmlDeviceGetBAR1MemoryInfo(device, &bar1);
    if (check_optional(result, "nvmlDeviceGetBAR1MemoryInfo"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS && bar1.bar1Free > bar1.bar1Total)
    {
        std::cout << "Unexpected BAR1 memory info: total=" << bar1.bar1Total
                  << " free=" << bar1.bar1Free << std::endl;
        return 1;
    }

    result = nvmlDeviceGetBoardId(device, &value);
    if (check_optional(result, "nvmlDeviceGetBoardId"))
    {
        return 1;
    }

    result = nvmlDeviceGetMultiGpuBoard(device, &value);
    if (check_optional(result, "nvmlDeviceGetMultiGpuBoard"))
    {
        return 1;
    }

    unsigned int current_mig_mode = 0;
    unsigned int pending_mig_mode = 0;
    result = nvmlDeviceGetMigMode(device, &current_mig_mode, &pending_mig_mode);
    if (check_optional(result, "nvmlDeviceGetMigMode"))
    {
        return 1;
    }

    unsigned long memory_affinity[16] = {};
    result = nvmlDeviceGetMemoryAffinity(device, 16, memory_affinity,
                                         NVML_AFFINITY_SCOPE_NODE);
    if (check_optional(result, "nvmlDeviceGetMemoryAffinity"))
    {
        return 1;
    }

    unsigned long cpu_affinity[16] = {};
    result = nvmlDeviceGetCpuAffinityWithinScope(device, 16, cpu_affinity,
                                                 NVML_AFFINITY_SCOPE_NODE);
    if (check_optional(result, "nvmlDeviceGetCpuAffinityWithinScope"))
    {
        return 1;
    }
    result = nvmlDeviceGetCpuAffinity(device, 16, cpu_affinity);
    if (check_optional(result, "nvmlDeviceGetCpuAffinity"))
    {
        return 1;
    }

    nvmlGpuThermalSettings_t thermal_settings = {};
    result = nvmlDeviceGetThermalSettings(device, 0, &thermal_settings);
    if (check_optional_indexed(result, "nvmlDeviceGetThermalSettings"))
    {
        return 1;
    }

    nvmlDramEncryptionInfo_t dram_current = {};
    nvmlDramEncryptionInfo_t dram_pending = {};
    dram_current.version = nvmlDramEncryptionInfo_v1;
    dram_pending.version = nvmlDramEncryptionInfo_v1;
    result = nvmlDeviceGetDramEncryptionMode(device, &dram_current, &dram_pending);
    if (check_optional(result, "nvmlDeviceGetDramEncryptionMode"))
    {
        return 1;
    }

    nvmlEccSramErrorStatus_t sram_status = {};
    sram_status.version = nvmlEccSramErrorStatus_v1;
    result = nvmlDeviceGetSramEccErrorStatus(device, &sram_status);
    if (check_optional(result, "nvmlDeviceGetSramEccErrorStatus"))
    {
        return 1;
    }

    nvmlClkMonStatus_t clk_mon = {};
    result = nvmlDeviceGetClkMonStatus(device, &clk_mon);
    if (check_optional(result, "nvmlDeviceGetClkMonStatus"))
    {
        return 1;
    }

    nvmlPlatformInfo_t platform_info = {};
    platform_info.version = nvmlPlatformInfo_v2;
    result = nvmlDeviceGetPlatformInfo(device, &platform_info);
    if (check_optional(result, "nvmlDeviceGetPlatformInfo"))
    {
        return 1;
    }

    nvmlPdi_t pdi = {};
    pdi.version = nvmlPdi_v1;
    result = nvmlDeviceGetPdi(device, &pdi);
    if (check_optional(result, "nvmlDeviceGetPdi"))
    {
        return 1;
    }

    nvmlHostname_v1_t hostname = {};
    result = nvmlDeviceGetHostname_v1(device, &hostname);
    if (check_optional(result, "nvmlDeviceGetHostname_v1"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS &&
        check_nonempty(hostname.value, "nvmlDeviceGetHostname_v1"))
    {
        return 1;
    }

    nvmlGpuVirtualizationMode_t virtualization_mode =
        NVML_GPU_VIRTUALIZATION_MODE_NONE;
    result = nvmlDeviceGetVirtualizationMode(device, &virtualization_mode);
    if (check_optional(result, "nvmlDeviceGetVirtualizationMode"))
    {
        return 1;
    }

    nvmlHostVgpuMode_t host_vgpu_mode = NVML_HOST_VGPU_MODE_NON_SRIOV;
    result = nvmlDeviceGetHostVgpuMode(device, &host_vgpu_mode);
    if (check_optional(result, "nvmlDeviceGetHostVgpuMode"))
    {
        return 1;
    }

    unsigned int is_mig_device = 0;
    result = nvmlDeviceIsMigDeviceHandle(device, &is_mig_device);
    if (check_optional(result, "nvmlDeviceIsMigDeviceHandle"))
    {
        return 1;
    }

    unsigned int gpu_instance_id = 0;
    result = nvmlDeviceGetGpuInstanceId(device, &gpu_instance_id);
    if (check_optional_indexed(result, "nvmlDeviceGetGpuInstanceId"))
    {
        return 1;
    }

    unsigned int compute_instance_id = 0;
    result = nvmlDeviceGetComputeInstanceId(device, &compute_instance_id);
    if (check_optional_indexed(result, "nvmlDeviceGetComputeInstanceId"))
    {
        return 1;
    }

    unsigned int max_mig_devices = 0;
    result = nvmlDeviceGetMaxMigDeviceCount(device, &max_mig_devices);
    if (check_optional(result, "nvmlDeviceGetMaxMigDeviceCount"))
    {
        return 1;
    }

    nvmlDevice_t mig_device = nullptr;
    result = nvmlDeviceGetMigDeviceHandleByIndex(device, 0, &mig_device);
    if (check_optional_list(result, "nvmlDeviceGetMigDeviceHandleByIndex"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS)
    {
        nvmlDevice_t parent_device = nullptr;
        result = nvmlDeviceGetDeviceHandleFromMigDeviceHandle(mig_device,
                                                              &parent_device);
        if (check_optional(result,
                           "nvmlDeviceGetDeviceHandleFromMigDeviceHandle"))
        {
            return 1;
        }
    }

    nvmlDeviceCapabilities_t caps = {};
    caps.version = nvmlDeviceCapabilities_v1;
    result = nvmlDeviceGetCapabilities(device, &caps);
    if (check_optional(result, "nvmlDeviceGetCapabilities"))
    {
        return 1;
    }

    nvmlEnableState_t nvlink_active = NVML_FEATURE_DISABLED;
    result = nvmlDeviceGetNvLinkState(device, 0, &nvlink_active);
    if (check_optional_indexed(result, "nvmlDeviceGetNvLinkState"))
    {
        return 1;
    }

    unsigned int nvlink_version = 0;
    result = nvmlDeviceGetNvLinkVersion(device, 0, &nvlink_version);
    if (check_optional_indexed(result, "nvmlDeviceGetNvLinkVersion"))
    {
        return 1;
    }

    unsigned int nvlink_capability = 0;
    result = nvmlDeviceGetNvLinkCapability(device, 0, NVML_NVLINK_CAP_VALID,
                                           &nvlink_capability);
    if (check_optional_indexed(result, "nvmlDeviceGetNvLinkCapability"))
    {
        return 1;
    }

    nvmlPciInfo_t nvlink_pci = {};
    result = nvmlDeviceGetNvLinkRemotePciInfo_v2(device, 0, &nvlink_pci);
    if (check_optional_indexed(result, "nvmlDeviceGetNvLinkRemotePciInfo_v2"))
    {
        return 1;
    }

    unsigned long long nvlink_counter = 0;
    result = nvmlDeviceGetNvLinkErrorCounter(device, 0,
                                             NVML_NVLINK_ERROR_DL_REPLAY,
                                             &nvlink_counter);
    if (check_optional_indexed(result, "nvmlDeviceGetNvLinkErrorCounter"))
    {
        return 1;
    }

    nvmlNvLinkUtilizationControl_t nvlink_control = {};
    result = nvmlDeviceGetNvLinkUtilizationControl(device, 0, 0,
                                                   &nvlink_control);
    if (check_optional_indexed(result,
                               "nvmlDeviceGetNvLinkUtilizationControl"))
    {
        return 1;
    }

    unsigned long long rx_counter = 0;
    unsigned long long tx_counter = 0;
    result = nvmlDeviceGetNvLinkUtilizationCounter(device, 0, 0,
                                                   &rx_counter, &tx_counter);
    if (check_optional_indexed(result,
                               "nvmlDeviceGetNvLinkUtilizationCounter"))
    {
        return 1;
    }

    nvmlIntNvLinkDeviceType_t nvlink_device_type =
        NVML_NVLINK_DEVICE_TYPE_UNKNOWN;
    result = nvmlDeviceGetNvLinkRemoteDeviceType(device, 0,
                                                 &nvlink_device_type);
    if (check_optional_indexed(result,
                               "nvmlDeviceGetNvLinkRemoteDeviceType"))
    {
        return 1;
    }

    unsigned int system_nvlink_bw_mode = 0;
    result = nvmlSystemGetNvlinkBwMode(&system_nvlink_bw_mode);
    if (check_optional(result, "nvmlSystemGetNvlinkBwMode"))
    {
        return 1;
    }

    nvmlNvlinkSupportedBwModes_t supported_bw_modes = {};
    supported_bw_modes.version = nvmlNvlinkSupportedBwModes_v1;
    result = nvmlDeviceGetNvlinkSupportedBwModes(device, &supported_bw_modes);
    if (check_optional(result, "nvmlDeviceGetNvlinkSupportedBwModes"))
    {
        return 1;
    }

    nvmlNvlinkGetBwMode_t nvlink_bw_mode = {};
    nvlink_bw_mode.version = nvmlNvlinkGetBwMode_v1;
    result = nvmlDeviceGetNvlinkBwMode(device, &nvlink_bw_mode);
    if (check_optional(result, "nvmlDeviceGetNvlinkBwMode"))
    {
        return 1;
    }

    nvmlNvLinkInfo_t nvlink_info = {};
    nvlink_info.version = nvmlNvLinkInfo_v2;
    result = nvmlDeviceGetNvLinkInfo(device, &nvlink_info);
    if (check_optional(result, "nvmlDeviceGetNvLinkInfo"))
    {
        return 1;
    }

    nvmlEventSet_t event_set = nullptr;
    result = nvmlEventSetCreate(&event_set);
    if (check_success(result, "nvmlEventSetCreate"))
    {
        return 1;
    }
    if (supported_event_types != 0)
    {
        result = nvmlDeviceRegisterEvents(device, supported_event_types, event_set);
        if (check_optional(result, "nvmlDeviceRegisterEvents"))
        {
            return 1;
        }
        if (result == NVML_SUCCESS)
        {
            nvmlEventData_t event_data = {};
            result = nvmlEventSetWait_v2(event_set, &event_data, 0);
            if (result != NVML_ERROR_TIMEOUT &&
                check_optional(result, "nvmlEventSetWait_v2"))
            {
                return 1;
            }
        }
    }
    result = nvmlEventSetFree(event_set);
    if (check_success(result, "nvmlEventSetFree"))
    {
        return 1;
    }

    unsigned int excluded_device_count = 0;
    result = nvmlGetExcludedDeviceCount(&excluded_device_count);
    if (check_success(result, "nvmlGetExcludedDeviceCount"))
    {
        return 1;
    }
    nvmlExcludedDeviceInfo_t excluded_device_info = {};
    result = nvmlGetExcludedDeviceInfoByIndex(0, &excluded_device_info);
    if (excluded_device_count > 0)
    {
        if (check_success(result, "nvmlGetExcludedDeviceInfoByIndex"))
        {
            return 1;
        }
    }
    else if (check_optional_indexed(result, "nvmlGetExcludedDeviceInfoByIndex"))
    {
        return 1;
    }

    return 0;
}
