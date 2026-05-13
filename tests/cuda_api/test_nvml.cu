#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <iostream>
#include <nvml.h>

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

    return 0;
}
