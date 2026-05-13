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

static int check_event_wait(nvmlReturn_t result, const char *label)
{
    if (result == NVML_SUCCESS || result == NVML_ERROR_TIMEOUT)
    {
        return 0;
    }
    return check_optional(result, label);
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

template <typename T>
static int check_counted_list(nvmlReturn_t (*get_entries)(nvmlDevice_t, unsigned int *, T *),
                              nvmlDevice_t device, const char *label,
                              unsigned int fallback_capacity = 64)
{
    unsigned int required = 0;
    nvmlReturn_t result = get_entries(device, &required, nullptr);
    if (check_optional_list(result, label))
    {
        return 1;
    }
    if (result != NVML_SUCCESS && result != NVML_ERROR_INSUFFICIENT_SIZE)
    {
        return 0;
    }

    unsigned int capacity = required > 0 ? required : fallback_capacity;
    std::vector<T> entries(capacity);
    unsigned int count = capacity;
    result = get_entries(device, &count, entries.data());
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

    const char *success_error_string = nvmlErrorString(NVML_SUCCESS);
    if (success_error_string == nullptr ||
        check_nonempty(success_error_string, "nvmlErrorString success"))
    {
        return 1;
    }
    const char *invalid_error_string = nvmlErrorString(NVML_ERROR_INVALID_ARGUMENT);
    if (invalid_error_string == nullptr ||
        check_nonempty(invalid_error_string, "nvmlErrorString invalid argument"))
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

    unsigned int hic_count = 0;
    result = nvmlSystemGetHicVersion(&hic_count, nullptr);
    if (check_optional_list(result, "nvmlSystemGetHicVersion"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS || result == NVML_ERROR_INSUFFICIENT_SIZE)
    {
        unsigned int hic_capacity = hic_count > 0 ? hic_count : 16;
        std::vector<nvmlHwbcEntry_t> hic_entries(hic_capacity);
        hic_count = hic_capacity;
        result = nvmlSystemGetHicVersion(&hic_count, hic_entries.data());
        if (check_optional_list(result, "nvmlSystemGetHicVersion"))
        {
            return 1;
        }
        if (result == NVML_SUCCESS && hic_count > hic_capacity)
        {
            std::cout << "HIC query returned " << hic_count
                      << " entries for capacity " << hic_capacity << std::endl;
            return 1;
        }
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

    nvmlUUID_t uuid_struct = {};
    uuid_struct.version = nvmlUUID_v1;
    uuid_struct.type = NVML_UUID_TYPE_ASCII;
    std::strncpy(uuid_struct.value.str, uuid, sizeof(uuid_struct.value.str) - 1);
    nvmlDevice_t by_uuid_v = nullptr;
    result = nvmlDeviceGetHandleByUUIDV(&uuid_struct, &by_uuid_v);
    if (check_success(result, "nvmlDeviceGetHandleByUUIDV"))
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
    if (result == NVML_SUCCESS)
    {
        nvmlDevice_t by_serial = nullptr;
        result = nvmlDeviceGetHandleBySerial(serial, &by_serial);
        if (check_success(result, "nvmlDeviceGetHandleBySerial"))
        {
            return 1;
        }
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
    if (unit_count > 0)
    {
        nvmlUnit_t unit = nullptr;
        result = nvmlUnitGetHandleByIndex(0, &unit);
        if (check_success(result, "nvmlUnitGetHandleByIndex"))
        {
            return 1;
        }
        nvmlUnitInfo_t unit_info = {};
        nvmlLedState_t led_state = {};
        nvmlPSUInfo_t psu_info = {};
        nvmlUnitFanSpeeds_t unit_fans = {};
        nvmlReturn_t unit_results[] = {
            nvmlUnitGetUnitInfo(unit, &unit_info),
            nvmlUnitGetLedState(unit, &led_state),
            nvmlUnitGetPsuInfo(unit, &psu_info),
            nvmlUnitGetTemperature(unit, 0, &value),
            nvmlUnitGetFanSpeedInfo(unit, &unit_fans),
        };
        const char *unit_labels[] = {
            "nvmlUnitGetUnitInfo",
            "nvmlUnitGetLedState",
            "nvmlUnitGetPsuInfo",
            "nvmlUnitGetTemperature",
            "nvmlUnitGetFanSpeedInfo",
        };
        static_assert(sizeof(unit_results) / sizeof(unit_results[0]) ==
                      sizeof(unit_labels) / sizeof(unit_labels[0]));
        for (size_t i = 0; i < sizeof(unit_results) / sizeof(unit_results[0]); ++i)
        {
            if (check_optional(unit_results[i], unit_labels[i]))
            {
                return 1;
            }
        }

        unsigned int unit_device_count = 0;
        result = nvmlUnitGetDevices(unit, &unit_device_count, nullptr);
        if (check_optional_list(result, "nvmlUnitGetDevices"))
        {
            return 1;
        }
        if (result == NVML_SUCCESS || result == NVML_ERROR_INSUFFICIENT_SIZE)
        {
            unsigned int unit_device_capacity =
                unit_device_count > 0 ? unit_device_count : device_count;
            std::vector<nvmlDevice_t> unit_devices(unit_device_capacity);
            unit_device_count = unit_device_capacity;
            result = nvmlUnitGetDevices(unit, &unit_device_count,
                                        unit_devices.data());
            if (check_optional_list(result, "nvmlUnitGetDevices"))
            {
                return 1;
            }
            if (result == NVML_SUCCESS && unit_device_count > unit_device_capacity)
            {
                std::cout << "Unit device query returned " << unit_device_count
                          << " entries for capacity " << unit_device_capacity
                          << std::endl;
                return 1;
            }
        }
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
    nvmlConfComputeSystemCaps_t cc_caps = {};
    nvmlConfComputeSystemState_t cc_state = {};
    nvmlConfComputeMemSizeInfo_t cc_mem_size = {};
    nvmlMemory_t cc_protected_memory = {};
    nvmlConfComputeGpuCertificate_t cc_certificate = {};
    nvmlConfComputeGpuAttestationReport_t cc_attestation = {};
    nvmlConfComputeGetKeyRotationThresholdInfo_t cc_key_rotation = {};
    cc_key_rotation.version = nvmlConfComputeGetKeyRotationThresholdInfo_v1;
    nvmlSystemConfComputeSettings_t cc_settings = {};
    cc_settings.version = nvmlSystemConfComputeSettings_v1;
    nvmlGridLicensableFeatures_t grid_features = {};
    unsigned int vgpu_capability = 0;
    nvmlVgpuHeterogeneousMode_t vgpu_heterogeneous = {};
    vgpu_heterogeneous.version = nvmlVgpuHeterogeneousMode_v1;
    nvmlVgpuSchedulerCapabilities_t vgpu_scheduler_caps = {};
    nvmlVgpuSchedulerStateInfo_v2_t vgpu_scheduler_state_v2 = {};
    nvmlVgpuSchedulerLogInfo_v2_t vgpu_scheduler_log_v2 = {};
    nvmlVgpuVersion_t supported_vgpu_version = {};
    nvmlVgpuVersion_t current_vgpu_version = {};
    nvmlGpmSupport_t gpm_support = {};
    gpm_support.version = NVML_GPM_SUPPORT_VERSION;
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
        nvmlSystemGetConfComputeCapabilities(&cc_caps),
        nvmlSystemGetConfComputeState(&cc_state),
        nvmlDeviceGetConfComputeMemSizeInfo(device, &cc_mem_size),
        nvmlSystemGetConfComputeGpusReadyState(&value),
        nvmlDeviceGetConfComputeProtectedMemoryUsage(device, &cc_protected_memory),
        nvmlSystemGetConfComputeKeyRotationThresholdInfo(&cc_key_rotation),
        nvmlSystemGetConfComputeSettings(&cc_settings),
        nvmlDeviceGetGridLicensableFeatures_v4(device, &grid_features),
        nvmlGetVgpuDriverCapabilities(NVML_VGPU_DRIVER_CAP_HETEROGENEOUS_MULTI_VGPU,
                                      &vgpu_capability),
        nvmlDeviceGetVgpuCapabilities(device, NVML_DEVICE_VGPU_CAP_FRACTIONAL_MULTI_VGPU,
                                      &vgpu_capability),
        nvmlDeviceGetVgpuHeterogeneousMode(device, &vgpu_heterogeneous),
        nvmlDeviceGetVgpuSchedulerCapabilities(device, &vgpu_scheduler_caps),
        nvmlDeviceGetVgpuSchedulerState_v2(device, &vgpu_scheduler_state_v2),
        nvmlDeviceGetVgpuSchedulerLog_v2(device, &vgpu_scheduler_log_v2),
        nvmlGetVgpuVersion(&supported_vgpu_version, &current_vgpu_version),
        nvmlGpmQueryDeviceSupport(device, &gpm_support),
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
        "nvmlSystemGetConfComputeCapabilities",
        "nvmlSystemGetConfComputeState",
        "nvmlDeviceGetConfComputeMemSizeInfo",
        "nvmlSystemGetConfComputeGpusReadyState",
        "nvmlDeviceGetConfComputeProtectedMemoryUsage",
        "nvmlSystemGetConfComputeKeyRotationThresholdInfo",
        "nvmlSystemGetConfComputeSettings",
        "nvmlDeviceGetGridLicensableFeatures_v4",
        "nvmlGetVgpuDriverCapabilities",
        "nvmlDeviceGetVgpuCapabilities",
        "nvmlDeviceGetVgpuHeterogeneousMode",
        "nvmlDeviceGetVgpuSchedulerCapabilities",
        "nvmlDeviceGetVgpuSchedulerState_v2",
        "nvmlDeviceGetVgpuSchedulerLog_v2",
        "nvmlGetVgpuVersion",
        "nvmlGpmQueryDeviceSupport",
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

    result = nvmlDeviceGetConfComputeGpuCertificate(device, &cc_certificate);
    if (result != NVML_ERROR_UNKNOWN &&
        check_optional(result, "nvmlDeviceGetConfComputeGpuCertificate"))
    {
        return 1;
    }
    result = nvmlDeviceGetConfComputeGpuAttestationReport(device, &cc_attestation);
    if (result != NVML_ERROR_UNKNOWN &&
        check_optional(result, "nvmlDeviceGetConfComputeGpuAttestationReport"))
    {
        return 1;
    }

    unsigned int gpm_stream_state = 0;
    result = nvmlGpmQueryIfStreamingEnabled(device, &gpm_stream_state);
    if (check_optional(result, "nvmlGpmQueryIfStreamingEnabled"))
    {
        return 1;
    }

    nvmlGpmSample_t gpm_sample1 = nullptr;
    nvmlGpmSample_t gpm_sample2 = nullptr;
    result = nvmlGpmSampleAlloc(&gpm_sample1);
    if (check_optional(result, "nvmlGpmSampleAlloc"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS)
    {
        result = nvmlGpmSampleAlloc(&gpm_sample2);
        if (check_optional(result, "nvmlGpmSampleAlloc"))
        {
            nvmlGpmSampleFree(gpm_sample1);
            return 1;
        }
        if (result != NVML_SUCCESS)
        {
            nvmlGpmSampleFree(gpm_sample1);
            gpm_sample1 = nullptr;
        }
    }
    if (gpm_sample1 != nullptr && gpm_sample2 != nullptr)
    {
        result = nvmlGpmSampleGet(device, gpm_sample1);
        if (check_optional(result, "nvmlGpmSampleGet"))
        {
            nvmlGpmSampleFree(gpm_sample1);
            nvmlGpmSampleFree(gpm_sample2);
            return 1;
        }
        usleep(150000);
        result = nvmlGpmSampleGet(device, gpm_sample2);
        if (check_optional(result, "nvmlGpmSampleGet"))
        {
            nvmlGpmSampleFree(gpm_sample1);
            nvmlGpmSampleFree(gpm_sample2);
            return 1;
        }
        result = nvmlGpmMigSampleGet(device, 0, gpm_sample1);
        if (check_optional_indexed(result, "nvmlGpmMigSampleGet"))
        {
            nvmlGpmSampleFree(gpm_sample1);
            nvmlGpmSampleFree(gpm_sample2);
            return 1;
        }

        nvmlGpmMetricsGet_t gpm_metrics = {};
        gpm_metrics.version = NVML_GPM_METRICS_GET_VERSION;
        gpm_metrics.numMetrics = 1;
        gpm_metrics.sample1 = gpm_sample1;
        gpm_metrics.sample2 = gpm_sample2;
        gpm_metrics.metrics[0].metricId = NVML_GPM_METRIC_SM_UTIL;
        result = nvmlGpmMetricsGet(&gpm_metrics);
        if (check_success(result, "nvmlGpmMetricsGet"))
        {
            nvmlGpmSampleFree(gpm_sample1);
            nvmlGpmSampleFree(gpm_sample2);
            return 1;
        }
        if (gpm_metrics.metrics[0].metricInfo.shortName == nullptr ||
            gpm_metrics.metrics[0].metricInfo.shortName[0] == '\0')
        {
            std::cout << "nvmlGpmMetricsGet returned empty metric metadata"
                      << std::endl;
            nvmlGpmSampleFree(gpm_sample1);
            nvmlGpmSampleFree(gpm_sample2);
            return 1;
        }
        if (check_success(nvmlGpmSampleFree(gpm_sample1), "nvmlGpmSampleFree") ||
            check_success(nvmlGpmSampleFree(gpm_sample2), "nvmlGpmSampleFree"))
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

    if (check_counted_list(nvmlDeviceGetEncoderSessions, device,
                           "nvmlDeviceGetEncoderSessions") ||
        check_counted_list(nvmlDeviceGetFBCSessions, device,
                           "nvmlDeviceGetFBCSessions"))
    {
        return 1;
    }

    if (check_counted_list(nvmlDeviceGetSupportedVgpus, device,
                           "nvmlDeviceGetSupportedVgpus") ||
        check_counted_list(nvmlDeviceGetCreatableVgpus, device,
                           "nvmlDeviceGetCreatableVgpus") ||
        check_counted_list(nvmlDeviceGetActiveVgpus, device,
                           "nvmlDeviceGetActiveVgpus"))
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

    unsigned int system_topology_count = topology_capacity;
    std::vector<nvmlDevice_t> system_topology_devices(topology_capacity);
    result = nvmlSystemGetTopologyGpuSet(0, &system_topology_count,
                                         system_topology_devices.data());
    if (result != NVML_ERROR_UNKNOWN &&
        result != NVML_ERROR_LIB_RM_VERSION_MISMATCH &&
        check_optional_list(result, "nvmlSystemGetTopologyGpuSet"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS && system_topology_count > topology_capacity)
    {
        std::cout << "System topology query returned " << system_topology_count
                  << " entries for capacity " << topology_capacity << std::endl;
        return 1;
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

    nvmlProcessesUtilizationInfo_t process_utilization_info = {};
    process_utilization_info.version = nvmlProcessesUtilizationInfo_v1;
    process_utilization_info.lastSeenTimeStamp = 0;
    result = nvmlDeviceGetProcessesUtilizationInfo(device,
                                                   &process_utilization_info);
    if (check_optional_list(result, "nvmlDeviceGetProcessesUtilizationInfo"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS || result == NVML_ERROR_INSUFFICIENT_SIZE)
    {
        unsigned int process_utilization_capacity =
            process_utilization_info.processSamplesCount > 0
                ? process_utilization_info.processSamplesCount
                : 64;
        std::vector<nvmlProcessUtilizationInfo_v1_t> process_utilization_entries(
            process_utilization_capacity);
        process_utilization_info.processSamplesCount =
            process_utilization_capacity;
        process_utilization_info.procUtilArray = process_utilization_entries.data();
        result = nvmlDeviceGetProcessesUtilizationInfo(
            device, &process_utilization_info);
        if (check_optional_list(result, "nvmlDeviceGetProcessesUtilizationInfo"))
        {
            return 1;
        }
        if (result == NVML_SUCCESS &&
            process_utilization_info.processSamplesCount >
                process_utilization_capacity)
        {
            std::cout << "Process utilization info query returned "
                      << process_utilization_info.processSamplesCount
                      << " entries for capacity "
                      << process_utilization_capacity << std::endl;
            return 1;
        }
    }

    for (unsigned int process_mode = 0; process_mode < 3; ++process_mode)
    {
        nvmlProcessDetailList_t detail_list = {};
        detail_list.version = nvmlProcessDetailList_v1;
        detail_list.mode = process_mode;
        result = nvmlDeviceGetRunningProcessDetailList(device, &detail_list);
        if (check_optional_list(result, "nvmlDeviceGetRunningProcessDetailList"))
        {
            return 1;
        }
        if (result == NVML_SUCCESS || result == NVML_ERROR_INSUFFICIENT_SIZE)
        {
            unsigned int detail_capacity =
                detail_list.numProcArrayEntries > 0
                    ? detail_list.numProcArrayEntries
                    : 64;
            std::vector<nvmlProcessDetail_v1_t> details(detail_capacity);
            detail_list.numProcArrayEntries = detail_capacity;
            detail_list.procArray = details.data();
            result = nvmlDeviceGetRunningProcessDetailList(device, &detail_list);
            if (check_optional_list(result,
                                    "nvmlDeviceGetRunningProcessDetailList"))
            {
                return 1;
            }
            if (result == NVML_SUCCESS &&
                detail_list.numProcArrayEntries > detail_capacity)
            {
                std::cout << "Process detail query returned "
                          << detail_list.numProcArrayEntries
                          << " entries for capacity " << detail_capacity
                          << std::endl;
                return 1;
            }
        }
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

    unsigned int pgpu_metadata_size = 0;
    result = nvmlDeviceGetPgpuMetadataString(device, nullptr, &pgpu_metadata_size);
    if (check_optional_list(result, "nvmlDeviceGetPgpuMetadataString"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS || result == NVML_ERROR_INSUFFICIENT_SIZE)
    {
        unsigned int pgpu_metadata_capacity =
            pgpu_metadata_size > 0 ? pgpu_metadata_size : 4096;
        std::vector<char> pgpu_metadata(pgpu_metadata_capacity);
        pgpu_metadata_size = pgpu_metadata_capacity;
        result = nvmlDeviceGetPgpuMetadataString(device, pgpu_metadata.data(),
                                                 &pgpu_metadata_size);
        if (check_optional_list(result, "nvmlDeviceGetPgpuMetadataString"))
        {
            return 1;
        }
        if (result == NVML_SUCCESS && pgpu_metadata_size > pgpu_metadata_capacity)
        {
            std::cout << "pGPU metadata string query returned "
                      << pgpu_metadata_size << " bytes for capacity "
                      << pgpu_metadata_capacity << std::endl;
            return 1;
        }
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

    nvmlGpuInstanceProfileInfo_t gpu_instance_profile = {};
    result = nvmlDeviceGetGpuInstanceProfileInfo(
        device, NVML_GPU_INSTANCE_PROFILE_1_SLICE, &gpu_instance_profile);
    if (check_optional(result, "nvmlDeviceGetGpuInstanceProfileInfo"))
    {
        return 1;
    }

    nvmlGpuInstanceProfileInfo_v2_t gpu_instance_profile_v = {};
    gpu_instance_profile_v.version = nvmlGpuInstanceProfileInfo_v2;
    result = nvmlDeviceGetGpuInstanceProfileInfoV(
        device, NVML_GPU_INSTANCE_PROFILE_1_SLICE, &gpu_instance_profile_v);
    if (check_optional(result, "nvmlDeviceGetGpuInstanceProfileInfoV"))
    {
        return 1;
    }

    nvmlGpuInstanceProfileInfo_v2_t gpu_instance_profile_by_id = {};
    gpu_instance_profile_by_id.version = nvmlGpuInstanceProfileInfo_v2;
    result = nvmlDeviceGetGpuInstanceProfileInfoByIdV(
        device, NVML_GPU_INSTANCE_PROFILE_1_SLICE, &gpu_instance_profile_by_id);
    if (check_optional(result, "nvmlDeviceGetGpuInstanceProfileInfoByIdV"))
    {
        return 1;
    }

    std::vector<nvmlGpuInstancePlacement_t> gpu_instance_placements(16);
    unsigned int gpu_instance_placement_count =
        static_cast<unsigned int>(gpu_instance_placements.size());
    result = nvmlDeviceGetGpuInstancePossiblePlacements_v2(
        device, NVML_GPU_INSTANCE_PROFILE_1_SLICE,
        gpu_instance_placements.data(), &gpu_instance_placement_count);
    if (check_optional_list(result,
                            "nvmlDeviceGetGpuInstancePossiblePlacements_v2"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS &&
        gpu_instance_placement_count > gpu_instance_placements.size())
    {
        std::cout << "GPU instance placement query returned "
                  << gpu_instance_placement_count << " entries for capacity "
                  << gpu_instance_placements.size() << std::endl;
        return 1;
    }

    unsigned int gpu_instance_remaining_capacity = 0;
    result = nvmlDeviceGetGpuInstanceRemainingCapacity(
        device, NVML_GPU_INSTANCE_PROFILE_1_SLICE,
        &gpu_instance_remaining_capacity);
    if (check_optional(result, "nvmlDeviceGetGpuInstanceRemainingCapacity"))
    {
        return 1;
    }

    std::vector<nvmlGpuInstance_t> gpu_instances(16);
    unsigned int gpu_instance_count =
        static_cast<unsigned int>(gpu_instances.size());
    result = nvmlDeviceGetGpuInstances(device, NVML_GPU_INSTANCE_PROFILE_1_SLICE,
                                       gpu_instances.data(),
                                       &gpu_instance_count);
    if (check_optional_list(result, "nvmlDeviceGetGpuInstances"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS && gpu_instance_count > gpu_instances.size())
    {
        std::cout << "GPU instance query returned " << gpu_instance_count
                  << " entries for capacity " << gpu_instances.size()
                  << std::endl;
        return 1;
    }

    nvmlGpuInstance_t gpu_instance = nullptr;
    result = nvmlDeviceGetGpuInstanceById(device, 0, &gpu_instance);
    if (check_optional_list(result, "nvmlDeviceGetGpuInstanceById"))
    {
        return 1;
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

    nvmlSystemEventSetCreateRequest_t system_event_create = {};
    system_event_create.version = nvmlSystemEventSetCreateRequest_v1;
    result = nvmlSystemEventSetCreate(&system_event_create);
    if (check_success(result, "nvmlSystemEventSetCreate"))
    {
        return 1;
    }

    nvmlSystemRegisterEventRequest_t system_event_register = {};
    system_event_register.version = nvmlSystemRegisterEventRequest_v1;
    system_event_register.eventTypes =
        nvmlSystemEventTypeGpuDriverUnbind | nvmlSystemEventTypeGpuDriverBind;
    system_event_register.set = system_event_create.set;
    result = nvmlSystemRegisterEvents(&system_event_register);
    if (check_success(result, "nvmlSystemRegisterEvents"))
    {
        return 1;
    }

    nvmlSystemEventData_v1_t system_event_data[4] = {};
    nvmlSystemEventSetWaitRequest_t system_event_wait = {};
    system_event_wait.version = nvmlSystemEventSetWaitRequest_v1;
    system_event_wait.timeoutms = 0;
    system_event_wait.set = system_event_create.set;
    system_event_wait.data = system_event_data;
    system_event_wait.dataSize =
        sizeof(system_event_data) / sizeof(system_event_data[0]);
    result = nvmlSystemEventSetWait(&system_event_wait);
    if (check_event_wait(result, "nvmlSystemEventSetWait"))
    {
        return 1;
    }
    if (result == NVML_SUCCESS &&
        system_event_wait.numEvent > system_event_wait.dataSize)
    {
        std::cout << "System event wait returned "
                  << system_event_wait.numEvent << " entries for capacity "
                  << system_event_wait.dataSize << std::endl;
        return 1;
    }

    nvmlSystemEventSetFreeRequest_t system_event_free = {};
    system_event_free.version = nvmlSystemEventSetFreeRequest_v1;
    system_event_free.set = system_event_create.set;
    result = nvmlSystemEventSetFree(&system_event_free);
    if (check_success(result, "nvmlSystemEventSetFree"))
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
