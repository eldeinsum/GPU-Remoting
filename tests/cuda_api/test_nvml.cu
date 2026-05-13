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
        result == NVML_ERROR_FUNCTION_NOT_FOUND || result == NVML_ERROR_NO_PERMISSION)
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
    for (size_t i = 0; i < sizeof(optional_results) / sizeof(optional_results[0]); ++i)
    {
        if (check_optional(optional_results[i], optional_labels[i]))
        {
            return 1;
        }
    }

    unsigned int min_limit = 0;
    unsigned int max_limit = 0;
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
