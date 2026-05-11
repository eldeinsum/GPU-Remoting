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
    if (NVML_SUCCESS != result)
    {
        std::cout << "Failed to initialize NVML through nvmlInit_v2: "
                  << nvmlErrorString(result) << std::endl;
        return 1;
    }
    // result = nvmlInitWithFlags(NVML_INIT_FLAG_NO_ATTACH);
    // if (NVML_SUCCESS != result)
    // {
    //     std::cout << "Failed to initialize NVML with NO_ATTACH: " << nvmlErrorString(result) << std::endl;
    //     return 1;
    // }

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

    nvmlMemory_t memory;
    result = nvmlDeviceGetMemoryInfo(device, &memory);
    if (NVML_SUCCESS != result)
    {
        std::cout << "Failed to get NVML memory info: " << nvmlErrorString(result) << std::endl;
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

    return 0;
}
