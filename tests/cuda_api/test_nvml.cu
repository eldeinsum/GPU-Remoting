#include <cuda_runtime.h>
#include <iostream>
#include <nvml.h>

int main()
{
    // const int iterations = 1;
    nvmlReturn_t result;
    result = nvmlInit_v2();
    if (NVML_SUCCESS != result)
    {
        std::cout << "Failed to initialize NVML: " << nvmlErrorString(result) << std::endl;
        return 1;
    }
    // result = nvmlInitWithFlags(NVML_INIT_FLAG_NO_ATTACH);
    // if (NVML_SUCCESS != result)
    // {
    //     std::cout << "Failed to initialize NVML with NO_ATTACH: " << nvmlErrorString(result) << std::endl;
    //     return 1;
    // }

    unsigned int device_count;
    result = nvmlDeviceGetCount_v2(&device_count);
    if (NVML_SUCCESS != result)
    {
        std::cout << "Failed to get device count: " << nvmlErrorString(result) << std::endl;
        return 1;
    }
    std::cout << "Found " << device_count << " devices" << std::endl;
    if (device_count == 0)
    {
        std::cout << "Expected at least one NVML device" << std::endl;
        return 1;
    }

    nvmlDevice_t device;
    result = nvmlDeviceGetHandleByIndex_v2(0, &device);
    if (NVML_SUCCESS != result)
    {
        std::cout << "Failed to get NVML device handle: " << nvmlErrorString(result) << std::endl;
        return 1;
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
