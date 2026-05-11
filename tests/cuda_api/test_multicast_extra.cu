#include <cuda.h>

#include <cstdio>

#define CHECK_DRV(call)                                                        \
    do {                                                                       \
        CUresult result = (call);                                              \
        if (result != CUDA_SUCCESS) {                                          \
            const char *name = nullptr;                                        \
            cuGetErrorName(result, &name);                                     \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,               \
                         name == nullptr ? "unknown" : name,                  \
                         static_cast<int>(result));                            \
            return 1;                                                          \
        }                                                                      \
    } while (0)

static int expect_result(CUresult actual, CUresult expected, const char *call)
{
    if (actual == expected) {
        return 0;
    }

    const char *actual_name = nullptr;
    const char *expected_name = nullptr;
    cuGetErrorName(actual, &actual_name);
    cuGetErrorName(expected, &expected_name);
    std::fprintf(stderr, "%s returned %s (%d), expected %s (%d)\n", call,
                 actual_name == nullptr ? "unknown" : actual_name,
                 static_cast<int>(actual),
                 expected_name == nullptr ? "unknown" : expected_name,
                 static_cast<int>(expected));
    return 1;
}

static int run_unsupported_checks(CUdevice device,
                                  const CUmulticastObjectProp &prop)
{
    size_t granularity = 0;
    if (expect_result(
            cuMulticastGetGranularity(
                &granularity, &prop, CU_MULTICAST_GRANULARITY_MINIMUM),
            CUDA_ERROR_NOT_SUPPORTED, "cuMulticastGetGranularity(minimum)")) {
        return 1;
    }
    if (granularity != 0) {
        std::fprintf(stderr, "unsupported multicast changed granularity\n");
        return 1;
    }

    if (expect_result(
            cuMulticastGetGranularity(
                &granularity, &prop, CU_MULTICAST_GRANULARITY_RECOMMENDED),
            CUDA_ERROR_NOT_SUPPORTED,
            "cuMulticastGetGranularity(recommended)")) {
        return 1;
    }

    CUmemGenericAllocationHandle mc_handle = 0;
    if (expect_result(cuMulticastCreate(&mc_handle, &prop),
                      CUDA_ERROR_NOT_SUPPORTED, "cuMulticastCreate")) {
        return 1;
    }
    if (mc_handle != 0) {
        std::fprintf(stderr, "unsupported multicast created handle\n");
        return 1;
    }

    if (expect_result(cuMulticastAddDevice(mc_handle, device),
                      CUDA_ERROR_NOT_SUPPORTED, "cuMulticastAddDevice") ||
        expect_result(cuMulticastBindMem(mc_handle, 0, 0, 0, prop.size, 0),
                      CUDA_ERROR_NOT_SUPPORTED, "cuMulticastBindMem") ||
        expect_result(cuMulticastBindMem_v2(mc_handle, device, 0, 0, 0,
                                            prop.size, 0),
                      CUDA_ERROR_NOT_SUPPORTED, "cuMulticastBindMem_v2") ||
        expect_result(cuMulticastBindAddr(mc_handle, 0, 0, prop.size, 0),
                      CUDA_ERROR_NOT_SUPPORTED, "cuMulticastBindAddr") ||
        expect_result(cuMulticastBindAddr_v2(mc_handle, device, 0, 0,
                                             prop.size, 0),
                      CUDA_ERROR_NOT_SUPPORTED, "cuMulticastBindAddr_v2") ||
        expect_result(cuMulticastUnbind(mc_handle, device, 0, prop.size),
                      CUDA_ERROR_NOT_SUPPORTED, "cuMulticastUnbind")) {
        return 1;
    }

    return 0;
}

int main()
{
    CHECK_DRV(cuInit(0));

    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));

    CUmulticastObjectProp prop = {};
    prop.numDevices = 1;
    prop.size = 65536;
    prop.handleTypes = 0;
    prop.flags = 0;

    int multicast_supported = 0;
    CHECK_DRV(cuDeviceGetAttribute(
        &multicast_supported, CU_DEVICE_ATTRIBUTE_MULTICAST_SUPPORTED,
        device));
    if (!multicast_supported) {
        int result = run_unsupported_checks(device, prop);
        if (result == 0) {
            std::puts("multicast API unsupported on this device");
        }
        return result;
    }

    size_t minimum_granularity = 0;
    CHECK_DRV(cuMulticastGetGranularity(
        &minimum_granularity, &prop, CU_MULTICAST_GRANULARITY_MINIMUM));
    if (minimum_granularity == 0) {
        std::fprintf(stderr, "invalid multicast granularity\n");
        return 1;
    }
    prop.size = minimum_granularity;

    CUmemGenericAllocationHandle mc_handle = 0;
    CHECK_DRV(cuMulticastCreate(&mc_handle, &prop));
    CHECK_DRV(cuMulticastAddDevice(mc_handle, device));

    CUmemAllocationProp allocation_prop = {};
    allocation_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    allocation_prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_NONE;
    allocation_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    allocation_prop.location.id = device;

    CUmemGenericAllocationHandle mem_handle = 0;
    CHECK_DRV(cuMemCreate(&mem_handle, prop.size, &allocation_prop, 0));
    CHECK_DRV(cuMulticastBindMem(mc_handle, 0, mem_handle, 0, prop.size, 0));
    CHECK_DRV(cuMulticastUnbind(mc_handle, device, 0, prop.size));
    CHECK_DRV(
        cuMulticastBindMem_v2(mc_handle, device, 0, mem_handle, 0, prop.size, 0));
    CHECK_DRV(cuMulticastUnbind(mc_handle, device, 0, prop.size));

    CUdeviceptr mapped = 0;
    CHECK_DRV(cuMemAddressReserve(&mapped, prop.size, 0, 0, 0));
    CHECK_DRV(cuMemMap(mapped, prop.size, 0, mem_handle, 0));
    CHECK_DRV(cuMulticastBindAddr(mc_handle, 0, mapped, prop.size, 0));
    CHECK_DRV(cuMulticastUnbind(mc_handle, device, 0, prop.size));
    CHECK_DRV(cuMulticastBindAddr_v2(mc_handle, device, 0, mapped, prop.size, 0));
    CHECK_DRV(cuMulticastUnbind(mc_handle, device, 0, prop.size));

    CHECK_DRV(cuMemUnmap(mapped, prop.size));
    CHECK_DRV(cuMemAddressFree(mapped, prop.size));
    CHECK_DRV(cuMemRelease(mem_handle));
    CHECK_DRV(cuMemRelease(mc_handle));

    std::puts("multicast API test passed");
    return 0;
}
