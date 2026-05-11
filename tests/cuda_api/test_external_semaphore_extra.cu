#include <cuda.h>
#include <cuda_runtime.h>
#include <vulkan/vulkan.h>
#include <unistd.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <vector>

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

#define CHECK_CUDA(call)                                                       \
    do {                                                                       \
        cudaError_t result = (call);                                           \
        if (result != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s failed: %s (%d)\n", #call,               \
                         cudaGetErrorName(result), static_cast<int>(result));  \
            return 1;                                                          \
        }                                                                      \
    } while (0)

#define CHECK_VK(call)                                                         \
    do {                                                                       \
        VkResult result = (call);                                              \
        if (result != VK_SUCCESS) {                                            \
            std::fprintf(stderr, "%s failed: %d\n", #call,                    \
                         static_cast<int>(result));                            \
            return false;                                                      \
        }                                                                      \
    } while (0)

struct VulkanContext {
    VkInstance instance = VK_NULL_HANDLE;
    VkPhysicalDevice physical_device = VK_NULL_HANDLE;
    VkDevice device = VK_NULL_HANDLE;
    uint32_t queue_family = 0;
    PFN_vkGetSemaphoreFdKHR get_semaphore_fd = nullptr;
};

static bool has_extension(const std::vector<VkExtensionProperties> &extensions,
                          const char *name)
{
    return std::any_of(extensions.begin(), extensions.end(),
                       [name](const VkExtensionProperties &extension) {
                           return std::strcmp(extension.extensionName, name) == 0;
                       });
}

static bool init_vulkan(VulkanContext *ctx)
{
    VkApplicationInfo app = {};
    app.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO;
    app.pApplicationName = "gpu-remoting-external-semaphore-test";
    app.apiVersion = VK_API_VERSION_1_2;

    VkInstanceCreateInfo instance_info = {};
    instance_info.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
    instance_info.pApplicationInfo = &app;
    CHECK_VK(vkCreateInstance(&instance_info, nullptr, &ctx->instance));

    uint32_t physical_device_count = 0;
    CHECK_VK(vkEnumeratePhysicalDevices(ctx->instance, &physical_device_count,
                                        nullptr));
    if (physical_device_count == 0) {
        return false;
    }
    std::vector<VkPhysicalDevice> physical_devices(physical_device_count);
    CHECK_VK(vkEnumeratePhysicalDevices(ctx->instance, &physical_device_count,
                                        physical_devices.data()));

    for (VkPhysicalDevice physical_device : physical_devices) {
        VkPhysicalDeviceProperties properties = {};
        vkGetPhysicalDeviceProperties(physical_device, &properties);
        if (properties.vendorID == 0x10de) {
            ctx->physical_device = physical_device;
            break;
        }
    }
    if (ctx->physical_device == VK_NULL_HANDLE) {
        ctx->physical_device = physical_devices[0];
    }

    uint32_t extension_count = 0;
    CHECK_VK(vkEnumerateDeviceExtensionProperties(
        ctx->physical_device, nullptr, &extension_count, nullptr));
    std::vector<VkExtensionProperties> extensions(extension_count);
    CHECK_VK(vkEnumerateDeviceExtensionProperties(
        ctx->physical_device, nullptr, &extension_count, extensions.data()));
    if (!has_extension(extensions, VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME) ||
        !has_extension(extensions, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME) ||
        !has_extension(extensions, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME)) {
        return false;
    }

    uint32_t queue_family_count = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(ctx->physical_device,
                                             &queue_family_count, nullptr);
    std::vector<VkQueueFamilyProperties> queue_families(queue_family_count);
    vkGetPhysicalDeviceQueueFamilyProperties(
        ctx->physical_device, &queue_family_count, queue_families.data());
    bool found_queue = false;
    for (uint32_t i = 0; i < queue_family_count; ++i) {
        if (queue_families[i].queueCount > 0 &&
            (queue_families[i].queueFlags &
             (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT))) {
            ctx->queue_family = i;
            found_queue = true;
            break;
        }
    }
    if (!found_queue) {
        return false;
    }

    float priority = 1.0f;
    VkDeviceQueueCreateInfo queue_info = {};
    queue_info.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    queue_info.queueFamilyIndex = ctx->queue_family;
    queue_info.queueCount = 1;
    queue_info.pQueuePriorities = &priority;

    const char *device_extensions[] = {
        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME,
        VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
        VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
    };

    VkPhysicalDeviceTimelineSemaphoreFeatures timeline_features = {};
    timeline_features.sType =
        VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_TIMELINE_SEMAPHORE_FEATURES;
    timeline_features.timelineSemaphore = VK_TRUE;

    VkDeviceCreateInfo device_info = {};
    device_info.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
    device_info.pNext = &timeline_features;
    device_info.queueCreateInfoCount = 1;
    device_info.pQueueCreateInfos = &queue_info;
    device_info.enabledExtensionCount = 3;
    device_info.ppEnabledExtensionNames = device_extensions;
    CHECK_VK(vkCreateDevice(ctx->physical_device, &device_info, nullptr,
                            &ctx->device));

    ctx->get_semaphore_fd = reinterpret_cast<PFN_vkGetSemaphoreFdKHR>(
        vkGetDeviceProcAddr(ctx->device, "vkGetSemaphoreFdKHR"));
    return ctx->get_semaphore_fd != nullptr;
}

static int export_vulkan_semaphore_fd(VulkanContext *ctx, bool timeline)
{
    VkSemaphoreTypeCreateInfo type_info = {};
    type_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO;
    type_info.semaphoreType =
        timeline ? VK_SEMAPHORE_TYPE_TIMELINE : VK_SEMAPHORE_TYPE_BINARY;
    type_info.initialValue = 0;

    VkExportSemaphoreCreateInfo export_info = {};
    export_info.sType = VK_STRUCTURE_TYPE_EXPORT_SEMAPHORE_CREATE_INFO;
    export_info.pNext = timeline ? &type_info : nullptr;
    export_info.handleTypes = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    VkSemaphoreCreateInfo semaphore_info = {};
    semaphore_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO;
    semaphore_info.pNext = &export_info;

    VkSemaphore semaphore = VK_NULL_HANDLE;
    if (vkCreateSemaphore(ctx->device, &semaphore_info, nullptr, &semaphore) !=
        VK_SUCCESS) {
        return -1;
    }

    VkSemaphoreGetFdInfoKHR fd_info = {};
    fd_info.sType = VK_STRUCTURE_TYPE_SEMAPHORE_GET_FD_INFO_KHR;
    fd_info.semaphore = semaphore;
    fd_info.handleType = VK_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD_BIT;

    int fd = -1;
    VkResult result = ctx->get_semaphore_fd(ctx->device, &fd_info, &fd);
    vkDestroySemaphore(ctx->device, semaphore, nullptr);
    return result == VK_SUCCESS ? fd : -1;
}

static int validate_driver_external_semaphore(int fd,
                                              CUexternalSemaphoreHandleType type)
{
    CUDA_EXTERNAL_SEMAPHORE_HANDLE_DESC desc = {};
    desc.type = type;
    desc.handle.fd = fd;

    CUexternalSemaphore semaphore = nullptr;
    CHECK_DRV(cuImportExternalSemaphore(&semaphore, &desc));

    CUDA_EXTERNAL_SEMAPHORE_SIGNAL_PARAMS signal_params = {};
    CUDA_EXTERNAL_SEMAPHORE_WAIT_PARAMS wait_params = {};
    if (type == CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD) {
        signal_params.params.fence.value = 1;
        wait_params.params.fence.value = 1;
    }

    CHECK_DRV(cuSignalExternalSemaphoresAsync(&semaphore, &signal_params, 1,
                                              nullptr));
    CHECK_DRV(cuWaitExternalSemaphoresAsync(&semaphore, &wait_params, 1,
                                            nullptr));
    CHECK_DRV(cuStreamSynchronize(nullptr));
    CHECK_DRV(cuDestroyExternalSemaphore(semaphore));
    return 0;
}

static int validate_runtime_external_semaphore(
    int fd, cudaExternalSemaphoreHandleType type)
{
    cudaExternalSemaphoreHandleDesc desc = {};
    desc.type = type;
    desc.handle.fd = fd;

    cudaExternalSemaphore_t semaphore = nullptr;
    CHECK_CUDA(cudaImportExternalSemaphore(&semaphore, &desc));

    cudaExternalSemaphoreSignalParams signal_params = {};
    cudaExternalSemaphoreWaitParams wait_params = {};
    if (type == cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd) {
        signal_params.params.fence.value = 1;
        wait_params.params.fence.value = 1;
    }

    CHECK_CUDA(cudaSignalExternalSemaphoresAsync(&semaphore, &signal_params, 1,
                                                 nullptr));
    CHECK_CUDA(cudaWaitExternalSemaphoresAsync(&semaphore, &wait_params, 1,
                                               nullptr));
    CHECK_CUDA(cudaStreamSynchronize(nullptr));
    CHECK_CUDA(cudaDestroyExternalSemaphore(semaphore));
    return 0;
}

int main()
{
    CHECK_DRV(cuInit(0));
    CUdevice device = 0;
    CHECK_DRV(cuDeviceGet(&device, 0));
    CUcontext context = nullptr;
    CHECK_DRV(cuDevicePrimaryCtxRetain(&context, device));
    CHECK_DRV(cuCtxSetCurrent(context));
    CHECK_CUDA(cudaSetDevice(0));

    VulkanContext vulkan = {};
    if (!init_vulkan(&vulkan)) {
        std::puts("external semaphore API test skipped");
        return 0;
    }

    int fd = export_vulkan_semaphore_fd(&vulkan, false);
    if (fd < 0 || validate_driver_external_semaphore(
                      fd, CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_OPAQUE_FD) != 0) {
        return 1;
    }

    fd = export_vulkan_semaphore_fd(&vulkan, true);
    if (fd < 0 || validate_driver_external_semaphore(
                      fd,
                      CU_EXTERNAL_SEMAPHORE_HANDLE_TYPE_TIMELINE_SEMAPHORE_FD) !=
                      0) {
        return 1;
    }

    fd = export_vulkan_semaphore_fd(&vulkan, false);
    if (fd < 0 || validate_runtime_external_semaphore(
                      fd, cudaExternalSemaphoreHandleTypeOpaqueFd) != 0) {
        return 1;
    }

    fd = export_vulkan_semaphore_fd(&vulkan, true);
    if (fd < 0 || validate_runtime_external_semaphore(
                      fd, cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd) !=
                      0) {
        return 1;
    }

    if (vulkan.device != VK_NULL_HANDLE) {
        vkDestroyDevice(vulkan.device, nullptr);
    }
    if (vulkan.instance != VK_NULL_HANDLE) {
        vkDestroyInstance(vulkan.instance, nullptr);
    }

    std::puts("external semaphore API test passed");
    return 0;
}
