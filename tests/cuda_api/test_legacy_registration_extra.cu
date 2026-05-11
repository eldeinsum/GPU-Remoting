#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <cstdio>

struct textureReference;
struct surfaceReference;

using register_texture_fn = void (*)(void **, const textureReference *,
                                     const void **, const char *, int, int,
                                     int);
using register_surface_fn = void (*)(void **, const surfaceReference *,
                                     const void **, const char *, int, int);

static bool is_remoted()
{
    const char *preload = std::getenv("LD_PRELOAD");
    return preload != nullptr && std::strstr(preload, "libclient.so") != nullptr;
}

int main()
{
    auto register_texture = reinterpret_cast<register_texture_fn>(
        dlsym(RTLD_DEFAULT, "__cudaRegisterTexture"));
    auto register_surface = reinterpret_cast<register_surface_fn>(
        dlsym(RTLD_DEFAULT, "__cudaRegisterSurface"));

    if (register_texture == nullptr || register_surface == nullptr) {
        if (!is_remoted()) {
            return 0;
        }
        std::fprintf(stderr, "legacy registration hooks are missing\n");
        return 1;
    }

    register_texture(nullptr, nullptr, nullptr, nullptr, 0, 0, 0);
    register_surface(nullptr, nullptr, nullptr, nullptr, 0, 0);
    return 0;
}
