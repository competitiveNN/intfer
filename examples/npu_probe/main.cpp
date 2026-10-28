// npu_probe: host-side probe of an Intel NPU via OpenVINO C API.
//
// Builds and runs whether or not the box actually has an NPU available to
// the container: it reports the device list OpenVINO sees (CPU/GPU/NPU"),
// whether "NPU" is among them, and the NPU node's visibility in
// /dev/accel/accel0. The NPU plugin lib is dlopen()'d at runtime so the
// binary never hard-links a host version and stays portable. With
// INTFER_ENABLE_NPU unset the body is compiled out and the program reports
// NPU absent.
//
// Hard exit codes: 0 = ran, whether or not an NPU was present (the probe is
// an observation tool, not a gate that demands hardware).
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <dlfcn.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
const char* kLibCandidates[] = {
    "/run/host/usr/lib64/libopenvino_c.so",
    "/run/host/usr/lib64/libopenvino_c.so.2510",
    "/run/host/usr/lib64/libopenvino_c.so.2025.1.0",
    "libopenvino_c.so",
    nullptr
};

void* open_lib() {
    void* t = dlopen("/run/host/usr/lib64/libtbb.so.12", RTLD_NOW | RTLD_GLOBAL);
    (void)t;
    void* b = dlopen("/run/host/usr/lib64/libopenvino.so.2025.1.0", RTLD_NOW | RTLD_GLOBAL);
    fprintf(stderr, "tbb=%p base=%p err=%s\n", t, b, dlerror());
    dlopen("/run/host/usr/lib64/openvino-2025.1.0/libopenvino_intel_npu_plugin.so",
           RTLD_NOW | RTLD_GLOBAL);
    void* p = dlopen("/run/host/usr/lib64/libopenvino_c.so.2025.1.0", RTLD_NOW | RTLD_LOCAL);
    (void)p;
    if (p) return p;
    fprintf(stderr, "fallback path err: %s\n", dlerror());
    for (int i = 0; kLibCandidates[i]; ++i) {
        void* h = dlopen(kLibCandidates[i], RTLD_NOW | RTLD_LOCAL);

        if (!h) {
            std::fprintf(stderr, "dlopen(%s) failed: %s\n", kLibCandidates[i], dlerror());
            h = dlopen(kLibCandidates[i], RTLD_LAZY | RTLD_LOCAL);
            if (h) std::fprintf(stderr, "  opened with RTLD_LAZY\n");
        }
        if (h) {
            std::fprintf(stderr, "NPU probe: loaded %s\n", kLibCandidates[i]);
            return h;
        }
    }
    std::fprintf(stderr, "NPU probe: could not dlopen libopenvino_c\n");
    return nullptr;
}

const char* accel_node() {
    return "/dev/accel/accel0";
}
bool accel_exists() {
    struct stat st{};
    if (stat("/dev/accel/accel0", &st) == 0) return true;
    if (stat("/run/host/dev/accel/accel0", &st) == 0) return true;
    return false;
}
} // namespace

int main() {
#if !defined(INTFER_ENABLE_NPU)
    std::fprintf(stderr, "[npu-probe] INTFER_ENABLE_NPU=0 (compiled out): NPU absent\n");
    std::fprintf(stdout, "NPU: absent (disabled at compile time)\\n");
    return 0;
#else
    setenv("LD_LIBRARY_PATH",
           "/run/host/usr/lib64:/run/host/usr/lib64/openvino-2025.1.0:"
           "/run/host/usr/local/lib", 0);

    char line[256]{};
    std::fprintf(stdout, "NPU probe report\\n");

    void* h = open_lib();
    if (!h) {
        std::fprintf(stderr, "[npu-probe] libopenvino_c not available; NPU absent\\n");
        std::fprintf(stdout, "NPU: absent (no libopenvino_c)\\n");
        return 0;
    }

    // ov_core_create(ov_core_t** core)
    int (*ov_core_create)(void**) = (int (*)(void**))dlsym(h, "ov_core_create");
    if (!ov_core_create) {
        std::fprintf(stderr, "[npu-probe] ov_core_create symbol missing\\n");
        std::fprintf(stdout, "NPU: absent (no ov_core_create)\\n");
        dlclose(h);
        return 0;
    }
    void* core = nullptr;
    int rc = ov_core_create(&core);
    std::fprintf(stderr, "[npu-probe] ov_core_create rc=%d core=%p\\n", rc, core);
    if (rc != 0 || !core) {
        std::fprintf(stdout, "NPU: absent (core create failed rc=%d)\\n", rc);
        dlclose(h);
        return 0;
    }

    // ov_core_get_available_devices(core, ov_available_devices_t* devices)
    int (*ov_core_get_available_devices)(void*, void*) =
        (int (*)(void*, void*))dlsym(h, "ov_core_get_available_devices");
    bool npu_present = false;
    if (ov_core_get_available_devices) {
        // ov_available_devices_t: { char** devices; size_t size; }
        struct { char** devices; size_t size; } devs{nullptr, 0};
        rc = ov_core_get_available_devices(core, &devs);
        std::fprintf(stderr, "[npu-probe] get_available_devices rc=%d size=%zu\\n", rc, devs.size);
        std::vector<std::string> names;
        if (rc == 0) {
            for (size_t i = 0; i < devs.size; ++i) {
                if (devs.devices[i]) names.emplace_back(devs.devices[i]);
            }
        }
        std::fprintf(stdout, "available_devices: ");
        for (size_t i = 0; i < names.size(); ++i) {
            if (i) std::fputs(" ", stdout);
            std::fputs(names[i].c_str(), stdout);
            if (names[i] == "NPU") npu_present = true;
        }
        std::fputs("\\n", stdout);
    } else {
        std::fprintf(stderr, "[npu-probe] ov_core_get_available_devices symbol missing\\n");
    }

    // accelerator node check
    bool accel = accel_exists();
    std::fprintf(stdout, "accel_node=%s exists=%s\\n", accel_node(), accel ? "yes" : "no");
    std::fprintf(stdout, "NPU: %s\\n", npu_present ? "present" : "absent");

    // cleanup (best effort)
    auto free_devs = (void (*)(void*))dlsym(h, "ov_available_devices_free");
    // ov_core_create allocated core; attempt ov_core_free if present
    auto core_free = (void (*)(void*))dlsym(h, "ov_core_free");
    if (core_free) core_free(core);

    dlclose(h);
    return 0;
#endif
}