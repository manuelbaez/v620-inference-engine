// HIP error checking and device allocation helpers for the engine.
#pragma once

#include <hip/hip_runtime.h>

#include <string>
#include <vector>

#include "core/common.hpp"

namespace qw {

inline void ck(hipError_t e, const char *what) {
    if (e != hipSuccess) fail(std::string("hip: ") + what + ": " + hipGetErrorString(e));
}
#define CK(x) ::qw::ck((x), #x)

template <typename T>
T *upload(const std::vector<T> &v) {
    T *d;
    CK(hipMalloc(&d, v.size() * sizeof(T)));
    CK(hipMemcpy(d, v.data(), v.size() * sizeof(T), hipMemcpyHostToDevice));
    return d;
}

// Debugging (QW_GUARD=1): every zeros() allocation gets a guard zone filled
// with a pattern after it; check_guards() reports allocations whose guard was
// overwritten (an out-of-bounds write), with the allocating file:line.
struct GuardedAlloc {
    void *ptr;
    size_t bytes;
    int device;
    const char *file;
    int line;
};
constexpr size_t GUARD_BYTES = 64 * 1024;
constexpr uint8_t GUARD_BYTE = 0xA5;
bool guards_enabled();
void register_guard(const GuardedAlloc &a);
// Returns the number of overwritten guards (and logs each).
int check_guards();

template <typename T>
T *zeros(size_t n, const char *file = __builtin_FILE(), int line = __builtin_LINE()) {
    T *d;
    const size_t bytes = n * sizeof(T);
    if (guards_enabled()) {
        CK(hipMalloc(&d, bytes + GUARD_BYTES));
        CK(hipMemset(reinterpret_cast<uint8_t *>(d) + bytes, GUARD_BYTE, GUARD_BYTES));
        int dev = 0;
        CK(hipGetDevice(&dev));
        register_guard({d, bytes, dev, file, line});
    } else {
        CK(hipMalloc(&d, bytes));
    }
    CK(hipMemset(d, 0, bytes));
    return d;
}

template <typename T>
T *pinned(size_t n) {
    T *h;
    CK(hipHostMalloc(reinterpret_cast<void **>(&h), n * sizeof(T), 0));
    return h;
}

}  // namespace qw
