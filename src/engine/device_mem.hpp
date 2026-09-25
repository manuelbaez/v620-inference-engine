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

template <typename T>
T *zeros(size_t n) {
    T *d;
    CK(hipMalloc(&d, n * sizeof(T)));
    CK(hipMemset(d, 0, n * sizeof(T)));
    return d;
}

template <typename T>
T *pinned(size_t n) {
    T *h;
    CK(hipHostMalloc(reinterpret_cast<void **>(&h), n * sizeof(T), 0));
    return h;
}

}  // namespace qw
