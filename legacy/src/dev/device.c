/* src/dev/device.c — GPU device layer for the qw inference engine.
 *
 * gfx1030-only (AMD Radeon Pro V620). Every HIP call goes through QW_HIP()
 * which converts hipError_t -> qw_err and logs file:line + error string.
 * Without -DQW_WITH_HIP the whole HIP surface degrades to
 * QW_ERR_UNSUPPORTED so the file compiles on machines without ROCm.
 */
#include "qw/device.h"
#include "qw/macros.h"
#include "qw/types.h"

#include <stdio.h>
#include <stdint.h>
#include <string.h>

#ifdef QW_WITH_HIP
#include <hip/hip_runtime_api.h>
#endif

/* --------------------------------------------------------- QW_HIP macro */
/* Wrap a HIP call: on non-success log file:line + hipGetErrorString and
 * return the mapped qw_err. Usage:
 *     QW_HIP(hipSetDevice(i));
 */
#ifdef QW_WITH_HIP
static inline qw_err qw_hip_map(hipError_t e) {
    switch (e) {
    case hipSuccess:
        return QW_OK;
    case hipErrorOutOfMemory:
        return QW_ERR_NOMEM;
    case hipErrorInvalidValue:
    case hipErrorInvalidDevice:
    case hipErrorInvalidDevicePointer:
    case hipErrorInvalidMemcpyDirection:
        return QW_ERR_RANGE;
    default:
        return QW_ERR_DEVICE;
    }
}

#define QW_HIP(expr)                                                        \
    do {                                                                    \
        hipError_t _e = (expr);                                             \
        if (_e != hipSuccess) {                                             \
            QW_LOGE("HIP %s failed at %s:%d: %s", #expr, __FILE__,          \
                    __LINE__, hipGetErrorString(_e));                       \
            return qw_hip_map(_e);                                          \
        }                                                                   \
    } while (0)
#else
#define QW_HIP(expr)                                                        \
    do {                                                                    \
        QW_LOGE("HIP call %s unavailable: built without -DQW_WITH_HIP "     \
                "(%s:%d)", #expr, __FILE__, __LINE__);                      \
        return QW_ERR_UNSUPPORTED;                                          \
    } while (0)
#endif

/* ------------------------------------------------------- device table */
static qw_device_info g_devs[QW_MAX_DEVICES];
static int            g_ndevs = 0;
static int            g_cur   = -1; /* current device (hipSetDevice) */
static bool           g_init  = false;

#ifdef QW_WITH_HIP

/* Derived theoretical memory bandwidth: bus_width bits / 8 * 2 (DDR) *
 * clock in Hz. GDDR6: 256-bit * 2 * 1800 MHz = 512 GB/s. */
static uint64_t derive_bw(int bus_width, int mem_clock_khz) {
    if (bus_width <= 0 || mem_clock_khz <= 0)
        return 0;
    return (uint64_t)(bus_width / 8) * 2u * (uint64_t)mem_clock_khz * 1000u;
}

static int attr_or_zero(int dev, hipDeviceAttribute_t attr) {
    int v = 0;
    if (hipDeviceGetAttribute(&v, attr, dev) != hipSuccess)
        v = 0;
    return v;
}

/* Fill one table entry from HIP queries. Returns QW_OK on success. */
static qw_err populate_device(int hip_dev, qw_device_info *info) {
    memset(info, 0, sizeof(*info));
    info->index = g_ndevs; /* stable slot in the visible table */

    QW_HIP(hipDeviceGetName(info->name, (int)sizeof(info->name), hip_dev));
    info->name[sizeof(info->name) - 1] = '\0';

    hipDeviceProp_t prop;
    QW_HIP(hipGetDeviceProperties(&prop, hip_dev));
    snprintf(info->gcn_arch_name, sizeof(info->gcn_arch_name), "%s",
             prop.gcnArchName);

    info->bus_width                = attr_or_zero(hip_dev,
                                   hipDeviceAttributeMemoryBusWidth);
    info->memory_clock_khz         = attr_or_zero(hip_dev,
                                   hipDeviceAttributeMemoryClockRate);
    info->memory_bandwidth         = derive_bw(info->bus_width,
                                               info->memory_clock_khz);
    info->max_threads_per_multi_processor =
        attr_or_zero(hip_dev, hipDeviceAttributeMaxThreadsPerMultiProcessor);
    info->warp_size =
        attr_or_zero(hip_dev, hipDeviceAttributeWarpSize); /* never hardcoded */
    info->multi_processor_count =
        attr_or_zero(hip_dev, hipDeviceAttributeMultiprocessorCount);

    /* hipDeviceGetPCIBusId takes a 13-byte buffer ("DDDD:BB:DD.D" + NUL);
     * our field is 16 bytes, which is fine. */
    {
        char bus[13] = {0};
        QW_HIP(hipDeviceGetPCIBusId(bus, (int)sizeof(bus), hip_dev));
        snprintf(info->pci_bus_id, sizeof(info->pci_bus_id), "%s", bus);
    }

    QW_HIP(hipMemGetInfo(NULL, &info->total_global_mem));

    /* Stream capture is a core feature of the HIP stream API on all
     * supported platforms; there is no device attribute to query it in
     * ROCm 10.0. Report true. */
    info->can_use_stream_capture = true;
    return QW_OK;
}

#endif /* QW_WITH_HIP */

/* -------------------------------------------------------------- init */
qw_err qw_device_init(void) {
    if (g_init)
        return QW_ERR_RANGE;

#ifdef QW_WITH_HIP
    int total = 0;
    QW_HIP(hipGetDeviceCount(&total));
    if (total <= 0) {
        QW_LOGW("no HIP devices found; device layer is empty");
        g_init = true;
        return QW_OK;
    }

    /* Pass 1: enumerate, keep only gfx1030. */
    for (int d = 0; d < total && g_ndevs < QW_MAX_DEVICES; d++) {
        hipDeviceProp_t prop;
        if (hipGetDeviceProperties(&prop, d) != hipSuccess) {
            QW_LOGW("hipGetDeviceProperties failed for device %d; skipping", d);
            continue;
        }
        if (strcmp(prop.gcnArchName, "gfx1030") != 0) {
            QW_LOGI("skipping device %d (%s): arch %s is not gfx1030; "
                    "this engine is gfx1030-only",
                    d, prop.name, prop.gcnArchName);
            continue;
        }
        if (populate_device(d, &g_devs[g_ndevs]) != QW_OK) {
            QW_LOGE("failed to fully populate device %d; skipping", d);
            continue;
        }
        QW_LOGI("device %d: %s [%s] %zu MiB VRAM, %d CUs, warp %d, %s, "
                "theoretical %.0f GB/s",
                g_ndevs, g_devs[g_ndevs].name, prop.gcnArchName,
                (size_t)g_devs[g_ndevs].total_global_mem / (1024 * 1024),
                g_devs[g_ndevs].multi_processor_count, g_devs[g_ndevs].warp_size,
                g_devs[g_ndevs].pci_bus_id,
                (double)g_devs[g_ndevs].memory_bandwidth / 1e9);
        g_ndevs++;
    }

    if (g_ndevs == 0) {
        QW_LOGW("no gfx1030 device found; this host is dev-only");
        g_init = true;
        return QW_OK;
    }

    /* Pass 2: enable peer access for every visible pair. hipSetDevice(a)
     * then hipDeviceEnablePeerAccess(b); tolerate already-enabled. */
    for (int a = 0; a < g_ndevs; a++) {
        for (int b = 0; b < g_ndevs; b++) {
            if (a == b)
                continue;
            int can = 0;
            hipError_t e = hipDeviceCanAccessPeer(&can, a, b);
            if (e != hipSuccess) {
                QW_LOGW("hipDeviceCanAccessPeer(%d,%d) failed: %s",
                        a, b, hipGetErrorString(e));
                continue;
            }
            if (!can)
                continue;
            QW_HIP(hipSetDevice(g_devs[a].index));
            hipError_t pe = hipDeviceEnablePeerAccess(g_devs[b].index, 0);
            if (pe == hipErrorPeerAccessAlreadyEnabled) {
                /* fine */
            } else if (pe != hipSuccess) {
                QW_LOGW("hipDeviceEnablePeerAccess(%d -> %d) failed: %s",
                        a, b, hipGetErrorString(pe));
                continue;
            }
            g_devs[a].p2p_supported = true;
            g_devs[b].p2p_supported = true;
            QW_LOGI("peer access enabled: device %d -> device %d", a, b);
        }
    }
    QW_HIP(hipSetDevice(g_devs[0].index)); /* leave a sane current device */
    g_cur = 0;
    g_init = true;
    return QW_OK;
#else
    QW_LOGW("device init without HIP: no devices available");
    g_init = true;
    return QW_OK;
#endif
}

/* --------------------------------------------------------- accessors */
qw_err qw_device_count(int *out_n) {
    if (!g_init)
        return QW_ERR_DEVICE;
    if (!out_n)
        return QW_ERR_NULL;
    *out_n = g_ndevs;
    return QW_OK;
}

const qw_device_info *qw_device_get(int i) {
    if (!g_init)
        return NULL;
    if (i < 0 || i >= g_ndevs)
        return NULL;
    return &g_devs[i];
}

qw_err qw_device_select(int i) {
    if (!g_init)
        return QW_ERR_DEVICE;
    if (i < 0 || i >= g_ndevs)
        return QW_ERR_RANGE;
    QW_HIP(hipSetDevice(g_devs[i].index));
    g_cur = i;
    return QW_OK;
}

qw_err qw_device_memory(int *free_out, size_t *total_out) {
    if (!g_init)
        return QW_ERR_DEVICE;
    if (!free_out && !total_out)
        return QW_ERR_NULL;
    {
        size_t free_sz = 0;
        QW_HIP(hipMemGetInfo(&free_sz, total_out));
        if (free_out)
            *free_out = (int)(free_sz > (size_t)INT32_MAX ? (size_t)INT32_MAX
                                                          : free_sz);
    }
    return QW_OK;
}

qw_err qw_device_synchronize_all(void) {
    if (!g_init)
        return QW_ERR_DEVICE;
#ifdef QW_WITH_HIP
    for (int i = 0; i < g_ndevs; i++) {
        QW_HIP(hipSetDevice(g_devs[i].index));
        QW_HIP(hipDeviceSynchronize());
    }
    if (g_cur >= 0)
        QW_HIP(hipSetDevice(g_devs[g_cur].index));
    return QW_OK;
#else
    QW_LOGE("synchronize_all unavailable without HIP");
    return QW_ERR_UNSUPPORTED;
#endif
}

/* -------------------------------------------------------- benchmark */
#ifdef QW_WITH_HIP

/* Time `iters` hipMemcpy(dst, src, bytes) on the current device.
 * Returns GB/s for the fastest timed run, or -1 on failure. */
static double bench_d2d(void *dst, const void *src, size_t bytes, int iters) {
    hipEvent_t ev0, ev1;
    if (hipEventCreate(&ev0) != hipSuccess ||
        hipEventCreate(&ev1) != hipSuccess) {
        return -1.0;
    }
    double best = -1.0;
    for (int i = 0; i < iters; i++) {
        if (hipEventRecord(ev0, 0) != hipSuccess)
            break;
        if (hipMemcpy(dst, src, bytes, hipMemcpyDeviceToDevice) != hipSuccess)
            break;
        if (hipEventRecord(ev1, 0) != hipSuccess)
            break;
        if (hipEventSynchronize(ev1) != hipSuccess)
            break;
        float ms = 0.0f;
        if (hipEventElapsedTime(&ms, ev0, ev1) != hipSuccess)
            break;
        if (ms <= 0.0f)
            continue;
        double gbps = ((double)bytes / (ms / 1000.0)) / 1e9;
        if (gbps > best)
            best = gbps;
    }
    hipEventDestroy(ev0);
    hipEventDestroy(ev1);
    return best;
}

/* Time H2D copy from pinned host memory into `dst`.
 * Returns GB/s for the fastest timed run, or -1 on failure. */
static double bench_h2d(void *dst, const void *src, size_t bytes, int iters) {
    hipEvent_t ev0, ev1;
    if (hipEventCreate(&ev0) != hipSuccess ||
        hipEventCreate(&ev1) != hipSuccess) {
        return -1.0;
    }
    double best = -1.0;
    for (int i = 0; i < iters; i++) {
        if (hipEventRecord(ev0, 0) != hipSuccess)
            break;
        if (hipMemcpy(dst, src, bytes, hipMemcpyHostToDevice) != hipSuccess)
            break;
        if (hipEventRecord(ev1, 0) != hipSuccess)
            break;
        if (hipEventSynchronize(ev1) != hipSuccess)
            break;
        float ms = 0.0f;
        if (hipEventElapsedTime(&ms, ev0, ev1) != hipSuccess)
            break;
        if (ms <= 0.0f)
            continue;
        double gbps = ((double)bytes / (ms / 1000.0)) / 1e9;
        if (gbps > best)
            best = gbps;
    }
    hipEventDestroy(ev0);
    hipEventDestroy(ev1);
    return best;
}

#endif /* QW_WITH_HIP */

qw_err qw_device_benchmark_bandwidth(int dev, size_t bytes, double *gbps_out,
                                     double *h2d_gbps_out) {
    if (!g_init)
        return QW_ERR_DEVICE;
    if (!gbps_out)
        return QW_ERR_NULL;
    if (dev < 0 || dev >= g_ndevs)
        return QW_ERR_RANGE;
    if (bytes < (size_t)(1 << 20)) {
        QW_LOGW("bandwidth benchmark: bytes=%zu < 1 MiB, result meaningless",
                bytes);
    }
    *gbps_out = 0.0;
    if (h2d_gbps_out)
        *h2d_gbps_out = 0.0;

#ifdef QW_WITH_HIP
    if (bytes == 0)
        return QW_ERR_RANGE;
    /* Cap the buffer so a pathological caller cannot OOM the 32 GB card. */
    size_t n = bytes;
    size_t cap = (size_t)4 * 1024 * 1024 * 1024; /* 4 GiB */
    if (n > cap)
        n = cap;

    QW_HIP(hipSetDevice(g_devs[dev].index));

    void *dbuf0 = NULL, *dbuf1 = NULL;
    QW_HIP(hipMalloc(&dbuf0, n));
    QW_HIP(hipMalloc(&dbuf1, n));

    const int iters = 10;
    double d2d = bench_d2d(dbuf1, dbuf0, n, iters);
    if (d2d > 0.0)
        *gbps_out = d2d;
    else
        QW_LOGW("D2D benchmark failed on device %d", dev);

    if (h2d_gbps_out) {
        void *pinned = NULL;
        if (hipHostMalloc(&pinned, n, hipHostAllocMapped) == hipSuccess) {
            double h2d = bench_h2d(dbuf0, pinned, n, iters);
            if (h2d > 0.0)
                *h2d_gbps_out = h2d;
            else
                QW_LOGW("H2D benchmark failed on device %d", dev);
            hipFreeHost(pinned);
        } else {
            QW_LOGW("pinned host allocation failed; H2D benchmark skipped");
        }
    }

    hipFree(dbuf0);
    hipFree(dbuf1);
    if (g_cur >= 0)
        hipSetDevice(g_devs[g_cur].index);
    return QW_OK;
#else
    QW_LOGE("benchmark_bandwidth unavailable without HIP (dev=%d)", dev);
    return QW_ERR_UNSUPPORTED;
#endif
}
