/* qw/device.h — GPU device enumeration, P2P setup, bandwidth benchmarking.
 *
 * Target hardware: AMD Radeon Pro V620 (gfx1030, RDNA2). This engine is
 * gfx1030-only: qw_device_init() skips (and logs) every device whose
 * gcnArchName is not "gfx1030".
 *
 * Build with -DQW_WITH_HIP to link against HIP; without it, every API
 * function returns QW_ERR_UNSUPPORTED so the code still compiles on
 * machines without ROCm installed (syntax-check / dev laptops).
 *
 * C17. No C++. Only libc + pthread.
 */
#ifndef QW_DEVICE_H
#define QW_DEVICE_H

#include <stddef.h>
#include <stdint.h>
#include "qw/types.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Upper bound on GPUs enumerated into the global table. */
#define QW_MAX_DEVICES 16

typedef struct qw_device_info {
    int      index;                    /* stable index into the qw table */
    char     name[64];                 /* from hipDeviceGetName */
    char     gcn_arch_name[192];       /* from hipGetDeviceProperties */
    int      bus_width;                /* bits, hipDeviceAttributeMemoryBusWidth */
    int      memory_clock_khz;         /* hipDeviceAttributeMemoryClockRate */
    uint64_t memory_bandwidth;         /* bytes/s, derived: bus_width/8 * 2 * khz * 1000 */
    size_t   total_global_mem;         /* bytes */
    int      max_threads_per_multi_processor; /* from hipDeviceGetAttribute */
    int      warp_size;                /* from hipDeviceGetAttribute; NEVER hardcoded */
    int      multi_processor_count;    /* == CU count on gfx1030 */
    char     pci_bus_id[16];           /* "XX:XX.X" from hipDeviceGetPCIBusId */
    bool     p2p_supported;            /* any peer reachable (hipDeviceCanAccessPeer) */
    bool     can_use_stream_capture;   /* hipDeviceGetAttributeCanUseStreamCapture */
} qw_device_info;

/* One-time: enumerate all HIP devices, keep only gfx1030, populate the
 * global table, and for every visible pair (a,b) call hipDeviceCanAccessPeer
 * and, if true, hipDeviceEnablePeerAccess (tolerating
 * hipErrorPeerAccessAlreadyEnabled). Safe to call exactly once; repeat calls
 * return QW_ERR_RANGE. */
qw_err qw_device_init(void);

/* Number of visible (gfx1030) devices. */
qw_err qw_device_count(int *out_n);

/* Pointer to the info record for table slot i. QW_ERR_RANGE if out of bounds. */
const qw_device_info *qw_device_get(int i);

/* hipSetDevice(i) and remember it as the current device. */
qw_err qw_device_select(int i);

/* Current device's hipMemGetInfo. free_out/total_out may be NULL. */
qw_err qw_device_memory(int *free_out, size_t *total_out);

/* hipDeviceSynchronize() on every visible device. */
qw_err qw_device_synchronize_all(void);

/* Measure real achievable bandwidth on device `dev` (table index).
 *
 * D2D: hipMemcpy(dst, src, bytes) over a device buffer, timed with
 *      hipEvent_t, repeated; reports max run-time in GB/s.
 * H2D: hipMemcpy from pinned (hipHostMalloc hipHostAllocMapped) memory if
 *      the allocation succeeds.
 * H2D result is stored in *h2d_gbps_out (may be NULL).
 *
 * QW_ERR_DEVICE if no gfx1030 device is initialized or dev is out of range.
 * QW_ERR_UNSUPPORTED when built without QW_WITH_HIP. */
qw_err qw_device_benchmark_bandwidth(int dev, size_t bytes, double *gbps_out,
                                     double *h2d_gbps_out);

#ifdef __cplusplus
}
#endif

#endif /* QW_DEVICE_H */
