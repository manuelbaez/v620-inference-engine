/* tools/probe.c — enumerate gfx1030 GPUs, print a device table + P2P
 * status, and benchmark real bandwidth (D2D, pinned H2D) against the
 * theoretical 512 GB/s VRAM and ~28 GB/s PCIe 4.0 x16 expectations.
 *
 * Exit code is 0 even when no gfx1030 GPU is present (laptop dev box).
 *
 * Build: make bin/probe   (or: cc -std=c17 -Iinclude tools/probe.c
 *          src/dev/device.c src/dev/alloc.c -o bin/probe -pthread
 *          [-DQW_WITH_HIP -lhip ...])
 */
#include <inttypes.h>
#include <stdio.h>
#include <string.h>

#include "qw/alloc.h"
#include "qw/device.h"
#include "qw/types.h"

#define QW_PROBE_BYTES ((size_t)256 * 1024 * 1024) /* 256 MiB test buffer */
#define QW_PROBE_ITERS 8

static void print_header(void) {
    printf("%-4s %-22s %-10s %10s %10s %4s %5s %9s %10s\n",
           "idx", "name", "arch", "VRAM GiB", "free GiB", "CU", "warp",
           "busW(b)", "BW GiB/s");
}

static void print_device(const qw_device_info *d, size_t free_bytes) {
    printf("%-4d %-22s %-10s %10.1f %10.1f %4d %5d %9d %10.1f\n",
           d->index, d->name, d->gcn_arch_name,
           (double)d->total_global_mem / (1024.0 * 1024.0 * 1024.0),
           (double)free_bytes / (1024.0 * 1024.0 * 1024.0),
           d->multi_processor_count, d->warp_size, d->bus_width,
           (double)d->memory_bandwidth / 1e9);
}

/* Per-device P2P state: p2p_supported is set during qw_device_init after
 * hipDeviceCanAccessPeer / hipDeviceEnablePeerAccess succeed. */
static void print_p2p_status(int n) {
    for (int i = 0; i < n; i++) {
        const qw_device_info *d = qw_device_get(i);
        if (!d)
            continue;
        printf("device %d (%s): p2p_supported=%s, can_use_stream_capture=%s, "
               "pcibus=%s\n",
               d->index, d->name, d->p2p_supported ? "yes" : "no",
               d->can_use_stream_capture ? "yes" : "no", d->pci_bus_id);
    }
}

int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    qw_err e = qw_device_init();
    if (e != QW_OK) {
        fprintf(stderr, "qw_device_init failed: %s\n", qw_errstr(e));
        return 0; /* still exit 0: dev-only host */
    }

    int n = 0;
    if (qw_device_count(&n) != QW_OK || n <= 0) {
        printf("no gfx1030 GPU present; this host is dev-only\n");
        return 0;
    }

    printf("== qw device probe: %d gfx1030 device(s) ==\n\n", n);
    print_header();
    for (int i = 0; i < n; i++) {
        const qw_device_info *d = qw_device_get(i);
        if (!d)
            continue;
        size_t free_bytes = 0;
        qw_device_memory(NULL, &free_bytes);
        print_device(d, free_bytes);
    }
    print_p2p_status(n);

    printf("\n== bandwidth benchmarks (%.0f MiB buffer, %d iters) ==\n",
           (double)QW_PROBE_BYTES / (1024.0 * 1024.0), QW_PROBE_ITERS);
    printf("theoretical: VRAM ~512 GB/s (256-bit x 2 x 1800 MHz), "
           "PCIe 4.0 x16 ~28 GB/s effective\n\n");

    for (int i = 0; i < n; i++) {
        const qw_device_info *d = qw_device_get(i);
        if (!d)
            continue;
        double d2d = -1.0, h2d = -1.0;
        e = qw_device_benchmark_bandwidth(i, QW_PROBE_BYTES, &d2d, &h2d);
        if (e != QW_OK) {
            printf("dev %d: benchmark failed: %s\n", i, qw_errstr(e));
            continue;
        }
        printf("dev %d (%s): D2D  %8.1f GB/s   (VRAM 512 GB/s spec)\n",
               i, d->name, d2d);
        printf("dev %d (%s): H2D  %8.1f GB/s   (PCIe ~28 GB/s spec)\n",
               i, d->name, h2d);
    }

    /* Quick pinned round-trip smoke test on device 0. */
    if (n > 0) {
        void *hp = NULL, *dp = NULL;
        size_t sz = 1 << 20;
        e = qw_pinned_alloc(&hp, &dp, sz);
        if (e == QW_OK) {
            printf("\npinned smoke test: %zu MiB pinned, host=%p dev=%p "
               "(%s)\n", sz / (1024 * 1024), hp, dp,
                   (hp == dp) ? "same pointer" : "DIFFERENT pointers");
            qw_pinned_free(hp);
        } else {
            printf("\npinned smoke test failed: %s\n", qw_errstr(e));
        }
    }

    printf("\nprobe complete\n");
    return 0;
}
