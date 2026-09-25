// Tiling shared by the grouped expert GEMMs (moe_experts.hip, moe_w4a8.hip):
// work is a device-built list of (expert, 32-token tile).
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace qw::gpu {

constexpr int MT = 32;  // tokens per tile
constexpr int BK = 64;  // K step (one LDS fill)

// The routed experts on v_dot4_i32_i8 (W4A8, lossy: see DESIGN.md), over the
// tile list built by moe_experts_T. q8 >= T*10*640 bytes, q8s >= T*10*5 floats.
void moe_experts_w4a8(const uint32_t *gw, const uint16_t *gs, const uint32_t *uw, const uint16_t *us,
                      const uint32_t *dw, const uint16_t *ds, const int32_t *counts, const int32_t *offsets,
                      const int32_t *pair_tok, const uint16_t *bin, int T, const int32_t *tile_e,
                      const int32_t *tile_m0, const int32_t *ntiles, int max_tiles, uint16_t *h, uint16_t *yp,
                      int8_t *q8, float *q8s, hipStream_t s);

}  // namespace qw::gpu
