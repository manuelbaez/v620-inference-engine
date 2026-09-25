// Types shared by the kernels and the engine: the batched-row description and
// the per-sequence state table.
//
// A batched decode step processes M rows (M <= MAX_ROWS). Row r is one token of
// sequence slot[r] at position pos[r]. Rows of the same slot are consecutive
// and in position order (a plain decode step has one row per slot; speculative
// verification has 1 + K rows per slot). run_first[r] is the row index where
// r's slot run starts, so a row can find earlier tokens of its own sequence in
// the same batch. Per-sequence state lives in slots, found through a device
// table of SlotPtrs indexed by slot.
#pragma once

#include <cstdint>

#include "core/shard.hpp"

namespace qw::gpu {

using namespace cfg;

constexpr int MAX_ROWS = 16;
constexpr int QSA_LAYERS_MAX = N_QSA + 1;  // + the MTP layer
constexpr int D_PAD = 328;                 // HC down outputs (320 + 4 inject) padded to 16 bytes
constexpr int LIST_W = 2052;               // QSA attention token-list width (512 groups x 4 + tail)

struct SlotPtrs {
    uint16_t *K[QSA_LAYERS_MAX], *V[QSA_LAYERS_MAX], *ck[QSA_LAYERS_MAX];
    float *raw_k[QSA_LAYERS_MAX];
    float *S;         // [N_GDN][12][128][128]
    float *ring;      // [N_GDN][GDN_RING][2560]
    float *ple;       // [PLE_RING][2560]
    float *mtp_pend;  // [HC][SH] pre-final-mixer hidden of the slot's last token (MTP input)
};

struct Rows {
    const int32_t *slot;       // [M]
    const int64_t *pos;        // [M]
    const int32_t *run_first;  // [M]
    int M;
};

}  // namespace qw::gpu
