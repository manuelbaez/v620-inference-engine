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
constexpr int MTP_HIST = 256;              // decoded rows' hiddens kept per slot (MTP catch-up)

// K/V/ck hold fp16 bits: store with __half_as_ushort (assigning a __half to a
// uint16_t converts the value to an integer).
//
// KV spill: positions < vt live in VRAM (K, V hold vt rows), positions >= vt in pinned host memory mapped into the
// device's address space (Kh, Vh, indexed from position vt). The compressed keys ck always stay in VRAM, sized for
// the whole logical length. A slot without spill has vt = NO_SPILL.
constexpr int32_t NO_SPILL = 1 << 30;
// The raw indexer keys are only read to build a group's compressed key, from the group's own 4 positions, so a slot
// keeps them in a ring of raw_ring rows (position p at row p % raw_ring): a prefill chunk, the 3 positions before it
// and the RAW_TAIL a snapshot captured inside the chunk takes. A snapshot carries the last RAW_TAIL positions' raw
// keys (the MTP layer can lag the committed tokens by up to Engine::MTP_HISTORY positions).
constexpr int RAW_TAIL = 256;

// One QSA layer's KV of one slot, as the prefill kernels take it.
struct KvSplit {
    uint16_t *K, *V;
    float *raw;  // ring [raw_ring][IDX_DIM]
    uint16_t *Kh, *Vh;
    int32_t vt, raw_ring;
};

// Decode-time VRAM cache of spilled K/V rows (EngineOptions::spill_cache_mb): one direct-mapped table shared by every
// slot and QSA layer, an entry per position (its K and V rows) tagged (slot epoch, slot, layer, position). Attention
// reads a spilled row there when the tag matches, else from host memory, and lists the miss; one kernel at the end of
// the step copies the listed rows in, so the table is read-only while attention runs. A slot's epoch moves when its
// spilled rows change other than through decode (prefill, an import, a reset); decode clears an entry it rewrites.
struct SpillCache {
    unsigned long long *tags = nullptr;   // [entries]; 0: empty (epochs start at 1)
    uint16_t *K = nullptr, *V = nullptr;  // [entries][HEAD_DIM]
    uint32_t entries = 0;
    const uint32_t *epoch = nullptr;  // [slots]
    uint2 *miss = nullptr;            // [miss_cap]: (entry, position | slot << 25 | layer << 28)
    uint32_t *miss_n = nullptr;
    uint32_t miss_cap = 0;
    unsigned long long *stats = nullptr;  // [2] hits, misses (QW_SPILL_CACHE_STATS), or null
};
constexpr unsigned long long CACHE_LOCK = ~0ull;
__host__ __device__ inline unsigned long long cache_tag(uint32_t epoch, int slot, int layer, int64_t pos) {
    return (static_cast<unsigned long long>(epoch) << 32) | (static_cast<unsigned long long>(slot) << 29) |
           (static_cast<unsigned long long>(layer) << 25) | static_cast<unsigned long long>(pos);
}
__host__ __device__ inline uint32_t cache_entry(int slot, int layer, int64_t pos, uint32_t entries) {
    return (static_cast<uint32_t>(pos) + static_cast<uint32_t>(slot * 16 + layer) * 2654435761u) % entries;
}

struct SlotPtrs {
    uint16_t *K[QSA_LAYERS_MAX], *V[QSA_LAYERS_MAX], *ck[QSA_LAYERS_MAX];
    float *raw_k[QSA_LAYERS_MAX];  // rings of raw_ring rows
    uint16_t *Kh[QSA_LAYERS_MAX], *Vh[QSA_LAYERS_MAX];
    int32_t vt, raw_ring;
    float *S;         // [N_GDN][12][128][128]
    float *ring;      // [N_GDN][GDN_RING][2560]
    float *ple;       // [PLE_RING][2560]
    float *mtp_pend;  // [HC][SH] pre-final-mixer hidden of the slot's last token (MTP input)
    float *mtp_hist;  // [MTP_HIST][HC][SH] the same for decoded positions, at pos % MTP_HIST
};

// Recurrent-state captures inside a prefill chunk: dst[i] receives the state
// as it stands after position pos[i] - 1 (a snapshot at pos[i] tokens),
// laid out like the slot's copy of that state.
constexpr int MAX_CAPTURES = 4;
struct Captures {
    int n = 0;
    int64_t pos[MAX_CAPTURES] = {};
    float *dst[MAX_CAPTURES] = {};
};

struct Rows {
    const int32_t *slot;       // [M]
    const int64_t *pos;        // [M]
    const int32_t *run_first;  // [M]
    int M;
};

}  // namespace qw::gpu
