// Per-rank shard shapes: 4-way tensor parallelism for dense layers, 4-way
// expert parallelism for the routed experts, residual sharded along H
// (docs/DESIGN.md). Derived from the model constants in config.hpp.
#pragma once

#include <cstddef>

#include "core/config.hpp"

namespace qw::cfg {

constexpr int RANKS = 4;

constexpr int SH = H / RANKS;               // 640 residual columns per rank per stream
constexpr int XW = HC * SH;                 // 2560: a rank's slice of a multi-stream row
constexpr int HC_DOWN = HC_RANK + HC;       // 324: low-rank rows + inject rows per stream
constexpr int HC_SEND = HC + HC * HC_DOWN;  // 1300: ssq[4] + down partials [4][324]

constexpr int GDN_VL = GDN_V_HEADS / RANKS;                            // 12 local v-heads
constexpr int GDN_KL = GDN_QK_HEADS / RANKS;                           // 4 local k-heads
constexpr int GDN_QKV_L = 2 * GDN_KL * GDN_DIM + GDN_VL * GDN_DIM;     // 2560
constexpr int GDN_PROJ_L = GDN_QKV_L + GDN_VL * GDN_DIM + 2 * GDN_VL;  // 4120: qkv | z | b | a
constexpr size_t GDN_STATE = size_t(GDN_VL) * GDN_DIM * GDN_DIM;       // floats per layer per sequence

constexpr int QH_L = Q_HEADS / RANKS;  // 6 local q heads (1 kv head)
constexpr int QSA_PROJ_L = QH_L * 2 * HEAD_DIM + 2 * HEAD_DIM + (IDX_HEADS + 1) * IDX_DIM;  // 4224

constexpr int EXP_L = N_EXPERTS / RANKS;  // 128 local experts
constexpr int SFFN_L = FFN / RANKS;       // 160 shared-expert columns
constexpr int VOCAB_L = VOCAB / RANKS;    // 62080 lm_head rows
constexpr int PLE_KV = HC * SH + SH;      // 3200

}  // namespace qw::cfg
