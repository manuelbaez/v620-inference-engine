// Qwen3.8-Flash-Next shape constants. The engine only serves this model, so the
// shapes are compile-time constants; check_config() verifies them against a
// checkpoint's config.json at load time. See docs/MODEL.md.
#pragma once

#include <cstdint>
#include <string>

namespace qw::cfg {

constexpr int H = 2560;            // hidden size
constexpr int HC = 4;              // hyper-connection streams
constexpr int HCH = HC * H;        // multi-stream width, 10240
constexpr int HC_RANK = 320;       // HC low rank
constexpr int N_LAYERS = 48;
constexpr int VOCAB = 248320;
constexpr int EOS = 248044;
constexpr float EPS = 1e-6f;

constexpr bool is_qsa(int layer) { return layer % 4 == 3; }
constexpr int N_QSA = N_LAYERS / 4;          // 12
constexpr int N_GDN = N_LAYERS - N_QSA;      // 36
constexpr int qsa_index(int layer) { return layer / 4; }
constexpr int gdn_index(int layer) { return layer - layer / 4; }

// QSA
constexpr int Q_HEADS = 24;
constexpr int KV_HEADS = 2;
constexpr int HEAD_DIM = 256;
constexpr int ROPE_DIM = 64;
constexpr double ROPE_THETA = 1e7;
constexpr int IDX_HEADS = 4;
constexpr int IDX_DIM = 128;
constexpr int IDX_BUDGET = 2048;             // tokens
constexpr int IDX_RATIO = 4;                 // tokens per compressed group
constexpr int IDX_TOPK_BLOCKS = IDX_BUDGET / IDX_RATIO;  // 512

// GDN
constexpr int GDN_QK_HEADS = 16;
constexpr int GDN_V_HEADS = 48;
constexpr int GDN_DIM = 128;
constexpr int GDN_KEY = GDN_QK_HEADS * GDN_DIM;          // 2048
constexpr int GDN_VAL = GDN_V_HEADS * GDN_DIM;           // 6144
constexpr int GDN_QKV = 2 * GDN_KEY + GDN_VAL;           // 10240
constexpr int GDN_CONV = 4;
// Per-sequence rings of recent pre-conv rows, indexed by position. Larger than
// the conv window so a speculative batch (up to 1 + 4 draft rows) never
// overwrites a row the next step still reads.
constexpr int GDN_RING = 8;

// MoE
constexpr int N_EXPERTS = 512;
constexpr int TOP_K = 10;
constexpr int FFN = 640;                     // routed and shared intermediate
constexpr int QGROUP = 128;                  // int4 group size along K

// PLE
constexpr int PLE_LAYER = 1;                 // 0-based
constexpr int NGRAM = 3;
constexpr int NGRAM_HEADS = 16;              // (NGRAM-1) * heads_per_ngram
constexpr int NGRAM_DIM = 160;               // H / NGRAM_HEADS
constexpr int PLE_CONV = 4;
constexpr int PLE_DILATION = 3;
constexpr int PLE_RING = 16;                 // >= (PLE_CONV-1)*PLE_DILATION + rows per step
constexpr int64_t NGRAM_VOCAB_BASE = 20000000;
constexpr uint64_t PLE_SEED = 1234;

// Throws if config.json does not describe the model these constants encode.
void check_config(const std::string &model_dir);

}  // namespace qw::cfg
