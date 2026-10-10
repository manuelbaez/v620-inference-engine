// Qwen3.8-Flash-Next shape constants. The engine only serves this model, so the
// shapes are compile-time constants; check_config() verifies them against a
// checkpoint's config.json at load time. See docs/MODEL.md.
#pragma once

#include <cstdint>
#include <string>

namespace qw::cfg {

constexpr int H = 2560;       // hidden size
constexpr int HC = 4;         // hyper-connection streams
constexpr int HCH = HC * H;   // multi-stream width, 10240
constexpr int HC_RANK = 320;  // HC low rank
constexpr int N_LAYERS = 48;
constexpr int VOCAB = 248320;
constexpr int EOS = 248044;
constexpr float EPS = 1e-6f;

// Vision tokens. Inside the engine a vision token is a negative id (its
// embedding is supplied by the prefill; the session derives the id from the
// image's content hash, so the prefix cache tells images apart): odd for
// images, even for videos. The n-gram table sees the pad token instead.
constexpr int32_t IMAGE_PAD = 248056, VIDEO_PAD = 248057;
constexpr int32_t real_token(int32_t t) {
    return t >= 0 ? t : ((-t) & 1 ? IMAGE_PAD : VIDEO_PAD);
}

constexpr bool is_qsa(int layer) {
    return layer % 4 == 3;
}
constexpr int N_QSA = N_LAYERS / 4;      // 12
constexpr int N_GDN = N_LAYERS - N_QSA;  // 36
constexpr int qsa_index(int layer) {
    return layer / 4;
}
constexpr int gdn_index(int layer) {
    return layer - layer / 4;
}

// QSA
constexpr int Q_HEADS = 24;
constexpr int KV_HEADS = 2;
constexpr int HEAD_DIM = 256;
constexpr int ROPE_DIM = 64;
constexpr double ROPE_THETA = 1e7;
constexpr int IDX_HEADS = 4;
constexpr int IDX_DIM = 128;
constexpr int IDX_BUDGET = 2048;                         // tokens
constexpr int IDX_RATIO = 4;                             // tokens per compressed group
constexpr int IDX_TOPK_BLOCKS = IDX_BUDGET / IDX_RATIO;  // 512

// GDN
constexpr int GDN_QK_HEADS = 16;
constexpr int GDN_V_HEADS = 48;
constexpr int GDN_DIM = 128;
constexpr int GDN_KEY = GDN_QK_HEADS * GDN_DIM;  // 2048
constexpr int GDN_VAL = GDN_V_HEADS * GDN_DIM;   // 6144
constexpr int GDN_QKV = 2 * GDN_KEY + GDN_VAL;   // 10240
constexpr int GDN_CONV = 4;
// Per-sequence rings of recent pre-conv rows, indexed by position. Larger than
// the conv window so a speculative batch (up to 1 + 4 draft rows) never
// overwrites a row the next step still reads.
constexpr int GDN_RING = 8;

// MoE
constexpr int N_EXPERTS = 512;
constexpr int TOP_K = 10;
constexpr int FFN = 640;     // routed and shared intermediate
constexpr int QGROUP = 128;  // int4 group size along K
// The routed experts of a checkpoint may instead have a scale and a zero point per QGROUP_Z inputs (an experiment,
// docs/DESIGN.md "Experts with a scale per 32 and zero points"). The engine keeps both in one 16-bit word per group:
// the fp16 scale, whose sign and low three mantissa bits are free (the checkpoint's scales are bf16, 7 mantissa
// bits), with the zero point's nibble (zero point + 8) in bit 15 and bits 2-0.
constexpr int QGROUP_Z = 32;
constexpr uint16_t QZ_SCALE_MASK = 0x7FF8;
inline uint16_t qz_word(uint16_t f16_scale, int zp_nibble) {
    return uint16_t(f16_scale | ((zp_nibble & 8) << 12) | (zp_nibble & 7));
}
inline int qz_nibble(uint16_t word) { return ((word >> 12) & 8) | (word & 7); }

// PLE
constexpr int PLE_LAYER = 1;  // 0-based
constexpr int NGRAM = 3;
constexpr int NGRAM_HEADS = 16;  // (NGRAM-1) * heads_per_ngram
constexpr int NGRAM_DIM = 160;   // H / NGRAM_HEADS
constexpr int PLE_CONV = 4;
constexpr int PLE_DILATION = 3;
constexpr int PLE_RING = 16;  // >= (PLE_CONV-1)*PLE_DILATION + rows per step
constexpr int64_t NGRAM_VOCAB_BASE = 20000000;
constexpr uint64_t PLE_SEED = 1234;

// Throws if config.json does not describe the model these constants encode.
// Returns the group size of the routed experts' scales (QGROUP or QGROUP_Z).
int check_config(const std::string &model_dir);

}  // namespace qw::cfg
