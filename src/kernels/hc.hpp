// Hyper-connections: the 4-stream residual's pre-block mix (norm, low-rank
// down/up projection, stream mixing into the block input) and the post-block
// combine. Each rank holds SH columns of every stream.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

namespace qw::gpu {

// ---- prefill, token-major [T] (two phases: ssq all-reduce, then the full down projection)
// Optionally X += pend * 2 sigmoid(inj/4), then ssq[t][s].
void hc_pre_T(float *X, const float *pend, const float *inj, int T, float *ssq, hipStream_t s);
// xn[t][s][j] = fp16(X * rsqrt(ssq/H + eps) * w1)
void hc_norm_T(const float *X, const float *ssq, const float *w1, int T, uint16_t *xn, hipStream_t s);
// u = silu(d[0:320]/4) (fp16), inj = d[320:324]
void hc_mid_T(const float *d, int T, uint16_t *u, float *inj, hipStream_t s);
// bin[t][j] = mean_s sigmoid(g[t][j*4+s]) * xn[t][s*640+j]; g fp16 [T][2560]
void hc_mix_T(const uint16_t *g, const uint16_t *xn, int T, uint16_t *bin, hipStream_t s);
// X += pend * 2 sigmoid(inj/4)
void hc_combine_T(float *X, const float *pend, const float *inj, int T, hipStream_t s);
// X[t][s][j] = emb[t][j] for all s; emb fp32 [T][640]
void embed_T(float *X, const float *emb, int T, hipStream_t s);

// ---- batched decode rows (one phase): send[m] = ssq[4] | down partial [4][324]
void hc_pre_B(float *X, const float *pend, const float *inj, const float *w1, uint16_t *y, float *send, int M,
              hipStream_t s);
void hc_up_mix_B(const float *red, const uint16_t *up, const uint16_t *y, uint16_t *block_in, float *inj_out, int M,
                 hipStream_t s);

}  // namespace qw::gpu
