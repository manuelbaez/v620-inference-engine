// MTP (multi-token prediction) head input and hidden-state plumbing.
//
// X_mtp[m][s] = fc_hidden(gnorm(X[m])[s]) + fc_embedding(gnorm(emb[m])), with
// the fc GEMMs row-parallel (each rank multiplies its SH columns; a
// reduce-scatter sums them). Work rows are stream-major: [5][rows] = 4 hidden
// streams, then the embedding.
#pragma once

#include <hip/hip_runtime.h>

#include <cstdint>

#include "kernels/types.hpp"

namespace qw::gpu {

// send[m] = {sum X[m]^2 over the rank's HC*SH, sum emb[m]^2 over its SH, 0, 0}
// (4 floats: collectives move 16-byte rows)
void mtp_ssq(const float *X, const float *emb, int rows, float *send, hipStream_t s);
// xn[5][rows][SH] fp16 from the all-reduced sums; nh [HC][SH], ne [SH] are 1 + w
void mtp_norm(const float *X, const float *emb, const float *red, const float *nh, const float *ne, int rows,
              uint16_t *xn, hipStream_t s);
// X[m][s][j] = out[s][m][j] + out[4][m][j]; out f32 [5][rows][SH]
void mtp_add(const float *out, int rows, float *X, hipStream_t s);
// Xm[t] = t < o ? pend : X[t - o] for t < rows (o = 0 or 1): prefill rows
// shifted by one so row t pairs a hidden with the next token
void mtp_shift(const float *pend, const float *X, int o, int rows, float *Xm, hipStream_t s);
// dst[m][0..n) = src[m][0..n) (src: device array of row pointers)
void gather_rows(const float *const *src, int rows, int n, float *dst, hipStream_t s);
// tab[run_slot].mtp_pend = X[run_start + run_len - 1] for each run
void save_last_rows(const float *X, const SlotPtrs *tab, const int32_t *run_start, const int32_t *run_len,
                    const int32_t *run_slot, int M, hipStream_t s);

}  // namespace qw::gpu
