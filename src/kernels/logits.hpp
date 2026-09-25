// Reductions over a rank's vocab shard of each row's logits.
#pragma once

#include <hip/hip_runtime.h>

namespace qw::gpu {

// out[m] = {max, sum exp(x - max)}: log-sum-exp pieces, combined across ranks on the host
void row_lse(const float *logits, int n, int M, float *out, hipStream_t s);
// out[m] = {max, index as int bits} (first index on ties)
void row_argmax(const float *logits, int n, int M, float *out, hipStream_t s);

}  // namespace qw::gpu
