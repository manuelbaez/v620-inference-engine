// Decoding of the low-precision formats of ModelOpt checkpoints (nvidia/Qwen3.8-Flash-Next-FP4-Local; docs/DESIGN.md
// "The FP4 checkpoint"), for the loaders: MXFP8 (E4M3 values, a UE8M0 scale per 32 inputs), NVFP4 (E2M1 values, the
// even input in the low nibble, an E4M3 scale per 16 inputs, an FP32 scale per tensor) and NVFP4 scales stored in the
// compressed "csf" form (per slab of 16 rows a base code per row and a 4-bit offset per scale, then exceptions:
// uint32 words, the flat position in the low 24 bits and the code in the high 8; as local-inference-lab/b12x writes it).
#pragma once

#include <cstddef>
#include <cstdint>

namespace qw {

constexpr float E2M1_VALUES[16] = {0, 0.5f, 1, 1.5f, 2, 3, 4, 6, -0.f, -0.5f, -1, -1.5f, -2, -3, -4, -6};

uint16_t f32_to_bf16(float f);  // round to nearest even

// csf scales of a [rows][cols] scale matrix -> E4M3 codes [rows][cols].
void csf_expand(const uint8_t *fixed, const uint32_t *exceptions, size_t n_exceptions, size_t rows, size_t cols,
                uint8_t *out);
// MXFP8 [rows][cols] -> bf16, exactly (E4M3 times a power of two fits bf16).
void mxfp8_to_bf16(const uint8_t *w, const uint8_t *scale, size_t rows, size_t cols, uint16_t *out);
// NVFP4 [rows][cols] with E4M3 codes per 16 (codes) and the tensor's FP32 scale -> fp32.
void nvfp4_to_f32(const uint8_t *w, const uint8_t *codes, float global, size_t rows, size_t cols, float *out);

}  // namespace qw
