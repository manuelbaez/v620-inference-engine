# ADR-002: int8 (W8A8) for MoE expert GEMMs during prefill only

Status: **Proposed** — gated on the benchmark in `tools/bench_int8.c` beating
the fp16 baseline by > 1.3x at M >= 32 on the V620 box.

## Context

The model is Qwen3.8-Flash-Next: 48 layers, hidden 2560, MoE with 512 routed
experts (+1 shared) per layer, 10 active experts/token,
`moe_intermediate_size` 640. Weights ship int4 (~57.75 GiB total), activations
compute in fp16 (the only hardware compute dtype on gfx1030).

Two regimes, two bottlenecks:

- **Decode is bandwidth-bound.** M = 1 per expert (a single token), and with a
  layer split only one GPU is active per stream, so per-GPU bandwidth times
  *add* (see `qw/shard.h`). int8 does not help a GEMV: halving the weight
  width halves the bytes read, but the op is already roofline-limited by the
  read, and the dequant + fp16 accumulate dominate. Decode keeps
  int4-weight / fp16-act GEMV.

- **Prefill is compute-bound, and the MoE structure is what makes it a real
  GEMM.** A prompt of L tokens activates 10 experts per token across 512
  experts, so each active expert sees `M = L * 10 / 512` tokens. At the
  default 4096-token context that is **M = 80** — a genuine GEMM shape
  (`[80 x 2560] * [2560 x 640]`), not a GEMV. At that shape, int8 dot
  throughput can roughly double fp16 FMA throughput *if* the hardware can
  actually execute the integer dots fast.

The hardware constraint that drives the whole decision: **gfx1030 (RDNA2) has
no `rocWMMA`, no CDNA-style MFMA, and no warp-matrix API, and no separate
matrix registers.** int8 is the *only* post-fp16 numeric type with hardware
support; bf16 is emulated and fp8/fp6/fp4 have no hardware support at all.
Therefore an int8 GEMM on gfx1030 must rely on `v_dot`-class integer
dot-product instructions that LLVM emits from a plain C loop (or on
hand-written intrinsics) — **not** on any tensor-core-style tile instruction.
Whether LLVM emits them well on gfx1030 is the open question the benchmark
exists to answer.

## Decision

Use **int8 W8A8 (weights int8 per-channel symmetric, activations int8
per-token symmetric) for the MoE expert GEMMs during prefill ONLY.** Decode
keeps the existing int4-weight / fp16-act GEMV path. The switch is **gated on
`tools/bench_int8.c` showing int8 beating the fp16/fp32-accumulate baseline by
> 1.3x at M >= 32** on the V620 box; below that threshold we do not ship the
second code path.

Concretely, when the gate passes:
- A prompt whose per-expert M ( = L * 10 / 512 ) is >= 32 routes the expert
  GEMM through the int8 blocked kernel (`qw_q8_gemm_blocked`).
- A prompt with M < 32 (short prompts / decode) keeps the fp16 GEMV path.
- This becomes a small **M-threshold dispatch table** in the expert forward
  pass, not a global flag.

The int8 kernel is the hand-tuned, cache-blocked (8x4 register tile, int32
accumulator, fused dequant epilogue) C17 reference in `src/core/quant.c` — the
shape we give LLVM the best chance to auto-vectorize into `v_dot` on the GPU.
The naive reference (`qw_q8_gemm_ref`, int64 accumulator) is the correctness
oracle.

## Alternatives considered

- **fp16 only (status quo).** Safe, one code path, but leaves the prefill
  compute-bound regime on the slower fp16 FMA rate. Rejected *if* the
  benchmark clears the 1.3x gate; otherwise this is the fallback.

- **int4 weights + int8 activations (W4A8).** Halves weight bytes vs W8A8
  (good for decode bandwidth) but the mixed-precision multiply
  (int4 x int8) is not a clean `v_dot` shape and needs a dequant-to-int8 of
  the weights first, which eats the compute saving. The int4 GEMV path already
  serves decode; W4A8 buys little for prefill where weights are not the
  bottleneck. Rejected.

- **GGUF Q8_0 blocks (32-element blocks, 1 fp32 scale per block).** Familiar
  format, but per-block (not per-channel) scaling is less accurate for the
  125B MoE and the block layout is not GEMM-friendly for a row-major
  `[N][K]` int8 kernel. We already have a clean per-channel W8A8 layout.
  Rejected.

- **bf16.** Rejected outright: emulated on gfx1030, no hardware path.

- **fp8 / fp6 / fp4.** Rejected outright: no hardware support on gfx1030 at
  all.

## Consequences

- **Quantization accuracy risk on a 125B MoE.** Per-channel symmetric int8
  needs calibration (per-channel max-abs) on representative data before we can
  trust it. `qw_q8_compare_fp16()` and `tests/test_quant.c` establish the MSE /
  cosine-similarity floor on synthetic data; a real-model accuracy pass
  (perplexity / downstream task delta) is required before the gate is
  considered fully met.
- **A second code path to maintain.** The expert forward pass now branches on
  M. The dispatch table, the int8 quantization of activations at runtime
  (per-token, cheap), and the int8 weight storage (an extra copy alongside
  int4, or a pre-converted int8 shard) all add surface area.
- **An M-threshold dispatch table.** The boundary (currently 32) is a knob,
  not a law; it should be set from the measured crossover in the benchmark,
  not assumed.
- **Memory.** int8 expert weights are 2x the int4 size. If we keep both
  int4 (decode) and int8 (prefill) resident, expert weight VRAM doubles for
  the MoE block (the dominant term). We must decide whether to store int8
  primary and dequant-to-int4 for decode, or store both — a VRAM budget
  question the shard planner must absorb.

## Open questions

1. **Does LLVM emit `v_dot`-class integer dot products from the blocked C loop
   on gfx1030, and at what efficiency?** This is unconfirmed. RDNA2 has
   `v_dot4_vopd` / `v_dot2cq_vopc` integer dot hardware, but the exact builtin
   (`__builtin_amdgcn_sdot4` / `__builtin_amdgcn_u4dot`) and whether the
   AMDGPU backend lowers a plain int8 dot loop to it on *this* arch is not
   verified. `src/core/quant.c` isolates the *only* unconfirmed intrinsic
   behind `#ifdef QW_ENABLE_DOT` and `#ifdef __AMDGCN__` so it can never
   silently break a non-GPU build. **`tools/bench_int8.c` settles this:** run
   it on the V620 box with `-DQW_WITH_HIP`; if the int8 kernel's GFLOP/s at
   M >= 32 clears the 1.3x gate, LLVM (or the intrinsic) is producing real
   dot throughput; if it does not, the int8 path is not worth shipping and we
   stay fp16.
2. **What is the true fp16 peak FLOP/s on the specific V620 silicon** (the
   1.96 TFLOP/s in the bench is a spec estimate)? The ridge point
   `peak_flops / 512e9 B/s` should be recomputed from a measured peak before
   the M-threshold is frozen.
3. **VRAM cost of dual int4+int8 expert weights** under the 32 GiB/GPU budget
   with the 12% safety margin (see `qw_model_estimate`).
