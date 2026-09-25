# qw

Single-model, single-hardware C inference engine for **Qwen/Qwen3.8-Flash-Next** on **3-4x AMD Radeon Pro V620** (RDNA2, `gfx1030`, 32GB GDDR6 each).

No external dependencies: libc + pthread. C17 host code; GPU kernels in HIP (`kernel/*.hip`) behind an `extern "C"` ABI shim. No cmake, no ggml, no llama.cpp, no torch.

## Model shape (baked in)

| | |
|---|---|
| Layers | 48, hidden 2560, 3:1 pattern — 36 GDN (gated-delta-net linear attn, no KV cache) + 12 QSA (full sparse attention) at indices 3,7,11,…,47 |
| QSA | 24 Q heads / 2 KV heads (GQA 12:1), head_dim 256, RoPE partial 0.25 (64/256 dims) |
| GDN | 48 V heads / 16 QK heads, head_dim 128 |
| MoE | 512 routed experts + 1 shared per layer, 10 active/token, expert_intermediate 640 |
| PLE n-gram table | 20,000,000 entries (~51B params, ~47.7 GiB fp8) — **resident in host RAM** |
| MTP module | ~4B params |
| Vocab / context | 248320 (padded) / 262144 |
| Transformer weights | ~125B params |

## Hardware constraints (ROCm 10.0.0 / HIP 7.15, gfx1030)

| Constraint | Consequence |
|---|---|
| No rocWMMA, no CDNA MFMA, no warp-matrix API | All matmul hand-written |
| fp16 has HW support; bf16/fp8/fp6/fp4 do **not** | Compute dtype is **fp16**; weights int4/int8/fp16 |
| 512 GB/s VRAM bandwidth | GEMM/attention bandwidth-bound — kernels tuned for this |
| PCIe 4.0 x16, ~25-28 GB/s effective (~1/20 of VRAM) | Minimize H2D/D2H traffic; nothing per-token crosses PCIe except prompt/decode I/O |
| 72 CUs, 1024 threads/CU max, warp size 32, no NVLink | Grid/occupancy tuned for 72 CUs; multi-GPU is layer-split over PCIe |
| Build | `--offload-arch=gfx1030` |

## Architecture decision log

- **Layer split, not tensor parallel.** The model has only **2 KV heads**; TP>2 forces KV-head replication (GQA 12:1 → 2 KV heads can't be divided across 3-4 GPUs without copying), adding PCIe traffic and no compute win. Instead: split 48 layers across GPUs (~12 per card), pass the 2560-dim hidden state (fp16, 5KB) between stages over PCIe — negligible vs weight traffic.
- **Weights resident, not streamed.** 125B params at int4 ≈ 62.5 GiB fits in 3-4 x 32 GiB. Streaming from host over 28 GB/s PCIe would make every layer O(GB/28GB/s) — residency is mandatory for reasonable latency.
- **int4 + fp16, not nvfp4/fp8.** gfx1030 has no fp8/fp6/fp4 hardware path; nvfp4 would dequantize to fp16 anyway with slower conversion. int4 (packed 2/byte) gives the best bytes/param with a hand-written dequant path; activations computed in fp16.
- **PLE n-gram table stays in host RAM.** ~47.7 GiB fp8 in 256 GB host RAM; looked up host-side, results staged to GPU. Moving it to GPU would eat ~1.5 of 3-4 cards and it's only touched sparsely.

## Layout

```
include/qw/     public C headers (types.h, macros.h, …)
src/            C17 host code by module (core/, loader/, runtime/, dev/, …)
kernel/         HIP kernels (*.hip), hipcc, --offload-arch=gfx1030
tools/          one .c per CLI tool -> bin/<name>
tests/          test_*.c -> bin/test_<name>
obj/ bin/ build/  generated (gitignored)
```

## Build

```sh
make            # build/libqw.a + tools into bin/
make test       # build + run all tests/test_*.c
make hipsyntax  # hipcc --offload-arch=gfx1030 -fsyntax-only kernel/*.hip (no-op if empty)
make clean
```

`make` succeeds **without** ROCm/hipcc installed (HIP targets are guarded; only `hipsyntax` needs it). Build a full ROCm image and rebuild to pull in GPU objects when available.

## Roadmap

- **P1** runtime + model loader (in progress)
- **P2** HIP kernels (matmul, dequant, elementwise)
- **P3** GDN + QSA attention
- **P4** MoE router + experts
- **P5** PLE n-gram host streamer
- **P6** sampler + tokenizer + HTTP server
