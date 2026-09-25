# qw engine design

Goal: serve Qwen3.8-Flash-Next (docs/MODEL.md) on 4x Radeon Pro V620 (gfx1030)
faster than the vllm-rdna2 fork, with an in-engine tiered prefix cache. Hardware
facts come from the [RDNA2/V620 guide](https://github.com/leapdragon/rdna2-v620-architecture-guide)
("guide §N" below) and from `home-infra/docs/llm-backend-amd-tuning.md`.

## The bar

vllm-rdna2 on this box (2026-09): ~56-65 tok/s single-stream decode (MTP=3 in
practice ~50-60), ~1,060 tok/s prefill, LMCache hits 62% of looked-up tokens
under agent load.

## Byte budget (from the checkpoint headers)

Bytes read per decoded token, before speculation:

| Part | GB/token | Notes |
|---|---|---|
| routed experts, int4 g128 | 1.22 | 10 of 512 per layer |
| GDN projections | 4.17 | bf16 in the checkpoint |
| HC mixers (96 x down+up+inject) | 1.27 | |
| QSA + indexer projections | 1.23 | |
| lm_head | 1.27 | |
| shared expert + router + PLE proj | 0.66 | |
| **total** | **~9.8** | 88% is dense, unquantized weight |

At 4-way split and 506 GB/s per card, that is ~4.9 ms per token (~200 tok/s)
with fp16 dense weights, and ~2.6 ms (~380 tok/s) with int8 dense copies. The
fork runs at ~30% of the fp16 figure. The gap is overhead: collectives, the
host-side PLE round trip, and thousands of small kernels per token.

## Parallelism

One process, one host thread per GPU, no Python.

- **Dense layers (GDN, QSA, HC, shared expert, lm_head): tensor parallel x4.**
  GDN: 12 v-heads / 4 qk-heads per card. QSA: 6 q-heads per card, KV head
  replicated on 2 cards each (only 12 KV layers, so the cost is small).
  lm_head: vocab-sharded, then a top-k merge instead of a full-logit gather.
- **Routed experts: expert parallel x4** (128 experts per card). TP would split
  the 640-wide intermediate into 160, which straddles the int4 group of 128
  (guide §4.6).
- **Residual stream: sharded along H.** Each card owns 640 of every stream's
  2560 columns (2560 of 10240). Replicating it would make every card stream
  every HC weight (1.27 GB, ~2.5 ms/token).

### The collective schedule: the main engineering problem

A naive schedule needs about 3 collectives per sublayer, so ~290 per token. At
~15 µs each (in-server RCCL, guide §4.1) that is ~4.3 ms, as much as all the
weight traffic. How to cut it:

1. **One small all-reduce per HC mix.** The down projection is linear in the
   un-normalized stream, `d = sum_s r_s * W_down[:, s] . (X_s * (1 + w_s))`, so
   each card computes 4 partial 324-vectors plus its 4 partial sums of squares.
   A single ~5 KB all-reduce gives `d`, the injection logits and every stream's
   rms scale.
2. Then an **all-gather** of `block_in` (640 -> 2560 fp16), the TP block, and a
   **reduce-scatter** of the block output back onto the sharded stream.
3. Custom push-based collectives over uncached VRAM staging buffers, following
   the guide's coherence and ordering rules (§4.2-4.3): payload and flag go to
   the same destination, reduction happens in a fixed rank order in fp32, and
   spins are bounded. RCCL is the correctness fallback.
4. Measure inside the engine before building anything fancier (guide §4.4: an
   out-of-server microbenchmark overstated RCCL cost 4x).

### P2P

Every collective is built on direct GPU-to-GPU PCIe transfers, not host
bounces:

- `hipDeviceEnablePeerAccess` for all 12 ordered pairs at startup. Clear the
  sticky `hipErrorPeerAccessAlreadyEnabled` with `hipGetLastError()`.
- **Kernel-initiated peer stores (push)**, never peer loads: the guide measured
  14.3 GB/s push against 5.7 GB/s pull on Gen3 (§4.1). All four cards here
  negotiate Gen4 x16, and `pcie_gen_cap` is deliberately not set, so expect
  more. The first GPU task is to measure the 4x4 push/pull bandwidth and the
  small-message latency matrix, because each card is on its own NUMA node and
  `85:00.0` has measured slower than its siblings.
- **Measured 2026-09-24** (`build/p2p_bench`, with production vLLM live on the
  same cards, so treat single pairs as noisy). HIP order in `llm-experiments`
  is 0=`c3`, 1=`85`, 2=`44`, 3=`03`. Kernel push runs at ~25 GB/s on most
  pairs, with 0->1 and 0->2 at ~14. Pull runs at 10-23 GB/s. SDMA copy runs
  at 10-25 GB/s. So push is the right primitive, and Gen4 gives about twice
  the guide's Gen3 bandwidth. Re-measure with production stopped, then
  measure the in-graph latency (the benchmark's latency column includes a
  host sync).
- The receive buffers are `hipExtMallocWithFlags(hipDeviceMallocUncached)` on
  the owning card, mapped into the peers. The payload and its sequence flag go
  to the same destination (flag last), and the owner polls its own VRAM with
  `s_sleep`.
- **Fuse the collective into its producer.** The kernel that computes a partial
  (a row-parallel GEMV's output tile, the HC down partials) stores its slice
  straight into each peer's staging buffer. The consumer's first kernel waits
  on the flags. No separate all-reduce kernel or launch is needed.
- Bulk moves use SDMA copies (`hipMemcpyPeerAsync`): prefix-cache KV blocks
  migrating between cards, and weight loading. Measure whether SDMA is safe to
  re-enable first, since production runs with `HSA_ENABLE_SDMA=0` after a wedge
  (home-infra tuning doc).

### Decode step

A fixed-shape program per batch size (1, 2, 4, 8), captured as a HIP graph per
GPU. All M ≤ 8 decode GEMVs cost the same as M=1 on this chip (guide §1.9), so
small batches and speculative verification are nearly free. Kernels are fused
wherever a sublayer's glue would otherwise be its own launch (guide §1.6: the
cost is the bubble between dependent kernels, not the launch call).

- GEMVs are wave-per-output-row with fp16 `v_dot2_f32_f16`, 2-4 independent
  16-byte loads in flight per lane, and split-K partials plus a combine instead
  of atomics (guide §1.7).
- MoE uses a skinny int4 expert GEMV with routing, silu*up and the weighted
  combine fused in (guide §8.3: 85% of bandwidth).
- GDN decode: one fused kernel per layer (conv step, l2norm, gating, delta rule,
  gated norm) with the recurrent state kept in fp32.
- QSA decode: indexer scoring over the compressed keys, top-512 selection, then
  flash-decode over the selected tokens with GQA 12:1 on the dot M dimension.
- PLE: the n-gram table stays in host RAM (int4 sidecar, ~30 GB, mmapped) and is
  never copied to a GPU. The host hashes the ids and gathers 16 rows (a few µs)
  **while the GPUs run layer 0**. The 10 KB result goes over PCIe into a buffer
  the layer-1 kernel waits on through a host-side handshake. Never use
  `hipStreamWaitValue32` for this: graph capture silently drops it (guide §4.5).

### Prefill

Prefill is compute-bound and served from Infinity Cache, so it wants different
kernels (guide §1.8): dequantize to dense fp16 and use tiled GEMMs, run the
chunked GDN form, and use a tiled attention kernel with `num_stages=1`-style
single buffering. It is a separate campaign from decode, measured separately.

## Memory plan (per card)

| | GiB |
|---|---|
| routed experts (128 x 48, int4 + scales) | ~14.5 |
| dense TP shards, fp16 | ~2.3 |
| lm_head shard | ~0.3 |
| MTP layer (bf16 experts, EP) | ~1.2 |
| QSA KV: 12.3 KB/token/card (fp16, 1 kv head x 12 layers) | 2.4 per 200k tokens |
| GDN state: 28 MB per sequence per card | |
| free for the GPU tier of the prefix cache | ~8-10 |

Embeddings stay on the host too (one 5 KB row per token).

## Prefix cache (the LMCache replacement)

The state that makes a prefix reusable:

| Part | Size | Notes |
|---|---|---|
| QSA K/V | 24.6 KB/token (fp16, all 12 layers, both heads) | append-only |
| indexer raw + compressed keys | ~3.8 KB/token | append-only |
| GDN recurrent state + conv tail | ~453 MB (fp32) per snapshot | fixed size, **not** append-only |
| PLE conv tail + last 2 tokens | ~0.4 MB per snapshot | |

- **Keys:** blocks of 256 tokens, each hashed with its parent's hash (a hash
  chain), kept in a radix tree. A lookup walks the prompt's block hashes.
- **KV blocks** live in three tiers: VRAM, then pinned host RAM, then NVMe on
  `/main-storage` (the ZFS pool, 1.2 TB free). They are evicted by LRU with
  prefix-aware refcounts, so a parent can't go while a live child needs it.
- **GDN snapshots** are only taken where they pay off. The recurrent state can't
  be rebuilt from KV, so a hit needs a snapshot at the exact hit boundary.
  Snapshots are taken at the end of every prompt, at the end of every generated
  turn, and every 4k tokens inside long prefills. They can be stored in fp16
  (~226 MB), which must pass a quality gate first.
- **Resume** = the longest cached prefix whose end has a snapshot. Load its KV
  blocks and its snapshot, then prefill only the rest. In chat and agent use,
  turn N+1 extends turn N exactly, so its end-of-turn snapshot is always the
  right one.
- **Tier traffic** runs in-process on dedicated copy streams with pinned
  staging, so there are no IPC events or server handshakes. That removes the
  whole failure class LMCache hit here (HIP interprocess events, TTL locks,
  chunk-size coupling, 4.8k of 8k tokens retrieved).
- Speculative drafts never enter the cache. Only committed tokens do, which
  sidesteps the relocation window that cost LMCache the tail of every request.

## CacheBlend-style non-prefix reuse (experimental, last)

CacheBlend reuses the KV of chunks that are not a prefix (RAG documents,
reordered tool outputs) and recomputes the ~15% of tokens whose KV deviates
most. That works because attention KV can be spliced. Here 36 of 48 layers are
GDN, whose state after a chunk depends on everything before it, so KV splicing
alone cannot work.

The one workable version: the gated delta rule is affine in its incoming state
(`S_out = S_in * T_chunk + U_chunk`, with `T` and `U` 128x128 per head), so a
chunk cached with its per-layer `(T, U)` (~450-900 MB) can be composed onto any
prefix state. QSA layers splice KV and recompute selected tokens as in
CacheBlend. Like CacheBlend, this assumes the chunk's activations computed out
of context are close enough to the in-context ones, and in the GDN layers
nothing later corrects that error. It ships only behind a flag, and only if a
quality eval (next-token agreement and long-context QA against a full
prefill) passes.

## Measured baseline (2026-09-24)

The first 4-GPU engine (`src/engine`) decodes one token at a time with plain
launches, 292 collectives per token and no graphs or fusion:

| | |
|---|---|
| decode, 57-token context | **56.4 tok/s** (production vllm-rdna2: ~56) |
| decode, 4.3k context | 54.6 tok/s |
| prompt, token by token | 40-48 tok/s (no prefill kernels yet) |
| logprobs vs fp32 reference | median \|Δ\| 0.0015 nats (57 tok), 0.015-0.023 (4.3k) |
| greedy vs vLLM | identical continuations on both test prompts |
| chained 5.2 KB all-reduce | 45.7 µs (push + receive kernels) |
| load time | ~95 s from page cache |

The collectives (~292 x ~45 µs ≈ 13 ms) are most of the ~18 ms token, so
the next work is cutting their count and latency. That means fusing the push
into producer kernels and the wait into consumers, capturing graphs, and
applying the down-projection algebra above.

## Prefill (measured 2026-09-24)

Chunked prefill (`Engine::prefill`, `src/kernels/prefill_ops.hip`), 2048-token
chunks, on the 4,266-token test prompt:

| step | prefill tok/s | notes |
|---|---|---|
| token by token through the decode graph | 50 | |
| batched: rocBLAS fp16 GEMMs, naive expert kernel | 770 | experts 52%, collectives 26% |
| tiled grouped int4 expert GEMM (dequant into LDS) | 1,322 | experts 2,437 -> 425 ms |
| fp16 output for the HC-down/shared GEMMs, fp16 HC-down all-reduce | 1,388 | router must stay fp32 |
| two micro-batches on two streams (collectives overlap compute) | 1,393 | +7% over the same halves run serially |
| 8192-token chunks + load-time warmup (rocBLAS loads kernels lazily) | **1,905** | 4.3k prompt; 1,996 on a 17k prompt |
| production vllm-rdna2 | ~1,060 | |

W4A8 experts (`QW_INT8_EXPERTS=1`, `v_dot4_i32_i8`, exact int8 weights,
per-128-group int8 activations): +2% prefill, but mean |Δlogprob| against the
fp32 reference goes from 0.075 to 0.104 (per-token activation scales: 0.114).
It stays opt-in. The experts are now under a fifth of prefill time, so the
faster dot buys little. For scale, production vLLM itself sits at 0.097.

Greedy continuations can differ between chunk sizes: fp16 rounding differs
with GEMM shapes, and near-ties flip (measured: after `88 4833`, tokens 13 and
25 are 0.07-0.18 nats apart). Teacher-forced decode after either chunking has
the same error against the reference (`qw_gpu --prefill-n`), so this is
rounding, not state corruption.

What was learned:
- **rocBLAS on gfx1030 is fast only with fp16 output**: 30-36 TFLOPS for the
  prefill shapes, against 6-11 TFLOPS with fp32 output (which falls back to
  `HSS` Tensile kernels).
- **Router logits must stay fp32.** fp16 logits flip near-tie expert
  choices: mean |Δlogprob| against the fp32 reference goes from 0.074 to 0.088.
  fp16 HC-down partials and fp16 shared-expert outputs are neutral.
- **Collectives on this box: never write to all peers at once for big
  payloads.** One kernel pushing to 3 peers simultaneously ran ~6x slower
  (10.6 MB all-reduce: 25 ms) than rotating destinations so that each GPU
  feeds one peer at a time and no two GPUs target the same one (4.1 ms). Small
  payloads still go to all peers at once (14 µs). Many receive blocks spinning
  on uncached flags must also be avoided; one small kernel waits instead.
- Accuracy: prefill logprobs sit at the same distance from the fp32 reference
  as the decode path's (mean ~0.074, median ~0.02 nats over 4,265 positions),
  with greedy output identical to vLLM.

Next for prefill: overlap the collectives (now the largest share) with
compute by splitting each chunk into two micro-batches on two streams; W4A8
expert GEMMs on `v_dot4_i32_i8`; a chunked (WY) GDN kernel.

## Roadmap

1. **Reference and spec.** CPU fp32 reference (`src/ref`), validated against
   vLLM's prompt logprobs. *In progress.*
2. **Single-GPU kernels.** Each kernel is checked against the reference, layer
   by layer.
3. **4-GPU runtime.** TP/EP, collectives, graph-captured decode; beat 65 tok/s.
4. **Prefix cache tiers.**
5. **MTP (K=3) speculative decoding.**
6. **Prefill kernels.**
7. **OpenAI-compatible server and tokenizer**, and a llama-swap entry in home-infra.
8. **CacheBlend experiment.**
