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
  chain). A lookup walks the prompt's block hashes. (Implemented: see "Block
  store" below.)
- **KV blocks** live in three tiers: VRAM, then pinned host RAM, then disk on
  `/main-storage` (the ZFS pool). That pool is a mirror of two 4 TB 5400 rpm
  HDDs (WD40EZRZ) with NVMe only as special vdev, log and L2ARC (an earlier
  version of this note said NVMe; the measured load speed, 160-205 MB/s, is
  the HDDs'; see "Disk-tier loads and host memory"). They are evicted by LRU
  with prefix-aware refcounts, so a parent can't go while a live child needs
  it.
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

## CacheBlend-style non-prefix reuse (measured 2026-09-25: rejected)

CacheBlend reuses the KV of chunks that are not a prefix (RAG documents,
reordered tool outputs) and recomputes the ~15% of tokens whose KV deviates
most. That works because attention KV can be spliced. Here 36 of 48 layers are
GDN, whose state after a chunk depends on everything before it.

The workable version was built as an experiment (`src/engine/blend.hip`,
`src/kernels/blend.hip`, `tools/qw_blend_eval.hip`, cases from
`tools/blend_cases.py`). The delta rule is affine in its incoming state, so a
chunk recorded after one prefix carries its transfer M = prod decay (I - beta
k k^T) (128 x 128 per v-head, accumulated by a v = 0 scan during its prefill),
and after another prefix the state is composed as S = S_out + M (S - S_in).
The chunk's KV is copied to its new positions with the keys re-rotated and the
compressed indexer keys rebuilt; its first W tokens are prefilled in context.

The transfer is exact where it can be: layer 0's inputs do not depend on the
context, and its composed state matches a full prefill (relative error 0.000,
against 0.05-0.19 without the correction). Deeper layers are not: their
activations over the whole chunk change with the prefix, and nothing corrects
that (selective recompute cannot, the recurrence needs every token). Composed
states are off by 2-29% at middle layers, 20-100x the noise of prefilling the
same prompt in different pieces, and W (32 to 512) barely matters.

Agent-style cases (a source file returned by a tool, reused after other files
or after an edited earlier message, then a question about it; KL over the
question's tokens against a full prefill, and greedy tokens that agree out of
48):

| case | rechunked (noise) | blend W=32 | no correction | chunk dropped |
|---|---|---|---|---|
| file moved after another file | 0.020, 45/48 | 0.048, 15/48 | 0.066, 15/48 | 1.45 |
| file moved (8k source prefix) | 0.057, 0/48 | 0.227, 18/48 | 0.125, 18/48 | 2.94 |
| file first in the source, third now | 0.004, 48/48 | **1.25**, 3/48 | 0.216, 0/48 | 1.47 |
| earlier message edited | 0.047, 48/48 | 0.520, 48/48 | 0.245, 48/48 | 2.18 |
| question about the file before it | 0.021, 0/48 | 0.062, 0/48 | 0.040, 4/48 | 0.63 |

The error is 2-300x the noise floor and erratic (in one case close to leaving
the file out), so the gate fails and non-prefix reuse is not served. The
experiment code stays for future attempts (e.g. recomputing only the deep
layers). `/health`'s `blend_candidate_tokens` keeps measuring how much such
reuse could have saved on real traffic.

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

## Batched decode and slots (2026-09-25)

The engine holds several sequences ("slots", default 131072 / 65536 / 32768 /
32768 tokens of KV), each with its own recurrent state (GDN state and conv
ring, PLE conv ring), QSA K/V, compressed keys and MTP input store. A decode
step is a batch of up to 16 rows; a row is (slot, position, token), rows of a
slot are consecutive (a "run"), and the kernels find per-sequence state
through a device table of slot pointers. Each row count has its own captured
graph, so a step costs one launch per rank.

- Rows share no arithmetic: logits of a sequence decoded alone and in a batch
  are bit-identical (`tests/gpu/test_batch_decode`), and so are k rows of one
  run vs k single steps (the speculative verification shape).
- Routed experts for small batches use GEMV-shaped kernels, one wave per
  (token, expert) pair and output row (`moe_experts_P`); the tiled prefill
  GEMM padded every expert to 32 tokens and cost 8.5 ms of a 17 ms step.
- Each row's log-sum-exp is computed per vocab shard on the GPUs and combined
  on the host, so sampling with logprobs needs no pass over 248k logits on
  the CPU (that pass cost ~10 ms per token in the server).

| rows per step | ms / step | aggregate tok/s |
|---|---|---|
| 1 | 17.3 | 57.9 |
| 2 (2 slots) | 19.3 | 103.5 |
| 4 (4 slots) | 24.2 | 165.0 |
| 8 (4 slots x 2 rows) | 34.0 | 235.3 |

## MTP speculative decoding (2026-09-25)

The checkpoint's MTP head is one more QSA decoder layer. Its input for
position p is fc_hidden(gnorm(X_p)) per stream + fc_embedding(gnorm(emb of
token p+1)), where X_p is the backbone's pre-final-mixer 4-stream hidden; its
own final mixer feeds the shared lm_head, and its combined hidden feeds the
next draft step. The engine keeps it exact and simple:

- The MTP layer has its own KV (QSA layer index 12 in every slot). Prefill
  runs it over every chunk, shifted by one row: row t pairs X_t with token
  t+1, and the chunk's last hidden waits in the slot's MTP input store until
  the next token is known. The fc projections are row-parallel (each rank
  multiplies its 640 columns, a reduce-scatter sums them).
- Its bf16 routed experts are quantized at load to the same int4 g128
  layout as the backbone's (only draft quality depends on them).
- A step feeds each request's pending token plus K = 3 drafts as one run.
  The GDN scan also writes the state after every row; `accept(slot, n)`
  copies back the state after row n-1 (KV, compressed keys and the conv rings
  are position-indexed, and the rings are large enough that rejected rows
  never overwrite a row still needed).
- Verification samples row j from the full model and continues only while
  the sample equals draft j, so every emitted token is an exact sample from
  the model given the true prefix (at any temperature); greedy output is
  identical to plain decoding (`tests/gpu/test_speculative`).
- Drafting: one MTP pass over the step's kept tokens, then K-1 single-row
  passes chained through the MTP output hidden; argmax per vocab shard on
  the GPUs.

Measured on short prompts: 2.5-2.6 tokens per step, 24.4 ms verification +
0.2 ms accept + 4.7 ms drafting, **86-87 tok/s** single stream (58 plain;
production vLLM with MTP: ~56-65). Through the server, 4 concurrent requests
reach 180 tok/s aggregate.

### MTP off while drafting does not pay (2026-09-25)

The step cost model is measured on this box: 15.0 ms plain, 18.8 / 22.6 /
26.4 ms with 1 / 2 / 3 drafts, i.e. 1.25 of a plain step for the first draft
and 0.25 per further one (`QW_SPEC_BASE`, `QW_SPEC_COST`). A request whose
running acceptance makes every K worth under one token per plain step decodes
plainly: no drafts and no MTP pass. The MTP layer's KV then falls behind, so
decode keeps every row's MTP input (the pre-final-mixer hidden) in a per-slot
ring of 256 positions (`mtp_hist`, 2.6 MB per slot per rank). After a plain
stretch (32 steps, doubling while drafting keeps not paying, at most 176) the
skipped MTP rows are run from the ring (`Engine::mtp_catch_up`, up to 16 rows
per pass) and drafting is retried from a neutral acceptance. A new turn in
the same slot catches up the same way before its prefill.

Through `Session::generate` (256 tokens, greedy, output identical):

| | plain | spec before | spec now |
|---|---|---|---|
| prompt 0 (predictable) | 67.8 tok/s | 100.9 | 100.7 |
| prompt 1 (hard to predict) | 67.5 tok/s | 63.1 | **74.5** |

## Block store: the shared host tier of the prefix cache (2026-09-25)

The first host tier kept whole conversations: a saved entry was reused only
when all of it was a prefix of the new prompt. Two agent conversations sharing
a 12k-token system prompt shared nothing. The block store
(`src/session/block_store.cpp`) replaces it.

- **Blocks.** Prompts are cut into 256-token blocks keyed by a hash chain
  (parent key + tokens; the tokens are compared on a hit, so collisions are
  harmless). A block holds its positions' KV for every rank: K, V, raw and
  compressed indexer keys of the 12 QSA layers and the MTP layer, 5.2 MB per
  rank (`Engine::export_kv` / `import_kv`).
- **Snapshots.** The recurrent state can't be rebuilt from KV, so a prompt
  resumes at a snapshot: the GDN states, conv and PLE rings and MTP input at
  the end of some block (31.5 MB per rank). The last block on a snapshot's
  path may be partial (a leaf), so snapshots sit at any position.
- **Where snapshots come from.** Prefill captures the state at any position
  inside a chunk without splitting it: the GDN scan writes its state after a
  given token, small kernels copy the conv and PLE rings as they stand there,
  and the MTP input is that token's hidden (`Engine::Capture`, at most 4 per
  chunk). The session captures before every chat message start
  (`<|im_start|>`, `qw_set_boundary_token`) at least `QW_SNAP_MIN_GAP` (1024)
  tokens apart, plus every chunk end and the prompt end, and saves each to the
  store as soon as its chunk is done.
- **Resume** = the deepest snapshot on the prompt's path, whoever stored it;
  only blocks the slot doesn't already hold are copied in.
- **Memory.** Pinned buffers come from arenas of ~250 MB per rank, allocated
  on demand and freed when empty (a spare is kept). Pinning was measured at
  ~0.05-0.1 s per arena on 2026-09-25; the pool now logs every pin over 0.5 s, and
  the first start of the 2026-10-01 deploy (108 GB of host memory available, the
  four pools pinning at once) logged 0.73-1.47 s per 251-256 MB arena. So a
  prefill tells the store what its saves will need and the
  arenas are pinned by a background thread while the GPUs work. Budget
  `QW_HOST_CACHE_GB` (default 128); LRU over nodes whose children hold nothing
  in RAM; evicting a snapshot also drops the ancestor blocks no other snapshot
  needs.
- **Disk tier** (`src/session/disk_tier.cpp`, `QW_DISK_CACHE_DIR`,
  `QW_DISK_CACHE_GB`, default 200): every block and snapshot is also written
  as a file by a background thread (`<hash>.qwb`, `<hash>.qws`; temp name +
  rename). The index is rebuilt from the files at startup; a restore reads
  whatever is only on disk. On shutdown the server saves every slot's newest
  snapshot (`qw_persist`).

Measured (`bench/prefix_cache_bench`, `tests/gpu/test_block_store`,
`test_snapshot_capture`, `test_host_tier`):

| | |
|---|---|
| second conversation sharing a 12k-token system prompt | 6.1 s -> **0.34 s** (0.06 s restore + 400 new tokens) |
| first conversation's saves (50 blocks, 3 snapshots) | ~40 ms on a 6.0 s prefill |
| prefill of 16k tokens with 8 captures (the maximum) | -1.3% tok/s |
| restore of 4k tokens from RAM / from disk after a restart | 0.02 s / 0.55 s |

A store restore is bit-identical to restoring the same capture in VRAM. A
mid-chunk capture differs from the state after a prefill split at that
position only by the rounding noise between different chunkings (relative
difference 0.04-0.07, the same as splitting the prefix itself differently).
Decode and speculative decoding are unchanged.

## Adaptive draft count

Each request keeps a running per-draft acceptance a (updated per verified
draft) and drafts K in [1, 3] tokens maximizing expected tokens per step cost:
(1 + a + ... + a^K) / (1 + beta (K - 1)), beta ~0.2 (one more verification row
plus one more MTP pass relative to a step; `QW_SPEC_COST`). K >= 1 because the
first MTP pass also keeps the MTP layer's KV current.

## Platform stability (main-srv, 2026-09-25)

The PCIe path above two of the cards (root port 80:03.1, switch port
83:00.0) logs correctable Data Link Layer AER events in bursts while the GPUs
are busy (87 in 3 hours; the kernel runs `pcie_aspm.policy=powersave`). Every
intermittent failure seen here fell inside such a burst: a rank stuck forever,
GPU memory aperture violations, and non-reproducible output differences in
multi-sequence speculative batches; the same tests pass repeatedly between
bursts. home-infra's tuning notes record the same class of failures for vLLM's
P2P all-reduce and SDMA copies on this box, and the amdgpu/ASPM kernel
parameters they recommend (aspm off, runpm=0, noretry=1, PCIe gen cap) are
not all in place. Engine-side mitigations: copies run as compute-shader
blits (`HSA_ENABLE_SDMA=0`, set by the engine unless overridden); a
collective that times out makes every later wait give up at once and the
engine fails with the waiting rank, missing peer and collective offset; a
dispatch watchdog logs the stuck job and each rank's phase every 60 s.

## Open issue: multi-request speculative batches (2026-09-25)

**Likely cause found (2026-09-26): the GPUs' -75 mV undervolt.** A detector
(`test_speculative --gen 256 --k 5 --repeat 6`: 18 greedy runs per prompt set,
plain vs speculative vs the engine's greedy reference, with the top-2 margin at
any divergence) found wrong tokens at -75 mV in plain *and* speculative paths,
including plain decoding differing from its own earlier run (impossible for
this deterministic engine on correct hardware) and flips of clear picks (top-2
margin 0.94). At 0 mV and -50 mV (same 160 W cap) the same test found none, and
every output stayed bit-identical before and after ~40 min of load:

| offset | detector | prefill | decode M=1/4/8 (ms) | power under load | sclk |
|---|---|---|---|---|---|
| 0 mV | 0 wrong | 2,101 tok/s | 14.70 / 21.79 / 26.89 | 550 W | 2,390 MHz |
| -50 mV | 0 wrong | 2,137 tok/s | 14.93 / 21.84 / 26.94 | 507 W | 2,391 MHz |
| -75 mV | 2 wrong (+3 in an earlier run) | 2,185 tok/s | 14.87 / 21.90 / 27.12 | 497 W | 2,392 MHz |

The undervolt buys no speed (decode is memory-bound, prefill collective-bound,
clocks equal), only power. The notes below predate this finding.

Speculative decoding of 3-4 requests at once (verification batches of 11-16
rows) intermittently goes wrong: tokens that differ from plain decoding, GPU
page faults (a data read past an allocation on one rank, or instruction
fetch faults on all), or a stuck step. Serializing every kernel
(`AMD_SERIALIZE_KERNEL=3`) makes it pass, so it is timing-dependent. Ruled out
so far, each with a test that passes in isolation:

- the per-pair expert kernels vs the tiled ones and an fp32 reference (`tests/gpu/test_moe`);
- `gemv_rows` for every row count and model shape, with stray-write guards (`tests/gpu/test_gemv`);
- the P2P collectives: 150k verified operations at the decode shapes, also
  with up to ~1 ms random skew between ranks (`tests/gpu/stress_comm`);
- out-of-bounds writes into any engine allocation: guard zones after every
  allocation, checked after every job (`QW_GUARD=1`), never tripped;
- data-derived indices: device-side assertions (`-DQW_DEVICE_CHECKS=ON`) never tripped;
- graph replay (also fails without graphs), SDMA vs blit copies, the
  concurrent-push threshold.

Single-request speculative decoding, two requests together, and plain
batched decoding at any size are reliable. Until the cause is found,
verification batches are capped at 8 rows (`QW_SPEC_MAX_ROWS`): one or two
requests draft 3 tokens, three or four draft 1. With the cap the failing case
passed 5/5 (it failed ~6/8 without), and it is also faster at 4 concurrent
requests: 223 tok/s aggregate vs 180-207 with 16-row steps.

## Roadmap

1. **Reference and spec.** Done: CPU fp32 reference (`src/ref`), validated against vLLM.
2. **Kernels.** Done, validated end to end against the reference and vLLM.
3. **4-GPU runtime.** Done: TP/EP, P2P collectives, graph-captured batched decode.
4. **Prefix cache tiers.** Done: per-slot reuse, VRAM snapshots, and the
   block store in RAM and on disk, shared by all conversations.
5. **MTP speculative decoding.** Done, with adaptive K. Next: draft steps inside
   one graph (device-side argmax and embedding lookup).
6. **Prefill kernels.** Done (two micro-batches, optional W4A8); chunked GDN next.
7. **OpenAI-compatible server and tokenizer.** Done (continuous batching),
   serving production through llama-swap and litellm.
8. **CacheBlend experiment.** Done: rejected on quality (see above); exact
   block-level reuse shipped instead.
9. **Performance, next** (details and estimates in "Next steps" below):
   - [~] int8 dense weights: built (+16% decode), fails the quality gate; opt-in `QW_INT8_DENSE=1`
   - [x] collectives: one kernel per small collective (small gain; see Next steps 2)
   - [x] fewer, bigger decode kernels: tried (SwiGLU into down GEMV, ring into scan, shared-expert graph branch), no gain; see Next steps 1
   - [x] one collective fewer per sublayer: not possible with this layout (see Next steps 9)
   - [x] MTP drafting without host round trips: a draft's steps are enqueued back to back, the
         drafted tokens picked on the GPUs (argmax parts all-gathered, `draft_pick`) and their
         embeddings read from the pinned, mapped embedding table (no VRAM); the collectives'
         sequence base advances on the device (`Comm::bump_base`). Same drafts; measured host
         round trip before: 0.27 ms of a 1.63 ms step. Speculative decoding 87 -> 91 and 115 -> 120
         tok/s on the test prompts
   - [x] longer drafts for a request decoding alone: server default `--mtp 5` (was 3); the
         draft count stays adaptive per request. Single stream, 400-token replies: T 0.7 / top-p
         0.8 / top-k 20 100-106 -> 111-112 tok/s, greedy 110 -> 117; `--mtp 7` no better (109-110 /
         116). Concurrent requests are unaffected (the 8-row verification cap leaves them 1 draft)
   - [ ] intermittent: greedy speculative output differing from plain decoding in
         `test_speculative` (different prompt from run to run, 4-12-row verification batches;
         also on the pre-GPU-sampling build), likely the open issue below; to root-cause
   - [~] KV spill: a slot holds more tokens than its VRAM (`EngineOptions::slot_spill`,
         `QW_SLOT_SPILL=0,131072,...`); the rest of its K/V/raw indexer keys lives in pinned host
         memory the kernels read and write directly. Built and measured 2026-10-03/04: bit-exact
         (`test_spill`), no cost without spill, decode +3-5% per spilled row, prefill into the
         spilled part at 97-99% of resident with the per-layer staging. See "KV spill"
   - [~] converted-weights cache: built 2026-10-04, off (`--weight-cache-dir`); a card's weights load in 7-13 s from
         the page cache against 48 s, but 340 s cold from the HDD pool. To enable when the model storage is on SSDs.
         See "Converted-weights cache"
   - [x] PLE n-gram table precision (`core/ple.cpp`: int4, int8 and bf16 layouts, chosen by the
         sidecar's META.json). Against a new fp32 reference with Qwen's original bf16 table
         (4,000 tokens): engine with the int4 table 0.065 mean |dlogprob| / 96.8% top-1, int8
         0.048 / 97.4% (-0.017, 95% CI -0.022..-0.013), bf16 0.044 / 98.0%. The old fp32
         reference (int4 table) is itself 0.064 from the new one: the int4 table was the largest
         single error source. int8: 54 GB of host RAM (int4 30, bf16 96)
   - [x] readiness: the engine returns from its constructor (and the server starts answering
         `/health`, which llama-swap polls) only after the PLE table is in RAM; the table reads
         while the ranks load, and warmup waits for it (a cold start used to serve requests while
         rows still came from disk, warmup 28-35 s)
   - [x] admission: a request that fits no free slot waits without holding up later ones that
         fit the free slots; freed slots are offered in arrival order, so it is not starved
         (`server/tests/test_admission.py`)
   - [x] no freeze behind a lone long prefill: with nothing decoding, a prompt went in whole,
         so a 128.8k-token prompt (63.8k restored from disk, 65k prefilled in ~95 s) kept an
         11-token request waiting until it was done; this looked like a lock-up during disk
         loads, which are what long prompts often start with. Now it goes a chunk at a time,
         yields when a request arrives, and prompts that fit whole in a piece go first
         (`server/tests/test_prefill_order.py`)
   - [~] decode slows during a disk-tier load: steps ~300 ms instead of ~22 ms while 8.6 GB
         loads (requests at 2.6-12.6 tok/s instead of ~65). Re-checked 2026-10-01 against the
         production log since the fp8 deploy (analysis below, "Disk-tier loads and host memory"):
         the one window with a load and running requests had 83 ms/step (0.3 GB load); the
         multi-second decode stalls in that log (5 windows over 500 ms/step, up to 3.6 s) have
         no load running. Found and fixed on the way: the pinned pool could leave a caller
         waiting forever (see next item), a hit that needs nothing from disk was held up by a
         load running for another prompt, and the load log now says where the time goes.
         Measured in production after the deploy of `qw-engine:15822e2` (2026-10-01, a freshly
         started engine with 108 GB of host memory available and an empty RAM cache): a
         replayed 29.8k-token chain from the disk tier (118 entries, 2.6 GB) loaded in 16.04 s
         (162 MB/s: 15.39 s reading, 0.65 s getting pinned buffers), restored in 0.155 s and
         was reported as 29,809 cached tokens; a decode running alongside it kept 79.4 tok/s
         (79.2 alone) with at most 30 ms between tokens. So with memory headroom a load of this
         size does not slow decode; the 83-300 ms/step windows were under memory pressure.
         Not yet measured: a larger load while the host cache is full. Still to run on the dev
         box (needs a GPU window): `tests/gpu/test_disk_load`, `tests/gpu/test_pinned_pool`
   - [x] pinned pool: `get()` could wait forever. A caller woken after another took the arena's
         units went back to sleep without asking for another arena, and `put()` never woke a
         waiting `get()`; with the scheduler and a load thread both taking buffers this left a
         load (or a save) waiting until some other allocation happened, which can look like a
         disk load taking 14 s for 0.3 GB (2026-09-28 18:02). `tests/gpu/test_pinned_pool`
         (8 threads) hangs on the old code and passes 120 of 120 runs now. Emptied arenas are
         also unpinned by the pool's thread instead of inside `put()` (on the scheduler thread,
         under the pool's lock), and a pin or free that takes over 0.5 s is logged
   - [x] `BlockStore::load`: a hit whose entries are all in RAM returns at once even while a
         load runs for another prompt (before: every hit read as "loading" until that load
         was attached, so a RAM-resident prompt waited out someone else's 20-40 s disk read; no
         such case in the production log since 2026-09-28, where requests are mostly alone).
         The load log line now gives the RAM cache, the pinned arenas, the disk writes queued,
         and, at the end, MB/s and the thread time spent getting pinned buffers and reading
   - [~] `QW_LOAD_THREADS` (default 1): load threads reading a load's files in parallel. A
         two-HDD mirror can serve two streams; to measure (needs files colder than the ARC)
   - [~] code review 2026-10-01 (6 of 24 applied the same day: S1, S2, S3, R1, S6, S7): 24 improvement points with where, evidence, fix, effort and a
         status line each, in `docs/CODE_REVIEW.md` (serving path S1-S8, reliability R1-R7,
         performance P1-P3, maintainability M1-M6). Measured ones first: the scheduler rebuilds a
         long prompt's ctypes array on every pass (5 ms at 100k tokens; the event loop that
         streams tokens is then 4.7 ms late per wakeup), tokenizing and rendering run on the
         event loop (4 MB/s), the dashboard shows the fp8 table as 28.8 GB instead of 51.2 GB, the
         tool-call parser mis-splits values containing `<parameter=`
   - [x] KV slots 256k + 128k + 64k + 32k in production: 4096-token prefill chunks
         (`QW_PREFILL_CHUNK`) free ~0.9 GB per card of prefill buffers for the 64k slot
         (prefill -2%); ~1.1 GB per card stays free
   - [x] startup: the four ranks load in parallel, each on its own thread and pool (warm cache
         105.6 -> 33.4 s; `QW_LOAD_SERIAL=1` for the old order)
   - [x] thinking controls (`server/qwserve/prompt.py` `resolve_thinking`, scheduler): this
         model's template takes `reasoning_effort` low / medium / xhigh (default xhigh; it adds
         an instruction to the system prompt) and `enable_thinking`. Requests may send
         `reasoning_effort` (OpenAI / vLLM values none, minimal, low, medium, high, xhigh, max,
         mapped onto the template's three; none turns thinking off; vLLM passes the raw value, so
         high or minimal would make this template raise), `chat_template_kwargs` (win, as in
         vLLM), a thinking-token budget as `thinking_token_budget` (vLLM, -1 unlimited),
         `thinking_budget_tokens` (opencode) or Anthropic's `thinking: {type, budget_tokens}`
         (LiteLLM turns the latter into reasoning_effort for OpenAI-type routes). A spent budget
         appends Qwen's closing text ("Considering the limited time by the user, I have to give
         the solution based on the thinking directly now.\n</think>\n\n", ~24 tokens, fed one
         per step without drafts); budget 0 = no thinking. Defaults: `--reasoning-effort` /
         `QW_REASONING_EFFORT` (xhigh), `--thinking-budget` / `QW_THINKING_BUDGET` (-1).
         `server/tests/test_thinking.py` covers the mapping
   - [x] logs like vLLM's (`server/qwserve/scheduler.py`): a stats line every `QW_LOG_INTERVAL`
         (10) s while busy (prompt and generation throughput, ms/step, running / prefilling /
         loading / waiting, slots and KV use, share of prompt tokens from the cache), one line per
         request (finish, prompt and cached tokens, generated, TTFT, decode speed, slot), errors
         with tracebacks, client disconnects as one line
   - [x] dashboard (`server/qwserve/dashboard/`, served at `/`, which llama-swap links as the
         model's upstream page): totals, charts of prompt processing speed, generation speed, demand, queue and KV
         (an hour of 5 s samples; lines are dotted across idle time; hovering shows the exact values
         at that time). Speed is
         the aggregate of all requests in that phase: tokens over the wall time during which
         at least one request was prefilling (or decoding), so interleaved work counts the
         way the requests experience it (a first version divided by the phase's own GPU
         time and read up to 2x high while prefill and decode alternated); demand is tokens
         over all wall time, which the charts showed first and read low when partly idle,
         KV in the GPUs, per-card VRAM and power, host cache, disk tier, PLE table, averages of
         TTFT, request time and decode speed, recent requests. A collector thread builds the
         snapshot every 2 s; the handler only returns it; figures to one decimal
   - [x] disk-tier loads off the scheduler thread (`Session::prefetch`, `BlockStore::load`): a
         97k-token prompt restored from disk after a restart blocked every request for ~54 s
         (TTFT 53.9 s, other streams stalled). Now the request waits in a loading state while a
         background thread reads the entries into RAM, then restores from RAM. First live run
         (cf08ac4, 80.8k-token prompt, 317 entries, 6.8 GB in 21.9 s): a short request was
         admitted at once (TTFT 0.11 s) but decoded at 1.5 tok/s, because the pinned buffers
         were still taken on the scheduler thread and the pools, empty after a start, pinned
         new arenas for ~14 s. Now the load thread takes them too (1a9510c): the next live run
         (9148b24, 101k-token prompt, 396 entries, 8.5 GB in 30.1 s) served a short request
         in the meantime at TTFT 0.11 s and 40 tok/s, done 0.7 s after the load started
   - [x] host memory: production (bf16 table 102 GB pinned + host cache up to 128 GB, container
         limit 240 GiB of the host's 251) was killed by the host's OOM killer on 2026-09-27
         17:40 (constraint none: the host itself ran out). Needs a smaller host cache budget
         (~96 GB) or the int8 table (54 GB). Again on 2026-09-28 00:26, while a disk-tier load
         pinned new arenas (`kfd_ioctl_alloc_memory_of_gpu` in the OOM trace); engine 90 GB
         anon + 100 GB table. The host's other big users: the vLLM LMCache server in
         llm-backend-amd (32 GB, unused since the vLLM entries were commented out) and the
         `lmcache` incus container (17 GB). At a full cache the sum is ~294 GB of 251. The
         memory pressure (swap 6 of 7 GB used) is also the likely cause of decode slowing to
         ~300 ms/step during disk loads: pinning has to reclaim first. Done 2026-09-28: the
         two LMCache servers stopped (host available 61 -> 109 GB), host cache 128 -> 96 GB
         until the fp8 table is in (then ~170 GB). Third kill 2026-10-01 01:54 with the fp8
         table and the 160 GB cache: the engine had 159 GiB anonymous (the cache full) plus
         the 47.7 GiB table, the `lmcache` incus container was running again (16.8 GiB), page
         cache down to 8 MB, 0.55 GB free on a 251 GiB host. Open: lower `--host-cache-gb`
         (128 leaves ~35 GB of headroom), stop the `lmcache` container again; see the analysis
   - [x] `scripts/prod-idle-unload.sh` also asks the engine: llama-swap's /api/metrics lists
         finished requests only, so a request in flight for over 60 s looked idle and an
         unload cut it off (502, 2026-09-28 00:46)
   - [x] fp8 PLE table from the official `Qwen/Qwen3.8-Flash-Next-FP8`: the 128 n-gram shards
         are F8_E4M3 with one bf16 scale for the whole table (51.2 GB, half of bf16).
         `tools/ple_download.py OUT Qwen/Qwen3.8-Flash-Next-FP8` writes them ready to serve
         (layout `f8e4m3_tensorscale`, decoded through a 256-entry table). Accuracy against
         the fp32 reference: 0.050 mean |dlogprob| (int8 0.048, bf16 0.044; see "PLE n-gram
         table precision")
   - [ ] NVFP4 experts (idea, not started): take only the routed experts from an NVFP4
         checkpoint (`nvidia/Qwen3.8-Flash-Next-NVFP4`; first check what it quantizes, dense
         layers must stay fp16). FP4 (E2M1) with an FP8 scale per 16 weights may round the
         experts closer to the original than AWQ int4 with a bf16 scale per 128, but AWQ's
         activation-aware scaling narrows that; unmeasured. Costs: no FP4/FP8 units on gfx1030
         (a 16-entry lookup and a scale per value, in the decode pair kernel and the prefill
         expert GEMM's LDS dequant; decode experts are ~1 ms of 14.7 ms, so ~+0.1 ms); ~4.5 vs
         ~4.1 bits per weight, experts 62.3 -> ~68 GB, ~+1.5 GB per card (one KV slot shrinks).
         Our fp32 reference also uses the AWQ experts, so first step: download the original bf16
         experts of a few layers and compare AWQ and NVFP4 reconstruction error weighted by
         typical inputs (hours); only if NVFP4 is clearly closer, build the kernels and measure
         against a reference with bf16 experts
   - [ ] MTP drafts over a reduced vocabulary: most of a draft step (1.36 ms) is the lm_head over
         all 248k tokens; drafting over the ~32k most frequent ones would cut that ~8x (~1-1.5 ms
         a step, ~5% estimated). Output unchanged (verification is exact); acceptance may drop
         slightly, so it needs measuring
   - [x] MTP off while acceptance stays low, with a catch-up pass on resume
   - [ ] root-cause multi-request speculative batches over 8 rows
   - [x] host sampling: one pass + parallel rows (16 -> 3 ms per 4-request step, +25% aggregate)
   - [x] GPU-side sampling (`kernels/sampling.hpp`, `QW_GPU_SAMPLING=0` for the host sampler):
         penalties from per-slot token counts on the GPUs, then per vocab shard the max,
         normalizer, top 256 candidates and a Gumbel-max draw; the host draws exactly as the CPU
         sampler from those (same tokens for the same seed with top-k/top-p/min-p, checked in
         `tests/gpu/test_sampling`), and fetches a row's full logits only when a nucleus reaches
         past the candidates (1 row in ~10k in real text, only at temperature 1, top-p 0.95).
         With the chained drafts, through the server (400-token replies, T 0.7 / top-p 0.8 /
         top-k 20): 1 stream 93-96 -> 107-112 tok/s, 4 streams 195 -> 231, 8 streams 180-198 ->
         210; temperature alone 217 -> 230 (4 streams); presence penalty 1.5 191 -> 225
   - [x] prefill profiled: collective-bound; flat pushes +4-10%
   - [x] scheduling A: prefill in pieces interleaved with decode (see "Interleaved prefill")
   - [x] batched prefill: several waiting prompts in one prefill pass (`Engine::prefill_batch`:
         segments of distinct slots as one chunk, row-wise work once, sequence kernels per
         segment; the scheduler packs prompts up to a piece, `QW_PREFILL_BATCH=0` off). 4 prompts
         (3,800 tokens) 2.05 -> 1.62 s; server bursts (2 x 8 concurrent ~300-token prompts, 4
         slots) 112 -> 116-119 tok/s, 4 x 400-token streams 185-195 -> 198-204 tok/s. Less than
         the +13% estimate: with 4 slots, later requests get slots one at a time, so most prefills
         still go alone. Numerically like a different chunking (GEMM row counts and a row's place
         in the GEMM change rounding, ~1e-3 after one layer; `tests/gpu/test_batch_prefill`)
   - [x] more drafts per request with 4 concurrent (QW_SPEC_MAX_ROWS=16): no gain at temperature
         0.7 (1.83 -> 2.35 tokens per request-step but 33.6 -> 41.7 ms per step), so the 8-row cap
         stays and its bug is not worth chasing for throughput
   - [x] int8 dense weights with GPTQ: built and measured (item 1 below); does not reach fp16
         accuracy even with group-16 scales, and the decode gain is small (+13% at M = 1, 0% at
         M = 8), so int8 stays off
   - [~] vision attention kernel: block size by image size (1080p 1.37 -> 1.23 s); a register-blocked redesign would be next
   - [x] vision: HF 3D M-RoPE positions measured; plain positions kept (as good or better)
   - [~] disk-tier write volume (the tier is moving to an SSD; measured 2026-10-01, section "Disk-tier
         write volume"): tens of GB/day (bound: 42 GB/day from the 4.7-day turnover); snapshots are
         47% of the bytes and ~80% of them were never restored from. Done 2026-10-03: the replicated rank
         pairs of a block are stored once (file format v2, -26.5% of all bytes, bit exact, 56 of 56 blocks
         checked, v1 files still read). Open, in order: a bytes-written
         counter; a disk snapshot policy (skip a prompt-end snapshot next to a
         capture: -10% of snapshot bytes at no measured cost; hold writes back and keep one per
         >= 4,096 tokens plus chain ends: -55% for ~0.05 s of extra prefill per request); byte
         shuffle + zstd-1 (-41% together with the pairs); a daily write cap
   - [~] prefill and TTFT in production (measured 2026-10-02, sections "Prefill and time to first token in
         production" and "Saving to the prefix cache and the prefill collectives"). Built and A/B-measured on the dev
         box: a 2-chunk rolling reserve, one RAM copy per KV replica pair and non-coherent arenas are the defaults
         (cold 100k prompt: 54-57 s against 73-93 s, no store 53 s, when the host is in a good state; the extra
         time over no store roughly halves when it is bad, but +49% remains), the collectives' 2D push (+5.7% at 8k
         tokens, +7.6% at 16k, bit-identical, decode untouched) and the input pipeline (+2.9%) are on. Rejected and
         recorded: huge-page arenas, kept arenas, the standing reserve as built (worse during long prefills; off),
         chunk 8192 (same as 4096), an expert permutation (real-text imbalance 1.14x: ~1.8% at most). Open: why a cold prompt still pays +49% in a bad host state (test with the
         n-gram table locked), a standing reserve that refills only while idle, per-request timing in the request
         log, the QSA attention kernel (20.7% of
         GPU time at 60k of context), the SSD move for disk loads (23% of the TTFT time)

## Disk-tier loads and host memory (analysis 2026-10-01)

Question: why do requests slow down while a KV block is loading from disk, and
how to fix it. Everything below is from the production log since the fp8
deploy (`journalctl -t qw`, 2026-09-28 02:30 to 2026-10-01 17:00: 938 stats
windows, 587 requests, 9 disk loads) and read-only looks at the host.

**Where the cache lives.** `/cache` in the engine's docker container is a
docker volume inside the `llm-backend-amd` incus container, whose root is the
ZFS dataset `main-storage/incus/llms` (lz4, 128K records) on pool
`main-storage`: a mirror of two WD40EZRZ HDDs, 87% full (frag 9%), with a
3-way NVMe special mirror, a mirrored NVMe SLOG and two NVMe L2ARC partitions
(1.08 TB cached, 4.5 M L2ARC hits). `l2arc_noprefetch=1`, so ZFS's
read-ahead of a streamed file is never served from the L2ARC (arcstats:
prefetch data 115 k hits, 3.0 M misses). The dev box's root is the same
dataset. The ARC is uncapped (`zfs_arc_max=0`: all RAM), 43 GB now.

**How fast loads are** (all 9 loads in the window; 285 entries = 6.2 GB):

| load | entries | size | time | rate |
|---|---|---|---|---|
| 09-28 15:30 | 1 | 0.1 GB | 0.57 s | 0.2 GB/s |
| 09-28 18:02 | 12 | 0.3 GB | **14.3 s** | 21 MB/s |
| 09-30 01:19 | 33 | 0.8 GB | 0.46 s | 1.7 GB/s (ARC) |
| 10-01 01:37 | 14 | 0.3 GB | 0.19 s | 1.6 GB/s (ARC) |
| 10-01 02:36 | 285 | 6.2 GB | 38.5 s | 161 MB/s |
| 10-01 03:37 | 201 | 4.4 GB | 21.4 s | 205 MB/s |
| 10-01 09:00 | 33 | 0.8 GB | 0.64 s | 1.3 GB/s (ARC) |
| 10-01 09:00 | 4 | 0.1 GB | 0.05 s | ARC |

Cold loads run at an HDD's streaming speed, about 1.7 times faster than
recomputing the same tokens (a 256-token block is 21.8 MB, 136 ms at 160 MB/s,
against ~250 ms of prefill at ~1,000 tok/s); what is in the ARC loads at GB/s.
The 12-entry load that took 14 s is the outlier: 0.3 GB cannot take that long
at any disk speed, and it ran during a prefill-heavy minute (prompt 700-900
tok/s, so a stream of saves being written to the same two spindles) with
three other requests in flight. Two candidate causes, not separated by this
log: the writes (a reader and a writer on two HDDs), and the pinned pool's
lost wakeup (fixed above). The new load log line splits the time into pinned
buffers and reading, which will tell.

**Which requests wait for a load.** The request that needs it: 6.2 GB took 40.5 s
TTFT, 4.4 GB 50 s (21 s load + 31.8k tokens prefilled at ~1,000 tok/s). A
request whose prompt was cached in RAM and arrived meanwhile would have waited
too (single-slot `load()`, fixed above); no request in the window did.

**Decode step time by situation** (stats windows, median ms/step):

| windows | n | median | p90 | max |
|---|---|---|---|---|
| nothing loading, nothing prefilling | 911 | 27.9 | 35.6 | 3,632 |
| a prefill running | 26 | 28.5 | 58.4 | 390 |
| a disk load running | 1 | 83.5 | | |

So in this window of the log a load slowed decode once (3x, a 0.3 GB load) and
there is no sample of the 6 GB loads that happened to run next to decoding. The
multi-second stalls are not loads: 3,633 ms/step (09-29 01:31, one request
decoding, no load), 2,741 (09-28 17:55), 890, 589, 513 ms. Host correctable
PCIe errors (AER BadTLP on the GPU's upstream port 83:00.0 and root port 80:03.1,
110-250 events per day) overlap two of them (09-28 17:41, 09-29 00:00) and not
the others. The earlier 300 ms/step with an 8.6 GB load was measured under the
bf16 table, when the host was swapping and being OOM-killed.

**The 32.5 s image encode** (09-27 19:06, 1,501 tokens, normally ~0.5 s): it
started right after a 1.4 GB disk load finished (19:05:58), on the bf16
configuration that was OOM-killed twice that day. The host kernel log has no
GPU reset, ring timeout or eviction message in that minute. Unexplained; the
same class as the decode stalls, a stall of the whole process.

**Host memory: the third OOM kill (2026-10-01 01:54).** The kernel's report
(constraint none: the host itself was out): anonymous 180 GiB (the engine's
159 GiB: the cache at its 160 GB budget, plus Python and staging), unevictable
47.7 GiB (the mlocked fp8 table), page cache 8 MB, free 0.55 GB, all four
NUMA nodes under their watermarks. `lmcache` (incus container, 16.8 GiB, back
in `incus list` although 09-28 stopped it) and the other containers hold the
rest. The sum is ~233 GiB plus kernel and ZFS on a 251 GiB host: the 160 GB
budget does not fit next to the table and the other tenants. An OOM kill
empties the RAM tier, so the next requests load from the HDDs: the 02:36
request (40 s TTFT) came after this kill; the process that is running now
started at ~14:00 with the cache empty again.

What the accounting misses (budget = bytes held by nodes): a load takes its
pinned buffers before `finish_load` evicts, a prefill pins ahead for all its
saves (`reserve`: ~8 GB for a 100k-token prompt), the buffers of evicted
nodes stay pinned until their disk write finishes (a 100k-token prefill
produces ~14 GB of files at ~140 MB/s, an HDD mirror's write speed), and an
arena is only unpinned when completely empty. So the pinned total runs above
the budget by several GB at the worst times. Setting the budget with that
slack in mind: engine anonymous ~ budget + 11 GB observed.

**Changed (this commit):** the pool fixes, `load()` per-hit needs, the load
log, `QW_LOAD_THREADS`, `tests/gpu/test_pinned_pool`, `test_disk_tier`,
`test_disk_load`. Not changed: production settings and the host (below).

**Recommended, in order of effect:**

1. Host cache 160 -> 128 GB (`--host-cache-gb` in llama-swap's qw command): at
   the observed slack the engine then tops out near 140 GB + 51 GB table,
   ~35 GB under the host. Stop the `lmcache` container (16.8 GiB).
2. Make streamed reads eligible for the L2ARC: `l2arc_noprefetch=0` (module
   parameter, runtime-settable; modprobe.d to keep). Whether the cache files
   are in the L2ARC is not known; with 1 TB cached and 640 GB written to it
   they may well be. Test: restore a cache file written in the last hours and
   watch `l2_hits` / `zpool iostat -v`.
3. Put the disk tier on NVMe (a dataset or filesystem on the NVMe devices the
   L2ARC uses; `--disk-cache-dir`): a cold 6 GB load would take ~3 s, not 38,
   and the writer stops competing with loads for two spindles. The cache is
   disposable, so no redundancy is needed.
4. Engine: bound the pinned total (count in-flight loads, reserve-ahead and
   queued writes in the budget; evict before a load pins), evaluate pausing
   the writer during a load, and a recompute-vs-load choice from the measured
   disk speed. All need a GPU window to measure.

## Disk-tier write volume (SSD endurance; measured 2026-10-01)

Question: the disk tier is moving to an SSD, how to write less. The engine does
not count bytes written, so this comes from the cache directory (6,075 files),
the production log (`journalctl -t qw`, 09-27 to 10-01, 989 requests) and
sampled files, all read-only. Implemented since (2026-10-03): the rank pairs of a block stored once (file format v2,
below); the rest is still open.

**What is on disk** (199.9 GB; the 200 GB budget is full):

| | files | GB | share | file size |
|---|---|---|---|---|
| blocks (`.qwb`) | 5,332 | 106.1 | 53% | 21.3 MB full; 656 partial leaves, 6.5 GB (mean 9.9 MB) |
| snapshots (`.qws`) | 743 | 93.8 | 47% | 125.7 MB (327 captures), 126.7 MB (416 with logits) |

A snapshot is the bytes of six blocks (1,536 tokens of KV): 12% of the files,
47% of the bytes. The oldest last-use time is 4.7 days ago, so the full tier
turns over about every 4.7 days: surviving writes are at most 200 GB / 4.7 d =
**42 GB/day (15 TB/year)**. The log's fresh prefill (prompt minus cached: 0.2 to
1.0 M tokens/day, 0.60 M mean over the three full days) bounds the blocks at 50
GB/day, plus the snapshots; it overcounts, because known content prefilled again
is in it. Both say tens of GB per day, which a 1 TB TLC drive rated 600 TBW
survives for decades. What matters for an SSD is the worst case (a batch of
100k-token prompts writes ~14 GB each) and write amplification on a nearly full
drive, not the average.

**Reads, for comparison.** 116 GB were read from the tier in the same window,
93 GB of it on 09-27/28 (restarts and replay experiments); 09-29 and 09-30 read
0.8 GB. Reads follow a restart or an OOM kill, when the RAM tier is empty, and
are for conversations that are still active.

**Free: the four ranks hold two copies.** In 56 of 56 sampled blocks (10
partial) rank 0 equals rank 1 and rank 2 equals rank 3 byte for byte, and rank
0 differs from rank 2: two KV heads on four cards (Parallelism: "KV head
replicated on 2 cards each"). The 8 sampled snapshots have no equal pair (the
recurrent state is sharded). Storing each pair once removes half of every block
file: -26.5% of all bytes, and a cold block load reads half as much. With a
`memcmp` guard a pair that ever differs is stored whole, and the compare is
also a cheap check of the two replicas against each other.

**Lossless compression** (zstd on real files, one thread, sizes relative to
the file):

| data | lz4 | zstd-1 | byte shuffle + zstd-1 | speed of shuffle + zstd-1 |
|---|---|---|---|---|
| block, four ranks as written | 0.995 | 0.84 | 0.805 (zstd-3: 0.757) | |
| block, one copy of each pair | | 0.420 | 0.401 (zstd-3: 0.379) | 1.1 GB/s compress, 1.8 GB/s decompress |
| snapshot | 0.99-1.00 | 0.90-0.91 | 0.80-0.81 (4-byte lanes) | 1.2 GB/s, 1.9-2.0 GB/s |

lz4, the dataset's setting, saves nothing on this data. Inside a block's rank
buffer: K and V 32% each (shuffled zstd 0.855), the fp32 raw indexer keys 32%
(0.46: their values are exactly representable in fp16, all 425,984 in each of
two sampled blocks), the compressed indexer keys 4% (0.885). Pairs plus
shuffled zstd-1: blocks x0.40, snapshots x0.80, **199.9 GB -> ~118 GB (-41%)**,
bit exact.

**Which snapshots are used.** 416 snapshots carry logits (prompt ends and
prefill chunk ends, 52.5 GB) and 327 are boundary captures (41.3 GB: at
`<|im_start|>`, at least 1,024 tokens apart); 656 sit on a partial-block leaf.
By token prefix, 636 (80.3 GB, 86%) were continued by a later snapshot of the
same chain (median distance to it 1,077 tokens; one conversation left 71) and
107 are the end of their chain. Matching the log's restore positions (792
restores; 854 requests with a cached prefix) to snapshot positions, only
106-140 of the 743 sit at a position that was restored from (11-12% of the
captures, 17-24% of the logits snapshots): 603-637 snapshots, 76-80 GB, 38-40%
of the whole tier, were not read in the up to 4.7 days they have lived. A cached
prefix equal to an earlier prompt's length: 156 requests; 5 tokens short of it:
67 (consistent with the chat template dropping `<think>` from the history turn,
so the next prompt diverges there and the boundary capture is its deepest
snapshot; not checked against the prompts).

Policies on that structure: a request restores from the deepest kept snapshot
on its path, every position the log shows in use is charged the distance to the
nearest kept ancestor at 2,100 tok/s, and a snapshot dropped from the disk is
taken as unavailable (pessimistic: the RAM tier would serve most of these):

| disk policy | snapshots | GB | saved | extra prefill over 989 requests |
|---|---|---|---|---|
| all (today) | 743 | 93.8 | | |
| skip the prompt-end snapshot when a capture is <= 16 tokens before it | 669 | 84.4 | 10% | 0 |
| chain ends + one per >= 4,096 tokens | 335 | 42.3 | 55% | 113 k tokens, 54 s (0.05 s/request) |
| chain ends + one per >= 8,192 tokens | 242 | 30.5 | 67% | 303 k tokens, 144 s (0.15 s/request) |
| chain ends + one per >= 16,384 tokens | 186 | 23.5 | 75% | 774 k tokens, 369 s (0.37 s/request) |
| chain ends only | 144 | 18.2 | 81% | 3.7 M tokens, 1,773 s (1.8 s/request) |

"Chain ends" are the deepest snapshot of a chain and those within 16 tokens
before it. As a write policy this needs writes held back: a snapshot goes to
disk when its chain has been idle for a while, when the RAM tier evicts it, or
at shutdown, and is dropped when a later snapshot of the chain arrives first
and is not far enough from the last written one; the partial block under a
dropped snapshot is not written either (6.5 GB of the tier). A crash loses at
most the held-back snapshots; the RAM tier serves them meanwhile.

**Plan, by value for risk** (none implemented):

1. Count bytes written and evicted (blocks and snapshots) in the stats and the
   dashboard, and read the SSD's SMART written-bytes counter after a few days:
   the figures above are bounds.
2. Store the replicated rank pair of a block once (new magic, old files stay
   readable, `memcmp` guard): -26.5%, and half the bytes on cold block loads.
3. Disk snapshot policy: skip a prompt-end snapshot next to a capture (free),
   then the held-back, thinned write (>= 4,096 tokens apart: -55% of snapshot
   bytes for ~0.05 s per request, priced pessimistically).
4. Shuffle + zstd-1 on blocks and snapshots (a libzstd dependency; ~1.8 GB/s
   decompression per thread, so `QW_LOAD_THREADS` >= 2 on NVMe).
5. A daily write cap as a guard for batch traffic (blocks before snapshots).
6. The SSD itself: keep 20-30% of it free or unpartitioned and TRIM it; no
   mirror (the cache is disposable and a mirror doubles the writes); `atime=off`.
   The engine writes whole files sequentially and never fsyncs, the easy case.
   On ZFS: recordsize 1M; lz4 gains nothing here.

Rejected: lossy encodings (fp8 KV, int8 snapshots): a cache hit would give
different outputs from a miss, and the accuracy rule keeps lossy off.

## Prefill and time to first token in production (measured 2026-10-02)

*Update 2026-10-03: plan item 1 (per-request timing in the request log) is built, see "Request timings" at the end of this section.*

Question: what can lower TTFT or raise prompt throughput. From the production
log (`journalctl -t qw`, 09-25 to 10-02: 975 requests with a TTFT, 1,275
ten-second stats windows), the cache directory and read-only probes of the
host. The GPUs belong to production, so there is no new GPU profile: the
compute-side items are candidates to measure, not results.

**Reading the log.** A request line has prompt, cached, generated, TTFT and
decode speed; fresh = prompt - cached is what was prefilled. The stats lines
average over 10 s, so their "prompt tok/s" understates the speed while a
prefill runs. A per-request speed (fresh / TTFT) is taken from requests that
did not overlap a disk load or another request's prefill (start = log time -
TTFT - decode time, give or take 1 s).

**Where it stands.** Requests with 1-8k fresh tokens, no load, no overlap:

| period | n | rate p25 / p50 / p75 (tok/s) | under 450 tok/s |
|---|---|---|---|
| bf16 table (before 09-28 00:31) | 51 | 398 / 808 / 1,563 | 25% |
| fp8, before the 10-01 19:11 deploy | 174 | 797 / 1,563 / 1,760 | 14% |
| fp8, `qw-engine:15822e2` (10-01 19:11 on) | 32 | 1,755 / 1,869 / 2,042 | 3% |

Ordinary prefills now run at the benchmark speed (~1,900-2,200 tok/s); the slow
tail went away with that deploy (the pool fix and the scheduler fixes; the log
cannot say which, and the sample is small and the traffic lighter). Since the
deploy: 183 requests, TTFT p50 0.53 s, p75 1.5, p90 2.3, p99 46, 346 s in all.
Where those 346 s went: disk loads 23% (an 81k-token restore from the HDDs:
46 s), one cold 105k-token prompt 30% (104.5 s), waiting behind another
prefill 10%, everything else 37%.

**Context length costs little up to 100k.** Clean requests with 1-16k fresh
tokens, per fresh token (p25 of each context bin): 0.51 ms at no context
(1,965 tok/s) plus 0.16 ms per 100k tokens of context: 1,500 tok/s at 100k,
1,290 at 170k. So the QSA indexer and attention are not what limits a typical
agent turn. The medians of the bins above 100k are much worse (1.0 ms/token at
100-130k, 1.6 at 130k+, p75 2.3): not attention arithmetic but the slow tail of
the pre-deploy engine and the next finding.

**A cold long prefill is limited by pinned-memory allocation** (consistent with
the log, not yet proven by an A/B). 10-02 20:56, 104,856 tokens, nothing cached,
nothing else running: 104.5 s (1,003 tok/s); the fit above gives ~62 s. The log
has 51 arena pins slower than 0.5 s in that window (13.1 GB; latency mean 2.8 s,
max 9.3 s, growing as they queued): the pool delivered ~126 MB/s, while a
prefill at full speed makes ~200 MB/s of new blocks and snapshots that need
pinned buffers (21.8 MB per 256 tokens, 126 MB per 4k chunk). It happens while
the RAM tier grows (an empty tier after a restart, or a burst past what
evictions free; at the budget evicted buffers are reused, except those the disk
writer still holds). Measured on the host: `hipHostMalloc` takes 0.7-1.5 s per
256 MB arena at best and several seconds with the four ranks' KV and snapshot
pools pinning at once; in the same container `mmap` + `mlock` of 256 MB takes
0.15-0.26 s with 4 KB pages and 0.03-0.05 s with 2 MB pages (`MADV_HUGEPAGE`;
THP is in `madvise` mode); the IOMMU is in passthrough (`iommu=pt`), so DMA
translation is not it. On a workstation with THP always on and one GPU every
allocation variant takes 10 ms, so a local microbenchmark cannot show it; the
test needs the dev box with four GPUs. No effect on other requests' decoding was
found (since the deploy 0 of 39 decode windows over 100 ms/step; windows with a
pin event: median 31 ms against 25).

Options: (a) arenas from `mmap` + `MADV_HUGEPAGE` + `hipHostRegister` (expected
4-7x cheaper); (b) reserve per chunk, two chunks ahead, instead of the whole
prompt at once (49 pins started together for 105k tokens), and keep emptied
arenas and a standing reserve; (c) store the replicated rank pairs once in RAM
too (see "Disk-tier write volume": -26% of the pinned bytes and half the D2H
export); (d) pin the budget at start-up in the background (the memory is
committed early). Expected: a cold 100k prompt in ~60 s instead of ~104 s, and a
faster refill of the cache after every restart. Test: the same 100k prefill with
a pre-pinned pool against today's, on the dev box.

**Result (same day, see "Saving to the prefix cache and the prefill collectives"):** the mechanism holds, the first guess
at the fix did not: huge pages are slower here, pins hold the process's memory-map lock and slow the GPUs' own work (no
caller waited for a buffer), and what helped was pinning less and later (the rolling reserve, rank pairs stored once).

**Checked and not worth it** (same log and the cache directory):

| idea | what the data says |
|---|---|
| Re-render the previous generation identically so the next turn does not prefill it | 272 requests continue an earlier prompt; the previous generation explains at most 7% of the fresh tokens (0.19 of 2.80 M), and it needs the reasoning echoed by the client or kept by the server (a prompt change) |
| Finer snapshots, so a diverging request resumes closer | 45 prompt pairs share >= 1,024 tokens: 29 diverge exactly at a snapshot; what is re-prefilled although cached is 1% of the shared tokens (8 s in all) |
| Faster restore import | median 12 ms for slow and fast requests alike; a fully cached 105-141k-token prompt answers in 0.12-0.28 s, which bounds render + tokenize + scheduling |
| Decoders alongside | 1,382 against 1,563 tok/s (median, 0.75-1.5 decoders against none): -12% |

**Compute side: candidates, each needs a GPU window.** The 8k-token profile
(2026-09-25) has collectives at 42% of GPU time, QSA attention 19%, GEMMs 12%,
MoE 12%, GDN scan 5%; there is none at 4k chunks or long contexts. The
collectives are not byte-bound: a chunk receives ~0.95 MB per token per rank
(19.4 KB per row per layer: two all-gathers and two reduce-scatters of 3.8 KB,
the HC all-reduces), ~3.9 GB per 4,096-token chunk, 1.9 GB/s at 2,000 tok/s,
against 14-25 GB/s for a kernel push between a pair; a chunk has ~800 of them
(8 per layer per micro-batch), and the notes' 10.6 MB all-reduce in 4.1 ms is
3.9 GB/s per rank.

1. The in-engine rate of the large gathers and reduce-scatters (`test_comm`
   sweep at 2-16 MB; block count; store width). A push serves one destination at
   a time with a system fence after each, and the receiver then copies the
   payload once more out of the uncached staging buffer. Ceiling ~35 of the 42
   points; realistic 10-20% of prefill.
2. MoE load imbalance under expert parallelism: max/mean tokens per rank per
   layer from the routing counts. Ranks wait for the slowest at the next
   collective, which the profile books as collective time. Above 1.2, a
   per-layer expert permutation is exact and costs nothing at run time.
3. Chunk and piece size: interleaved pieces are 2,048 tokens while others
   decode (the chunk is 4,096): the rate at 1k/2k/4k/8k for the production
   shapes (the notes only have 8k against 4k, +2%).
4. Three or four micro-batches instead of two.
5. The host between chunks (embedding conversion, PLE gather, the saves): ~1-3%
   by estimate.
6. The QSA attention kernel (19% at 8k, +31% per token at 100k).
7. SDMA for the big payloads (frees the CUs; SDMA wedged a rank once).

**Plan, by value for effort:**

1. Per-request timing in the request log line (queue, load, pinned-buffer wait,
   restore, prefill, first token): no GPU needed, and it makes the pinning
   finding provable in production.
2. The SSD for the disk tier with the rank pairs stored once ("Disk-tier write
   volume"): disk loads were 23% of the TTFT time since the deploy.
3. The pool changes (a)-(c), A/B on the dev box.
4. One GPU window for the compute candidates: `prefill_bench` at chunk
   1k/2k/4k/8k with and without micro-batches; a context option for
   `prefill_bench` (prefill N tokens, then time a chunk); `rocprofv3` of one 4k
   chunk at 60k context; routing counts per rank per layer.

## Saving to the prefix cache and the prefill collectives (built and measured 2026-10-02)

**Decision (the owner, 2026-10-02):** the RAM and disk cache stays on even where saving costs more than recomputing
would, because it saves energy: a recompute of a 64k-token prompt runs the four GPUs at ~620 W for 34 s (21 kJ), a
restore is a copy; an SSD for the disk tier is planned. This section is about making saves and loads cheaper; "no
store" below is the floor the overhead is measured against, never a candidate.

Measured on the dev box with production stopped, through the real Session and block store
(`bench/cold_prefill_bench.hip`: a cold 64k or 100k-token chat-like prompt, 4,096-token chunks, a store budget that holds
it unless stated, a fresh store for every run, the start of the configuration order rotating between repetitions; GPU
energy from the cards' power sensors, GPUs only, not the host). `bench/run_suite.sh` reruns every number in stages
(`correctness`, `pool`, `pool2`, `pool3`, `pool4`, `compute`, `stall`, `final`). Candidates were built behind switches and
kept only where the numbers pay.

| switch | what it does | code default |
|---|---|---|
| `QW_RESERVE_AHEAD=N` | pinned buffers for the saves of the next N chunks, taken at the start of a prefill and after every chunk (0: the whole prompt at once, as before) | 2 |
| `QW_KV_PAIRS` | a block's KV kept once per replica group of ranks: -24% RAM per block, half the device-to-host export | on |
| `QW_KV_PAIR_CHECK=N` | also exports the replicas of every Nth shared block and compares them (counts mismatches) | off |
| `QW_POOL_ARENA` | `noncoherent` (`hipHostMallocNonCoherent`) or `malloc` (default flags: coherent, fine-grained memory) | noncoherent |
| `QW_POOL_SERIAL=1` | one arena pinned at a time in the whole process | off (no clear gain) |
| `QW_POOL_RESERVE_GB=G` | a standing reserve of pinned memory, refilled after 1.5 s without a request; the Session waits up to 30 s for it before the server reports ready | 0 (harmful during long prefills, below) |
| `QW_COMM_PUSH2D` | strided and tiny-row collectives pushed by all threads over a flat (row, column) range, for 64 rows or more | on |
| `QW_PREFILL_PIPELINE` | the next chunk's host inputs prepared while the GPUs run this one | on |
| `QW_PROFILE=1`, `QW_MOE_STATS=1` | diagnostics: host/GPU split of a prefill, routing balance per layer | off |

### Pinning, and what a cold prefill pays for saving

**The host.** Its four NUMA nodes are full of page cache and ARC (`numactl -H`: 1.6 and 3.7 GB free of 64 GB on two
nodes, 5-11 GB each at the end of the window) and the kernel's counters since boot show a machine that lives in reclaim:
248k allocation stalls, 39.5 M pages scanned directly, 614k compaction stalls of which 93% failed, 39% of
transparent-huge-page faults falling back to 4 KB pages. Getting pages for a pin means reclaiming them first, so the
cost of pinning is the host's memory state, and that state drifts: the same configuration took 173 s and 37 s of pin
time in two runs, the first store run after an engine load was the slow one twice, and the whole machine was
much slower late at night than early in the window (below).

**Pinning in isolation** (`bench/pin_bench.hip`, four GPUs idle, 256 MB arenas):

| way | 1 thread | 4 threads | 8 threads | small mmap/munmap on another thread while pinning |
|---|---|---|---|---|
| `hipHostMalloc`, default flags | 0.050 s | 0.083 s (max 0.41) | 0.287 s (max 2.0) | 45 ms (8 threads: 170 ms) |
| `hipHostMalloc`, non-coherent | 0.045 s | 0.127 s (max 0.68) | 0.216 s (max 1.4) | 44-58 ms |
| `mmap` + `hipHostRegister`, 4 KB pages | 0.353 s | 1.257 s | 2.232 s | 315-357 ms |
| `mmap` + `MADV_HUGEPAGE` + register | 0.089 s | 0.479 s | 0.886 s | 129-200 ms |

Pins serialize (at 8 threads the aggregate rate falls) and every pin holds the process's memory-map lock for its
length: a thread that maps or faults memory waits for it. Huge pages do not help here (compaction fails) and
registering memory is slower than allocating it, so the guess of the previous section (huge-page arenas, "4-7x
cheaper") was wrong and the arena kind was removed from the code. The default allocation is coherent, fine-grained memory
(a 64 MB host-to-device copy from it ran at 1.1 GB/s, 13-15 from non-coherent memory; the engine's many small restore
copies reach 4-5 GB/s per rank from either), which is why the non-coherent kind is the default. With the GPUs busy
computing, pinning itself is not slower (`stall` stage: 0.047 s per arena at one thread, 0.09 s at eight, probe stalls
~45 ms); its cost to a prefill shows only in the runs below.

**Cold prefill through the store.** "old" = the behaviour before today (the whole prompt's buffers reserved at once, one
KV buffer per rank, default-flag arenas); "defaults" = non-coherent arenas, two chunks ahead, one KV buffer per replica
group (bench name `nc+pairs`; `ahead2+pairs` is the same with default-flag arenas). Seconds, min / median / max over the
repetitions; kJ is GPU energy; pins = arenas pinned, pin s = pool-thread time in `hipHostMalloc`; no caller ever waited
for a buffer in any run (`waits` 0): the pins slow the GPUs' own work.

Early in the window (host in its good state), 64k tokens, four repetitions (`pool2`):

| configuration | s | mean vs no store | kJ | pins | pin s | RAM GB |
|---|---|---|---|---|---|---|
| no store | 34.01 / 34.19 / 34.21 | | 21.3 | 0 | 0 | 0 |
| old | 34.99 / 35.41 / **42.91** | +8.8% | 21.9 | 60-67 | 16-81 | 12.3 |
| defaults | 34.70 / 34.85 / 35.37 | +2.2% | 21.6 | 42 | 8.5-12.6 | 9.3 |
| default-flag arenas (`ahead2+pairs`) | 34.66 / 34.75 / 34.97 | +1.9% | 21.6 | 42 | 8.1-14.4 | 9.3 |
| one pin at a time (`serial+ahead2+pairs`) | 34.59 / 34.95 / 36.82 | +3.3% | 21.6 | 42 | 2.3-2.8 | 9.3 |

and 100k tokens, two repetitions: no store 52.9 / 53.2 s (33.3 kJ); old **93.0 / 73.5 s** (+57%, 37.3 kJ, 95 pins,
106-248 s of pin time, the GPUs at 400-490 W instead of 620); `ahead2+pairs` 57.5 / 54.4 s (+5.5%, 34.0 kJ, 60 pins,
11-24 s). The old behaviour reproduces production's slow cold prompt (105k tokens: 104.5 s against ~62 s expected).

Later in the window (23:00 to 02:00; the host's free memory per node had shrunk, and the numbers are much worse for
everything that stores):

| run | no store | old | defaults / `ahead2+pairs` | other |
|---|---|---|---|---|
| 100k, 3 reps (`pool4`) | 52.8 / 53.0 / 53.3 s, 33.3 kJ | not run | **72.6 / 76.9 / 87.2 s** (+49%), 36.5 kJ, 60 pins, 74-127 s pin time; `ahead2+pairs` 74.3 / 79.7 / 81.4 | serial pins 73.1 / 74.0 / 79.8 (pin time 23-30 s); standing reserve 10 GB 100.1 / 100.2 / 103.4 (94 pins) |
| 64k, 4 reps, 6 GB budget (`churn_64k`: arenas given back and pinned again) | 33.6 / 34.1 / 34.3 s, 21.2 kJ | 63.2 / 76.0 / 78.4 s (+115%), 25.6 kJ, 65-70 pins | `ahead2+pairs` 35.8 / 46.0 / 46.6 (+26%), 22.4 kJ, 32 pins | arenas kept: old+keep 69.7 / 81.1 / 84.4, `ahead2+pairs+keep` 36.8 / 44.8 / 44.9: no effect |
| 64k, 4 reps (`reserve_64k`) | 33.9 / 34.1 / 34.4 s, 21.3 kJ | 70.9 / 78.3 / 84.9 s (+124%), 25.7 kJ | `ahead2+pairs` 49.8 / 54.3 / 54.9 (+55%), 23.5 kJ | standing reserve 10 GB 55.5 / 61.3 / 64.5 |

What to take from it, honestly:

- The new defaults always beat the old behaviour, in every stage, by a wide margin when the host is bad (the extra time over
  no store roughly halves: 64k +42 s -> +20 s in `churn_64k`, +44 s -> +20 s in `reserve_64k`), and cost 2-5% when it
  is good. They also pin fewer and lighter (42 against 60-67 arenas at 64k, 9.3 against 12.3 GB).
- They do not remove the cost in a bad state: +49% over no store for a cold 100k prompt late at night, +3.2 kJ (+10%) of
  GPU energy. A cold prompt that long is one the cache would have saved the next time anyway (a 100k recompute is 33 kJ).
  Saving costs 0.3-3 kJ per cold prompt against 21-33 kJ for the recompute, so the cache pays back from a hit rate of
  1-10% of what is saved (GPU energy only; the host's CPUs and memory are not metered; 272 of 975 production requests
  continued an earlier prompt).
- The slowdown is not proportional to the time spent in `hipHostMalloc`: one pin at a time cut pin time to a quarter (23-30
  s against 74-127 s) and the 100k prompt by about 4% (medians 74.0 against 76.9 / 79.7, three repetitions: not
  distinguishable), so the rest comes from the memory state itself (reclaim and compaction work competing with the
  threads that drive the GPUs). A suspect for the dev box only, untested: its user may not lock memory (`ulimit -l` 8 MB,
  "mlock refused"), so the n-gram table is page cache, and pinning 14 GB under node pressure can evict part of it, making
  every token's gather a disk read; production locks the table, so its numbers may be better. Test: a box with
  `ulimit -l unlimited`, the same 100k prompt with the table locked.
- The standing reserve (`QW_POOL_RESERVE_GB`) makes long prefills worse (100 s against 77 s at 100k; 61 s against 54 s at
  64k) and stays off. It pins 10 GB in the first seconds, and its "refill after 1.5 s without a get()" test is met between
  the chunks of a prefill (a chunk takes ~2 s), so the refill pins while the GPUs compute. A fix would refill only while the
  engine is idle (the Session knows). What it does help is a disk load: 13 entries (0.4 GB) took 0.31 s (800 MB/s) with the
  reserve against 1.6 s (225 MB/s) without, 1.5-4.5 s of thread time waiting for buffers against 0.3-0.5 s.
- Keeping emptied arenas (`QW_POOL_KEEP`, built and measured in `churn_64k`) changed nothing and was removed.

**Replica sharing.** Two KV heads on four ranks: ranks (0,1) and (2,3) compute the same K, V and indexer keys from the
same inputs, and the bytes are identical (56 of 56 disk blocks incl. a partial one in `tests/gpu/test_kv_replicas`; no
mismatch in a 64k prefill with `QW_KV_PAIR_CHECK=1`; snapshots have no identical pairs). The store keeps one buffer per
group (`Payload` aliases the replica to the primary), exports it once (`export_kv` skips an aliased rank), and reads a
disk file's replica bytes by skipping them; the disk files followed on 2026-10-03: a block written from grouped buffers is a v2 file ("QWBLOCK2") with one buffer per
group, 10.9 MB less than a full 21.8 MB block (-50% of its KV bytes; -26.5% of all bytes of the tier, snapshots being unchanged); v1 files
stay readable (index scan and reads), and a store that keeps a buffer per rank (`QW_KV_PAIRS=0`) still writes v1 and
reads a v2 file by copying the group's bytes to the replicas (`tests/gpu/test_disk_tier`).

**Rolling reserve.** `Session::prefill_range` asked the store for the pinned memory of the whole prompt at the start (105k
tokens: 49 arenas at once). Now it asks for the next two chunks' worth at the start and after every chunk
(`QW_RESERVE_AHEAD`): the first saves find buffers, the rest are pinned in step with the prefill. The interleaved path
reserves per piece, as before. A cancelled request no longer leaves memory pinned for saves it never makes.

### The collectives, and the other compute candidates

**Collectives** (`bench/comm_bench.hip`, the engine's `Comm` at a 2,048-row micro-batch, i.e. a 4,096-token chunk; raw P2P
ceiling from plain push kernels: ~14 GB/s per GPU for 2.6 MB messages, 22-24 for 8 MB):

| collective | plain | 2D push | per rank sent |
|---|---|---|---|
| all-gather of the block input (fp16 640 -> 2,560) | 0.454 ms (17.3 GB/s) | 0.451 ms | 7.9 MB |
| reduce-scatter of the block output (fp16) | **1.307 ms (6.0 GB/s)** | **0.518 ms (15.2 GB/s)** | 7.9 MB |
| all-reduce of the HC down projection (fp16) | 0.491 ms (8.2 GB/s) | 0.477 ms | 4.0 MB |
| all-reduce of the HC sums of squares (16 B rows) | **0.914 ms** | **0.046 ms** | 0.1 MB |

The all-gather is at the link's rate. The other two were far below it: the reduce-scatter's source rows are 5,120 B apart
with a 1,280 B slice per destination, and the push looped over rows with a 256-thread block per 80 16-byte columns (69%
of the threads idle); the sums of squares are 2,048 rows of one 16-byte store, which the small-payload path gave to one
block per destination, one row at a time. `Comm::Tuning::push2d` copies a flat index over (row, column) pairs with every
thread instead: bit-identical results (`tests/gpu/test_comm_variants`: 27 MB of outputs per rank, all collectives,
256-2,048 rows). It applies from 64 rows, so decode keeps its row-block push (with it also on for 4-8 rows, the 4-slot decode
step read 21.12-21.38 against 21.00-21.03 ms, perhaps +0.8%, so it was restricted). A vectorized receive changed nothing
and was removed.

**End to end** (`prefill_bench`, chunk 4,096, best of 3, two alternating repetitions of each variant):

| tokens | plain | 2D push | gain | GPU energy |
|---|---|---|---|---|
| 8,192 | 2,082 / 2,064 tok/s | 2,188 / 2,193 | **+5.7%** | 2.45-2.49 -> 2.32-2.33 kJ |
| 16,384 | 2,001 / 1,996 | 2,145 / 2,158 | **+7.6%** | 5.10 -> 4.78 kJ |

Decode (`test_batch_decode`, ms per step at 1 / 2 / 4 / 8 rows): 14.75-14.79 / 16.66-16.79 / 21.00-21.03 / 26.37-26.57
plain against 14.75-14.79 / 16.63-16.75 / 21.12-21.38 / 26.50-26.53 with it applied to every row count. The greedy
determinism run (`test_speculative --gen 256 --k 5 --repeat 6`, two prompts) gives identical tokens in all repeats with
and without it.

**Prefill input pipeline** (`QW_PREFILL_PIPELINE`): the embeddings and fp16 PLE rows of the next chunk are prepared on a
helper thread while the GPUs run the current chunk (they depend on the tokens only), instead of between chunks with the
GPUs idle. 16,384 tokens: 2,155 -> 2,219 tok/s (**+2.9%**, host inputs 320 -> 75 ms, 4.77 -> 4.69 kJ). Bit-identical logits
and next decode steps, with captures in the middle (`tests/gpu/test_prefill_pipeline`).

**Chunk size** (2D push on): 2,048 / 4,096 / 8,192 tokens give 2,057 / 2,188 / 2,197 tok/s at 8k and 1,998 / 2,153 / 2,157
at 16k: 4,096 and 8,192 are the same (the old note of +2% for 8,192 is gone with the better collectives), 2,048 costs 6%
(the price of the 2,048-token pieces that interleave with decoding). Production stays at 4,096.

**Long context.** 4,096 tokens after 60,000 of context run at 1,935 tok/s (2,188 with none: -12%); the 60,000 tokens
themselves took 29.3 s (2,047 tok/s). The kernel trace of that run (all four ranks summed, 255 s of kernel time): the
collectives 34% (waiting for the peers 20.2%, pushes 11.8%, small exchanges 1.0%, receives 1.0%), QSA attention 20.7% (it
grows with context), rocBLAS GEMMs 16.6%, routed experts 15.1%, GDN scan 4.8%.

**MoE balance** (`QW_MOE_STATS=1`): the busiest rank's expert tiles over the mean's, per layer summed. With a fixed
arithmetic sequence of token ids: 1.21 for 4.4k tokens, 1.35 over four chunks of 16k (1.44 in token pairs; worst layer 1.93),
1.33 over the 60k run. With real text (60,655 token ids of the docs and engine sources, `prefill_bench --tokens-file`,
`tools/text_tokens.py`): **1.137** over four chunks of 16k (1.19 in token pairs; worst layers 40: 1.41, 31, 15: 1.26). That is
the factor the MoE phase (15% of GPU time) and the waiting at the next collective grow by: perfect balance would save at
most 15% x (1 - 1/1.137) = ~1.8% of prefill, and only the part of the imbalance that is the same on every prompt can be
removed by a static expert permutation. Not worth building.

**The defaults as built** (`final` stage, after the rebuild; 2D push and pipeline on): prefill 8,192 tokens 2,243 tok/s
(2.32 kJ), 16,384 tokens 2,221 tok/s (4.67 kJ, 285 mJ per token; 2,001 before this work, +11%); decode 14.73 / 16.60 /
21.03 / 26.42 ms per step at 1 / 2 / 4 / 8 rows (unchanged); the 6-repeat greedy determinism run, the pool, comm-variant,
pipeline, block-store and disk-load tests all pass.

**Tried and not kept:** huge-page arenas (slower to pin, and the registered mapping cost the other threads more: probe
stalls 129-200 ms against 45); `hipHostRegister` of ordinary memory; keeping emptied arenas; the vectorized receive in the
collectives; the standing reserve as built (above; off, not removed).

**Open:** why the cold prefill still pays +49% over no store when the host is bad (test with the n-gram table locked);
a standing reserve that refills only while the engine is idle; per-request timing in the request log (queue, load,
pinned-buffer wait, restore, prefill, first token); a v2 disk format with the replica pairs stored once; the QSA attention kernel (20.7% of GPU time at 60k of context).

### Request timings (2026-10-03)

The request line of the log now says where the time before the first token went, after the old fields (which keep their
meaning and order, so scripts that read them still work):

```
request ab12: stop, prompt 12000 (11264 cached), generated 300, ttft 0.80 s, decode 71.2 tok/s, slot 1 | queue 0.01 s, load 0.00 s, restore 0.02 s, prefill 0.70 s (saves 0.04 s, pin wait 0.00 s), interleaved 0.05 s
```

- `queue`: arrival until a slot was taken (the `ttft` field counts from the slot, as before).
- `load`: the cached prompt loading from disk in the background (slot taken until its restore starts).
- `restore`: copying what the caches hold into the slot (`Session::begin_prompt`).
- `prefill`: the engine's prefill calls the request was part of (a batched pass counts in full for each prompt in it),
  including `saves` (the block store's export and queueing of the new blocks and snapshots, inside the chunk callbacks)
  and in them `pin wait`, the time the saves waited for an arena to be pinned (per thread, so other requests' waits are
  not mixed in).
- `interleaved`: the rest from the start of the restore to the first token: decode steps and prefills of other requests
  that ran in between, and the first token's sampling.

`Session::slot_timing`, `qw_get_slot_timing` and `Engine.slot_timing` carry it; `server/tests/test_request_log.py` checks
the line (a disk load shows as `load`). Not in the line: the disk load's thread time getting buffers (it is in the load log
line) and anything after the first token.

## Vision: where image time goes (2026-10-01)

CPU preprocessing in the server (`server/qwserve/vision.py`, PIL + numpy,
off the event loop) for a 1080p screenshot to 1920x1088 (2,040 tokens, 50 MB of
fp32 patches), measured on a Ryzen 5 9600X (the server's EPYC is slower per
thread): decode 3-24 ms (JPEG 3.5, PNG 13-24), bicubic resize 6.5-18,
normalize 15.5, two-frame copy 2.4, patchify 7.3, sha256 digest 21, total
56-90 ms against 1.23 s of vision tower on one card (720p: 26 ms against
0.34 s). Inside `VisionEncoder::encode` the scalar fp32 to fp16 conversion of
the 12.5 M patch values takes 41 ms on the same CPU, before the first kernel
starts. So the tower is the cost, not the Python. Facts that bound the options:

- attention is ~80% of the tower at ~20% of the packed-dot peak (the
  roadmap's register-blocked redesign: at 50% the 1080p tower would take
  ~0.7 s);
- a single image uses one card of four; the encode is synchronous on the
  scheduler thread, so every other request waits for it (1.2 s per 1080p
  image, longer for several slices or a video);
- `comm.hpp` already has an all-gather: a sequence-parallel encode (each card
  takes a quarter of the rows through the GEMMs and the queries, K and V
  gathered per layer, 9.4 MB per card per layer at 1080p) should cut a single
  1080p image to ~0.45 s with identical attention results; an estimate, not a
  measurement;
- the first image of a size allocates the workspace (~330 MB per card at
  1080p, with a `hipFree` that waits for the device).

Open, in the order I would try them (all need the GPUs): the attention kernel,
the sequence-parallel single-image encode, layer-sliced encoding between
decode steps so other requests are delayed by ~50 ms rather than ~1.2 s, a
vectorized fp16 conversion.

## PLE n-gram table precision: int4, int8 or bf16 (2026-09-27)

The PLE n-gram table (layer 1, docs/MODEL.md "PLE") is 320,001,536 rows of 160
values, read from host RAM one row per n-gram head per token: its precision
costs RAM, not GPU memory or speed. The AWQ checkpoint does not include it; the
int4 sidecar used until now came separately. Qwen's original bf16 table is in
`Qwen/Qwen3.8-Flash-Next` (tensors
`model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0..127`,
in 33 of its 131 files).

**Options** (all read by `core/ple.cpp`, layout from the sidecar's META.json):

| table | layout | host RAM | error vs the original |
|---|---|---|---|
| int4 (the old sidecar) | group of 16 values, fp16 scale, int4 | 30 GB | 9% relative (L2, 100k rows of shard 0) |
| int8 | group of 16 values, fp16 scale, int8 (max/127) | 54 GB | worst value 0.39% of its row's max |
| bf16 (original) | as released | 96 GB | exact |

Built with `tools/ple_download.py` (HTTP ranges of just those tensors, ~102 GB)
and `tools/ple_convert.py` (bf16 and int8 sidecars; checks the rows line up
with the int4 sidecar). The int8 table lives at
`/mnt/llms/qwen3.8-flash-next-ple/ples_int8`.

**Next-token distributions** (4,000 held-out tokens: DESIGN.md prose, Python
code, license text; teacher-forced decode; against a new fp32 CPU reference
with the original bf16 table; paired bootstrap over tokens):

| engine with | mean \|dlogprob\| | top-1 agreement | vs int4 (95% CI) |
|---|---|---|---|
| int4 table | 0.065 | 96.8% | |
| int8 table | 0.048 | 97.4% | -0.017 (-0.022, -0.013) |
| bf16 table | 0.044 | 98.0% | -0.021 (-0.026, -0.015) |
| fp8 table (official FP8 checkpoint) | 0.050 | 97.3% | |

The fp8 row is from a rerun on 2026-09-28 (the same 4,000 tokens, rebuilt; the
fp32 reference recomputed): bf16 0.0444 / 98.0% and int8 0.0476 / 97.4%
reproduce the first run, fp8 0.0503 / 97.3%. Paired: fp8 - bf16 +0.0059 (95% CI
+0.0023, +0.0095), int8 - bf16 +0.0032 (+0.0002, +0.0062), fp8 - int8 +0.0027
(-0.0007, +0.0063). `Qwen/Qwen3.8-Flash-Next-FP8` stores the table as e4m3 with
one scale for all 320M rows; our int8 has an fp16 scale per 16 values, and is at
least as close for 6 GB more (57.6 vs 51.2 GB).

The old fp32 reference, built with the int4 table, is itself 0.064 from the
new one: the int4 table was the largest single source of error, more than all
of the engine's fp16 arithmetic (0.044 with the full table). int8 recovers
~85% of it.

**Tasks** (`tools/bench_tasks.py` through the server, greedy, thinking off,
slots and prefill chunk as in production; GSM8K test set, MMLU 25 questions
per subject, ARC-Challenge test set; `tools/bench_compare.py` pairs the runs,
McNemar's exact test):

| task | int4 table | int8 table | right only with int4 / int8 | McNemar p |
|---|---|---|---|---|
| GSM8K (1,319) | 94.8% | 94.6% | 10 / 8 | 0.82 |
| MMLU (1,425) | 86.6% | 86.0% | 14 / 6 | 0.12 |
| ARC-Challenge (1,172) | 97.2% | 97.2% | 2 / 2 | 1.00 |
| all (3,916) | 92.5% | 92.3% | 26 / 16 | 0.16 |
| bf16 table | (to run) | | | |

With ~1,200-1,400 questions per task, only accuracy differences of about 1-2
points can show. The int8 table changed 42 of 3,916 outcomes, split 26/16:
no measurable task effect. The distribution result and the task result measure
different things: int8 brings the next-token probabilities measurably closer
to the original model's, but greedy final answers on these tasks almost never
depend on that difference.

**Decision (2026-09-28): production runs the official fp8 table** (`ples_fp8`,
51.2 GB) with a 160 GB host prefix cache. The bf16 table plus a 128 GB cache
and the vLLM LMCache servers passed the host's 251 GB and the host OOM killer
took the engine twice; with bf16 the cache had to shrink to 96 GB. fp8 costs
+0.006 mean |dlogprob| against bf16 (0.050 vs 0.044) and is within noise of
int8 (0.048; 6 GB more); the owner chose the official table for the RAM.

Earlier decision (2026-09-27): production runs the original bf16 table
(`ples_bf16`, 102.4 GB pinned in RAM; the container limit went to 240 GiB and
the host prefix cache stays at 128 GB): at worst it scores like int4 on tasks,
and its next-token distributions are the model's own. The bf16 task run is
still to do. Before that, production ran the int8 table on 2026-09-27 (+24 GB of host RAM; the container's limit was raised
to 236 GiB for it): closer to the trained model at no GPU or speed cost, with
no task gain shown. If bf16 also shows no task effect, the choice is between
fidelity (int8) and 24 GB of RAM (int4).

## Vision: images and video (2026-09-25)

The checkpoint carries a Qwen3-VL vision tower (`model.visual.*`, 0.41 B
parameters): 16x16 patches over 2 frames, a learned 48x48 position table
interpolated to each image's grid, 27 pre-norm blocks (width 1152, 16 heads of
72, 2D rotary positions, GELU-tanh MLP 4304), then a 2x2 patch merger to the
language model's width (2560). Every card holds its own fp16 copy (0.9 GB) and
encodes whole slices: an image, or one temporal slice of a video (attention
never spans slices), so up to four slices encode at once with no traffic
between cards (`Engine::encode_vision`; tensor-parallel splitting would spend
more time in all-reduces than in compute).

- `src/vision/vision_encoder.hip`: fp32 residual stream (activations reach
  ~1e4), fp16 rocBLAS GEMMs, and a fused flash-style attention kernel (one
  thread per query, keys and transposed values through LDS as half2 pairs,
  online softmax over groups of 16 keys). Matches HF's `Qwen4ExpVisionModel`
  in fp32 to 2-6e-3 relative error (`tests/gpu/test_vision_encoder` against
  `tools/vision_ref.py` dumps).
- Encode time on one card: 720p 0.34 s, 1080p 1.23 s (attention is ~80% of
  it, at ~20% of the packed-dot peak; 256 queries per block from 6k patches,
  128 below; tile and group sizes swept); four 1080p images ~1.6 s. Images are scaled to at most ~1920x1088 by default
  (`QW_VISION_MAX_PIXELS`; the model allows 16.7 MP).
- Inputs: the server mirrors the Qwen3-VL processor (PIL backend) exactly for
  images (pixels within 1.2e-7, identical token ids) and video (ffmpeg decode,
  2 fps sampling, timestamps between temporal slices; frames within 8e-3
  because HF resizes video frames with torchvision)
  (`server/tests/test_vision_preprocess.py`).
- Language model: vision tokens are negative ids inside the engine, derived
  from each item's content hash and position within it; prefill takes their
  embeddings, the n-gram table and the sampler's penalties see the pad token.
  The prefix cache therefore tells images apart with no special casing, and
  the vision tower only runs for vision tokens that are prefilled, with an LRU
  cache of its outputs (`QW_VISION_CACHE_GB`, default 2). A resent screenshot
  costs nothing once its prefix is cached.
- Positions: the vLLM fork gives vision tokens plain token indices (what
  production vLLM served); HF transformers gives them 3D M-RoPE positions
  (interleaved sections 11/11/10; text after an image resumes at the image's
  start + max(rows, cols); compressed indexer keys at their first token's
  position). Measured with prefill support for per-token 3D positions
  (`Engine::prefill`'s rope3, experiment only) on 100 generated image QA
  cases with known answers (`tools/vision_pos_cases.py`,
  `tools/qw_vision_pos_eval`, teacher-forced answer log-probability and
  exact greedy match):

  | task (20 each) | plain positions | HF 3D positions |
  |---|---|---|
  | count objects | -0.135, 19/20 | -0.217, 19/20 |
  | grid cell color | -0.042, 20/20 | -0.070, 20/20 |
  | read a text line | -0.184, 19/20 | -0.369, 17/20 |
  | table lookup | -0.004, 20/20 | -0.005, 20/20 |
  | where is the shape | -0.114, 20/20 | -0.052, 20/20 |

  Plain positions are as good or better on four of five tasks (OCR most
  clearly), so the engine keeps them.

Measured through the server: OCR of rendered text, shapes/colors/layout, and
a question about one line of a 1080p code screenshot all answered correctly;
two different images at the same prompt position never share cached state.

## Interleaved prefill (2026-09-26)

A new request's prefill used to run to the end before anything else decoded,
so a long prompt froze every running stream (12.4 s for a 30k-token prompt).
`Session::begin_prompt` restores what the caches hold and `prefill_some`
prefills the rest in pieces; the scheduler advances the oldest prefill by one
piece (`QW_PREFILL_PIECE`, 2048 tokens) and then lets the running requests
decode for `QW_DECODE_SHARE` (0.25) of the piece's time. With nothing
decoding, a prompt still goes in whole. Snapshots reach the block store at
the same points as before (chunk ends, message boundaries, the prompt end),
not at every piece. Two streams decoding while a 30k-token prompt arrives:

| decode share | longest stall | their speed meanwhile | the prompt's TTFT |
|---|---|---|---|
| (before: whole prefill) | 12.4 s | 0 | 12.5 s |
| 0 (one step per piece) | 1.15 s | 3.7 tok/s | 13.8 s |
| 0.25 (default) | 1.15 s | 17 tok/s | 16.6 s |
| 1.0 | 1.15 s | 25.7 tok/s | 20.3 s |

Bursts of short prompts are unchanged (8 concurrent 300-token prompts: 92
tok/s aggregate either way): each still pays the ~80 ms of collectives that
make a prefill; batching several prompts into one prefill is the fix for that.

## Next steps: where the time goes (profiled 2026-09-25)

Single-stream decode is 15.0 ms per step (66 tok/s plain, 90-106 with MTP at
temperature 0). A `rocprofv3 --kernel-trace` of 300 steps (`qw_gpu --gen
300`), per step on one rank (tracing inflates the collective waits, not the
kernel durations):

| | per step |
|---|---|
| dense GEMVs (`gemv_rows`, 339 launches) | 4.9 ms: ~86% of the card's bandwidth for the fp16 weights |
| other compute (MoE pairs 1.0, HC mixers 0.9, QSA 0.4, GDN 0.3, routing 0.3 ms) | ~3.5 ms |
| collectives: 291 (3 per HC mix: all-reduce, all-gather, reduce-scatter; 97 mixes) | push + receive kernels, 582 launches |
| kernel launches in the graph | 1,585 |

So ~8.5 ms is compute and ~6.5 ms is collective latency and launch gaps. The
GEMVs are near the bandwidth limit, so the remaining gains are in bytes,
launches and collectives, in this order of expected payoff:

1. **int8 dense weights (W8A16).** Built and measured (`QW_INT8_DENSE=1`,
   `gemv_rows_i8`, one fp32 scale per 32 weights along K): single-stream
   decode 14.9 -> 12.8 ms per step (67 -> 78 tok/s), but against the fp32
   reference over 500 tokens (teacher-forced decode) mean |dlogprob| goes
   0.094 -> 0.127 and top-1 agreement 95.4% -> 92.6% (per-row scales were
   worse: 0.161 against fp16). Every weight group contributes (bisected with
   `QW_INT8_SKIP`), so it stays opt-in: a speed/accuracy choice.
   **GPTQ (2026-09-26).** Calibration Hessians H = XᵀX of every int8 matrix's
   input from 203k tokens of code, docs and chat text (`tools/gptq_calib_data.py`,
   `tools/qw_calibrate`, per-stream Hessians for the HC down rows), then GPTQ
   (`tools/gptq_int8.py`: block 128, damp 0.01) into `.q8` sidecars the engine
   loads from `QW_INT8_DIR`. Per-matrix output error on the calibration inputs
   falls 1.5-18x, but the model-level gain is modest. Measured on a held-out
   4,000-token set (DESIGN.md, Python code, license text) against the fp32
   reference, paired bootstrap over tokens:

   | variant | mean \|dlogprob\| | top-1 | vs fp16 (95% CI) |
   |---|---|---|---|
   | fp16 | 0.042 | 98.0% | |
   | GPTQ int8, group 32 | 0.054 | 97.1% | +0.012 (+0.009, +0.016) |
   | GPTQ int8, lm_head + PLE only | 0.048 | 97.1% | +0.006 (+0.003, +0.010) |
   | GPTQ int8, group 16 (first 1,594 tokens) | | | +0.018 (+0.010, +0.026); group 32 there: +0.029 |

   Per group (group 32, 500 tokens), every dense group alone costs about as much
   as all of them together (GDN alone +0.023, all +0.022): the errors do not add,
   the recurrent state turns any weight error into about the same divergence, so
   no subset of fp16 groups closes the gap. Treating the errors as independent,
   group-16 int8 adds a perturbation about 60% the size of fp16 arithmetic's own;
   matching fp16 would need several times less weight error than 8 bits give.
   The speed is also smaller than first measured: `test_batch_decode` 14.69 ->
   12.99 ms/step at M = 1 (+13%), 21.4 -> 20.0 at M = 4 (+7%), 26.9 -> 26.9 at
   M = 8; int8 lm_head alone gains nothing. Kept as tooling (`QW_I8_GROUP` build
   option for the scale group), off by default. (The group-16 build with every
   group int8 also stalled under HIP graphs, ~0.9 s per step, fine with
   `QW_NOGRAPH=1`; not chased since group 16 is not used.) Original
   note: Dense fp16 weights are 88% of the bytes
   read per token. Per-channel int8 copies halve the GEMV time (~-2.4 ms per
   step, ~66 -> ~80 tok/s plain, MTP scales with it). Gate: mean |dlogprob|
   against the fp32 reference (`qw_gpu --logprobs`) must stay near today's
   0.074 (fp16), and greedy output should match on the test prompts.
   Kernel fusion (item 3) was tried on the two safe candidates and measured
   against the separate kernels, three alternating runs each, output
   bit-identical: the shared expert's SwiGLU inside its down projection
   (recomputed in every output row's wave: no gain at M = 1, slower from
   M = 4) and the GDN conv-ring update inside the scan (M = 1 within noise,
   M = 4 22.0 vs 21.4 ms: the scan is on the critical path). The ~3.3 µs per
   launch is not simply additive: moving work into a critical kernel costs
   about what the launch saved. Neither was kept.
   Parallel graph branches do not help either: forking the shared expert
   onto a second stream inside the decode graph (it is independent of the
   routed experts) made the step 14.6 -> 16.4 ms; cross-stream graph edges
   cost more on ROCm than the launch latency they hide.
2. **Fuse the collectives into their producers and consumers.** Measured
   first (`bench/graph_launch_bench`): a dependent kernel in a graph costs
   ~3.3 µs on this card. Step one, done: a small collective is one kernel
   (four push blocks and a waiting/reducing block) instead of two
   (`exchange_small_kernel`, `QW_COMM_SPLIT=1` restores two): 18.8 -> 15.7 µs
   per chained all-reduce, but decode only goes 15.0 -> 14.8 ms per step,
   because each collective's time is mostly the cross-GPU round trip, which a
   launch gap overlaps. Folding the push into the producer and the wait into
   the consumer would save as little, so collectives are better attacked by
   count (item 4) or by overlapping them with compute. Original note: Each
   collective is a separate push kernel and a separate receive kernel. The
   producing GEMV or mixer can store its slice straight into the peers'
   buffers, and the consuming kernel can wait on the flags itself, as the
   design intended (section P2P). That removes ~580 launches per step and
   their gaps; estimate -1.5 to -3 ms.
3. **Fewer, bigger kernels.** The ~1,000 small kernels (HC pre/norm/mix,
   routing, combine, swiglu, GDN conv/scan/norm) each cost a launch gap of a
   few µs in the graph. Fusing each sublayer's glue into its neighbours
   (e.g. routing + combine into the expert kernels, GDN conv + scan + norm in
   one kernel) should save another ~1-2 ms.
4. **One collective fewer per sublayer.** The next HC mix's down projection
   is linear in the residual, so each rank could apply it to its unreduced
   block output instead of waiting for the reduce-scatter (97 fewer
   collectives per step), at the cost of reading the HC-down weights for the
   full width. Needs measuring: ~-1.5 ms of latency against ~+0.3 ms of reads.
5. **MTP drafting inside one graph.** Drafting costs ~4 ms per step (3 MTP
   passes with a host round trip each for argmax and the next embedding).
   Device-side argmax and embedding lookup make it one graph launch: ~-2.5 ms
   per step, ~10% of MTP throughput.
6. **Stop drafting when acceptance stays low.** On hard-to-predict text MTP
   can be slower than plain decoding (63 vs 68 tok/s on the second test
   prompt, 1.7 tokens per step), and temperature 1.0 is probably also below
   plain. The adaptive draft count never goes below 1, because the first MTP
   pass also keeps the MTP layer's KV current. Plan: when a request's
   acceptance EMA stays under a threshold, switch it to plain decoding
   (K = 0); probe again every N steps, first running one MTP pass over the
   tokens decoded since drafting stopped to bring the MTP KV up to date (the
   same catch-up path prefill uses, `mtp_pend`).
7. **Multi-request speculative batches over 8 rows.** Root-cause the open
   issue above; lifting the 8-row cap lets 3-4 concurrent requests draft 3
   tokens each instead of 1.
8. **Sampling.** Profiled with 4 concurrent requests at temperature 0.7
   (`QW_TRACE` splits `Session::generate`): of a 48 ms step, 29 ms were the
   verification batch and 16-18 ms host sampling (~2 ms per row: three passes
   over the 248k vocabulary with `exp` twice). Now one pass collects the
   drawable tokens with their weights (same draw for the same random number)
   and a step's rows are sampled in parallel (8 threads; each row draws from
   its own generator seeded in order from the session's, still exact):
   3 ms per step. 4 x 400-token requests: 133-141 -> 171-176 tok/s
   aggregate; bursts of 8 x 300-token prompts: 92 -> 114 tok/s. GPU-side
   sampling (below) would remove the rest. Original note on GPU sampling: Top-k/top-p/min-p sampling costs 0.2-0.7 ms per
   token on the host at temperature > 0; doing it per vocab shard on the GPUs
   (like the log-sum-exp) saves most of that.
9. **Prefill.** Profiled (`bench/prefill_bench` under rocprofv3): 300 tokens
   run at ~1,350 tok/s, 8k at ~2,200: every prefill pays a fixed ~80 ms. The
   collectives are ~75% of GPU time at 300 tokens and ~42% at 8k (the two
   micro-batches hide part of it); then QSA attention 19%, GEMMs ~12%, MoE
   ~12%, GDN scan 5% (8k). Done: large multi-row payloads whose rows are
   contiguous (all-gathers, all-reduces) are pushed as one flat copy instead
   of a block per 1-2 KB row: +4-10% prefill (300 tokens 0.226 -> 0.213 s,
   8k 3.80 -> 3.66 s), decode unchanged. Item 4 (one collective fewer) is not
   possible as stated: the next mixer's RMS norm needs the updated residual,
   and squares do not distribute over the ranks' partial sums; it needs a
   different parallel layout. For many concurrent requests the bigger lever
   is scheduling: batch waiting prompts into one prefill (the fixed cost is
   paid once) and interleave prefill chunks with decode steps. Original note:
   ~2,000 tok/s. Profile a long prefill the same way before
   choosing: candidates are the chunked (WY) GDN kernel, the fp32-output
   router GEMM (rocBLAS falls back to a slow HSS kernel), and QSA attention.

## KV spill: slots larger than their VRAM (built and measured 2026-10-03)

**What.** A slot's capacity becomes `slot_tokens` (the tokens held in VRAM, as before) plus `slot_spill` (tokens held
in pinned host memory). Positions below the VRAM part live where they always did; positions from it on have their K, V
and raw indexer keys in host memory mapped into the device's address space, and the kernels address them with one
compare per row (`kv_row`, `kernels/types.hpp` `SlotPtrs::vt`, `KvSplit`). The compressed keys stay in VRAM for the
whole logical length, so scoring and the top-512 selection run unchanged on the GPU: only the K/V rows the selected
groups name are read from the host. A slot without spill has `vt = NO_SPILL` and takes the first branch.

**Why it can work.** Decode reads about 2,052 tokens per layer whatever the context: 2 MiB of K+V per layer, ~24 MiB per
row and rank over the 12 QSA layers (25 with the MTP layer). The cost of a spilled slot is that traffic over PCIe, not a
function of its length. In VRAM a spilled token costs 0.83 KB (compressed keys) instead of 20.8 KB.

**Host memory.** `SpillPool` (`src/engine/spill.cpp`) pins everything once at start (pinning on demand stalled the
process for seconds, see "Pinning"). One copy per KV pair: both ranks of a pair compute the same bytes and both write
them, each reading what it wrote (no cross-rank ordering). 13.3 KB (K, V) + 6.7 KB (raw keys) per spilled token and
KV group. A layer's pages are placed near one of the pair's two GPUs, alternating by layer
(`hipHostMallocNumaUser` plus `set_mempolicy(MPOL_PREFERRED)`; `QW_SPILL_NUMA=0` leaves it to HIP). Placement is
read back with `move_pages` and logged at start. `QW_SPILL_NC=1` maps the memory non-coherent (cacheable).
Spill is allocated from the host's RAM budget: it comes out of what `QW_HOST_CACHE_GB` can use.

**What was measured (`bench/spill_gather_bench`, dev box, production's model resident but idle, 2026-10-03).**
Kernels shaped like the decode attention read scattered 2 KiB groups (512 per row) from the pool:

| | per GPU |
|---|---|
| one GPU alone, 1 row, attention-shaped | 22-23 GB/s (a "wide" kernel with 8x the blocks: 22-27) |
| the two GPUs of a pair reading the same buffers | 21-23 |
| all four GPUs together, pages placed near their GPUs | **22-23 each, ~90 GB/s in all** |
| all four together, every page on one node (HIP's default from device 0, `QW_SPILL_NUMA=0`) | **11.7 each, ~47 GB/s in all** |
| layers on the GPU's own node vs on its pair's node | 22.4 vs 23.3: within noise |

So a 4-block-per-row attention-shaped read already reaches the link's rate (Gen4 x16, ~25 GB/s; same as the P2P
push), and neighbouring NUMA nodes cost nothing measurable: the node is a memory-controller domain on one IO die.
What matters is spreading the pages: one node's controllers cap all four GPUs together at ~47 GB/s. A spilled row
at 22 GB/s is ~1.1 ms per step per rank (24 MiB), against 17.3 ms for a plain step: ~6%, if every selected token
were spilled; spilled slots decoding in the same step add up on the link. Mapped host memory was verified in both
directions (device writes read by the host, host writes read by the device: no bad words on any GPU).
**Checks.** `tests/gpu/test_spill.hip` passes: spilled slots (boundaries at 4096 and 8192, chunks that straddle them,
appended prefills, one-row decode, 4-row verification runs with `accept`) give bit-identical logits to a resident slot,
and `export_kv` / `import_kv` across the boundary bit-identical bytes.

**Engine cost (`bench/spill_bench`, dev box, 2026-10-03, chunk 4096, cap 262144, B = 32,768 tokens in VRAM + 229,376
spilled; random-token prompts).**

Without spill in use (the same source built on the unmodified tree against the new tree with spill off, two runs each,
alternating): decode 16.9 / 17.1 / 18.0 ms at 8k / 32k / 131k on both (differences under 0.2 ms, the noise); prefill
2,202 / 2,169 / 1,991 tok/s on the old tree against 2,192 / 2,147 / 1,979 on the new: -0.5 to -1%, the same sign at every
context, about the size of the run-to-run spread (0.3-0.6%). The compare in the attention kernels is not free in prefill.

A resident slot (A) against the spilled one (B) in one process:

| context | decode, 1 row: A / B | 4-row runs: A / B | prefill increment tok/s: A / B |
|---|---|---|---|
| 8k, 32k (inside B's VRAM part) | 17.1 / 17.1, 17.2 / 17.2 | 24.2 / 24.2, 24.4 / 24.4 | 2,201 / 2,192, 2,151 / 2,132 |
| 65k | 17.5 / 18.0 (+3%) | 24.7 / 28.3 (+14%) | 2,074 / 950 |
| 131k | 18.0 / 18.8 (+5%) | 25.5 / 29.9 (+18%) | 1,910 / 734 |
| 200k | 18.9 / 19.9 (+5%) | 26.3 / 30.9 (+18%) | 1,627 / 680 |

A and B decoding in the same step: 20.3 / 21.0 / 22.2 ms at 65k / 131k / 200k (A alone 17.5 / 18.0 / 18.9). Reading:
- Inside the VRAM part B is identical to A: the spill path costs nothing there.
- Decode costs +0.5 / +0.8 / +1.0 ms per spilled row at 65k / 131k / 200k, about the 1.1 ms the link would take for
  every selected token (24 MiB at 22 GB/s). The fraction of spilled positions at those contexts is 50 / 75 / 84%,
  so on these random-token prompts the top-512 selection is spread about evenly over the positions; real text may
  concentrate on recent tokens (which are in the spilled part, being the newest) or on the start. Not measured.
- A 4-row run costs 4x a single row (+3.6 / +4.5 / +4.6 ms): every row reads its own selection and nothing is shared
  between the rows of a run, whose selections are nearly the same. A union fetch per run would cut that to about
  one row's worth.
- **Prefill into the spilled part is 2.2-2.6x slower** (950 / 734 / 680 tok/s against 2,074 / 1,910 / 1,627): the
  attention kernels read the host for every query of the chunk. This is the case the staging plan below is for.

**Settings (2026-10-03).** `--kv-spill on|off` (default off; `QW_KV_SPILL`), `--slots` = the tokens each slot keeps in
VRAM, `--slot-max-tokens N` (default 524288; `QW_SLOT_MAX_TOKENS`) = the capacity of every slot, spilling what its VRAM
tokens do not cover; `QW_SLOT_SPILL` names the spill of each slot explicitly, `QW_SPILL=0` forces everything off. At
start the engine logs the plan (VRAM tokens + spilled per slot, GB of pinned RAM), refuses a plan the host cannot hold
(spill + `QW_SPILL_MARGIN_GB`, default 24, against `MemAvailable`) or that the cards cannot hold after the weights, and
logs each card's free VRAM (the early estimate is `19.4 GiB + slots + 6144 B per compressed-key group of the longest slot`,
within 0.05 GiB of the measured value on two configurations; `QW_VRAM_CHECK=0` skips both VRAM checks); `bench/slots_probe` checks a configuration loads and decodes across all its slots.

**Sizing (per card, measured on the dev box, int4 PLE table; 12.24 GiB of VRAM free after the weights, vision tower
and buffers with one 8k slot).** A slot costs 19.97 KB per resident token (K, V, fp32 raw keys; 13 layers), 832 B per
token of capacity (compressed keys: 436 MB for a 512k slot) and ~38 MB of state; the 512k scoring buffers add ~0.8 GB
for the prefill; keep ~0.5 GiB free. Host RAM per spilled token: 39.9 KB (K, V and raw keys, both KV groups).
- Warmup (graphs for every batch size, rocBLAS kernels) takes ~1.1 GiB of VRAM on top of what the load leaves, and
  ~0.5 GiB should stay free, so a plan needs ~1.6 GiB free after the load. 6 slots, 1 x 256k + 5 x 32k in VRAM, all
  512k, left 0.97 GiB: it loaded, and warmup died with `hipMalloc: out of memory`. (The constructor then hung forever:
  an exception left it while the rank threads waited, and destroying the condition variable under them never
  returned, which also lost the error message. The constructor now stops the threads first.)
- **6 slots, 1 x 256k + 5 x 24k in VRAM, all 512k: loads and runs** (1.73 GiB free after the load, 0.62 after warmup and a
  run, 110 GB of pinned RAM pinned in 36.6 s, 6-row decode step 25.3 ms against 25.1 without spill, 27.8 ms with 40k
  tokens in every slot, five of them 15k tokens into their spilled part). That is the largest of this shape on these
  cards: the next 8k tokens in each of the five small slots would not leave room for warmup.
- 8 slots (256k + 7 x 16k, all 512k): fits the VRAM by arithmetic (~7 GiB resident + 3.3 GiB compressed keys) but
  needs ~152 GB of pinned RAM: not on this host next to the PLE table and the prefix cache.
- Pinning and placement (`QW_SPILL_POLICY`, default `auto`: a node's layers are bound to it when it can give that much
  counting the page cache it can reclaim, otherwise preferred). Pin times vary with the host's memory state: 108.6 GB
  took 73 s and 210 s with the nodes' free lists nearly empty, 55 GB 10.5-11.2 s and 110 GB 36.6 s later. With the old
  "preferred" policy 45-55% of the sampled pages landed on the intended node at 108 GB (the kernel falls back to
  another node when the node's free list runs low, instead of reclaiming page cache); with `auto` 80-92% (42% of the
  pages on node 0 where 25% were meant; the fallback goes there). Placement is not exact, and need not be: the reads
  run at the link's rate either way as long as the pages are spread (below).
- **numactl does not help.** HIP places a whole buffer on one node, chosen from the thread's policy, so a process-wide
  `numactl --interleave=all` or `--membind=0,1` put everything on node 0 (`QW_SPILL_POLICY=process` leaves the policy to
  the process): four GPUs reading together reach 15.2 GB/s each instead of 22.7-23.0 with the per-buffer placement
  (preferred 22.7, auto 23.0, bind 23.0). The engine's per-buffer binding is what spreads the pool.
- With the raw indexer keys in a ring (below, 2026-10-04): **6 slots, 1 x 256k + 5 x 56k in VRAM, all 512k, load and
  run**: 69.1 GB pinned in 12.1 s (was 110 GB with 5 x 24k), 89% of the pages on the intended node, 1.76 GiB of VRAM
  free after the load (the estimate said 2.0: it is now 0.25 GiB optimistic), 0.66 after warmup and a run, 6-row step
  25.2 ms. 5 x 64k would leave 1.45 GiB and is refused.

**Raw indexer keys as a ring (2026-10-04).** A slot used to keep every position's raw indexer key (fp32, 128 per
token and QSA layer: 6.7 KB per token and rank, a third of its KV), but only a group's own 4 positions are ever read, to
build its compressed key. Now each slot keeps a ring of `prefill_chunk + RAW_TAIL + 64` rows per layer (position p at
row p % rows; 30 MB per slot at a 4096-token chunk), and a snapshot carries the raw keys of its last `RAW_TAIL` = 256
positions for every layer (1.7 MB per rank; the MTP layer can lag the committed tokens by up to 192 positions, so a
group's last 3 would not do). Prefill captures copy the tail at the end of the chunk's job; `snapshot_save` /
`snapshot_restore` and `export_recurrent` / `import_recurrent` carry it. Blocks no longer hold raw keys: 5.3 -> 3.6 MB per
256-token block and rank (-32%). The state layout id changed (5), so cache files written before are ignored (one cold
disk tier after the deploy). `blend_splice` (the rejected CacheBlend experiment) needs every raw key and now refuses.
KV spill holds K and V only (13.3 KB per spilled token and KV group, was 20).
Exactness: `bench/logits_dump` (13k-token prompt in 4k chunks, 48 decode steps, four 4-row runs rolled back with
`accept`, a snapshot at 13,057 = 1 mod 4, 6,000 more tokens so the ring wraps, the snapshot restored and 32 steps; on a
resident and a spilled slot) gives bit-identical logits on the builds before and after (212 rows); `test_spill`,
`test_snapshot_capture`, `test_block_store`, `test_host_tier` pass.
Speed (old and new build alternating, dev box): resident decode 16.9-18.0 ms and prefill 2,156-2,192 / 2,135-2,144 /
1,968-1,972 tok/s at 8k / 32k / 131k on both; a slot spilled at 32k the same on both (decode 17.0-18.9 ms, prefill 945 /
734 tok/s at 65k / 131k): no change beyond the noise. A cold 65k-token prompt through the block store (`cold_prefill_bench`,
nc+pairs): 31.44-31.45 s against 31.37-31.40 s, saves +1.5-1.7% over no store against +1.2-1.5%, pinning 5.5-5.8 s
against 5.1-5.4 s, the cache's RAM 9.3 -> 8.6 GB: the blocks are a third smaller, but for one long prompt the
snapshots (33 MB per rank each, now 1.7 MB more) are most of the cache. What it buys is VRAM: 6.7 KB less per resident
token.

**Non-coherent spill memory** (`QW_SPILL_NC=1`: the GPUs' L2 may cache the host's lines): prefill into the spilled part
825 -> 973 tok/s at 65k and 703 -> 790 at 98k (resident: ~2,100 / ~2,000), decode 0.1-0.2 ms better; `test_spill`
passes. Not the default: a CPU write into the spill (an `import_kv`) under a GPU's cached lines is not covered by a test.

Coherence (`tests/gpu/test_spill_coherence.hip`, 200 repetitions of each, coherent and non-coherent mapping): a GPU
reads a 1 MiB region (so it sits in its L2), the CPU rewrites it, the GPU reads again; a GPU writes, the stream is
synchronized, the CPU reads; the first sequence inside captured graphs; GPU 0 reads, GPU 1 writes, GPU 0 reads again.
No stale word in any of them, in either mapping. With the prefill staging below the non-coherent mapping only buys
decode ~0.15 ms per step with a spilled row (~1%); it stays opt-in.

**Prefill staging (2026-10-04, `EngineOptions::spill_stage`, on; `QW_SPILL_STAGE=0` off).** Each card has a VRAM buffer
as large as the largest slot's spill (K and V) and a copy stream. Before each QSA layer (and the MTP layer) of a chunk
that reaches a slot's spilled part, the rows of that layer already in host memory (from the slot's VRAM boundary to the
chunk's first position) are copied in; the layer's kernels then use the buffer in place of the host part (same
indexing), the chunk's own rows are written into it, and after both micro-batches' attention they go back to host
memory, which frees the buffer for the next QSA layer's copy. The copy for a QSA layer runs while the three GDN layers
before it compute, so one buffer is enough. Segments of a batched chunk are staged while their regions fit the buffer;
the others read host memory directly. Bit-identical to the direct path (`bench/logits_dump` against the build before:
identical; `test_spill`'s batched case prints the same hash with the staging on and off).

| context (B: 32k in VRAM) | resident A | B staged | B direct (before) |
|---|---|---|---|
| 65k | 2,081 tok/s | **2,061** | 935 |
| 131k | 1,911 | **1,872** | 722 |
| 200k | 1,622 | **1,580** | 671 |

Prefill into the spilled part now runs at 97-99% of a resident slot (was 40-45%); decode unchanged. Cost: VRAM for the
buffer, 1 KiB per token of the largest spill (456 MiB for a 512k slot with 56k in VRAM). The 6-slot plan becomes
1 x 256k + 5 x 48k in VRAM, all 512k: loads and runs (1.82 GiB free after the load, 0.71 after warmup and a run, 70.2 GB
pinned in 14.0 s, 6-row decode step 25.1 ms).

**Fetch once per verification run (built 2026-10-04, measured, reverted).** Per QSA layer, after the selection: mark
the spilled groups any row of a run selected (a bitmap per run), number them, copy each once into a VRAM scratch, and
let the rows read them there (graph variants used only for batches with such a run). Bit-identical, but slower on
4-row runs: 24.2 / 28.5 / 30.3 / 31.2 ms per step at 32k / 65k / 131k / 200k without it, 25.1 / 28.9 / 31.5 / 33.3 ms with
it. Three more kernels and a memset per layer cost ~0.8 ms even with nothing spilled, more than the saved reads.
Non-coherent mapping (the L2 serving the rows' repeats) does better on the same runs: +2.5 / 3.5 / 3.6 ms over the
resident slot against +3.9 / 4.9 / 4.9 coherent.

**Infinity Cache and host memory** (`spill_gather_bench --tokens N --rows 4 --reps 100`, a region read over and over):
2 MiB (fits the 4 MiB L2): coherent 34 GB/s, non-coherent ~1,250 GB/s; 32 MiB (fits the 128 MB Infinity Cache, not
the L2): 23 and 26 GB/s; 1.4 GiB: 23 and 23. The Infinity Cache does not hold host memory's lines: only what is
copied into VRAM benefits from it.

**Locality of the spilled selection (`bench/spill_locality`, `QW_SPILL_LOCALITY=1`, real text: this repository's docs
and sources, 230,646 tokens; a slot with 8,192 tokens in VRAM; 256 greedy decode steps at each context).** Of the groups
a step selects, the share the same layer selected within the last K steps, i.e. the hit rate of a cache holding the
groups of the last K steps, and its size per spilled slot (12 layers, 4 KiB per group and layer):

| context | spilled share | K=1 | K=2 | K=4 | K=8 | K=16 | K=32 |
|---|---|---|---|---|---|---|---|
| 32k | 59% | 61.4% (14 MB) | 70.4% (20) | 77.8% (30) | 83.6% (52) | 88.1% (96) | 91.3% (183) |
| 100k | 97% | 57.5% (23) | 67.6% (33) | 76.2% (53) | 82.7% (93) | 87.9% (172) | 91.2% (330) |
| 200k | 98% | 61.4% (24) | 71.0% (33) | 77.6% (51) | 82.7% (88) | 86.7% (161) | 89.6% (307) |

Per layer the K=1 rate ranges 54-73%. Runs of 4 rows with the text's next tokens select 1.86-1.94x one row's distinct
spilled groups (a fetch once per run would read 47-48% of what the rows read now). So a cache of the last few steps'
spilled rows in VRAM would serve ~3/4 of the reads at ~50 MB per spilled slot, and it would also take the repeats
within a run; to pay, it has to live inside the attention kernel (a tag check per row, fill on miss), not in extra
kernels.

**Decode cache of spilled rows (2026-10-04, `EngineOptions::spill_cache_mb`, default 256 MB per card; `QW_SPILL_CACHE_MB`,
0 off).** One direct-mapped table per card shared by every slot and QSA layer: an entry holds one position's K and V
rows (1 KiB + an 8-byte tag: slot epoch, slot, layer, position). The attention kernel checks the tag of each spilled row
it reads: a hit reads VRAM (and so the Infinity Cache), a miss reads host memory as before and is listed. One kernel at
the end of the step (every decode, verification and MTP graph) copies the listed rows in, claiming an entry with a
compare-and-swap so concurrent misses for one entry do not mix rows; the table is never written while attention reads
it. `qsa_prep_B` clears the entry of a spilled position it rewrites (a verification row after a rollback), and a slot's
epoch moves on prefill, `import_kv`, `import_recurrent`, `snapshot_restore` and `slot_reset`, which drops all its rows.
Bit-identical (`bench/logits_dump` against the original build; `test_spill`, including the batched hash).

Measured with real text (`spill_bench --tokens-file`, the prompt and the decoded tokens from this repository's docs and
sources; B = 32k in VRAM; ms per step):

| context | 1 row: A / B no cache / B cache | hits | 4-row runs: A / B no cache / B cache | hits |
|---|---|---|---|---|
| 65k | 17.32 / 18.55 / **17.52** | 87.5% | 24.52 / 29.93 / **25.00** | 89.7% |
| 131k | 17.83 / 19.69 / **18.45** | 79.1% | 25.20 / 32.42 / **26.26** | 86.0% |
| 200k | 18.61 / 20.34 / **18.91** | 86.8% | 26.21 / 33.44 / **26.72** | 91.5% |

The cost of a spilled row drops 70-84% for one-row steps and 85-93% for verification runs (a run's rows hit what earlier
steps filled; within a step the table does not change). Resident slot A is the same
with and without the cache (the fill kernel exits at once). With real text the uncached cost is higher than with the
random-token prompts above (+1.2-1.9 ms per row against +0.5-1.0). The 6-slot plan with the cache: 1 x 256k + 5 x 44k
in VRAM (1.81 GiB free after the load, 0.71 after warmup); 5 x 48k is refused (1.56 GiB after the load, 1.6 needed).

Cache size (4-row runs, real text, one spilled slot decoding; B's cost over A at 131k / 200k): no cache +7.2 / +7.2 ms;
64 MB 72-76% hits, +1.9 / +1.5 ms; 128 MB 85%, +0.7 / +0.8; 256 MB 86-92%, +1.1 / +0.5; 512 MB 94%, +0.3 / +0.2. Most of
it comes by 128 MB; the pool is shared, so several spilled slots decoding at once divide it, which is why the default
is 256 MB (VRAM for ~19k resident tokens across the slots).

**Not done.** One host copy shared with the block store.

**Production (2026-10-04, image `qw-engine:06ce991`).** `QW_SLOTS=262144,45056,45056,45056,45056,45056`,
`QW_KV_SPILL=on`, `QW_SLOT_MAX_TOKENS=524288`, `--host-cache-gb 64` (was 160). The start: 70.8 GB pinned in 14.4 s (91%
of the sampled pages on the intended node), 1.81 GiB of VRAM free per card after the load, ready in 154 s, of which
the fp8 PLE table's read took 143 s (the ranks were loaded ~58 s before it finished); smoke test passed; the host has
101 GiB available with the engine up and the cache empty. Server check on the dev box before it (`--kv-spill on`, the
same slots): a 78.6k-token prompt in a 44k-resident slot decoded at 71.7 tok/s, follow-up turns reused 56k and 78.7k
cached tokens, a new conversation sharing the long prefix reused 77.8k. Seen there: pinning the prefix cache's arenas
took 1.7-4.9 s each with 122 GB already pinned (spill + table), and steps of up to 419 ms next to them.

**A possibility, not built: the prefill staging buffer as cache space.** The staging buffer (1 KiB per token of the
largest spill, ~456 MB in the 6-slot plan) is only used while a prefill chunk reads a slot's spilled part; during pure
decoding it is idle VRAM holding the same kind of rows as the decode cache. The cache could use it then: either a
larger cache at no VRAM cost (256 + 456 MB: ~94% hits instead of 86-92%, a few tenths of a ms per spilled step), or the
dedicated pool dropped (256 MB back per card, ~4k more resident tokens per small slot). The cost: every prefill chunk
that touches spill overwrites it, so with interleaved prefill (long prompts go in pieces between decode steps) the
decoders start cold after each piece, and the handover needs its own invalidation and bit-exact tests with prefill and
decode alternating. Under mixed load it would do worse than the two separate buffers. The simpler lever when VRAM is
short is `QW_SPILL_CACHE_MB=128` (85% hits).

## Converted-weights cache (built 2026-10-04; off, waiting for SSDs)

Every start converts the checkpoint into the engine's layouts for each card: bf16 -> fp16 slices of the dense
matrices, the int4 experts repacked (~14.5 GB per card), the MTP head's bf16 experts quantized. Measured per card on the
dev box (checkpoint in the page cache): 44.7-48.8 s, of which 36.6-41.3 s is conversion and 5.5-6.2 s the upload.

`--weight-cache-dir DIR` (`QW_WEIGHT_CACHE_DIR`, `EngineOptions::weight_cache_dir`; default off) keeps the converted
pieces. Every upload of the loader goes through `WeightCache::item` (`src/engine/weights.hip`): the piece is built by
its lambda and uploaded, and appended to `DIR/rank<r>.qww` when recording; with a matching file it is read and uploaded
and the lambda does not run. The file's key covers the checkpoint's files (path, size, modification time), the rank,
the MTP head and `WEIGHT_CACHE_VERSION` (bump it when a conversion changes); it is written as `.tmp` and renamed when
the loader finishes, so an interrupted start leaves nothing that would be used; a replay that does not consume exactly
the recorded pieces fails with "delete it". Runs with `QW_INT8_*` set do not use it. 18.4 GB per card, 73.6 GB in all.

| start | weights per card |
|---|---|
| no cache | 47.9-48.8 s (convert 39.7-41.3, upload 5.5-6.2) |
| recording | 734-744 s: the four files written to the HDD pool took ~700 s (once) |
| from the cache, file in the page cache | **7.4-12.7 s** (read 0.4-1.7, upload 7.0-11.3) |
| the same before the file was pulled in by all loader threads | 30-34 s (page faults under `hipMemcpy`, one thread) |
| from the cache, cold on the HDD pool | 334-343 s |

Logits are bit-identical to the build before in every mode (`bench/logits_dump`: no cache, recording, replay).
Why it is off: on this host the files cannot stay in the page cache next to the PLE table (51 GB), the pinned spill
(71 GB) and the prefix cache, so a start would read them from the 2-HDD mirror at ~215 MB/s, 7x slower than
converting. And at the 2026-10-04 production start the cards were loaded ~58 s before the PLE table's 143 s read
finished: until the table reads faster, the weights are not what a start waits for. Both change with the model
storage on SSDs (73.6 GB at 2 GB/s is ~37 s cold; the table ~25 s).

## The real workload, and three changes measured against it (2026-10-04)

**What production serves** (1,310 requests, 2026-09-27 to 10-04, from the request log): prompts median 44k tokens, p90
156k, max 219k, 49% over 45k; 96% of prompt tokens come from the cache (a turn adds ~350 tokens, median); 357 tokens
generated (median); mostly one request at a time (two or more in ~20% of the busy samples). TTFT median 0.5 s, p90 7.3
s, p99 54 s; decode median 79 tok/s (2.26 tokens per step at 28 ms). 92 requests (7%) with 8k or more uncached tokens
hold 52% of all TTFT (40 of 76 minutes), at a median of 1,144 uncached tokens per second of TTFT; only 12 of them came
within 30 minutes of an engine start. 60 engine starts in the period, 17 of them failed with out-of-memory (16 on
09-25..27, one on 10-04 when a request arrived while a test on the dev box held the cards).

**Slot choice (on).** With every slot at the same capacity, a new conversation went to the least recently used slot:
production's first session after the KV spill deploy sat in a 44k-resident slot at 44.5k tokens with the 256k-resident
slot idle. `Session::acquire` now prefers, among slots with equal reuse, the one whose VRAM part covers the prompt
(the smallest such; the largest VRAM part when none does). Measured: a 49k-token conversation starts in slot 0.

**Moving a conversation that outgrows its slot's VRAM part (`QW_SLOT_MOVE=1`; off).** When the slot holding a
prompt's prefix would spill and a free slot holds the prompt in VRAM, acquire takes that slot if the block store can
restore the conversation there (the snapshot at its last prompt's end). Server test (a conversation from 29k tokens,
+2.7k per turn, 16 turns, slots 256k + 5 x 44k): the move happened at 46,217 tokens: restore 0.17 s, 2,692 tokens
prefilled after it (43,525 reused), TTFT 2.02 s against ~1.4 s for its neighbours, once. After the boundary: decode
mean 88.9 tok/s moved against 82.1 spilled (10 turns each, 54-83 tokens per turn: noisy; the step-level measurements
above say 1-3%). A spilled slot has no fixed cost per turn: 512 tokens at 60k context prefill in 0.38-0.39 s spilled
against 0.36 s resident, 2,048 in 1.11 against 1.06-1.10 (`spill_bench --ctxs 60000,60512,...`).

**Waiting for a shared prefix (`QW_SHARE_WAIT`, on; `QW_SHARE_MIN` 2048).** A waiting request whose prompt starts
like one being prefilled waits while that one still has 2,048 or more of the shared tokens to go, then restores them
from the block store (`Scheduler._shares_prefill`; `server/tests/test_admission.py`). Server test, cold prompts:

| | off | on |
|---|---|---|
| 4 requests sharing a 16k-token system prompt, all done after | 26.4 / 27.5 s | **8.4 / 8.1 s** |
| ... their TTFTs | 9.4, 13.1, 18.5, 23.9 s | 7.1 s, then 0.3-0.6 s after 7.2 s in the queue |
| the same 26k-token prompt twice at once | 14.1 and 24.4 s | 14.1 s both (the second: 0.08 s after waiting) |

Without it the four are prefilled side by side (the later ones pick up 4,096 cached tokens on the way).

**Pinning the prefix cache's memory at start (`QW_POOL_PREPIN=1`; off, not viable here).** `ArenaBank` pins the
RAM budget (plus 12%) as arenas the pools take and return, so serving pins nothing. The stalls it is meant to remove
are real: in the server tests a turn's TTFT goes from ~1.2 s to 3-5.6 s when an arena is pinned during it, 4-31 pins
per 22-turn session, 1.7-4.9 s per 265 MB arena with the spill and the table pinned. But pinning the 64 GB budget after
them took 335 s and 570 s (0.13-0.22 GB/s: the host has to reclaim page cache for every arena; the spill's 71 GB pins
in ~10 s because it goes first), the server's start timed out once, and on the dev box (which cannot mlock the PLE
table) the table was pushed out of the page cache: cold prefill 153 s against 26-31, decode 28-77 tok/s. With 0 pins
while serving, as designed. Left in as an option; what to try instead: pin ahead only while the engine is idle (the
standing reserve refilled on the scheduler's idle signal, not on a timer), or smaller arenas.

**A start waits for busy cards (`QW_START_WAIT_S`, off).** Instead of failing with out-of-memory when another
process holds the cards, the engine waits up to that long for them. Not deployed.
