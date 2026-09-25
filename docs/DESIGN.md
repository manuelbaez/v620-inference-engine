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
  on demand and freed when empty (a spare is kept). Pinning costs ~0.05-0.1 s
  per arena, so a prefill tells the store what its saves will need and the
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

1. **int8 dense weights (W8A16).** Dense fp16 weights are 88% of the bytes
   read per token. Per-channel int8 copies halve the GEMV time (~-2.4 ms per
   step, ~66 -> ~80 tok/s plain, MTP scales with it). Gate: mean |dlogprob|
   against the fp32 reference (`qw_gpu --logprobs`) must stay near today's
   0.074 (fp16), and greedy output should match on the test prompts.
2. **Fuse the collectives into their producers and consumers.** Each
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
8. **GPU-side sampling.** Top-k/top-p/min-p sampling costs 0.2-0.7 ms per
   token on the host at temperature > 0; doing it per vocab shard on the GPUs
   (like the log-sum-exp) saves most of that.
9. **Prefill.** ~2,000 tok/s. Profile a long prefill the same way before
   choosing: candidates are the chunked (WY) GDN kernel, the fp32-output
   router GEMM (rocBLAS falls back to a slow HSS kernel), and QSA attention.
