# The qw engine: how it works, why, and how to build the next one

This document explains the qw inference engine end to end: what it does, every
feature with its effect on speed and output quality, the defaults, and the
decisions (kept and rejected) that got it here. It is written so that an agent
can read it and build an engine of the same quality for a newer open-weight
model on similar hardware. The detailed measurements live in
[DESIGN.md](DESIGN.md); the exact forward pass of the current model is in
[MODEL.md](MODEL.md). Numbers here are from 2026-09-24..27.

- [1. The method that worked](#1-the-method-that-worked)
- [2. Hardware and model](#2-hardware-and-model)
- [3. Architecture](#3-architecture)
- [4. Features, their effects and defaults](#4-features-their-effects-and-defaults)
- [5. KV slots](#5-kv-slots)
- [6. The prefix cache: VRAM, RAM and disk](#6-the-prefix-cache-vram-ram-and-disk)
- [7. Speculative decoding with the MTP head](#7-speculative-decoding-with-the-mtp-head)
- [8. Kernel and graph merges: what paid and what did not](#8-kernel-and-graph-merges-what-paid-and-what-did-not)
- [9. Precision decisions](#9-precision-decisions)
- [10. Scheduling and serving](#10-scheduling-and-serving)
- [11. Platform lessons](#11-platform-lessons)
- [12. Where it stands](#12-where-it-stands)
- [13. Porting to a new model: the plan](#13-porting-to-a-new-model-the-plan)

---

## 1. The method that worked

These practices mattered more than any single kernel. Reuse them.

1. **Write the spec first.** [MODEL.md](MODEL.md) derives the exact forward
   pass from the checkpoint headers and a reference implementation. When the
   two disagree, the checkpoint wins. Every shape, norm variant, rotary layout,
   hashing rule and quantization layout is written down before any kernel.
2. **Build an fp32 CPU reference** (`src/ref`, `tools/qw_ref`). It is slow
   (4,000 tokens in ~10 minutes) but it is the ground truth for everything
   else. Validate it against the upstream implementation (here: vLLM's greedy
   output and logprobs) once.
3. **One accuracy metric, used everywhere:** teacher-forced next-token
   log-probabilities of a fixed held-out text, compared with the reference:
   mean |Δlogprob| and top-1 agreement (`qw_gpu --logprobs`, then
   `tools/`-style paired comparisons). Use 4,000 tokens of mixed text (prose,
   code, license text), not 500: at 500 tokens the 95% confidence interval is
   about ±0.015 nats, too wide to tell 8-bit from 16-bit weights. Always report
   paired differences with a bootstrap CI.
4. **Know the noise floor before judging a change.** fp16 rounding depends on
   GEMM shapes, so the same prompt prefilled in two different chunkings
   differs by up to ~1-2 in max |Δlogit| and can flip greedy near-ties. A
   change is only suspicious if it moves the metric beyond that floor. For
   exactness checks use paths that must be bit-identical (the same kernels on
   the same shapes).
5. **Determinism is a hardware detector.** The engine is deterministic for a
   fixed path, so a greedy run that does not repeat itself exactly means the
   hardware computed something wrong. This found the undervolt problem
   (section 11).
6. **Measure before building, A/B after.** Every optimization here was
   estimated from a profile or a microbenchmark, then measured against the
   build without it, alternating runs, several repetitions. Many "obvious"
   wins measured as zero or negative and were reverted (section 8).
7. **Accuracy before speed.** Output-changing optimizations (lower precision)
   stay off unless they are within the fp16 noise of the reference. The owner
   of this engine chose that explicitly; int8 weights were rejected on it.
8. **Record everything** in DESIGN.md with numbers, including what was
   rejected and why, so nobody repeats it.

---

## 2. Hardware and model

**Hardware:** 4x AMD Radeon Pro V620 (gfx1030, RDNA2, 32 GB each, ~506 GB/s,
no matrix cores, no bf16), PCIe Gen4 x16 each on its own NUMA node, no
NVLink-style link: GPU-to-GPU traffic is PCIe peer-to-peer (~25 GB/s kernel
push). Host: 48 threads, 256 GB RAM (containers limited to 240 GiB), ZFS pool
for models and the disk cache. Power-capped to 160 W per card with a -50 mV
undervolt (section 11).

**Model:** Qwen3.8-Flash-Next (details in [MODEL.md](MODEL.md)):

- 48 layers: 36 **GDN** (gated delta-net, a linear-attention recurrent layer
  with a fixed-size state) and 12 **QSA** (sparse attention: an indexer scores
  compressed keys, the top blocks are attended). Only QSA layers have a
  growing KV cache.
- **Hyper-connections (HC):** 4 residual streams of width 2560 mixed by small
  low-rank mixers before every sublayer (97 mixes per token).
- **MoE:** 512 routed experts, top-10, intermediate 640, plus a shared expert.
- **PLE:** a per-layer n-gram embedding at layer 1: hashed 2- and 3-gram ids
  index a 320M-row table (16 rows of 160 values per token).
- **MTP head:** one extra QSA layer that predicts the next token from the last
  hidden state; used for speculative decoding.
- **Vision tower:** a Qwen3-VL ViT (27 layers, 0.41 B parameters).
- Checkpoint: routed experts int4 (AWQ, group 128), everything else bf16
  (run as fp16). The PLE table is a separate sidecar (section 9).

---

## 3. Architecture

**Layers of the software:**

| layer | where | role |
|---|---|---|
| kernels | `src/kernels` (HIP) | GEMVs, MoE, GDN, QSA, HC, PLE, vision, sampling |
| collectives | `src/comm` | custom P2P push collectives |
| engine | `src/engine` | per-GPU rank threads, weights, prefill, decode graphs, MTP, host-tier copies |
| session | `src/session` | slots, prefix cache (snapshots, block store, disk tier), sampling, speculative generation |
| C API | `include/qw/capi.h`, `src/capi` | the library boundary |
| server | `server/qwserve` (Python) | OpenAI-compatible HTTP, chat template, tokenizer, vision preprocessing, scheduler |

One process, one host thread per GPU. The Python server calls the C library
through ctypes from a single scheduler thread.

**Parallelism** (DESIGN.md "Parallelism"):

- **Dense layers: tensor parallel x4.** Each card holds a quarter of every
  GDN/QSA/HC/shared-expert matrix and of the vocabulary (lm_head).
- **Routed experts: expert parallel x4** (128 per card). TP would split the
  640-wide intermediate into 160, straddling the int4 group of 128.
- **The residual stream is sharded along the hidden dimension:** each card
  owns 640 of every stream's 2560 columns, so no card streams all HC weights.
- Per HC mix this needs three collectives: one small all-reduce (the down
  projection is linear, so partial products plus partial sums of squares give
  the result in one ~5 KB exchange), an all-gather of the block input, and a
  reduce-scatter of the block output. That is ~291 collectives per decoded
  token, which is why collective latency is the main engineering problem on
  a PCIe-only box.

**Collectives** (`src/comm`): kernel-initiated **pushes** into uncached
receive buffers on the peer (pull is 2-3x slower on this box), payload and
sequence flag to the same destination, the owner polls its own VRAM, fixed
rank order reduction in fp32. Measured rules: small payloads go to all peers
at once; big payloads rotate destinations (one kernel pushing to 3 peers at
once was ~6x slower); big contiguous payloads go as one flat copy (+4-10%
prefill). A chained small all-reduce costs ~15 µs, mostly the PCIe round trip.

**Decode step** (`src/engine/decode.hip`): one HIP graph per batch size
(1-16 rows) and kind (plain, state-saving for verification, MTP), so a step is
one launch per GPU. Dense GEMVs are wave-per-row fp16 with `v_dot2_f32_f16`
at ~86% of the card's bandwidth. Routed experts for small batches use a
per-(token, expert) GEMV kernel (the tiled prefill GEMM padded every expert to
32 tokens and cost half the step). Each row's log-sum-exp is computed per
vocabulary shard on the GPUs.

**Prefill** (`src/engine/prefill.hip`): rocBLAS fp16 GEMMs (fast on gfx1030
only with fp16 output: 30-36 vs 6-11 TFLOPS), a tiled grouped int4 expert GEMM
that dequantizes into LDS, the router kept in fp32, each chunk split into two
micro-batches on two streams with separate collective channels so one half's
collectives overlap the other's compute, snapshot captures inside the chunk,
and segments of several slots in one pass (batched prefill).

**The PLE table** stays in host RAM (never on a GPU): the host hashes the
n-gram ids and gathers 16 rows per token; the result rides along with the
token embedding.

---

## 4. Features, their effects and defaults

"Default" is the code default (server arguments / environment); "production"
is what the deployed engine runs (llama-swap command + compose environment).
Speed figures are single-stream decode unless stated.

| feature | what it does | speed effect | quality effect | default | production | knob |
|---|---|---|---|---|---|---|
| Decode graphs | one HIP graph per batch size | ~3.3 µs saved per dependent kernel (1,585 kernels/step) | none | on | on | `QW_NOGRAPH=1` off |
| Fused small collectives | push + wait + reduce in one kernel | 15.0 -> 14.8 ms/step | none (bit-identical) | on | on | `QW_COMM_SPLIT=1` old |
| Flat pushes for big payloads | one copy instead of a block per row | prefill +4-10% | none | on | on | `QW_COMM_ROWWISE=1` old |
| Two prefill micro-batches | collectives of one half overlap the other | prefill +7% | rounding only | on | on | `QW_NO_SPLIT=1` |
| Prefill chunk | tokens per prefill pass | 8k vs 4k: ~2% faster prefill | rounding only | 8192 | **4096** (frees ~0.9 GB/card for KV) | `QW_PREFILL_CHUNK` |
| Batched prefill | several waiting prompts in one pass | bursts +4-7%, 4 prompts 2.05 -> 1.62 s | rounding only | on | on | `QW_PREFILL_BATCH=0` |
| Interleaved prefill | long prompts go in pieces between decode steps; short prompts go first, and a lone long prefill yields to arrivals | longest stall 12.4 -> 1.15 s; decoders keep ~22 tok/s | none | piece 2048, decode share 0.25 | same | `QW_PREFILL_PIECE`, `QW_DECODE_SHARE` |
| KV slots | sequences held at once, each with a KV capacity | more slots = more concurrency | none | 262144,65536,32768,32768 | **262144,131072,65536,32768** | `QW_SLOTS` / `--slots` |
| Prefix cache, VRAM | reuse a slot's own history and snapshots | skips prefill of reused tokens | none (exact) | on | on | |
| Prefix cache, RAM (block store) | 256-token KV blocks + state snapshots shared by all conversations | 12k shared system prompt: 6.1 -> 0.34 s | none (exact) | 128 GB | 160 GB, which the host did not hold (OOM kill 2026-10-01); 128 GB recommended (see Host RAM below) | `--host-cache-gb`, `QW_HOST_CACHE_GB` |
| Prefix cache, disk | blocks and snapshots survive restarts | 4k tokens restored in 0.55 s after restart (dev box); production's cache is on a 2-HDD ZFS mirror: cold loads 160-205 MB/s, 4-6 GB in 21-38 s, ARC-resident GB/s | none (exact) | 200 GB | 200 GB (`/cache` volume, ZFS on HDDs) | `--disk-cache-dir`, `--disk-cache-gb` |
| Snapshot spacing | captures at chat message starts at least N apart | | none | 1024 | 1024 | `QW_SNAP_MIN_GAP` |
| MTP speculative decoding | drafts up to K tokens, verified exactly | 67 -> 107-112 tok/s (Qwen sampling) | none (exact sampling) | K = 5 | 5 | `--mtp` (0 off) |
| Adaptive draft count | K per request from its running acceptance | avoids drafting that does not pay | none | on | on | `QW_SPEC_BASE`, `QW_SPEC_COST` |
| MTP off when it does not pay | plain decoding stretches, then catch up | hard text 63 -> 74.5 tok/s | none | on | on | |
| Chained drafts | draft steps without host syncs | +~5% speculative | none | on | on | |
| Verification row cap | at most 8 rows per verification batch | | none | 8 | 8 | `QW_SPEC_MAX_ROWS` |
| GPU sampling | penalties + candidates + Gumbel draw on the GPUs | +11-18% under sampling settings | same distribution (different random draws) | on | on | `QW_GPU_SAMPLING=0` |
| Parallel sampling on host | fallback/host path samples rows in parallel | host sampling 16 -> 3 ms/step | none | on | on | |
| Parallel weight loading | the 4 ranks load at once | startup 105 -> 33 s (warm) | none | on | on | `QW_LOAD_SERIAL=1` |
| PLE table | n-gram table precision in RAM | none | int4 0.065, int8 0.048, bf16 0.044 mean \|Δlogprob\|; no task difference int4 vs int8 | int4 path in `--ple-dir` default | **bf16** (102 GB RAM) | `--ple-dir` |
| PLE pinning | table locked in RAM at startup | avoids disk reads per token | none | on | on | `QW_PLE_PIN=0` |
| Thinking effort | `reasoning_effort` none..max mapped onto the template's low/medium/xhigh; none = no thinking | fewer thinking tokens at lower effort (an instruction, not a limit) | the model's own levels | xhigh | **medium** (compose `QW_REASONING_EFFORT`) | `--reasoning-effort`, `QW_REASONING_EFFORT`; per request `reasoning_effort` / `chat_template_kwargs` |
| Thinking budget | hard cap on thinking tokens, then Qwen's closing text | bounds latency | tight caps cost correctness on hard problems | unlimited | unlimited | `--thinking-budget`, `QW_THINKING_BUDGET`; per request `thinking_token_budget`, `thinking_budget_tokens`, `thinking.budget_tokens` |
| Vision | ViT copy on every card, lazy encoding, LRU cache | 1080p image ~1.2 s | as HF | on | on | `QW_NO_VISION`, `QW_VISION_CACHE_GB` (2), `QW_VISION_MAX_PIXELS` (1920x1088) |
| int8 dense weights | W8A16 copies of dense matrices | +13% at 1 row, 0% at 8 | **worse**: +0.012..0.033 | off | off | `QW_INT8_DENSE=1` (+ `QW_INT8_DIR` GPTQ) |
| W4A8 experts | int8 activations for the expert GEMM in prefill | prefill +2% | **worse** (0.075 -> 0.104) | off | off | `QW_INT8_EXPERTS=1` |

| Logs | stats line every N s while busy, one line per request, errors with tracebacks | none | none | 10 s | 10 s | `QW_LOG_INTERVAL` (0 off) |
| Dashboard | page at `/` (llama-swap's model link), JSON at `/metrics.json`, collected on its own thread | none measurable | none | on | on | |
| Background disk loads | a prompt whose cache is on disk waits while threads read it into RAM; others keep running, including prompts whose cache is all in RAM | removes a ~54 s stall per 97k-token disk restore | none | on, 1 thread | on, 1 thread | `QW_LOAD_THREADS` (parallel file reads, to be measured) |
| Health and watchdog | `/health` answers 503 with a reason when the scheduler thread died, an engine call has run over `QW_STUCK_SECONDS` (a wedged GPU) or a rank failed / a collective timed out (permanent); a watchdog exits the process after `QW_EXIT_GRACE` s so the supervisor restarts it | a failed or wedged engine is restarted instead of failing every request behind a healthy-looking server | none | on (300 s, 30 s) | on | `QW_STUCK_SECONDS`, `QW_EXIT_GRACE`, `QW_WATCHDOG_EXIT=0` |
| Media by URL | an image or video URL is fetched only when every address of its host is public, and each redirect is checked; videos decode as a stream (one pass counts the frames, one keeps the sampled ones) | a 60 s 1080p30 video: ~11 GB of server memory -> ~0.1 GB; 300 frames of 720p: 1,631 -> 69 MB | none (identical pixels) | public addresses only | public addresses only | `QW_MEDIA_FETCH=0` (data: URLs only), `QW_MEDIA_ALLOW_PRIVATE=1` |

Diagnostics: `QW_TRACE` (step phase timings, sampling fallbacks), `QW_PROFILE`
(prefill timings), `QW_GUARD` (allocation guard zones), `QW_NOGRAPH`.

---

## 5. KV slots

A **slot** is one sequence the engine can hold on the GPUs: its QSA K/V and
indexer keys (grow with length), its GDN recurrent state and conv rings, its
PLE conv ring and its MTP state (fixed size). Every decode step is a batch of
rows from any slots, so the number of slots is the number of requests that
can run at once. Each slot has its own capacity in tokens.

- **KV cost:** ~20.8 KB per token per card (13 QSA layers including the MTP
  layer: K and V fp16 of one head, fp32 raw indexer keys, fp16 compressed
  keys). 256k tokens = ~5.5 GB per card.
- **Fixed cost per slot:** GDN state (~28 MB per card) plus rings and a
  256-position MTP history.
- **Why mixed sizes:** one very long context (opencode sends up to 262k) plus
  smaller slots for parallel side requests. All-large slots do not fit: a
  second 256k slot needs 2.7 GB more per card. The current set
  (256k + 128k + 64k + 32k) fits because the prefill chunk was halved to 4k,
  which freed ~0.9 GB per card of prefill buffers (prefill 2% slower); ~1.1 GB
  per card stays free. Running a card within ~0.2 GB of full caused stalls and
  out-of-memory errors in testing, so keep ~0.5 GB spare.
- **Admission** (`server/qwserve/scheduler.py`): a request gets the free slot
  that can hold its prompt plus `max_tokens` (default reserve 2048), preferring
  the one that can reuse most of the prompt (section 6). A request that fits
  no free slot waits, and later requests may take the free slots that do fit
  them; waiting requests are offered freed slots in arrival order, so a
  request waiting for the big slot is not starved.
- **Why 4 slots:** throughput grows with rows per step (1 row 14.7 ms, 4 rows
  21.4 ms, 8 rows 26.9 ms), but more slots cost KV memory; 4 matched the use.

---

## 6. The prefix cache: VRAM, RAM and disk

GDN layers make this model's cache different from a transformer's: the
recurrent state **cannot be rebuilt from the KV**, so a prompt can only resume
at a point where the state was saved (a snapshot). Everything below follows
from that.

**What is saved:**

| part | size | property |
|---|---|---|
| QSA K/V + indexer keys | ~20.8 KB/token/card | append-only, position-indexed |
| GDN state + conv rings + PLE ring + MTP input | ~31.5 MB/card per snapshot | fixed size, only valid at its position |

**Tiers:**

1. **VRAM (the slot itself).** A new prompt that extends what a slot holds
   (the usual chat or agent turn) continues in place. VRAM snapshots (8, any
   slot) let a slot rewind to a saved point (e.g. the reply was re-rendered
   differently by the chat template).
2. **Host RAM: the block store** (`src/session/block_store.cpp`). Prompts are
   cut into 256-token blocks keyed by a hash chain (parent key + tokens;
   tokens compared on a hit). Blocks hold the KV of every card; snapshots hold
   the state at a block end or any leaf position. A prompt resumes from the
   deepest snapshot on its path, whichever conversation stored it, so two
   conversations sharing a long system prompt share its cache. Snapshots are
   **captured inside prefill chunks without splitting them** (the GDN scan
   writes its state after a given token; small kernels copy the rings) at
   chat message starts (`<|im_start|>`) at least 1,024 tokens apart, at chunk
   ends and at the prompt end. Pinned host memory comes from ~250 MB arenas
   pinned in the background while the GPUs prefill (pinning on demand stalled
   prefill 0.2-0.4 s; an arena took 0.7-1.5 s to pin on 2026-10-01 with the four
   pools pinning at once), and unpinned by the pool's thread once empty (never in
   the caller). The pool must serve several callers at once (the scheduler's
   saves and the disk-load threads): a caller that wakes to find the arena's
   units taken has to ask again, and a returned unit has to wake a waiting
   caller, or a request waits forever (found 2026-10-01). LRU eviction;
   budget 128 GB. The budget counts what nodes hold: loads, saves' reserved
   buffers and buffers still waiting for their disk write come on top (observed
   ~11 GB at 160 GB).
3. **Disk** (`src/session/disk_tier.cpp`): every block and snapshot is also
   written to a file by a background thread; the index is rebuilt at startup;
   the server persists every slot's newest snapshot at shutdown. Budget 200 GB.
   Loads run on threads (`QW_LOAD_THREADS`, default 1) into pinned buffers
   they take themselves, one load at a time; a prompt whose entries are all in
   RAM is never held up by another prompt's load. Speed is the medium's: on
   production's two-HDD mirror a cold load streams at 160-205 MB/s (1.7x
   faster than recomputing, and the HDDs are shared with the cache writer);
   fast storage for `--disk-cache-dir` is the largest lever. The load log line
   splits time into pinned buffers and reading.

**Measured:** a second conversation sharing a 12k-token system prompt: 6.1 ->
0.34 s. Restore of 4k tokens: 0.02 s from RAM, 0.55 s from disk after a
restart. In production (a day of opencode use) **96% of prompt tokens came
from the cache**; prefill was ~20% of engine time.

**Exactness:** a restore is bit-identical to the saved state. A mid-chunk
capture differs from a prefill split at that point only by chunking rounding.

**Rejected: CacheBlend-style reuse of non-prefix chunks** (reusing a document's
KV after a different prefix). For GDN layers the state transfer across a chunk
is an affine map S_out = M S_in + U that can be computed exactly for one layer,
but the recomposed deep states differed from true prefill by 2-300x the
noise floor. Exact prefix reuse only. See DESIGN.md.

---

## 7. Speculative decoding with the MTP head

The checkpoint's MTP head is one more QSA layer that predicts token t+2 from
the hidden state at t and the embedding of t+1. The engine uses it for
speculative decoding:

- **Drafting:** one MTP pass over the step's kept tokens, then single-row
  passes chained through the MTP output. Argmax per vocabulary shard on the
  GPUs.
- **Verification:** the pending token plus K drafts go through the full model
  as consecutive rows of one batch (the "state-saving" graph writes the GDN
  state after every row). Row j is sampled from the full model and kept only
  while it equals draft j, so every emitted token is an exact sample from the
  model (at any temperature); greedy output equals plain decoding.
  `accept(slot, n)` restores the state after row n-1 (KV and rings are
  position-indexed; the rings are large enough that rejected rows never
  overwrite a needed row).
- **The MTP layer runs in prefill too** (shifted by one row), so drafting can
  start right after a prompt.
- **Adaptive K:** each request keeps a running per-draft acceptance a and uses
  the K that maximizes expected tokens per step cost,
  (1 + a + ... + a^K) / (1.25 + 0.25 (K - 1)), with the cost model measured on
  this box (15.0 ms plain; 18.8 / 22.6 / 26.4 ms with 1 / 2 / 3 drafts).
- **MTP off when it does not pay:** when no K is worth it, the request decodes
  plainly for a stretch (32 steps, doubling while it keeps not paying). Decode
  keeps each row's MTP input in a 256-position ring per slot, so the skipped
  MTP rows are caught up before drafting resumes.
- **Draft limit 5** (was 3): single-stream 100-106 -> 111-112 tok/s with Qwen's
  sampling settings, greedy 110 -> 117; 7 was no better. With several requests
  the 8-row verification cap leaves each one draft.
- **Acceptance** measured greedy: ~60% per draft on prose, ~85% on code, ~92%
  on boilerplate (2.35-3.53 tokens per step with K = 3).

**Why the draft steps were merged ("chained drafts").** Profiling a
speculative step showed each draft step = 1.36 ms of GPU work + 0.27 ms of
host round trip (copy the argmax parts back, pick the token on the host, look
up its embedding, stage the next step). The fix keeps the per-step graphs but
enqueues all draft steps back to back: the ranks all-gather their argmax parts
and a small kernel (`draft_pick`) picks the token and reads its embedding row
straight from the embedding table in pinned, device-mapped host memory (no
VRAM spent on a 1.27 GB table). The collectives' sequence base is advanced on
the device between steps (`Comm::bump_base`), because the host-side staging
word can only be rewritten after a sync. One sync per draft instead of one per
step: +~5% speculative decoding, identical drafts. A single graph for all
steps would add little over this and needs a graph per (rows, requests, steps)
combination.

**Open idea:** drafting over a reduced vocabulary (the ~32k most frequent
tokens) would cut the lm_head share of a draft step ~8x; output unchanged,
acceptance to be measured.

---

## 8. Kernel and graph merges: what paid and what did not

On this card a dependent kernel inside a graph costs ~3.3 µs, and a decode
step has ~1,585 kernels, so merging looked like a big lever. It mostly was
not: the time is in the collectives' PCIe round trips and in kernels on the
critical path, and moving work into a critical kernel costs about what the
launch saved.

| merge | result | kept |
|---|---|---|
| Whole decode step as one HIP graph per batch size | the baseline for everything; ~56 -> 67 tok/s with fusions below | yes |
| Collective push + wait + reduce in one kernel (`exchange_small_kernel`) | 18.8 -> 15.7 µs per chained all-reduce; step 15.0 -> 14.8 ms | yes |
| Big multi-row payloads as one flat copy | prefill +4-10% | yes |
| HC down-projection algebra: one all-reduce per mix for d, injection logits and all rms scales | 3 collectives per mix instead of ~5 | yes |
| MoE routing + silu*up + weighted combine in the expert kernels | expert GEMV near bandwidth | yes |
| GDN conv step, l2norm, gating, delta rule, gated norm in one decode kernel | | yes |
| Snapshot captures inside the GDN scan and ring kernels | prefill 16k with 8 captures -1.3% | yes |
| Shared expert SwiGLU inside its down projection | no gain at 1 row, slower from 4 | no |
| GDN conv-ring update inside the scan | 1 row within noise, 4 rows 22.0 vs 21.4 ms (slower) | no |
| Shared expert on a second stream inside the graph (fork/join) | 14.6 -> 16.4 ms: cross-stream graph edges cost more than they hide | no |
| One collective fewer per sublayer (apply the next mix's down projection to the unreduced output) | not possible: the next mixer's RMS norm needs the reduced residual | no |
| Collective push fused into the producer GEMV, wait into the consumer | estimated small: the round trip dominates and a launch gap already overlaps it | not built |
| Chained MTP draft steps without host syncs | +~5% speculative | yes |
| Sampling on the GPUs after the decode graph | +11-18% under sampling settings | yes |

---

## 9. Precision decisions

The bar: stay within the fp16 noise of the fp32 reference. Results
(teacher-forced, 4,000 held-out tokens unless stated):

| choice | mean \|Δlogprob\| | decision |
|---|---|---|
| fp16 dense weights, fp32 accumulate, fp32 router, fp32 GDN state | 0.042-0.044 | baseline |
| fp16 router logits | 0.074 -> 0.088 (older reference) | router stays fp32 |
| fp16 HC-down partials and all-reduce, fp16 shared-expert output | neutral | kept |
| int8 dense weights, round to nearest (group 32) | +0.033 (500 tokens) | rejected |
| int8 dense weights, GPTQ group 32 | +0.012 (CI +0.009..+0.016) | rejected |
| int8 dense weights, GPTQ group 16 | +0.018 on 1,594 tokens | rejected |
| W4A8 experts in prefill | 0.075 -> 0.104 | rejected (opt-in) |
| PLE table int4 -> int8 | 0.065 -> 0.048 | **int8 adopted** |
| PLE table bf16 | 0.044 | ran in production 2026-09-27/28 (102 GB RAM) |
| PLE table fp8 (official) | 0.050 | **production** (51 GB RAM, cache 160 GB) |

Why int8 weights failed here: every dense group alone costs about as much as
all of them together. The 36 recurrent GDN layers carry any weight error
forward, so the errors do not add, they saturate, and no subset kept in fp16
closes the gap. The speed was also smaller than estimated (+13% at 1 row, 0%
at 8 rows) because decode is only partly weight-bound.

Why the PLE table mattered: it was the one int4 component that did not have to
be int4 (it lives in host RAM), and its error (9% relative) was larger than
all of the engine's fp16 arithmetic. See DESIGN.md "PLE n-gram table
precision" for the task benchmarks.

**GPU sampling** keeps the exact distribution: penalties from per-slot token
counts on the GPUs (bit-identical penalized logits), then per vocabulary shard
the max, normalizer, top 256 candidates and a Gumbel-max draw. The host draws
exactly like the CPU sampler from the candidates (same token for the same seed
with top-k/top-p/min-p); plain temperature uses the Gumbel draws (same
distribution, different random numbers); a nucleus wider than the candidates
fetches that row's full logits (~1 row in 10,000 in real text).

---

## 10. Scheduling and serving

- **Continuous batching** (`scheduler.py`): one scheduler thread admits
  requests to slots, advances prefills, and runs batched generate steps (up
  to 16 rows).
- **Interleaved prefill:** while others decode, a prefill advances one 2,048
  token piece at a time, then the running requests decode for 25% of the
  piece's time. A 30k-token prompt no longer freezes other streams for 12 s.
  With nothing decoding a prefill goes a chunk at a time and yields when a
  request arrives; prompts that fit whole in a piece go before a long one in
  progress (a 65k-token prefill once kept new requests waiting for 95 s).
- **Batched prefill:** prompts waiting together go into one prefill pass up
  to a piece (a prefill chunk when nothing decodes), paying the pass's fixed
  cost (~80 ms of collectives) once.
- **Per-slot prompt logits:** each slot keeps its own prompt logits (a fully
  cached prompt used to take another admission's logits).
- **Vision:** images and video are preprocessed in the server exactly as HF's
  Qwen3-VL processor (tested against dumps), encoded lazily only for tokens
  that are actually prefilled, and cached by content hash; vision tokens get
  content-derived ids so the prefix cache never confuses two images. Videos
  are decoded as a stream, so a long one never sits in memory, and media URLs
  are fetched from public addresses only (section 4, "Media by URL").
- **The event loop stays free** (each of these used to delay every stream by
  5-400 ms): the chat template and the tokenizer run on a worker thread
  (`Server.prepare`; `encode_batch`, because `Tokenizer.encode` keeps the GIL
  even there), a request's prompt is converted to int32 once (`Tokens`, 5 ms
  per call at 100k tokens with the GIL held, and the scheduler used to do it
  every pass for every waiting or loading request), and a request that found no
  slot is not offered to the engine again until one is released.
- **Failure isolation and health:** a request the engine cannot admit fails
  alone (it used to fail every request in flight and sit at the head of the
  waiting list, raising on every pass). `/health` reports the scheduler thread,
  an engine call that runs for minutes and the engine's own sticky failure
  (`Engine::failure()`), and a watchdog exits the process when that persists.
- **Tool calls:** a parameter's value may contain the format's own tags
  (a file that documents it); a value ends at a `</parameter>` followed by the
  next parameter or the end of the call, a forgotten closing tag is tolerated
  for declared parameter names, and a call cut off by `max_tokens` keeps its
  partial last value.
- **Timings** in llama.cpp format per request (llama-swap's activity log), so
  real use can be analyzed (prefill vs decode share, cache hit rate).

---

## 11. Platform lessons

- **Undervolt:** at a 160 W cap, -75 mV produced silent wrong tokens (plain
  decoding not repeating itself, clear picks flipped) and was very likely the
  cause of a long-standing "intermittent wrong tokens / GPU page faults" issue.
  0 and -50 mV were clean. Speed was identical at 0 / -50 / -75 mV (decode is
  memory-bound, prefill collective-bound); -50 mV saves ~8% power. Test
  undervolts with a determinism detector, not by "did it crash".
- **170 vs 160 W:** +2-4% on benchmarks, nothing in real use.
- **SDMA disabled** (`HSA_ENABLE_SDMA=0`): an SDMA code-object wedge hung a
  rank; blit copies are used.
- **PCIe peer push** is the right primitive; avoid many blocks spinning on
  uncached flags.
- **Memory headroom:** keep ~0.5 GB free per card; running within ~0.2 GB caused
  stalls.
- **Host RAM:** PLE table (bf16: 102.4 GB pinned) + block store (up to 128 GB)
  + model page cache; the container limit was raised to 240 GiB. With the table
  pinned, the model files no longer all fit in the page cache, so a warm start
  reads part of them from disk again (~120 s instead of ~60 s with int8). The
  limit that matters is the host's (251 GB, shared with other services): the
  host OOM killer took the engine twice when everything together passed it.
  Production runs the fp8 table (51.2 GB) with a 160 GB cache, and it was
  OOM-killed a third time on 2026-10-01 (the cache full: engine 159 GiB
  anonymous + 47.7 GiB table + the `lmcache` container back at 16.8 GiB + other
  services on 251 GiB; page cache 8 MB). 128 GB is the recommended budget.
  Budget = host RAM - table - ~11 GB engine - other services - margin, where
  the margin also covers a disk-tier load (up to ~9 GB taken before the budget
  is enforced). A kill empties the RAM tier, so the next requests load from
  the disk tier. Inside the engine's container `/proc/meminfo` is the host's.
- **Readiness:** the engine reports ready (and the server answers `/health`)
  only after the table is in RAM; before, a cold start served requests while
  rows still came from disk (warmup 28-35 s instead of ~5 s).

---

## 12. Where it stands

| | |
|---|---|
| single stream, Qwen sampling (T 0.7, top-p 0.8, top-k 20) | 107-112 tok/s |
| single stream, greedy | ~117 tok/s |
| plain decode (no drafts) | ~67 tok/s (14.7 ms/step) |
| 4 concurrent requests | ~231 tok/s aggregate |
| prefill | ~2,050-2,200 tok/s |
| startup (warm) | ~120 s with the bf16 table (~60 s with int8; the weights themselves ~33-45 s, loaded in parallel) |
| accuracy vs fp32 reference with the original PLE table | 0.050 mean \|Δlogprob\| (production's fp8 table; 0.044 with bf16) |
| tasks (int4 / int8 table) | GSM8K 94.8 / 94.6%, MMLU 86.6 / 86.0%, ARC-Challenge 97.2 / 97.2% (no significant difference) |
| the previous production engine (vLLM fork) | ~56-65 tok/s, ~1,060 tok/s prefill |

Where the remaining time goes (decode step, one card): ~4.9 ms dense GEMVs at
~86% of bandwidth, ~3.5 ms other compute, the rest collective latency and
launch gaps. Remaining ideas are small: reduced-vocabulary drafts (~5%), a
chunked GDN prefill kernel, fp8 KV (capacity, not speed; changes output), and
NVFP4 experts (possibly closer to the original than AWQ int4; costs ~1.5 GB
per card and software FP4 decode on RDNA2; see DESIGN.md roadmap).

---

## 13. Porting to a new model: the plan

Follow this order; each step has a gate before the next.

1. **Read the checkpoint** (config, safetensors headers) and the reference
   implementation. Write MODEL.md: every shape, norm, rotary layout,
   attention variant, recurrent layer, MoE routing rule, quantization layout,
   special embeddings, MTP/draft heads, vision tower.
2. **Byte and memory budget:** bytes read per decoded token per component and
   per card (DESIGN.md "Byte budget"), weights per card, KV bytes per token
   per card, fixed state per sequence. This decides the parallel layout and
   the slot sizes.
3. **fp32 CPU reference** and its validation against upstream greedy output
   and logprobs. Build the held-out 4,000-token evaluation set and the paired
   comparison tools now.
4. **Parallel layout:** TP for dense parts, EP for experts, shard the residual
   if there are multiple streams; count the collectives per token and design
   them down first (algebra like the HC all-reduce).
5. **Collectives:** measure the P2P matrix (push vs pull, latency, the
   all-to-all-at-once penalty) before writing them.
6. **Kernels, correctness first:** each against the reference, then speed.
   Decode: bandwidth-bound GEMVs and fused small layers, one graph per batch
   size. Prefill: vendor GEMMs with the fast output type, tiled expert GEMMs,
   two micro-batches.
7. **Gate: accuracy** within the fp16 noise of the reference, greedy
   continuations matching upstream.
8. **Slots, batched decode** (bit-identical rows), then **speculative
   decoding** if the model has a draft head (exact verification, adaptive K,
   off when it does not pay, chained drafts), then the **prefix cache** (for
   recurrent layers: snapshots are mandatory; capture inside prefill chunks),
   RAM and disk tiers.
9. **Server:** OpenAI API, chat template, continuous batching, interleaved and
   batched prefill, GPU sampling, per-request timings for later analysis.
10. **Precision choices last**, each measured against the reference on 4,000
    tokens and, for anything shipped, on task benchmarks (`tools/bench_tasks.py`).
11. **Platform:** validate power and undervolt settings with a determinism
    detector (`test_speculative --repeat`), not crash tests.
12. **Record** every result, kept or rejected, in the design doc.
