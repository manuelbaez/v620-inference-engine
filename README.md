# qw

An inference engine for exactly one model on exactly one machine:
**Qwen3.8-Flash-Next** (180B hybrid MoE: GDN linear attention, QSA sparse
attention, hyper-connections, a 51B-entry n-gram table) on **4x AMD Radeon Pro
V620** (RDNA2, `gfx1030`, 32 GB each).

Every shape is a compile-time constant and every kernel is written for this
model on this chip. The goal is to beat the vllm-rdna2 fork (~56-65 tok/s
decode on the same box) and to keep prompt state, KV plus recurrent state,
reusable across turns without re-prefilling.

- [`docs/MODEL.md`](docs/MODEL.md): the exact forward pass (the spec).
- [`docs/DESIGN.md`](docs/DESIGN.md): parallelism, kernels, memory plan, the prefix
  cache, CacheBlend notes and the roadmap.
- [`docs/ENGINE_GUIDE.md`](docs/ENGINE_GUIDE.md): how the engine works, every feature
  with its effect and defaults, the decisions made, and a plan for porting to a new model.
- [`docs/PENDING_TESTS.md`](docs/PENDING_TESTS.md): benchmarks that gave no result yet and measurements still to do.
- [`docs/CODE_REVIEW.md`](docs/CODE_REVIEW.md): open improvement points found by reading the
  code, each with evidence, fix, effort and status.

## Status

| Phase | State |
|---|---|
| 1. Spec + CPU fp32 reference (`src/ref`) | done; matches vLLM (greedy identical at 57 and 4,266 tokens) |
| 2. Decode kernels, P2P collectives | done (`tests/gpu`) |
| 3. 4-GPU runtime (TP4 dense + EP4 experts) | plain decode **67 tok/s** single stream (HIP graphs per batch size), **prefill ~2,050-2,200 tok/s** (vLLM: ~56 / ~1,060); the four ranks load in parallel |
| 4. Prefix cache | per-slot reuse, VRAM snapshots, and a block store shared by all conversations in host RAM (128 GB) and on disk (survives restarts): any stored prefix up to a chat message boundary is restored in 0.02-0.5 s instead of re-prefilled (a shared 12k system prompt: 6.1 s -> 0.34 s); a restore from a cold disk takes as long as the disk reads (production's HDD mirror: 160-205 MB/s), see docs/DESIGN.md "Disk-tier loads and host memory"; the cache stays on even where saving costs more than recomputing (energy: a 64k-token recompute is ~21 kJ on the GPUs), and saving it is cheap since 2026-10-02 (buffers pinned two chunks ahead, one RAM copy per KV replica pair, non-coherent arenas): a cold 100k-token prompt through the store takes 54-57 s against 73-93 s before and 53 s with no store when the host's memory is in a good state (+49% in a bad one, DESIGN.md "Saving to the prefix cache and the prefill collectives") |
| 5. MTP speculative decoding | done: exact, adaptive draft count up to 5 (off while drafting does not pay), draft steps chained on the GPUs; **107-112 tok/s** single stream with Qwen's sampling settings (~117 greedy), **~231 tok/s** at 4 concurrent; on by default (`--mtp 5`) |
| 6. Prefill kernels and scheduling | the big collectives pushed by all threads (+6-8%), the next chunk's inputs prepared during the current one (+3%), two micro-batches per chunk, several waiting prompts per pass (batched prefill), long prompts in pieces between decode steps (interleaved prefill); from 65k tokens of context on the indexer's top-512 selection is split over the four cards (+26% at 230k tokens of context, +46% at 360k) |
| 7. OpenAI server, tokenizer, llama-swap entry | done: serving production through llama-swap and litellm; images and video (vision tower on every card, 720p in 0.33 s) |
| 8. CacheBlend-style reuse (experimental) | built and measured, rejected on quality (docs/DESIGN.md) |
| 9. Sampling | on the GPUs (penalties, candidates, Gumbel-max), exact; host fallback |
| 10. Thinking controls | `reasoning_effort` and thinking-token budgets per request, with server defaults |

The n-gram (PLE) table never goes to the GPUs. It stays pinned in host RAM and
the host gathers 16 rows per token. It comes in three precisions (int4 30 GB,
int8 54 GB, the original bf16 102 GB, and the official fp8 51 GB;
`--ple-dir`); production runs the int8 table since 2026-10-08 (docs/DESIGN.md, "PLE
n-gram table precision").

## Serving

`server/qwserve` is an OpenAI-compatible server (`/v1/chat/completions`,
`/v1/completions`, `/v1/models`, `/tokenize`, `/health`) in front of the engine
(`build/libqw_engine.so`, C API in `include/qw/capi.h`). It renders the
model's own `chat_template.jinja` and parses output like production vLLM's
`--reasoning-parser qwen3 --tool-call-parser qwen3_coder`. Token ids match
vLLM's `/tokenize` exactly on the test conversations (`server/tests/test_text.py`).

Requests run concurrently with continuous batching, one per engine slot
(default slots 262144, 65536, 32768, 32768 tokens; production since 2026-10-04: 6 slots of 512k tokens with KV
spill, 262144 tokens in VRAM in one and 45056 in each of the other five, 4096-token prefill chunks). Each slot keeps its
conversation's state (`--slots` are the tokens each slot keeps in VRAM; with `--kv-spill on` every slot holds up to
`--slot-max-tokens`, default 524288, the part beyond its VRAM tokens in pinned host RAM that the GPUs read directly:
docs/DESIGN.md "KV spill"; off by default), and a new request goes to the slot holding the longest
prefix of its prompt; one that fits no free slot waits without holding up
later ones that fit.

Thinking: `reasoning_effort` (none, minimal, low, medium, high, xhigh, max;
none turns it off) and a thinking-token budget (`thinking_token_budget`,
`thinking_budget_tokens` or Anthropic's `thinking.budget_tokens`) per request;
defaults `--reasoning-effort` (xhigh; production sets medium) and `--thinking-budget`
(-1, unlimited).
Sampling: the settings a request does not send (temperature, top_p, top_k, min_p, the
penalties) come from the model's `generation_config.json` (temperature 1.0, top-k 20,
top-p 0.95), as with vLLM; without that file: temperature 1, nothing else.
All settings and their defaults: docs/ENGINE_GUIDE.md, section 4.

`/health` answers 503 with a reason when the engine can no longer serve (a rank
failed, a collective timed out, an engine call has run for over 300 s, the
scheduler thread died), and the process then exits after 30 s so its supervisor
restarts it (`QW_WATCHDOG_EXIT=0` only reports). Media by URL is fetched from
public addresses only (`QW_MEDIA_ALLOW_PRIVATE=1` allows the LAN,
`QW_MEDIA_FETCH=0` allows `data:` URLs only); videos decode as a stream.

The server logs a stats line every 10 s while busy and one line per request,
and serves a dashboard at `/` (prompt and generation speed while busy, demand, queue, KV and memory use, request
times; JSON at `/metrics.json`). Through llama-swap:
`http://llm-backend-amd.local.net:8080/upstream/qw/qwen3.8-flash-next/`.

```sh
python3 -m venv ~/qwenv && ~/qwenv/bin/pip install tokenizers jinja2 aiohttp numpy pillow
server/ctl.sh start                             # port 8000; stop | restart | status
for t in admission prefill_order thinking marshal event_loop isolation health tool_calls fetch video api request_log; do
  ~/qwenv/bin/python server/tests/test_$t.py; done   # no GPU, no checkpoint (test_video needs ffmpeg)
server/tests/test_e2e.py --url http://host:8000 --model qw --compare ref.json
server/tests/test_concurrency.py --url http://host:8000 --model qw -n 4
```

Measured through HTTP (2026-09-24/25):

| | |
|---|---|
| 5.4k-token prompt | TTFT 2.97 s (1,816 tok/s), decode 58 tok/s |
| 36k-token prompt | TTFT 16.5 s (2,201 tok/s), decode 53 tok/s |
| follow-up turn on the same conversation | fully cached, TTFT 0.10-0.18 s |
| 4 concurrent requests | 223 tok/s aggregate with MTP (174 without) |

## Inputs

- `/mnt/llms/qwen3.8-flash-next-awq`: AWQ checkpoint (int4 experts, bf16 rest)
- `/mnt/llms/qwen3.8-flash-next-ple/ples_int4`, `ples_int8`, `ples_fp8`, `ples_bf16`: the
  n-gram table sidecar in four precisions (int4 is the `--ple-dir` default; bf16 is Qwen's
  original, `tools/ple_download.py` + `tools/ple_convert.py`; fp8 is from Qwen's FP8
  checkpoint, `tools/ple_download.py OUT Qwen/Qwen3.8-Flash-Next-FP8`; production runs int8)

## Layout

```
include/qw/capi.h      C API of the engine (libqw_engine.so)
src/core/              model constants and shard shapes, safetensors, JSON, n-gram hashing and table
src/ref/               fp32 CPU reference of the forward pass
src/kernels/           HIP kernels by component (hc, gdn, qsa, moe_*, ple, mtp, logits, gemm, gemv)
src/comm/              push-based P2P collectives
src/engine/            the 4-GPU engine: weights, buffers, prefill, batched decode, speculative decoding
src/session/           slots, prefix cache (snapshots, block store, disk tier), sampling, speculative generation
src/vision/            the vision tower (a copy on every card)
src/capi/              C API implementation
tools/                 qw_gpu / qw_ref runners, vLLM ground-truth helper, GPTQ and PLE table tools, task benchmarks
bench/                 P2P, GEMM and uncached-memory microbenchmarks; pinning, collectives, cold prefill through the store and GPU energy (run_suite.sh runs them in stages on the dev box)
tests/unit, tests/gpu  host unit tests; GPU tests (collectives, batched decode and prefill, speculative decoding, sampling, prefix cache, vision)
server/                OpenAI server (qwserve package), its tests, ctl.sh
docker/                container image for llama-swap (built on the serving box by scripts/build-image.sh)
scripts/               deploy to the GPU dev box, build the image
```

## Build and run

```sh
make                      # CMake build into build/ (HIP for gfx1030, C++17); make test runs host tests
scripts/deploy.sh         # tar the tree to hermes@llm-experiments and build there

# 4-GPU engine (needs the cards free: stop production first)
build/qw_gpu --tokens 760,6511,314,9338,369 --gen 32 --logprobs /tmp/gpu.txt
build/test_batch_decode --a 1,2,3 --b 4,5,6       # batched rows == single rows, bit-exact
build/test_speculative --p 1,2,3 --p 4,5,6        # MTP speculative == plain greedy

# CPU reference: greedy continuation plus per-position logprobs
build/qw_ref --tokens 760,6511,314,9338,369 --gen 8 --logprobs /tmp/ref.txt

# ground truth from the production vLLM
tools/vllm_logprobs.py tokenize "The capital of France is"
tools/vllm_logprobs.py logprobs 760,6511,314,9338,369 /tmp/vllm.txt
tools/vllm_logprobs.py compare /tmp/ref.txt /tmp/vllm.txt
```

Tests: `make test`, plus `tests/unit/ngram_ref.py build/test_ngram`, which checks
the n-gram hashing against a direct port of the reference implementation.
