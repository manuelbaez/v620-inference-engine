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

## Status

| Phase | State |
|---|---|
| 1. Spec + CPU fp32 reference (`src/ref`) | done; matches vLLM (greedy identical at 57 and 4,266 tokens) |
| 2. Decode kernels, P2P collectives | done (`tests/gpu`) |
| 3. 4-GPU runtime (TP4 dense + EP4 experts) | plain decode **67 tok/s** single stream (HIP graphs per batch size), **prefill ~2,050-2,200 tok/s** (vLLM: ~56 / ~1,060); the four ranks load in parallel |
| 4. Prefix cache | per-slot reuse, VRAM snapshots, and a block store shared by all conversations in host RAM (128 GB) and on disk (survives restarts): any stored prefix up to a chat message boundary is restored in 0.02-0.5 s instead of re-prefilled (a shared 12k system prompt: 6.1 s -> 0.34 s) |
| 5. MTP speculative decoding | done: exact, adaptive draft count up to 5 (off while drafting does not pay), draft steps chained on the GPUs; **107-112 tok/s** single stream with Qwen's sampling settings (~117 greedy), **~231 tok/s** at 4 concurrent; on by default (`--mtp 5`) |
| 6. Prefill kernels and scheduling | two micro-batches per chunk, several waiting prompts per pass (batched prefill), long prompts in pieces between decode steps (interleaved prefill) |
| 7. OpenAI server, tokenizer, llama-swap entry | done: serving production through llama-swap and litellm; images and video (vision tower on every card, 720p in 0.33 s) |
| 8. CacheBlend-style reuse (experimental) | built and measured, rejected on quality (docs/DESIGN.md) |
| 9. Sampling | on the GPUs (penalties, candidates, Gumbel-max), exact; host fallback |
| 10. Thinking controls | `reasoning_effort` and thinking-token budgets per request, with server defaults |

The n-gram (PLE) table never goes to the GPUs. It stays pinned in host RAM and
the host gathers 16 rows per token. It comes in three precisions (int4 30 GB,
int8 54 GB, the original bf16 102 GB; `--ple-dir`); production runs bf16
(docs/DESIGN.md, "PLE n-gram table precision").

## Serving

`server/qwserve` is an OpenAI-compatible server (`/v1/chat/completions`,
`/v1/completions`, `/v1/models`, `/tokenize`, `/health`) in front of the engine
(`build/libqw_engine.so`, C API in `include/qw/capi.h`). It renders the
model's own `chat_template.jinja` and parses output like production vLLM's
`--reasoning-parser qwen3 --tool-call-parser qwen3_coder`. Token ids match
vLLM's `/tokenize` exactly on the test conversations (`server/tests/test_text.py`).

Requests run concurrently with continuous batching, one per engine slot
(default slots 262144, 65536, 32768, 32768 tokens; production 262144, 131072,
65536, 32768 with 4096-token prefill chunks). Each slot keeps its
conversation's state, and a new request goes to the slot holding the longest
prefix of its prompt; one that fits no free slot waits without holding up
later ones that fit.

Thinking: `reasoning_effort` (none, minimal, low, medium, high, xhigh, max;
none turns it off) and a thinking-token budget (`thinking_token_budget`,
`thinking_budget_tokens` or Anthropic's `thinking.budget_tokens`) per request;
defaults `--reasoning-effort` (xhigh; production sets medium) and `--thinking-budget`
(-1, unlimited).
All settings and their defaults: docs/ENGINE_GUIDE.md, section 4.

The server logs a stats line every 10 s while busy and one line per request,
and serves a dashboard at `/` (prompt and generation throughput, queue, KV and memory use, request
times; JSON at `/metrics.json`). Through llama-swap:
`http://llm-backend-amd.local.net:8080/upstream/qw/qwen3.8-flash-next/`.

```sh
python3 -m venv ~/qwenv && ~/qwenv/bin/pip install tokenizers jinja2 aiohttp
server/ctl.sh start                             # port 8000; stop | restart | status
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
- `/mnt/llms/qwen3.8-flash-next-ple/ples_int4`, `ples_int8`, `ples_bf16`: the n-gram
  table sidecar in three precisions (int4 is the `--ple-dir` default; bf16 is Qwen's
  original, `tools/ple_download.py` + `tools/ple_convert.py`)

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
bench/                 P2P, GEMM and uncached-memory microbenchmarks
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
