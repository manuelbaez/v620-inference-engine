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

## Status

| Phase | State |
|---|---|
| 1. Spec + CPU fp32 reference (`src/ref`) | done; matches vLLM (greedy identical at 57 and 4,266 tokens) |
| 2. Decode kernels, P2P collectives | done (`tests/gpu`) |
| 3. 4-GPU runtime (TP4 dense + EP4 experts) | **decode 58 tok/s** single stream, **165 tok/s** at 4 concurrent (HIP graphs per batch size), **prefill ~1,900-2,000 tok/s** (vLLM: ~56 / ~1,060) |
| 4. Prefix cache | per-slot in-place continuation + recurrent-state snapshots; LMCache offload tier next |
| 5. MTP speculative decoding | engine done: greedy output identical, 2.5-2.6 tokens/step, 70-74 tok/s single stream; server integration next |
| 6. Prefill kernels | batched prefill done; overlap and int8 next |
| 7. OpenAI server, tokenizer, llama-swap entry | server works (matches vLLM's template, tokenization and parsers); llama-swap entry pending |
| 8. CacheBlend-style reuse (experimental) | |

The n-gram (PLE) table never goes to the GPUs. It stays mmapped in host RAM as
the int4 sidecar, and the host gathers 16 rows per token.

## Serving

`server/qwserve` is an OpenAI-compatible server (`/v1/chat/completions`,
`/v1/completions`, `/v1/models`, `/tokenize`, `/health`) in front of the engine
(`build/libqw_engine.so`, C API in `include/qw/capi.h`). It renders the
model's own `chat_template.jinja` and parses output like production vLLM's
`--reasoning-parser qwen3 --tool-call-parser qwen3_coder`. Token ids match
vLLM's `/tokenize` exactly on the test conversations (`server/tests/test_text.py`).

Requests run concurrently with continuous batching, one per engine slot
(default slots 131072, 65536, 32768, 32768 tokens). Each slot keeps its
conversation's state, and a new request goes to the slot holding the longest
prefix of its prompt.

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
| 4 concurrent requests | 158 tok/s aggregate (57 tok/s single) |

## Inputs

- `/mnt/llms/qwen3.8-flash-next-awq`: AWQ checkpoint (int4 experts, bf16 rest)
- `/mnt/llms/qwen3.8-flash-next-ple/ples_int4`: int4 n-gram table sidecar

## Layout

```
include/qw/capi.h      C API of the engine (libqw_engine.so)
src/core/              model constants and shard shapes, safetensors, JSON, n-gram hashing and table
src/ref/               fp32 CPU reference of the forward pass
src/kernels/           HIP kernels by component (hc, gdn, qsa, moe_*, ple, mtp, logits, gemm, gemv)
src/comm/              push-based P2P collectives
src/engine/            the 4-GPU engine: weights, buffers, prefill, batched decode, speculative decoding
src/session/           slots, prefix reuse, sampling
src/capi/              C API implementation
tools/                 qw_gpu / qw_ref command-line runners, vLLM ground-truth helper
bench/                 P2P, GEMM and uncached-memory microbenchmarks
tests/unit, tests/gpu  host unit tests; GPU tests (collectives, batched decode, speculative decoding)
server/                OpenAI server (qwserve package), its tests, ctl.sh
scripts/               deploy to the GPU box
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
