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
- `legacy/`: the earlier C17 attempt, kept for reference only; its model facts are wrong.

## Status

| Phase | State |
|---|---|
| 1. Spec + CPU fp32 reference (`src/ref`) | done; matches vLLM (greedy identical at 57 and 4,266 tokens) |
| 2. Decode kernels, P2P collectives | done; unit-tested against CPU (`build/gpu_test_*`) |
| 3. 4-GPU runtime (TP4 dense + EP4 experts) | **decode 63 tok/s** (HIP graphs), **prefill ~1,900-2,000 tok/s** (vLLM: ~56 / ~1,060) |
| 4. Prefix cache | in-place continuation + recurrent-state snapshots (single sequence); multi-session tiers / LMCache next |
| 5. MTP speculative decoding | |
| 6. Prefill kernels | batched prefill done; overlap and int8 next |
| 7. OpenAI server, tokenizer, llama-swap entry | server works (matches vLLM's template, tokenization and parsers); llama-swap entry pending |
| 8. CacheBlend-style reuse (experimental) | |

The n-gram (PLE) table never goes to the GPUs. It stays mmapped in host RAM as
the int4 sidecar, and the host gathers 16 rows per token.

## Serving

`server/qw_server.py` is an OpenAI-compatible server (`/v1/chat/completions`,
`/v1/completions`, `/v1/models`, `/tokenize`, `/health`) in front of the engine
(`build/libqw_engine.so`, C API in `src/engine/capi.hpp`). It renders the
model's own `chat_template.jinja` and parses output like production vLLM's
`--reasoning-parser qwen3 --tool-call-parser qwen3_coder`. Token ids match
vLLM's `/tokenize` exactly on the test conversations (`server/test_text.py`).

```sh
python3 -m venv ~/qwenv && ~/qwenv/bin/pip install tokenizers jinja2 aiohttp
server/ctl.sh start --max-model-len 65536      # port 8000; stop | restart | status
server/test_e2e.py --url http://host:8000 --model qw --compare ref.json
```

Measured through HTTP (2026-09-24):

| | |
|---|---|
| 5.4k-token prompt | TTFT 2.97 s (1,816 tok/s), decode 58 tok/s |
| 36k-token prompt | TTFT 16.5 s (2,201 tok/s), decode 53 tok/s |
| follow-up turn on the same conversation | fully cached, TTFT 0.10-0.18 s |
| short answers | 0.24 s vs 0.58 s on production vLLM |

One request runs at a time. The engine holds one sequence, so an unrelated
request in between wipes the previous conversation's cache.

## Inputs

- `/mnt/llms/qwen3.8-flash-next-awq`: AWQ checkpoint (int4 experts, bf16 rest)
- `/mnt/llms/qwen3.8-flash-next-ple/ples_int4`: int4 n-gram table sidecar

## Build and run

```sh
make                      # host code: g++ C++17; kernels: hipcc --offload-arch=gfx1030
./deploy.sh               # tar the tree to hermes@llm-experiments and build there

# 4-GPU engine (needs the cards free: unload production first)
curl -X POST http://llm-backend-amd.local.net:8080/api/models/unload
build/qw_gpu --tokens 760,6511,314,9338,369 --gen 32 --logprobs /tmp/gpu.txt

# CPU reference: greedy continuation plus per-position logprobs
build/qw_ref --tokens 760,6511,314,9338,369 --gen 8 --logprobs /tmp/ref.txt

# ground truth from the production vLLM
tools/vllm_logprobs.py tokenize "The capital of France is"
tools/vllm_logprobs.py logprobs 760,6511,314,9338,369 /tmp/vllm.txt
tools/vllm_logprobs.py compare /tmp/ref.txt /tmp/vllm.txt
```

Tests: `make test`, plus `tests/ngram_ref.py build/test_ngram`, which checks the
n-gram hashing against a direct port of the reference implementation.
