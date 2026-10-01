# Code review: improvement points

Reviewed 2026-10-01 against commit `da15529` (master). The review read `src/`,
`server/`, `tests/`, `scripts/` and `docker/` (about 20k lines), the production
log since 2026-09-28 (`journalctl -t qw`: 587 requests, 585 stop and 2 abort,
no scheduler error, watchdog line or collective timeout), looked read-only at
the production host, and measured on the desktop (Ryzen 5 9600X, Python 3.14;
the server's EPYC is slower per thread). Lint is clean (ruff E9/F/B: 13 trivial
findings, none functional). The review changed no code.

**How to read it.** Evidence is **measured** (a number from a run; the recipe
is in the appendix), **read** (from the code, not reproduced) or **estimate**.
Effort is S (under an hour), M (a few hours) or L (days). Every point ends with
a status line: when one is fixed, put the commit and the before/after numbers
there and tick the roadmap bullet in `docs/DESIGN.md`.

**Applied 2026-10-01:** S1, S2, S3, R1, S6 and S7 (commits cb63dd4, 22e3839, 65e3c30, 553a5db, d793c7b,
dba9bef), each with a test that fails on the previous code; their status lines carry the before and after
numbers. The rest are open. Deployed to production the same day as `qw-engine:15822e2` (rollback:
`qw-engine:e7f5a3f`): the 11 tests pass inside that image, the smoke test and a replay of a cached 30k-token
chain through the disk tier pass on the live engine, and `/health` answers 200.

The reliability points are latent: nothing in the production log shows them
firing. The measured serving-path points are not: they cost latency whenever a
long prompt waits or loads.

Not covered: kernel-level performance (see DESIGN "Next steps: where the time
goes"), the host and storage (see DESIGN "Disk-tier loads and host memory"),
the model spec, quantization quality.

## Summary

In the order I would act on them:

1. **The serving loop re-marshals long prompts every pass (S1) and tokenizes on
   the event loop (S2).** Measured: 5.1 ms per call at 100k tokens, and with it
   in a loop the event loop that streams tokens is 4.7 ms late per wakeup
   (0.06 ms idle). This is a concrete cause of other requests slowing while a
   long prompt waits for a slot or loads from disk. Small, no GPU needed.
2. **One request's exception fails every request (S3), and a failed or stuck
   engine looks healthy (R1).** Small changes that turn latent outages into
   restarts.
3. **Agent-facing correctness: the tool-call parser (S6) and stop strings in
   streams (S5).** Measured edge cases that an agent editing files about this
   very format hits.
4. **Media and input handling (S7, S4).** A whole video is decoded into RAM
   (60 s of 1080p30 is about 11 GB), any URL is fetched, nothing limits the
   queue.
5. **Cache identity, counter wrap, launch errors (R4, R2, R3).** Small C++
   changes; they need a GPU window to verify.
6. **Tests, pins and dead experiment code (M4, M5, M1).** What keeps the next
   change safe.
7. **Small speed items (P1, P2):** about 3% of prefill and 1% of multi-request
   decode.

**Already good** (so it is not "fixed" by accident): exactness is tested, not
assumed (determinism repeats, fp32 reference gates, bit-identical restores);
`check_config` rejects a checkpoint that does not match the engine's constants
at start; prefill errors fail only their batch; client disconnects are handled;
cache files are written to a temporary name and renamed, validated by header on
read, and an unreadable entry is dropped instead of served; the chat template
runs in a sandboxed Jinja environment; no prompt text is logged; rejected
ideas are recorded with numbers.

**Suggested order**

- *Batch 1, server (Python), testable without a GPU:* S1, S2, S3, S4, S5, S6,
  S8, S7, M5, and the model-free tests of M4. No numeric change anywhere.
- *Batch 2, engine (C++), needs a GPU window:* R2, R3, R4, R5, R6, P1, P2, M2,
  M6, then M3. Each is small and exact; verify with `test_comm`,
  `test_batch_decode`, `test_speculative`, `test_host_tier` and alternating
  A/B runs.
- *Batch 3, larger:* R1 (engine, server and the supervisor's health settings),
  P3 (vision), M1 (remove or gate the experiments; bit-identical outputs are
  the check), R7 (the open correctness issue).

## Index

| id | point | evidence | effort | status |
|---|---|---|---|---|
| S1 | prompts re-marshalled on every scheduler pass | measured | S-M | done |
| S2 | tokenize and render on the event loop | measured | S | done |
| S3 | one request's exception fails every request | read | S | done |
| S4 | no admission control, silent ignores, 500s on bad input | read | S-M | open |
| S5 | streaming with `stop` leaks the start of a stop string | read | S | open |
| S6 | tool-call parser edge cases | measured | S-M | done |
| S7 | media: SSRF, whole video in RAM, bomb error type | read | M | done |
| S8 | dashboard shows the fp8 PLE table as 28.8 GB | measured | S | open |
| R1 | a failed or stuck engine is never reported or restarted | read | S-M | done |
| R2 | collective sequence numbers wrap (unsigned compare) | read | S | open |
| R3 | kernel launch errors are never read | read | S | open |
| R4 | saved-state identity ignores the weights | read | S | open |
| R5 | the checkpoint stays in the page cache after upload | measured | S | open |
| R6 | startup scans every cache file serially | measured | S-M | open |
| R7 | speculative batches over 8 rows (known) | known | L | open |
| P1 | prefill host input prep is serial with the GPUs | measured / estimate | S-M | open |
| P2 | `accept()` syncs every rank per request per step | read | S | open |
| P3 | vision warmup, first-image cost, encode stalls others | read | M-L | open |
| M1 | code of rejected experiments in hot paths | measured | M | open |
| M2 | `count_reuse` hashes every prompt for a rejected experiment | read | S | open |
| M3 | 41 environment reads, no effective-config log line | measured | S-M | open |
| M4 | tests: unregistered, hidden arguments, no CI, parsers untested | measured | M | open |
| M5 | no pinned dependencies, root image, no healthcheck | read | S | open |
| M6 | C API structs mirrored by hand, no version check | read | S | open |

## 1. Serving path

### S1. Prompts are re-marshalled on every scheduler pass

Evidence: **measured**. Effort: S-M. Status: **done** (cb63dd4). `Request.prompt` is a `Tokens` (a list that caches its int32 array, built once: 0.9 ms
per 100k tokens, nothing after) and `Engine.c_array` uses it; a request that found no slot is offered again
only after a slot was released (`Scheduler.slot_epoch`); the loop sleeps 20 ms instead of spinning when only
loads or blocked waiters remain (it spun at 100% while a request waited for a slot and another loaded).
Measured with a fake C library: `Engine.prefetch` at 100k tokens 7.4 ms to 0.001 ms per call; event-loop
lateness while polling, median 4.68 ms to 0.06 ms (max 5.64 to 0.51). `test_marshal`, `test_admission`.

**Where.** `server/qwserve/engine.py:124-160`: `acquire`, `set_prompt`,
`begin_prompt` and `prefetch` each build `(ctypes.c_int32 * len(tokens))(*tokens)`.
`server/qwserve/scheduler.py:205`: `_admit` calls `acquire` for every waiting
request on every pass, before it knows a slot is free. `:233`: `_poll_loading`
calls `prefetch` for every loading request on every pass (30-50 times a
second). The C side copies the prompt into a `std::vector` per call
(`src/capi/capi.cpp:94-176`), `Session::prefetch` copies it again and
`BlockStore::lookup` hashes it (`src/session/prefix_cache.cpp:33-38`).

**What happens.** Building the array costs 0.49 / 2.53 / 5.10 / 10.18 ms at
10k / 50k / 100k / 200k tokens and holds the GIL for the whole call (it is one
C-level call, so the interpreter's 5 ms switch interval never interrupts it).
The C side is small: 0.36 ms per poll at 100k tokens (two copies and the hash
chain). While a 100k-token prompt loads from disk (20-40 s on the production
disks) or waits for a slot, every scheduler pass spends about 5.5 ms on this,
next to a 25-35 ms decode step, and the asyncio thread that writes every
stream's tokens gets the GIL only after each rebuild: event-loop wakeup
lateness is a median 4.70 ms (p99 4.98, max 6.23) with the rebuild running in
a loop, against 0.06 ms idle and 0.08 ms with the buffer built once (probe in
the appendix). Every client streaming at that moment sees each token up to
5 ms late, and the scheduler spends about a fifth of each pass on it.

**Fix.** Build the token buffer once per request and keep it on the `Request`
(`array.array('i', tokens)`: 0.91 ms at 100k tokens; numpy int32: 1.19 ms; the
pointer to a prebuilt numpy array: 22 us), and pass the same buffer to
`acquire`, `prefetch` and `begin_prompt`. That removes the 5 ms and the GIL
hold; the 0.36 ms C side is not worth a new API (a cheap `qw_load_pending` call
would remove it too, optional). In `_admit`, skip `acquire` while every slot is
busy and nothing has finished since the last pass.

**Verify.** `server/tests/test_admission.py` and `test_prefill_order.py` (their
`FakeEngine` takes the new buffer type); the appendix probe before and after;
in production, TTFT and the `ms/step` stats line of a request that arrives
while another prompt loads.

### S2. Tokenizing and rendering run on the event loop

Evidence: **measured**. Effort: S. Status: **done** (22e3839). `chat`, `/tokenize` and `/v1/completions` run render + encode through
`Server.offload` / `Server.prepare`. `test_event_loop` (tiny tokenizer from `fixtures.py`): a 2 MB request
stalled the loop 392 ms on the loop and 22 ms through `prepare()`, same tokens.

**Where.** `server/qwserve/api.py:133-136` (`chat`), `:116-119` (`tokenize`),
`:272` (`completions`): `self.render(body)` (Jinja) and `self.encode(text)` run
synchronously inside the coroutine. Media preprocessing already goes through
`run_in_executor` (`:64`).

**What happens.** Hugging Face `tokenizers` encoded 649 KB of code and docs text
(203,298 tokens) in 161 ms, 4.0 MB/s (Qwen2.5-0.5B's tokenizer as a stand-in
for the model's; the EPYC is slower). A 100k-token prompt (about 320-400 KB)
therefore blocks the loop for roughly 80-100 ms here, plus the Jinja render of
a long history (not measured). Every other stream stalls meanwhile and its
tokens arrive in a burst. Agents send such a request every turn.

**Fix.** `await loop.run_in_executor(None, ...)` for render and encode; both are
pure functions of the request body. An executor alone is not enough: `Tokenizer.encode`
keeps the GIL for the whole call (measured: with `encode` in an executor the loop still
stalled 157 ms), while `encode_batch([text])` returns identical ids at the same speed and
releases it (4.3 ms). So `ChatPrompt.encode` uses `encode_batch` and the handlers call it
through `Server.offload`.

**Verify.** Event-loop lateness probe (appendix) while a large request is
tokenized; stream inter-token times of a concurrent request.

### S3. One request's exception fails every request in flight

Evidence: **read**. Effort: S. Status: **done** (65e3c30). `acquire` has its own try: the request gets its error and end events and holds no slot.
`test_isolation` runs the real scheduler thread: next to a poisoned prompt the long request decodes to its
length and the next one finishes; against the previous `scheduler.py` the long request is failed after 21
tokens and the other two are never answered.

**Where.** `server/qwserve/scheduler.py:205`: `self.e.acquire(...)` is outside
the `try` that starts at `:211`. The catch-all at `:392-405` fails everything in
`loading + prefilling + active`. In the engine, `src/session/generate.cpp:164`
(`QW_CHECK(rq.budget >= 1 && room(rq.slot) >= 1, ...)`) and `:124` throw from
inside the batched step, which only has the same catch-all above it.

**What happens.** `_prefill` handles a failing batch by failing only that
batch's requests, and `_begin` has its own handler, so only `acquire` and the
decode step are exposed. A malformed request that makes `acquire` throw (for
example the media-key checks in `Session::media_keys`) ends every active
request with an error, and, found while testing, it
stays at the head of the waiting list and raises again on every pass, so no
request behind it is admitted either until its client disconnects. No
scheduler error occurred in the 587 requests since 09-28, so this is latent.

**Fix.** Move `acquire` into the `try` and fail only that request. Validate
budget and room per request in Python before `generate` (`r.limit` already
bounds them) so the C++ checks stay assertions. Keep the catch-all for engine
failures, where failing everything is right.

**Verify.** A `FakeEngine` whose `acquire` raises for one request, next to a
running one (the existing admission tests have the scaffolding).

### S4. No admission control, silent ignores, 500 on bad input

Evidence: **read**. Effort: S-M. Status: open.

**Where.** `scheduler.py:119` (`submit` only rejects `n >= max_tokens`);
`api.py:129` and `:267` (`await request.json()` raises on invalid JSON and
aiohttp answers 500); `scheduler.py:13-47` (`Request.__init__`: `int(mt)`,
`float(...)` raise `ValueError` on bad types, also a 500); `prompt.py:81` (only
`tool_choice == "none"` is looked at). The server reads the body keys listed by
`grep 'body.get' server/qwserve`; `response_format`, `logit_bias`,
`parallel_tool_calls` and `tool_choice` as `required` or a named function are
ignored without a message.

**What happens.** The queue is unbounded: each queued 100k-token request holds
about 3.5 MB of Python ints plus its body, there is no 429, and a client that
asked for JSON mode gets free text without being told.

**Fix.** A queue cap with 429 and `Retry-After`; 400 for malformed JSON and bad
field types; reject or log once per unsupported field (`response_format` with
`json_schema` needs guided decoding, which would be a feature).

**Verify.** Requests with bad types and a burst larger than the cap against the
fake engine.

### S5. Streaming with `stop` leaks the start of a stop string

Evidence: **read**. Effort: S. Status: open.

**Where.** `api.py:164-217` (`check_stop` looks for a *complete* stop string in
`parser.content`; the delta written before it completes is already sent).

**What happens.** With stop `"STOP"` and tokens `"ST"`, `"OP"`: `"ST"` is
streamed at the first token; at the second, `k <= emitted_content`, so nothing
more is sent and the client keeps `"ST"`. Non-streaming answers are truncated
correctly.

**Fix.** Hold back the longest suffix of the content that is a prefix of any
stop string; release it when it is disproved or at the end of the stream.

**Verify.** A scripted token stream through `OutputParser` and the stop logic
(see M4 for a model-free fixture).

### S6. Tool-call parser edge cases

Evidence: **measured** (16-case probe, appendix). Effort: S-M. Status: **done** (d793c7b). `test_tool_calls`: the 16 cases and new ones, a differential fuzz of 3,000 random
well-formed calls against the previous implementation (identical), and the cases through `OutputParser`
(a call cut off by `max_tokens` now yields its partial value). 12 of its checks fail on the old parser;
the whole-sample-call case gave `path = /x, content = <function=edit>`.

**Where.** `server/qwserve/tool_calls.py:8` (`_PARAM_RE`), `:75-101`
(`parse_tool_call`), `server/qwserve/output_parser.py:57` (`finish`).

**What happens.**

- A value containing `<parameter=` splits into extra parameters: `see
  <parameter=count> in docs` became `{"path": "see ", "count": " in docs"}`.
- A value containing `</parameter>` ends there: `foo</parameter>bar` became
  `"foo"`.
- A truncated last parameter (no `</parameter>`, no `</function>`, as when
  `max_tokens` cuts the call) is dropped and the call returns `{}`.

What worked: plain calls, integer / number / boolean / null coercion against
the schema, JSON objects, `<` and `>` inside values, HTML-like values, dotted
and dashed function names, one trailing newline stripped. The module says it
follows vLLM's qwen3_coder parser; the two were not diffed here. The cases
matter for agents that edit files documenting this format (this repository's
own docs and tests are such files).

**Fix.** A value ends at the first `</parameter>` that is followed, after whitespace, by `<parameter=` or
the end of the call, and the call at its final `</function>`. Two exceptions keep the old tolerance:
a declared parameter that has not appeared yet and starts a line ends the value before it (the model
forgot the closing tag; needs the tool's schema, without one the two cannot be told apart), and a
value with no such closing tag runs to the end of the call, minus a cut-off fragment of the closing
tag. The 16 cases are a unit test.

**Verify.** The probe cases as a test; `server/tests/test_text.py` (needs the
checkpoint) for the cases the current parser already gets right.

### S7. Media: any URL is fetched, a whole video is held in RAM, one error escapes

Evidence: **read** (arithmetic and a checked exception class). Effort: M.
Status: **done** (dba9bef). `test_video`: output identical to the previous decoder on five clips (patches, grid,
hash, timestamps); peak memory for 300 frames of 720p 69 MB against 1,631 MB, and 73-83 MB flat for 5, 10
and 20 s clips; garbage and empty input are ValueErrors. `test_fetch`: 13 refusals without contacting the
local server, public addresses allowed, redirects checked, the overrides, the bomb error.

**Where.** `server/qwserve/vision.py:175-186` (`fetch`: any http(s) URL through
`urllib.request.urlopen`, redirects followed, 64 MB cap, 30 s timeout);
`:160-172` (`video_bytes`: `ffmpeg ... -f rawvideo -pix_fmt rgb24 -` with
`capture_output=True`, then `np.frombuffer`); `:188-197` (`media_of_part`);
`api.py:64-66` and `:137-140` (catches `ValueError` and `OSError` only).

**What happens.**

- The server fetches whatever URL a request names, including loopback and LAN
  addresses (server-side request forgery). In production the server sits
  behind llama-swap and litellm, which limits who can use it.
- `ffmpeg` decodes the whole video at its native frame rate into memory to keep
  about 2 frames per second (the Qwen3-VL sampling). 1080p rgb24 is 6.2 MB per
  frame, so 60 s at 30 fps is 1,800 frames, about 11 GB, in the server process,
  on a host that has already run out of memory three times.
- `PIL.Image.DecompressionBombError` derives from `Exception` only (checked in
  Pillow 12), so an oversize image gives a 500 instead of a 400.

**Fix.** Remote fetch is refused unless every address the host resolves to is public (decimal and
hex spellings of 127.0.0.1, `::ffff:127.0.0.1`, 169.254.169.254 and 100.64/10 are caught), every
redirect is checked, `QW_MEDIA_ALLOW_PRIVATE=1` allows the LAN and `QW_MEDIA_FETCH=0` allows `data:`
URLs only; DNS rebinding between the check and the connection is not covered. Videos decode as a
stream: one pass counts the frames, a second keeps only the sampled ones, each resized and
normalized as it arrives (the video goes to a file, not a pipe, so an mp4 with its index at the end
also decodes). The bomb error and ffmpeg failures are `ValueError`s (400).

**Verify.** `server/tests/test_vision_preprocess.py` (compares with HF's
processor) must stay identical; peak RSS of a long video before and after.

### S8. The dashboard shows the fp8 PLE table as 28.8 GB

Evidence: **measured** (live). Effort: S. Status: open.

**Where.** `server/qwserve/dashboard/collector.py:54`:
`{"bf16": 2.0, "group16_int8_fp16scale": 1.125}.get(layout, 0.5625)`.

**What happens.** The fp8 layout `f8e4m3_tensorscale` is not in the map and
falls back to int4's 0.5625 bytes per value. The production `/metrics.json`
reports `memory.ple_table` = 28,800,138,240 for `ples_fp8`
(320,001,536 rows x 160 values); the table is 51.2 GB.

**Fix.** Add `"f8e4m3_tensorscale": 1.0` and fail loudly (or show "unknown")
for an unknown layout instead of guessing int4.

**Verify.** `curl localhost:5800/metrics.json` on the engine.

## 2. Reliability

### R1. A failed or stuck engine is never reported or restarted

Evidence: **read**. Effort: S-M. Status: **done** (553a5db), deployed in `qw-engine:15822e2`: the getter returns "" on the live engine (`/health`
answers 200); its failure path has not run on hardware. `test_health`: 503 with the reason, exit after the grace, no exit when it recovers inside it, a call
stuck in `generate()` reported and exited for, a dead scheduler thread, the binding. Still to check: that
llama-swap restarts the model on the next request after the exit (as it did after the 10-01 OOM kill).

**Where.** `src/engine/engine.hip:202` (`error_` is set by the first failing
rank and never cleared), `:246` (every later `dispatch()` fails with it),
`:228-236` (the watchdog logs every 60 s and keeps waiting),
`server/qwserve/api.py:101` (`/health` returns `"status": "ok"` always),
`server/qwserve/scheduler.py:392-405` (the loop continues after an error).

**What happens.** After a rank error or a collective timeout
(`Comm::describe_error`), every later request fails, but the process answers
`/health` and looks fine to llama-swap. A rank stuck forever (the documented
GPU wedge on this platform) hangs the scheduler thread without a sound beyond
the 60 s log line. A dead scheduler thread is not noticed either.

**Fix.** `Engine::failure()` (a mutex-guarded getter of `error_` and the comm timeout flags) is exported as
`qw_engine_failure` and bound as `Engine.failure()`, callable from any thread. `Scheduler.health()` is
not ok when the scheduler thread died, an engine call has run for over `QW_STUCK_SECONDS` (300; the
scheduler's engine is wrapped to record the call in progress), or the engine reports a failure.
`/health` answers 503 with the reason, and a watchdog thread exits the process (code 3) once health
has failed for `QW_EXIT_GRACE` seconds (30) so the supervisor restarts it (`QW_WATCHDOG_EXIT=0`: only
report). A failure streak heuristic was rejected: a few malformed requests in a row would look like a
failed engine.

**Verify.** A fake engine that raises or sleeps; a dev-box run that kills a
rank's stream with a bad launch.

### R2. Collective sequence numbers are unsigned 32-bit and compared unsigned

Evidence: **read** plus arithmetic. Effort: S. Status: open.

**Where.** `src/comm/comm.hip:110` (`while (load(flag) < seq)`),
`src/engine/rank.hpp:97,188` (`uint32_t seq_base`, `seq_base2`),
`comm.hip:48,129,135` (`seq = *base + k`).

**What happens.** A plain decode step runs about 291 collectives (DESIGN,
"where the time goes"), about 300-350 with the draft steps. 2^32 is reached
after 12-14 million steps, about 4 days of continuous decoding at 35 steps/s.
After the wrap the flags still hold values near 2^32, so the first waits of
each (parity, source) slot pass at once and read stale slots: one step's
output is wrong, then it recovers. Not reproduced; production restarts more
often than every 4 days of busy time.

**Fix.** `(int32_t)(flag - seq) < 0` in `wait_flags`.

**Verify.** `test_comm` and `stress_comm` with the base preset to `0xFFFFFF00`
(`Comm::set_base` exists) so a run crosses the wrap. Needs a GPU window.

### R3. Kernel launch errors are never read

Evidence: **read** (grep). Effort: S. Status: open.

**Where.** Every `<<<...>>>` launch (88 sites in `src/kernels`,
`src/comm`, `src/vision`); `hipGetLastError` appears once, in
`src/comm/comm.hip:199`, to clear a sticky flag.

**What happens.** A launch error (bad configuration, out of resources) does not
throw in HIP: it sets the last error and the kernel does not run. Inside a
captured graph it can surface late or not at all. A skipped kernel leaves stale
buffers and gives wrong tokens, the symptom class of the open issue R7.

**Fix.** `CK(hipGetLastError())` after each graph capture and after each rank
job (after the stream sync, one call per job), and a `QW_DEBUG_LAUNCH=1` mode
that checks after every wrapper and synchronizes per kernel.

**Verify.** Run the GPU test suite once with the debug mode.

### R4. The saved-state identity ignores the weights

Evidence: **read**. Effort: S. Status: open.

**Where.** `src/engine/host_tier.hip:137-148` (`state_layout_id`: FNV over shapes,
ring sizes, layer count, ranks, vocabulary and the MTP flag);
`src/session/disk_tier.cpp:76,174` (files are accepted when `h.layout == layout_`).

**What happens.** The id does not change with the checkpoint, the quantization
or the PLE table's precision and scale. A changed `--model-dir` or `--ple-dir`
reuses KV built by other weights. Production did exactly that on 2026-09-28:
the table went from bf16 to fp8 and the disk entries computed with bf16 stayed
valid (within noise, 0.044 against 0.050 mean |dlogprob|, but by accident); a
different checkpoint would be silently wrong.

**Fix.** Add a fingerprint to the id: a hash of `model.safetensors.index.json`
and the shard sizes, the PLE `META.json` text, and an engine "numerics version"
constant that is bumped when a kernel change moves values beyond noise.
Entries with another id are already removed at scan. The first start after the
change drops the cache; say so in the commit.

**Verify.** `test_host_tier` (restart path); start with a changed `META.json`
and see the entries removed.

### R5. The checkpoint stays in the page cache after the upload

Evidence: **measured**. Effort: S. Status: open.

**Where.** `src/engine/engine.hip:103` (`st_.clear()` unmaps); the shards are
`mmap`ed by `src/core/safetensors.cpp:78-93`.

**What happens.** The engine's cgroup (the incus container, 2026-10-01,
`memory.stat`) holds `inactive_file` 41.9 GB and `active_file` 21.8 GB, 63.8 GB
of clean file cache, next to the 51.3 GB mapped PLE table. The AWQ checkpoint
is 74 GB and is the likely source (not attributed per file). The cache is
reclaimable, but every pinned-arena allocation near the limit has to reclaim it
first, and on a host with the cache full that is where allocations stall (see
DESIGN "Disk-tier loads and host memory").

**Fix.** `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` on each shard after the
upload (open, advise, close); test on ZFS (ARC plus page cache) whether the
cache really drops. Alternative: read with `O_DIRECT` (ZFS 2.4 supports it).

**Verify.** The cgroup's `memory.stat` file lines before and after a start.

### R6. Startup scans every cache file serially

Evidence: **measured** (log). Effort: S-M. Status: open.

**Where.** `src/session/disk_tier.cpp:58-99` (`scan`: `fopen`, read the header
and the tokens of each file), `src/session/block_store.cpp:62-63`.

**What happens.** The time from "warmup done" to "disk tier holds ..." in the
log was 5, 5, 14, 5, 4 and 3 s on the six starts of 09-27 and 09-28, and 58 s
on the cold start after the 10-01 OOM kill (about 5,980 files on the HDD
mirror, ARC and page cache cold): each header is a random read.

**Fix.** Scan with 8-16 threads (the mirror serves two spindles, NCQ more), or
write an index (hash, parent, tokens, sizes, mtime) at persist time and
periodically, trusted while the directory listing (metadata only) matches, with
the existing validation of a header on read.

**Verify.** Start time with a cold ARC (after a memory-pressure event) before
and after.

### R7. Speculative batches over 8 rows (known, open)

Evidence: documented. Effort: L. Status: open.

**Where.** `src/session/generate.cpp:128` (cap of 8 rows,
`QW_SPEC_MAX_ROWS`); DESIGN "Open issue: multi-request speculative batches".

**What happens.** Batches of 11-16 rows intermittently produced wrong tokens or
GPU faults; the cause is unknown. PCIe AER events on 83:00.0 overlap some of
the failures. R2 cannot explain it (not time dependent); R3 could.

**Fix.** `test_speculative --repeat` with `QW_SPEC_MAX_ROWS=16` plus R3's launch
checks, and a `rocprofv3` kernel trace around a failure, compared with the AER
log.

## 3. Performance

### P1. Prefill's host input preparation is serial with the GPUs

Evidence: **measured** (conversion rate), **estimate** (total). Effort: S-M.
Status: open.

**Where.** `src/engine/prefill.hip:409-435` (`prefill_inputs`: bf16 to fp32
embedding rows, `ple_->gather` of 16 rows per token, fp32 to fp16), called
before `dispatch()` in `Engine::prefill` and `prefill_batch` on the scheduler
thread; `src/core/ple.cpp:176-211`.

**What happens.** The scalar `f32_to_f16` takes 40.9 ms for 12.5 M values
(3.3 ns each); a 4096-token chunk converts 10.5 M, about 34 ms. With the
embedding conversion and the table gather the total is roughly 60 ms per chunk
single-threaded, about 3% of a 2 s chunk, during which the GPUs wait.

**Fix.** `ThreadPool::parallel_for` over tokens (the pool is in
`src/core/threadpool.hpp`) and F16C (`_mm256_cvtps_ph`), or prepare chunk n+1
while chunk n runs (needs `dispatch` split into begin and wait). No numeric
change: the same conversions.

**Verify.** `QW_PROFILE` per-chunk times, alternating A/B runs on the dev box.

### P2. `accept()` syncs every rank once per request per step

Evidence: **read**. Effort: S. Status: open.

**Where.** `src/engine/speculative.hip:20-43`, called at
`src/session/generate.cpp:284` inside the per-request loop.

**What happens.** Each call enqueues the state copies and runs
`hipStreamSynchronize` on all four ranks, for every request whose drafts were
rejected. About 1% at four concurrent requests (estimate: 50-100 us per sync
plus a state copy of about 20 MB).

**Fix.** An `accept_batch` that enqueues all copies and syncs once.

**Verify.** `test_speculative` (greedy equals plain) and `QW_TRACE` accept time.

### P3. Vision: warmup, first-image cost, encode stalls everything

Evidence: **read** (numbers in DESIGN "Vision: where image time goes").
Effort: M-L. Status: open.

**Where.** `src/engine/engine.hip:126-133` (warmup encodes one 32x32-patch slice
per rank), `src/vision/vision_encoder.hip:30-48` (`reserve(L)` frees and
reallocates the workspace for a larger image; the `hipFree` waits for the
device), `src/engine/vision_job.hip` (synchronous).

**What happens.** The first real image pays the workspace (about 330 MB per
card at 1080p) and rocBLAS kernel loads; the tower then takes 1.23 s per 1080p
image on one card of four and every other request waits for it.

**Fix.** Warm up at the production maximum (1920x1088 is 8,160 patches) and
reserve the workspace there; then the options in DESIGN (attention kernel,
sequence-parallel encode with `Comm::allgather`, layer-sliced encode between
decode steps).

**Verify.** `test_vision_encoder`, first-image latency after a start.

## 4. Maintainability and tests

### M1. Code of rejected experiments sits in hot paths

Evidence: **measured** (grep and `wc`). Effort: M. Status: open.

**Where.** Experiment-only files, 1,274 lines: `src/engine/blend.hip`,
`src/kernels/blend.{hip,hpp}`, `src/engine/calibrate.hip`,
`src/kernels/moe_w4a8.hip`, `tools/qw_blend_eval.hip`, `tools/blend_cases.py`,
`tools/qw_calibrate.hip`, `tools/gptq_int8.py`, `tools/qw_vision_pos_eval.hip`,
`tools/vision_pos_cases.py`. Hooks inside hot files: `prefill.hip` (17
calibration, 13 `rope3`, 6 transfer), `weights.hip` (24 int8), `decode.hip` (7
int8), members in `engine.hpp` and `rank.hpp`. The calibration hooks build a
`std::string` name per layer of every prefill chunk.

**What happens.** The experiments were measured and rejected (CacheBlend,
GPTQ int8, W4A8 experts, 3D vision positions) and the results are recorded in
DESIGN, but their code makes the hot files longer than they need to be and the
engine harder to read.

**Fix.** A CMake option (`QW_EXPERIMENTS`) around them, or delete them (the docs
and git history keep the results). Keep the int8-dense opt-in only if you still
want it.

**Verify.** Bit-identical outputs: `test_batch_prefill`, `test_speculative`,
`test_block_store` before and after.

### M2. `count_reuse` hashes every admitted prompt for a rejected experiment

Evidence: **read**. Effort: S. Status: open.

**Where.** `src/session/prefix_cache.cpp:179-196`, `src/session/session.hpp:207`
(`seen_chunks_`, up to 1,048,576 entries), `include/qw/capi.h`
(`blend_candidate_tokens`).

**What happens.** `chunk_prompt` runs over the whole prompt on every admission,
O(prompt), only to count `blend_candidate_tokens`, a statistic of the rejected
CacheBlend work; the map costs tens of MB.

**Fix.** Remove it and the field (the C struct and `engine.py`'s `CacheStats`
change together; see M6).

### M3. Configuration by 41 environment reads, no effective-config line

Evidence: **measured** (grep). Effort: S-M. Status: open.

**Where.** 31 `getenv` sites in C++ (`session.cpp` 5, `engine.hip` 5,
`comm.hip` 5, `generate.cpp` 4, `prefill.hip` 3, `decode.hip` 3,
`block_store.cpp` 2, four files with 1) and 10 `os.environ` reads in Python
(`scheduler.py` 4, `__main__.py` 4, `vision.py` 2); many are function-local
statics evaluated once.

**What happens.** Nothing logs which knobs a running engine uses, so a
production engine's settings can only be inferred from the compose file and
llama-swap's command.

**Fix.** One struct, parsed once, validated, logged at start as a single line,
and passed to the components.

### M4. Tests: unregistered, hidden arguments, no CI, parsers untested

Evidence: **measured** (ran them). Effort: M. Status: open.

**Where.** `CMakeLists.txt:41,44,139-140` (`add_test` for `ngram`, `chunker`
and `comm` only; `make test` runs `-LE gpu`); `tests/gpu/*` (several need
arguments, for example `test_batch_decode --a --b --gen` and
`test_speculative --p ... --gen --k`, and hang without them); `server/tests/*.py`
(scripts with their own `check`, no runner).

**What happens.** In a fresh venv `test_admission`, `test_prefill_order` and
`test_thinking` pass without a model; `test_text` and `test_vision_preprocess`
need the checkpoint, `test_concurrency` and `test_e2e` need a server.
`qwserve.scheduler` imports `qwserve.dashboard.metrics`, and that package's
`__init__` imports aiohttp, so the scheduler's tests need it too. No test covers
`tool_calls`, `output_parser`, `detok`, stop handling or the API's streaming
format without a model. There is no CI.

**Fix.** `scripts/gpu-tests.sh` holding the arguments and prompts (and a
reminder of the production idle check); register the GPU tests with
`LABELS gpu`; a tiny `tokenizers` fixture (WordLevel with the special tokens)
for parser, detokenizer and API tests; move `Metrics` out of `dashboard`;
GitHub Actions for the CPU tests plus a `hipcc -c` compile check in the ROCm
image.

**Verify.** The CPU suite runs on a clean checkout with one command.

### M5. No pinned dependencies, root image, no healthcheck

Evidence: **read**. Effort: S. Status: open.

**Where.** `docker/Dockerfile:27` (unpinned `tokenizers jinja2 aiohttp numpy
pillow`), `:14-19` (`tools`, `bench` and `tests` are copied before the compile
layer), `:36` (runs as root, no `HEALTHCHECK`); no `requirements.txt` or
`pyproject.toml`.

**What happens.** PIL's resize and the tokenizer decide the pixels and token ids
the model sees, so a rebuild with newer packages can change outputs silently
(`test_vision_preprocess` and `test_text` exist to catch it, but only when
run). Editing a test or a tool rebuilds the engine layer.

**Fix.** `server/requirements.txt` with exact versions and `pip install -r`; a
`HEALTHCHECK` on `/health` (after R1); copy only what the two built targets
need (a CMake option to skip tools, bench and tests); a non-root user.

### M6. C API structs are mirrored by hand

Evidence: **read**. Effort: S. Status: open.

**Where.** `include/qw/capi.h` (`qw_sampling`, `qw_step_req`, `qw_media`,
`qw_cache_stats`) and `server/qwserve/engine.py:7-50`.

**What happens.** There is no version or size check; a field added on one side
(`capi.h` was touched by 11 commits, `qw_cache_stats` has grown more than once)
shifts the other silently.

**Fix.** `int qw_api_version(void)` and `size_t qw_struct_size(int id)`, checked
in `Engine.__init__`.

## Appendix: how the numbers were measured

Run from the repository root with a venv that has `numpy aiohttp jinja2
tokenizers pillow`. The numbers in this document came from a Ryzen 5 9600X
under Python 3.14; rerun on the target before and after a fix. The three
Python scripts below are complete: save each as a file and run it.

**A1. Marshalling cost (S1).**

```python
import ctypes, random, time
for n in (10_000, 50_000, 100_000, 200_000):
    toks = [random.randrange(150000) for _ in range(n)]
    best = 1e9
    for _ in range(5):
        t = time.perf_counter()
        arr = (ctypes.c_int32 * n)(*toks)          # what Engine.acquire / prefetch do per call
        best = min(best, time.perf_counter() - t)
    print(n, f"{best * 1e3:.2f} ms")
```

Results: 0.49 / 2.53 / 5.10 / 10.18 ms. The prebuilt alternatives at 100k tokens:
`array.array("i", toks)` 0.91 ms, `np.asarray(toks, np.int32)` 1.19 ms, and
`arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32))` on the prebuilt array 22 us.

**A2. Event-loop lateness while a thread rebuilds the buffer (S1).** One thread
stands in for the scheduler (rebuild, then `sleep(1.5 ms)` for the C call and
the rest of a pass); the main thread runs `asyncio` and records how late
`await asyncio.sleep(0.002)` wakes, for 4 s per mode.

```python
import asyncio, ctypes, random, statistics as st, threading, time

N_TOK = 100_000
toks = [random.randrange(150000) for _ in range(N_TOK)]
pre = (ctypes.c_int32 * N_TOK)(*toks)

def spin(mode, stop):
    while not stop.is_set():
        if mode == "rebuild":
            (ctypes.c_int32 * N_TOK)(*toks)                                   # the per-call rebuild
        else:
            ctypes.memmove(ctypes.addressof(pre), ctypes.addressof(pre), 8)  # a cheap C call
        time.sleep(0.0015)

async def probe(duration=4.0):
    late, end = [], time.perf_counter() + duration
    while time.perf_counter() < end:
        t0 = time.perf_counter()
        await asyncio.sleep(0.002)
        late.append((time.perf_counter() - t0 - 0.002) * 1e3)
    return sorted(late)

for mode in ("idle", "rebuild", "cached"):
    stop = threading.Event()
    if mode != "idle":
        threading.Thread(target=spin, args=(mode, stop), daemon=True).start()
    late = asyncio.run(probe())
    stop.set()
    print(f"{mode:8s} median {st.median(late):5.2f} ms  p99 {late[int(.99 * len(late))]:5.2f}  max {late[-1]:6.2f}")
```

Results: idle median 0.06 ms; rebuild 4.70 ms (p99 4.98, max 6.23); cached 0.08 ms.

**A3. C side of one poll (S1).** Two copies of a 100k-token `std::vector<int32_t>`
and the `BlockStore::key` hash chain (`src/session/block_store.cpp`) over 390
blocks of 256 tokens, `g++ -O3 -march=native`, best of 9: 0.36 ms.

**A4. Tokenizer rate (S2).** `Tokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")`
(a stand-in for the model's tokenizer), `encode(text, add_special_tokens=False)`
over the repository's `src/**` and `docs/*.md` concatenated: 649 KB, 203,298
tokens, 161 ms; twice the text, 318 ms.

**A5. Tool-call parser probe (S6).** Run with `server/` on the path.

```python
import sys
sys.path.insert(0, "server")
from qwserve.tool_calls import parse_tool_call

tools = [{"type": "function", "function": {"name": "edit", "parameters": {"properties": {
    "path": {"type": "string"}, "count": {"type": "integer"}, "flag": {"type": "boolean"},
    "opts": {"type": "object"}, "items": {"type": "array"}, "ratio": {"type": "number"},
    "maybe": {"type": ["string", "null"]}}}}}]

def show(label, body, name="edit"):
    c = parse_tool_call(f"<function={name}>\n{body}</function>", tools)
    print(f"{label:46s}", None if c is None else (c["function"]["name"], c["function"]["arguments"]))

P = lambda k, v: f"<parameter={k}>\n{v}\n</parameter>\n"
show("plain", P("path", "/a/b.py") + P("count", "3"))
show("value contains </parameter>", P("path", "foo</parameter>bar"))
show("value contains <parameter=", P("path", "see <parameter=count> in docs"))
show("value with two trailing newlines", "<parameter=path>\nx\n\n</parameter>\n")
show("leading spaces in value", "<parameter=path>  x</parameter>\n")
show("integer written 1e3", P("count", "1e3"))
show("number 3.0", P("ratio", "3.0"))
show("boolean True", P("flag", "True"))
show("nullable string null", P("maybe", "null"))
show("object from invalid JSON", P("opts", "{'a': 1}"))
show("unknown function", P("x", "1"), name="nope")
show("duplicate parameter", P("path", "a") + P("path", "b"))
show("truncated: no closing tags", "<parameter=path>\n/a\n")
show("code with < and >", P("path", "if a < b and c > d: pass"))
show("html-like value", P("path", '<div class="x">hi</div>'))
show("dotted and dashed name", P("a", "1"), name="mcp.server-tool_1")
```

Results are the table in S6's evidence: the three failures are the second,
third and truncated cases (`{"path": "foo"}`; `{"path": "see ", "count": " in docs"}`;
`{}`); the rest parse as expected (one newline stripped, `1e3` stays a string,
`True` becomes `true`, invalid JSON stays a string, the last duplicate wins).

**A6. fp32 to fp16 conversion rate (P1).** `qw::f32_to_f16` from
`src/core/common.hpp` over 12,533,760 random floats (a 1080p image's patch
count), `g++ -O3 -march=native`, best of 7: 40.9 ms.

**A7. Startup phases (R6).** In the engine container,
`journalctl -t qw | grep -E 'warmup done|disk tier holds|engine ready'`; the gap
between the first two lines is the scan (`DiskTier::scan` and `load_index`).

**A8. Page cache of the engine's cgroup (R5).** Inside the incus container,
`grep -E '^(file|file_mapped|active_file|inactive_file) ' /sys/fs/cgroup/memory.stat`
and `du -sh /mnt/llms/qwen3.8-flash-next-awq`.

**A9. Dashboard size (S8).** Inside the engine container,
`curl -s localhost:5800/metrics.json`, field `memory.ple_table`.
