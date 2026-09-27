# Working on the qw engine

## Keep the docs current (do this without being asked)

Every change that affects behavior, a default, performance or output quality
updates the docs in the same commit or the one right after it:

- `docs/DESIGN.md`: measurements and decisions, including what was tried and
  rejected, with numbers. New work goes in the roadmap (item 9) as `[x]` done,
  `[~]` partial or `[ ]` open; larger topics get their own section (e.g. "PLE
  n-gram table precision").
- `docs/ENGINE_GUIDE.md`: the explanation of the engine for someone porting it
  to a new model. Update the feature table in section 4 (what it does, speed
  effect, quality effect, code default, production setting, knob) whenever a
  feature, default or production setting changes, and the numbers in section
  12 when they move.
- `README.md`: the status table, serving notes and layout.
- `docs/MODEL.md`: only when the understanding of the model itself changes.

Record production settings that live outside this repo (llama-swap command,
compose environment, container limits in home-infra) in ENGINE_GUIDE's
"production" column.

## How work is done here

- **Accuracy before speed.** Anything that changes outputs is measured against
  the fp32 reference on the 4,000-token set (mean |dlogprob|, top-1, paired
  bootstrap CI) and stays off unless it is within fp16 noise; the owner opts in
  explicitly otherwise.
- **Measure before and after.** Profile or microbenchmark first; A/B against the
  build without the change, alternating runs; revert what does not pay and
  record it.
- **Know the noise floor.** Different GEMM shapes (chunking, batching) change
  rounding; only differences beyond that floor mean something. Greedy paths that
  must be bit-identical are the exactness checks.
- **Determinism detects hardware errors.** `test_speculative --gen 256 --k 5
  --repeat 6` repeats greedy runs; any mismatch between runs of the same path
  means wrong arithmetic (it found the -75 mV undervolt problem).
- Commit and push regularly; small, single-concern files; code matches the
  surrounding style.

## GPUs, dev box and production

- Build and test on `hermes@llm-experiments.local.net` (`scripts/deploy.sh`
  syncs and builds). The 4 GPUs are shared with production.
- Unloading production for GPU tests is allowed, but only once it is idle:
  `ssh main-srv.local.net 'incus exec llm-backend-amd --project llms -- bash
  /tmp/idle_unload.sh'` (waits for 60 s without requests). Production reloads on
  the next request. Do not rebuild the dev box's `build/` while a benchmark
  there is using it.
- GPU tests need arguments (prompts); see README "Build and run".
- Deploy: `scripts/build-image.sh` builds `qw-engine:<sha>` and `:latest`;
  unload production so the next request starts the new image, then send a test
  request and check the log (`journalctl -t qw` in llm-backend-amd).
- Power and undervolt are set by the owner (`home-infra/set-v620-power.sh WATTS
  MV`, needs their sudo); production runs 160 W / -50 mV.
