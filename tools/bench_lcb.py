"""LiveCodeBench code generation through the OpenAI API, with the benchmark's own prompt and code extraction
(github.com/LiveCodeBench/LiveCodeBench on PYTHONPATH; datasets < 3). Thinking on at the given effort, the model's own
sampling settings, one sample per problem. Writes the file lcb_runner.runner.custom_evaluator takes; resumes.
  PYTHONPATH=~/lcb python bench_lcb.py OUT.json [--url http://localhost:8000] [--release v6] [--effort xhigh]
                                                [--workers 6] [--max-tokens 65536]
Then scores them with the benchmark's checker (the generated programs run on this machine against the hidden tests):
pass@1 overall and by difficulty. (v6 = the 175 problems of 2025-01..04 that release_v6 added.)"""
import argparse, json, os, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from lcb_runner.benchmarks.code_generation import load_code_generation_dataset
from lcb_runner.lm_styles import LMStyle
from lcb_runner.prompts.code_generation import format_prompt_generation
from lcb_runner.utils.extraction_utils import extract_code

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--release", default="v6")
ap.add_argument("--effort", default="xhigh")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--max-tokens", type=int, default=65536)
ap.add_argument("--model", default="qw/qwen3.8-flash-next")
a = ap.parse_args()
URL = a.url.rstrip("/") + "/v1/chat/completions"
problems = sorted(load_code_generation_dataset(release_version=a.release), key=lambda p: str(p.question_id))
raw_path = a.out + ".raw"
raw = json.load(open(raw_path)) if os.path.exists(raw_path) else {}
lock, t0 = threading.Lock(), time.time()

def run(p):
    if p.question_id in raw and "error" not in raw[p.question_id]:
        return
    body = {"model": a.model, "max_tokens": a.max_tokens, "reasoning_effort": a.effort,
            "messages": format_prompt_generation(p, LMStyle.OpenAIChat)}
    out = {"error": "?"}
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL, json.dumps(body).encode(), {"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=7200) as r:
                d = json.load(r)
            c = d["choices"][0]
            out = {"text": c["message"].get("content") or "", "finish": c.get("finish_reason"),
                   "tokens": d.get("usage", {}).get("completion_tokens"), "difficulty": str(p.difficulty)}
            break
        except Exception as ex:
            out = {"error": str(ex)}
            time.sleep(10 * (attempt + 1))
    with lock:
        raw[p.question_id] = out
        json.dump(raw, open(raw_path, "w"))
        if len(raw) % 10 == 0:
            print(f"{len(raw)}/{len(problems)} done, {time.time() - t0:.0f} s", flush=True)

with ThreadPoolExecutor(a.workers) as ex:
    list(ex.map(run, problems))
json.dump([{"question_id": p.question_id, "code_list": [extract_code(raw[p.question_id].get("text", ""), LMStyle.OpenAIChat)]}
           for p in problems], open(a.out, "w"))
n = len(problems)
print(f"{n} problems: no code found in {sum(not extract_code(raw[p.question_id].get('text', ''), LMStyle.OpenAIChat) for p in problems)}, "
      f"cut off in {sum(raw[p.question_id].get('finish') == 'length' for p in problems)}, errors "
      f"{sum('error' in raw[p.question_id] for p in problems)}, mean completion tokens "
      f"{sum(raw[p.question_id].get('tokens') or 0 for p in problems) / n:.0f}", flush=True)

from lcb_runner.evaluation.compute_code_generation_metrics import codegen_metrics
codes = [[extract_code(raw[p.question_id].get("text", ""), LMStyle.OpenAIChat)] for p in problems]
metrics = codegen_metrics([p.get_evaluation_sample() for p in problems], codes, k_list=[1], num_process_evaluate=12, timeout=6)
detail = metrics[0]["detail"]["pass@1"]
ok = {p.question_id: float(detail[i]) for i, p in enumerate(problems)}
json.dump(ok, open(a.out + ".passed", "w"))
print(f"LiveCodeBench {a.release}: pass@1 {100 * metrics[0]['pass@1']:.1f}% of {n}", flush=True)
for d in sorted(set(str(p.difficulty) for p in problems)):
    v = [ok[p.question_id] for p in problems if str(p.difficulty) == d]
    print(f"  {d}: {100 * sum(v) / len(v):.1f}% of {len(v)}", flush=True)
