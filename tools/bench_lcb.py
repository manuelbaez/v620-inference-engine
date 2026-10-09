"""LiveCodeBench code generation through the OpenAI API, with the benchmark's own prompt and code extraction
(github.com/LiveCodeBench/LiveCodeBench on PYTHONPATH; datasets < 3). Thinking on at the given effort, the model's own
sampling settings, one sample per problem. Writes the file lcb_runner.runner.custom_evaluator takes; resumes.
  PYTHONPATH=~/lcb python bench_lcb.py OUT.json [--url http://localhost:8000] [--release v6] [--effort xhigh]
                                                [--workers 6] [--max-tokens 65536] [--retry-cut]
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
ap.add_argument("--retry-cut", action="store_true", help="ask again the problems whose answer was cut off at max_tokens")
ap.add_argument("--api-key-file", help="a file holding the API key of a hosted endpoint (sent as a Bearer token)")
ap.add_argument("--extra", default="", help="JSON merged into every request body, e.g. a hosted API's own thinking "
                "and sampling fields; with it, reasoning_effort is not sent")
ap.add_argument("--stream", action="store_true", help="read the answer as a stream (long answers through a proxy)")
ap.add_argument("--limit", type=int, default=0, help="only the first N problems (a trial)")
a = ap.parse_args()
URL = a.url.rstrip("/") + "/v1/chat/completions"
HEADERS = {"content-type": "application/json"}
if a.api_key_file:
    HEADERS["Authorization"] = "Bearer " + open(a.api_key_file).read().strip()
EXTRA = json.loads(a.extra) if a.extra else None
problems = sorted(load_code_generation_dataset(release_version=a.release), key=lambda p: str(p.question_id))
if a.limit:
    problems = problems[:a.limit]


def complete(body):
    """One chat completion: (text, finish reason, completion tokens)."""
    req = urllib.request.Request(URL, json.dumps(body).encode(), HEADERS)
    with urllib.request.urlopen(req, timeout=6 * 3600) as r:
        if not body.get("stream"):
            d = json.load(r)
            if "choices" not in d:
                raise RuntimeError(json.dumps(d)[:300])
            c = d["choices"][0]
            return c["message"].get("content") or "", c.get("finish_reason"), d.get("usage", {}).get("completion_tokens")
        text, finish, tokens = [], None, None
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            d = json.loads(line[6:])
            if d.get("error"):
                raise RuntimeError(json.dumps(d["error"])[:300])
            if d.get("usage"):
                tokens = d["usage"].get("completion_tokens")
            for c in d.get("choices") or []:
                text.append((c.get("delta") or {}).get("content") or "")
                finish = c.get("finish_reason") or finish
        if finish is None:
            raise RuntimeError("the stream ended without a finish reason")
        return "".join(text), finish, tokens

raw_path = a.out + ".raw"
raw = json.load(open(raw_path)) if os.path.exists(raw_path) else {}
lock, t0 = threading.Lock(), time.time()

def run(p):
    if (p.question_id in raw and "error" not in raw[p.question_id]
            and not (a.retry_cut and raw[p.question_id].get("finish") == "length")):
        return
    body = {"model": a.model, "max_tokens": a.max_tokens, "messages": format_prompt_generation(p, LMStyle.OpenAIChat)}
    if EXTRA is None:
        body["reasoning_effort"] = a.effort
    else:
        body.update(EXTRA)
    if a.stream:
        body["stream"] = True
        body.setdefault("stream_options", {"include_usage": True})
    out = {"error": "?"}
    for attempt in range(4):
        try:
            text, finish, tokens = complete(body)
            out = {"text": text, "finish": finish, "tokens": tokens, "difficulty": str(p.difficulty)}
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
