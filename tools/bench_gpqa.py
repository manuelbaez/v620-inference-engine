"""GPQA Diamond (198 questions) through the OpenAI API, in the simple-evals format: four shuffled options, the answer
on the last line as "Answer: X". Thinking on at the given effort and the model's own sampling settings (what the
model card's 91.7 was measured with as far as it says: thinking mode, temperature 1.0, top_p 0.95, top_k 20).
Writes per-question results (correct, tokens) to OUT.json; resumes from it.
  python bench_gpqa.py OUT.json [--url http://localhost:8000] [--csv gpqa_diamond.csv] [--effort xhigh]
                       [--workers 4] [--seed 0] [--max-tokens 32768] [--retry-cut]
The questions: https://openaipublic.blob.core.windows.net/simple-evals/gpqa_diamond.csv"""
import argparse, csv, json, os, random, re, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--csv", default="gpqa_diamond.csv")
ap.add_argument("--effort", default="xhigh")
ap.add_argument("--workers", type=int, default=4)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--max-tokens", type=int, default=32768)
ap.add_argument("--model", default="qw/qwen3.8-flash-next")
ap.add_argument("--retry-cut", action="store_true", help="ask again the questions whose answer was cut off at max_tokens")
a = ap.parse_args()
URL = a.url.rstrip("/") + "/v1/chat/completions"
TEMPLATE = ("Answer the following multiple choice question. The last line of your response should be of the following "
            "format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before "
            "answering.\n\n{q}\n\nA) {A}\nB) {B}\nC) {C}\nD) {D}")

rows = list(csv.DictReader(open(a.csv)))
rnd = random.Random(a.seed)
items = []
for i, r in enumerate(rows):
    opts = [r["Correct Answer"], r["Incorrect Answer 1"], r["Incorrect Answer 2"], r["Incorrect Answer 3"]]
    perm = rnd.sample(range(4), 4)
    shuffled = [opts[p].strip() for p in perm]
    gold = "ABCD"[perm.index(0)]
    items.append((i, TEMPLATE.format(q=r["Question"].strip(), A=shuffled[0], B=shuffled[1], C=shuffled[2], D=shuffled[3]), gold))

res = json.load(open(a.out)) if os.path.exists(a.out) else {}
lock, t0 = threading.Lock(), time.time()

def ask(prompt):
    body = {"model": a.model, "max_tokens": a.max_tokens, "messages": [{"role": "user", "content": prompt}],
            "reasoning_effort": a.effort}
    for attempt in range(4):
        try:
            req = urllib.request.Request(URL, json.dumps(body).encode(), {"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=3600) as r:
                return json.load(r)
        except Exception as ex:
            err = str(ex)
            time.sleep(10 * (attempt + 1))
    return {"error": err}

def run(it):
    i, prompt, gold = it
    if str(i) in res and "error" not in res[str(i)] and not (a.retry_cut and res[str(i)].get("finish") == "length"):
        return
    d = ask(prompt)
    if "error" in d:
        out = {"error": d["error"], "correct": False}
    else:
        c = d["choices"][0]
        text = c["message"].get("content") or ""
        m = re.findall(r"(?i)Answer[ \t]*:[ \t]*\$?([A-D])\$?", text)
        out = {"correct": bool(m) and m[-1].upper() == gold, "answer": m[-1].upper() if m else None, "gold": gold,
               "finish": c.get("finish_reason"), "tokens": d.get("usage", {}).get("completion_tokens")}
    with lock:
        res[str(i)] = out
        json.dump(res, open(a.out, "w"))
        n = len(res)
        if n % 10 == 0:
            ok = sum(v["correct"] for v in res.values())
            print(f"{n}/{len(items)} done, {ok} correct ({100 * ok / n:.1f}%), {time.time() - t0:.0f} s", flush=True)

with ThreadPoolExecutor(a.workers) as ex:
    list(ex.map(run, items))
ok = sum(v["correct"] for v in res.values())
n = len(res)
se = (ok / n * (1 - ok / n) / n) ** 0.5 * 100
print(f"GPQA Diamond: {ok}/{n} = {100 * ok / n:.1f}% (standard error {se:.1f}); no answer found in "
      f"{sum(v.get('answer') is None for v in res.values())}, cut off at max_tokens in "
      f"{sum(v.get('finish') == 'length' for v in res.values())}, errors {sum('error' in v for v in res.values())}; "
      f"mean completion tokens {sum(v.get('tokens') or 0 for v in res.values()) / n:.0f}", flush=True)
