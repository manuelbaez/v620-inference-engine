"""Task benchmarks through the OpenAI API (greedy, thinking off): GSM8K (exact final number),
MMLU (stratified sample, letter answer), ARC-Challenge (letter answer). Writes per-question
correctness to OUT.json for paired comparisons.
  python bench_tasks.py OUT.json [--url http://localhost:8000] [--mmlu-per-subject 25] [--limit N]"""
import argparse, json, random, re, threading, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from datasets import load_dataset

ap = argparse.ArgumentParser()
ap.add_argument("out")
ap.add_argument("--url", default="http://localhost:8000")
ap.add_argument("--mmlu-per-subject", type=int, default=25)
ap.add_argument("--limit", type=int, default=0)
ap.add_argument("--workers", type=int, default=4)
a = ap.parse_args()
URL = a.url.rstrip("/") + "/v1/chat/completions"

def ask(prompt, max_tokens):
    body = {"model": "x", "temperature": 0, "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}], "chat_template_kwargs": {"enable_thinking": False}}
    for attempt in range(5):
        try:
            req = urllib.request.Request(URL, json.dumps(body).encode(), {"content-type": "application/json"})
            with urllib.request.urlopen(req, timeout=900) as r:
                return json.load(r)["choices"][0]["message"]["content"]
        except Exception as ex:
            time.sleep(5 * (attempt + 1))
    return ""

items = []  # (task, id, prompt, max_tokens, gold, checker)
def num(s):
    s = s.replace(",", "").replace("$", "")
    m = re.findall(r"-?\d+(?:\.\d+)?", s)
    return m[-1] if m else None
def same_num(x, y):
    try:
        return x is not None and abs(float(x) - float(y)) < 1e-6
    except ValueError:
        return False

gsm = load_dataset("openai/gsm8k", "main", split="test")
for i, r in enumerate(gsm):
    gold = r["answer"].split("####")[-1].strip()
    p = r["question"] + "\n\nSolve step by step, then give the final answer on the last line as: #### <number>"
    items.append(("gsm8k", i, p, 768, gold, lambda out, g: same_num(num(out.split("####")[-1]) if "####" in out else num(out), g)))

LET = "ABCDEFGHIJ"
def letter(out):
    m = re.search(r"\b([A-J])\b", out.strip().upper()[:40])
    return m.group(1) if m else None

mmlu = load_dataset("cais/mmlu", "all", split="test")
by_subj = {}
for i, r in enumerate(mmlu):
    by_subj.setdefault(r["subject"], []).append(i)
rnd = random.Random(0)
for subj in sorted(by_subj):
    for i in sorted(rnd.sample(by_subj[subj], min(a.mmlu_per_subject, len(by_subj[subj])))):
        r = mmlu[i]
        opts = "\n".join(f"{LET[k]}. {c}" for k, c in enumerate(r["choices"]))
        p = f"{r['question']}\n\n{opts}\n\nAnswer with only the letter of the correct option."
        items.append(("mmlu", i, p, 8, LET[r["answer"]], lambda out, g: letter(out) == g))

arc = load_dataset("allenai/ai2_arc", "ARC-Challenge", split="test")
for i, r in enumerate(arc):
    labels, texts = r["choices"]["label"], r["choices"]["text"]
    opts = "\n".join(f"{LET[k]}. {t}" for k, t in enumerate(texts))
    gold = LET[labels.index(r["answerKey"])]
    p = f"{r['question']}\n\n{opts}\n\nAnswer with only the letter of the correct option."
    items.append(("arc", i, p, 8, gold, lambda out, g: letter(out) == g))

if a.limit:
    items = [it for t in ("gsm8k", "mmlu", "arc") for it in [x for x in items if x[0] == t][:a.limit]]
print(f"{len(items)} questions", flush=True)
res, lock, t0 = {}, threading.Lock(), time.time()
def run(it):
    task, i, p, mt, gold, check = it
    out = ask(p, mt)
    ok = bool(check(out, gold))
    with lock:
        res[f"{task}:{i}"] = ok
        n = len(res)
        if n % 200 == 0:
            print(f"{n}/{len(items)} done, {time.time() - t0:.0f} s", flush=True)
with ThreadPoolExecutor(a.workers) as ex:
    list(ex.map(run, items))
json.dump(res, open(a.out, "w"))
for t in ("gsm8k", "mmlu", "arc"):
    v = [ok for k, ok in res.items() if k.startswith(t + ":")]
    print(f"{t}: {sum(v)}/{len(v)} = {100 * sum(v) / max(1, len(v)):.1f}%", flush=True)
