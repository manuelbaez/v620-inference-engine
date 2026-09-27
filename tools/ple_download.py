"""Downloads the bf16 PLE n-gram table (the 128 ngram_embedding shards) of Qwen/Qwen3.8-Flash-Next
by HTTP range requests into OUT/shard_<i>.bf16 (raw [rows][160] bf16). Resumable."""
import json, os, struct, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
REPO = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"
OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
idx = json.load(urllib.request.urlopen(REPO + "model.safetensors.index.json", timeout=60))["weight_map"]
PFX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"
tensors = {k: v for k, v in idx.items() if k.startswith(PFX)}

def get(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"})
    return urllib.request.urlopen(req, timeout=120)

headers = {}
def header(f):
    if f not in headers:
        r = get(REPO + f, 0, 7).read()
        n = struct.unpack("<Q", r)[0]
        headers[f] = (json.loads(get(REPO + f, 8, 8 + n - 1).read()), 8 + n)
    return headers[f]

def fetch(name):
    i = int(name[len(PFX):].split(".")[0])
    path = f"{OUT}/shard_{i}.bf16"
    f = tensors[name]
    h, base = header(f)
    meta = h[name]
    assert meta["dtype"] == "BF16" and meta["shape"][1] == 160, meta
    a, b = meta["data_offsets"]
    size = b - a
    have = os.path.getsize(path) if os.path.exists(path) else 0
    for attempt in range(20):
        if have >= size:
            return i, size
        try:
            with open(path, "ab") as out:
                r = get(REPO + f, base + a + have, base + b - 1)
                while True:
                    chunk = r.read(8 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    have += len(chunk)
        except Exception as ex:
            print(f"shard {i}: {ex}, retrying", flush=True)
            time.sleep(5)
    raise RuntimeError(f"shard {i} incomplete")

t0 = time.time()
done = 0
with ThreadPoolExecutor(6) as ex:
    for i, size in ex.map(fetch, sorted(tensors, key=lambda k: int(k[len(PFX):].split(".")[0]))):
        done += size
        print(f"shard {i} ok  total {done / 1e9:.1f} GB  {done / 1e6 / (time.time() - t0):.0f} MB/s", flush=True)
print("DONE", flush=True)
