"""Downloads the PLE n-gram table (the 128 ngram_embedding shards) of a Qwen3.8-Flash-Next checkpoint
by HTTP range requests. Resumable.

  python3 tools/ple_download.py OUT                      # bf16 from Qwen/Qwen3.8-Flash-Next:
                                                         #   OUT/shard_<i>.bf16 (raw [rows][160] bf16),
                                                         #   then tools/ple_convert.py
  python3 tools/ple_download.py OUT Qwen/Qwen3.8-Flash-Next-FP8
                                                         # fp8 (e4m3, one scale for the whole table):
                                                         #   OUT/shard_<i>.safetensors (weight_f8 u8
                                                         #   [rows][160]) and META.json, ready to serve
"""
import json, os, struct, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
OUT = sys.argv[1]
REPO = f"https://huggingface.co/{sys.argv[2] if len(sys.argv) > 2 else 'Qwen/Qwen3.8-Flash-Next'}/resolve/main/"
os.makedirs(OUT, exist_ok=True)
idx = json.load(urllib.request.urlopen(REPO + "model.safetensors.index.json", timeout=60))["weight_map"]
PFX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_"
SCALE = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.weight_scale"
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

def st_header(rows):  # the safetensors header of an fp8 shard: weight_f8 u8 [rows][160]
    h = json.dumps({"weight_f8": {"dtype": "U8", "shape": [rows, 160], "data_offsets": [0, rows * 160]}}).encode()
    h += b" " * (-len(h) % 8)
    return struct.pack("<Q", len(h)) + h

def fetch(name):
    i = int(name[len(PFX):].split(".")[0])
    f = tensors[name]
    h, base = header(f)
    meta = h[name]
    assert meta["dtype"] in ("BF16", "F8_E4M3") and meta["shape"][1] == 160, meta
    fp8 = meta["dtype"] == "F8_E4M3"
    path = f"{OUT}/shard_{i}.safetensors" if fp8 else f"{OUT}/shard_{i}.bf16"
    lead = st_header(meta["shape"][0]) if fp8 else b""
    if lead and not os.path.exists(path):
        with open(path, "wb") as out:
            out.write(lead)
    a, b = meta["data_offsets"]
    size = b - a
    have = os.path.getsize(path) - len(lead) if os.path.exists(path) else 0
    for attempt in range(20):
        if have >= size:
            return i, size, meta
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
done, rows, fp8 = 0, 0, False
with ThreadPoolExecutor(6) as ex:
    for i, size, meta in ex.map(fetch, sorted(tensors, key=lambda k: int(k[len(PFX):].split(".")[0]))):
        done += size
        rows += meta["shape"][0]
        fp8 = meta["dtype"] == "F8_E4M3"
        print(f"shard {i} ok  total {done / 1e9:.1f} GB  {done / 1e6 / (time.time() - t0):.0f} MB/s", flush=True)
if fp8:  # the table's single scale (bf16 [1]), kept in META.json
    h, base = header(idx[SCALE])
    a, _ = h[SCALE]["data_offsets"]
    assert h[SCALE]["dtype"] == "BF16" and h[SCALE]["shape"] == [1], h[SCALE]
    raw = get(REPO + idx[SCALE], base + a, base + a + 1).read()
    scale = struct.unpack("<f", b"\0\0" + raw)[0]
    json.dump({"layout": "f8e4m3_tensorscale", "shards": len(tensors), "rows": rows, "width": 160, "scale": scale},
              open(f"{OUT}/META.json", "w"))
    print(f"scale {scale}", flush=True)
print("DONE", flush=True)
