"""Downloads the original (bf16) routed experts of some layers of Qwen/Qwen3.8-Flash-Next by HTTP range requests,
for comparing the engine's int4 experts with them (docs/DESIGN.md "The int4 experts against the original"). Resumable.

  python3 tools/expert_download.py OUT LAYER [LAYER ...]
      OUT/L<layer>.gate_up_proj.bf16   raw [512][1280][2560] bf16 (gate rows, then up rows)
      OUT/L<layer>.down_proj.bf16      raw [512][2560][640] bf16
      OUT/META.json                    shapes per file
"""
import json, os, struct, sys, time, urllib.request
OUT = sys.argv[1]
LAYERS = [int(a) for a in sys.argv[2:]]
REPO = "https://huggingface.co/Qwen/Qwen3.8-Flash-Next/resolve/main/"
os.makedirs(OUT, exist_ok=True)
idx = json.load(urllib.request.urlopen(REPO + "model.safetensors.index.json", timeout=60))["weight_map"]

def get(url, a, b):
    return urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b}"}), timeout=120)

meta_path = OUT + "/META.json"
metas = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
for L in LAYERS:
    for part in ("gate_up_proj", "down_proj"):
        name = f"model.language_model.layers.{L}.mlp.experts.{part}"
        f = idx[name]
        n = struct.unpack("<Q", get(REPO + f, 0, 7).read())[0]
        m = json.loads(get(REPO + f, 8, 8 + n - 1).read())[name]
        assert m["dtype"] == "BF16", m
        a, b = m["data_offsets"]
        path = f"{OUT}/L{L}.{part}.bf16"
        metas[os.path.basename(path)] = m["shape"]
        json.dump(metas, open(meta_path, "w"))
        t0, start = time.time(), os.path.getsize(path) if os.path.exists(path) else 0
        for attempt in range(50):
            have = os.path.getsize(path) if os.path.exists(path) else 0
            if have >= b - a:
                break
            try:
                with open(path, "ab") as out:
                    r = get(REPO + f, 8 + n + a + have, 8 + n + b - 1)
                    while True:
                        buf = r.read(1 << 22)
                        if not buf:
                            break
                        out.write(buf)
            except Exception as ex:
                print(f"  retry {attempt}: {ex}", flush=True)
                time.sleep(5)
        size = os.path.getsize(path)
        print(f"L{L} {part} {m['shape']}: {size >> 20} MB, {(size - start) / 1e6 / max(time.time() - t0, 1e-3):.0f} MB/s", flush=True)
        assert size == b - a, "incomplete"
print("done", flush=True)
