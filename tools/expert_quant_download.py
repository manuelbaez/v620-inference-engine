"""Downloads the routed experts of some layers from another 4-bit quantization of the model on Hugging Face
(compressed-tensors `pack-quantized`, with or without zero points, any group size), dequantizes them to bf16 as a
loader of that checkpoint would, and writes them in tools/expert_download.py's layout, so that the fp32 reference runs
with them through QW_REF_EXPERTS_DIR (docs/DESIGN.md "Another 4-bit quantization of the experts"). With ORIG (the
original experts, tools/expert_download.py) it also prints the relative error of the weights against them.

  python3 tools/expert_quant_download.py REPO OUT [--orig ORIG] LAYER [LAYER ...]
      REPO e.g. cyankiwi/Qwen3.8-Flash-Next-AWQ-INT4
"""
import json, os, struct, sys, time, urllib.request
import numpy as np
args = sys.argv[1:]
ORIG = args.pop(args.index("--orig") + 1) if "--orig" in args else None
if ORIG:
    args.remove("--orig")
REPO, OUT, LAYERS = f"https://huggingface.co/{args[0]}/resolve/main/", args[1], [int(a) for a in args[2:]]
H, FFN, NEXP = 2560, 640, 512
os.makedirs(OUT, exist_ok=True)

def get(f, a, b):
    for attempt in range(50):
        try:
            return urllib.request.urlopen(urllib.request.Request(REPO + f, headers={"Range": f"bytes={a}-{b - 1}"}), timeout=120).read()
        except Exception as ex:
            print(f"  retry {attempt}: {ex}", flush=True)
            time.sleep(5)
    raise SystemExit("download failed")

idx = json.load(urllib.request.urlopen(REPO + "model.safetensors.index.json", timeout=60))["weight_map"]
_hdr = {}
def header(f):
    if f not in _hdr:
        n = struct.unpack("<Q", get(f, 0, 8))[0]
        _hdr[f] = (json.loads(get(f, 8, 8 + n)), 8 + n)
    return _hdr[f]

def fetch_layer(L):  # {tensor name: bytes}, the layer's expert tensors in as few ranges as they lie in
    prefix = f"model.language_model.layers.{L}.mlp.experts."
    by_file = {}
    for name, f in idx.items():
        if name.startswith(prefix):
            by_file.setdefault(f, []).append(name)
    out, t0, total = {}, time.time(), 0
    for f, names in by_file.items():
        h, base = header(f)
        spans = sorted((h[n]["data_offsets"][0], h[n]["data_offsets"][1], n) for n in names)
        i = 0
        while i < len(spans):
            j = i
            while j + 1 < len(spans) and spans[j + 1][0] == spans[j][1]:
                j += 1
            a, b = spans[i][0], spans[j][1]
            buf = get(f, base + a, base + b)
            assert len(buf) == b - a, "short read"
            total += len(buf)
            for s, e, n in spans[i:j + 1]:
                out[n] = (buf[s - a:e - a], h[n])
            i = j + 1
    print(f"L{L}: {total >> 20} MB in {len(by_file)} files, {total / 1e6 / (time.time() - t0):.0f} MB/s", flush=True)
    return out

def nibbles(buf, meta, n):  # packed int32 words [rows][words] -> [rows][n] values - 8
    w = np.frombuffer(buf, dtype=np.uint32).reshape(meta["shape"])
    v = (w[:, :, None] >> (4 * np.arange(8, dtype=np.uint32))) & 0xF
    return v.reshape(w.shape[0], -1)[:, :n].astype(np.int8) - 8

def bf16(a):  # uint16 bf16 -> float32
    return (a.astype(np.uint32) << 16).view(np.float32)

def to_bf16(x):  # float32 -> uint16 bf16, round to nearest even
    u = x.view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)

def matrix(t, p):  # dequantized [out][in] fp32
    out, n = (int(v) for v in np.frombuffer(t[p + "weight_shape"][0], dtype=np.int64))
    q = nibbles(*t[p + "weight_packed"], n)
    sbuf, smeta = t[p + "weight_scale"]
    assert smeta["dtype"] == "BF16", smeta
    scale = bf16(np.frombuffer(sbuf, dtype=np.uint16)).reshape(smeta["shape"])
    groups = scale.shape[1]
    if p + "weight_zero_point" in t:  # packed along the rows: [out / 8][groups]
        zbuf, zmeta = t[p + "weight_zero_point"]
        zw = {"shape": zmeta["shape"][::-1]}
        zp = nibbles(np.frombuffer(zbuf, dtype=np.uint32).reshape(zmeta["shape"]).T.tobytes(), zw, out).T
        q = q.astype(np.int16) - np.repeat(zp, n // groups, axis=1)
    return q.astype(np.float32) * np.repeat(scale, n // groups, axis=1)

rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
metas = {}
for L in LAYERS:
    t = fetch_layer(L)
    gu = np.memmap(f"{OUT}/L{L}.gate_up_proj.bf16", dtype=np.uint16, mode="w+", shape=(NEXP, 2 * FFN, H))
    dn = np.memmap(f"{OUT}/L{L}.down_proj.bf16", dtype=np.uint16, mode="w+", shape=(NEXP, H, FFN))
    if ORIG:
        ogu = np.memmap(f"{ORIG}/L{L}.gate_up_proj.bf16", dtype=np.uint16, mode="r", shape=gu.shape)
        odn = np.memmap(f"{ORIG}/L{L}.down_proj.bf16", dtype=np.uint16, mode="r", shape=dn.shape)
    err = {}
    for e in range(NEXP):
        p = f"model.language_model.layers.{L}.mlp.experts.{e}."
        for part, dst, src in (("gate_proj", gu[e, :FFN], ORIG and ogu[e, :FFN]), ("up_proj", gu[e, FFN:], ORIG and ogu[e, FFN:]),
                               ("down_proj", dn[e], ORIG and odn[e])):
            w = matrix(t, p + part + ".")
            dst[:] = to_bf16(w)
            if ORIG:
                o = bf16(np.asarray(src))
                err.setdefault(part, []).append((rel(w, o), rel(bf16(np.asarray(dst)), o)))
    gu.flush(); dn.flush()
    metas[f"L{L}.gate_up_proj.bf16"], metas[f"L{L}.down_proj.bf16"] = list(gu.shape), list(dn.shape)
    for part, v in err.items():
        a = np.array(v)
        print(f"  L{L} {part:9s} relative error against the original: {a[:, 0].mean():.4f} ({a[:, 0].min():.4f}-{a[:, 0].max():.4f}), "
              f"as bf16 {a[:, 1].mean():.4f}", flush=True)
json.dump(metas, open(OUT + "/META.json", "w"))
print("done", flush=True)
