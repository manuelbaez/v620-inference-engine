"""The engine's int4 routed experts (the AWQ checkpoint: int4, a bf16 scale per 128 inputs) against the original bf16
ones (tools/expert_download.py), weight by weight: the relative error of each matrix, and what plain rounding of the
original to b bits would give for comparison (symmetric, a scale per group of inputs, the clip that minimizes the
squared error among a few).

  python3 tools/expert_compare.py ORIG_DIR LAYER [--experts N=32] [--awq /mnt/llms/qwen3.8-flash-next-awq]
"""
import json, struct, sys, numpy as np
ORIG, L = sys.argv[1], int(sys.argv[2])
NEXP = int(sys.argv[sys.argv.index("--experts") + 1]) if "--experts" in sys.argv else 32
AWQ = sys.argv[sys.argv.index("--awq") + 1] if "--awq" in sys.argv else "/mnt/llms/qwen3.8-flash-next-awq"
H, FFN, G = 2560, 640, 128

def bf16(a):  # uint16 bf16 -> float32
    return (a.astype(np.uint32) << 16).view(np.float32)

idx = json.load(open(AWQ + "/model.safetensors.index.json"))["weight_map"]
_hdr = {}
def awq(name):
    f = AWQ + "/" + idx[name]
    if f not in _hdr:
        with open(f, "rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            _hdr[f] = (json.loads(fh.read(n)), 8 + n)
    h, base = _hdr[f]
    m = h[name]
    dt = {"I32": np.int32, "BF16": np.uint16}[m["dtype"]]
    return np.memmap(f, dtype=dt, mode="r", offset=base + m["data_offsets"][0], shape=tuple(m["shape"]))

def awq_matrix(e, part):  # dequantized [out][in]
    p = f"model.language_model.layers.{L}.mlp.experts.{e}.{part}."
    packed = np.asarray(awq(p + "weight_packed")).view(np.uint32)
    scale = bf16(np.asarray(awq(p + "weight_scale")))
    out = packed.shape[0]
    nib = ((packed[:, :, None] >> (4 * np.arange(8, dtype=np.uint32))) & 0xF).astype(np.float32).reshape(out, -1) - 8.0
    return nib * np.repeat(scale, G, axis=1)

def rtn(w, bits, group):  # round to nearest, symmetric, a scale per `group` inputs, best of a few clips
    out, n = w.shape
    x = w.reshape(out, n // group, group)
    lv = 2 ** (bits - 1) - 1
    best, err = None, None
    m = np.abs(x).max(axis=2, keepdims=True) + 1e-30
    for clip in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5):
        s = m * clip / lv
        q = np.clip(np.round(x / s), -lv - 1, lv) * s
        e = ((q - x) ** 2).sum(axis=2, keepdims=True)
        if best is None:
            best, err = q, e
        else:
            take = e < err
            best, err = np.where(take, q, best), np.where(take, e, err)
    return best.reshape(out, n)

gu = np.memmap(f"{ORIG}/L{L}.gate_up_proj.bf16", dtype=np.uint16, mode="r", shape=(512, 2 * FFN, H))
dn = np.memmap(f"{ORIG}/L{L}.down_proj.bf16", dtype=np.uint16, mode="r", shape=(512, H, FFN))
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))
# which half of gate_up_proj is the gate
g0 = awq_matrix(0, "gate_proj")
first, second = bf16(np.asarray(gu[0, :FFN])), bf16(np.asarray(gu[0, FFN:]))
print(f"layer {L}: int4 gate_proj of expert 0 against the first / second half of gate_up_proj: {rel(g0, first):.3f} / {rel(g0, second):.3f}")
gate_first = rel(g0, first) < rel(g0, second)
rows = {}
rng = np.random.default_rng(0)
for e in rng.choice(512, NEXP, replace=False):
    orig = {"gate_proj": bf16(np.asarray(gu[e, :FFN] if gate_first else gu[e, FFN:])),
            "up_proj": bf16(np.asarray(gu[e, FFN:] if gate_first else gu[e, :FFN])), "down_proj": bf16(np.asarray(dn[e]))}
    for part, w in orig.items():
        r = rows.setdefault(part, {})
        r.setdefault("int4 (checkpoint)", []).append(rel(awq_matrix(e, part), w))
        for bits, group in ((4, 128), (4, 16), (5, 128), (6, 128), (6, 16), (8, 128)):
            r.setdefault(f"rounded to {bits} bits, group {group}", []).append(rel(rtn(w, bits, group), w))
print(f"relative error of the weights, mean over {NEXP} experts (min-max):")
for part, r in rows.items():
    for k, v in r.items():
        print(f"  {part:9s} {k:28s} {np.mean(v):.4f} ({np.min(v):.4f}-{np.max(v):.4f})")
