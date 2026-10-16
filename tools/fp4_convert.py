"""nvidia/Qwen3.8-Flash-Next-FP4-Local -> a checkpoint the engine reads (docs/DESIGN.md "The FP4 checkpoint"):
the AWQ checkpoint's tensor names and types, with that model's (QAD-trained) values.

  - MXFP8 matrices (E4M3, a UE8M0 scale per 32 inputs: GDN, QSA, shared experts, vision attention) -> bf16, exact
  - NVFP4 matrices outside the routed experts (vision MLP, the MTP head's experts) -> bf16, rounded to nearest
  - F32 tensors where the engine reads bf16 (the hyper-connection norms) -> bf16, rounded to nearest
  - the routed experts stay NVFP4: weight_packed I32 [out][in / 8] (the E2M1 nibbles as they are, the even input
    in the low nibble), weight_scale U8 [out][in / 16] (E4M3; the checkpoint's compressed "csf" scales expanded),
    weight_global F32 [1] (its weight_scale_2), weight_shape I64 [2]
  - the n-gram table -> a PLE sidecar, layout group16_nvfp4_e4m3scale: weight_f4 U8 [rows][80], weight_scale U8
    [rows][10] (E4M3), weight_global F32 [1] per shard (the table has one FP32 scale: the same in every shard)
  - config.json: the AWQ one with quantization_config {"quant_method": "qw-nvfp4", ... "group_size": 16}

  python3 tools/fp4_convert.py FP4_DIR AWQ_DIR OUT_DIR PLE_OUT_DIR [--ple-only]
"""
import json, os, shutil, struct, sys
import numpy as np

FP4, AWQ, OUT, PLE_OUT = sys.argv[1:5]
PLE_ONLY = "--ple-only" in sys.argv  # the checkpoint is converted already: only the PLE sidecar
SHARD_BYTES = 4 << 30
os.makedirs(OUT, exist_ok=True)
os.makedirs(PLE_OUT, exist_ok=True)


class Checkpoint:
    def __init__(self, d):
        self.d, self.idx, self.hdr = d, json.load(open(d + "/model.safetensors.index.json"))["weight_map"], {}

    def meta(self, name):
        f = self.d + "/" + self.idx[name]
        if f not in self.hdr:
            with open(f, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                self.hdr[f] = (json.loads(fh.read(n)), 8 + n)
        h, base = self.hdr[f]
        return f, h[name], base

    def raw(self, name):
        f, m, base = self.meta(name)
        dt = {"BF16": np.uint16, "F32": np.float32, "F8_E4M3": np.uint8, "U8": np.uint8, "U32": np.uint32,
              "I32": np.int32, "I64": np.int64, "F16": np.uint16}[m["dtype"]]
        return np.memmap(f, dtype=dt, mode="r", offset=base + m["data_offsets"][0], shape=tuple(m["shape"])), m


fp4, awq = Checkpoint(FP4), Checkpoint(AWQ)
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6], dtype=np.float32)
E4M3 = None


def e4m3(b):
    global E4M3
    if E4M3 is None:
        x = np.arange(256, dtype=np.int32)
        s, ex, m = np.where(x & 0x80, -1.0, 1.0), (x >> 3) & 15, x & 7
        E4M3 = (s * np.where(ex == 0, m / 8.0 * 2.0 ** -6, (1 + m / 8.0) * 2.0 ** (ex - 7))).astype(np.float32)
    return E4M3[b]


def to_bf16(x):  # float32 -> uint16 bf16, round to nearest even
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


def mxfp8(name):  # base name (without .weight) -> fp32, exact in bf16
    w, _ = fp4.raw(name + ".weight")
    s, _ = fp4.raw(name + ".weight_scale")
    v = e4m3(np.asarray(w)) * np.repeat(np.ldexp(1.0, np.asarray(s).astype(np.int32) - 127).astype(np.float32), 32, axis=1)
    assert not (v.view(np.uint32) & 0xFFFF).any(), name + ": MXFP8 value not exact in bf16"
    return v


def csf_scales(p, rows, cols):  # tools/expert_quant_download.py csf_scales, on memmaps
    f = np.asarray(fp4.raw(p + "weight_scale.nvfp4_csf_fixed")[0]).reshape(rows // 16, 16 + 8 * cols)
    base, packed = f[:, :16].reshape(rows, 1), f[:, 16:].reshape(rows, cols // 2)
    diff = np.empty((rows, cols), dtype=np.uint8)
    diff[:, 0::2], diff[:, 1::2] = packed & 15, packed >> 4
    s = (base.astype(np.int32) + diff).astype(np.uint8)
    x = np.asarray(fp4.raw(p + "weight_scale.nvfp4_csf_exceptions")[0])
    s.ravel()[x & 0xFFFFFF] = (x >> 24).astype(np.uint8)
    return s


def nvfp4_scales(p, rows, cols):
    if p + "weight_scale.nvfp4_csf_fixed" in fp4.idx:
        return csf_scales(p, rows, cols)
    return np.asarray(fp4.raw(p + "weight_scale")[0]).reshape(rows, cols)


def nvfp4(p):  # prefix ending in "." -> fp32 [out][in]
    q = np.asarray(fp4.raw(p + "weight")[0])
    rows, n = q.shape[0], 2 * q.shape[1]
    v = np.empty((rows, n), dtype=np.float32)
    v[:, 0::2], v[:, 1::2] = E2M1[q & 15], E2M1[q >> 4]
    g = float(np.asarray(fp4.raw(p + "weight_scale_2")[0]).ravel()[0])
    return v * np.repeat(e4m3(nvfp4_scales(p, rows, n // 16)), 16, axis=1) * np.float32(g)


# --- the output, in shards of ~4 GiB
shard, shard_bytes, files, weight_map, used = [], 0, [], {}, set()


def flush():
    global shard, shard_bytes
    if not shard:
        return
    name = f"model-{len(files) + 1:05d}.safetensors"
    hdr, off = {}, 0
    for n, dt, shape, b in shard:
        hdr[n] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
        weight_map[n] = name
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(f"{OUT}/{name}.tmp", "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for *_, b in shard:
            f.write(b)
    os.replace(f"{OUT}/{name}.tmp", f"{OUT}/{name}")
    files.append(name)
    print(f"{name}: {len(shard)} tensors, {off / 2**30:.2f} GiB", flush=True)
    shard, shard_bytes = [], 0


def put(name, dt, shape, b):
    global shard_bytes
    if shard_bytes + len(b) > SHARD_BYTES:
        flush()
    shard.append((name, dt, tuple(shape), b))
    shard_bytes += len(b)


def take(*names):
    for n in names:
        if n in fp4.idx:
            used.add(n)


EXPERT_PARTS = ("weight_packed", "weight_scale", "weight_shape")
done_experts = set()
for name in [] if PLE_ONLY else sorted(awq.idx):
    _, am, _ = awq.meta(name)
    if ".mlp.experts." in name and name.startswith("model.language_model."):  # the routed experts: stay NVFP4
        p = name.rsplit(".", 1)[0] + "."
        if p in done_experts:
            continue
        done_experts.add(p)
        q, qm = fp4.raw(p + "weight")
        rows, n = qm["shape"][0], 2 * qm["shape"][1]
        put(p + "weight_packed", "I32", (rows, n // 8), np.asarray(q).tobytes())
        put(p + "weight_scale", "U8", (rows, n // 16), nvfp4_scales(p, rows, n // 16).tobytes())
        put(p + "weight_global", "F32", (1,), np.asarray(fp4.raw(p + "weight_scale_2")[0]).astype(np.float32).tobytes())
        put(p + "weight_shape", "I64", (2,), np.array([rows, n], dtype=np.int64).tobytes())
        take(p + "weight", p + "weight_scale_2", p + "input_scale", p + "weight_scale.nvfp4_csf_fixed",
             p + "weight_scale.nvfp4_csf_exceptions")
        continue
    if name.startswith("mtp.layers.0.mlp.experts."):  # fused bf16 [E][2 FFN][H] and [E][H][FFN]
        E = am["shape"][0]
        parts = ("gate_proj", "up_proj") if name.endswith("gate_up_proj") else ("down_proj",)
        out = np.empty(am["shape"], dtype=np.uint16)
        for e in range(E):
            ws = []
            for part in parts:
                p = f"mtp.layers.0.mlp.experts.{e}.{part}."
                ws.append(nvfp4(p))
                take(p + "weight", p + "weight_scale", p + "weight_scale_2", p + "input_scale")
            out[e] = to_bf16(np.concatenate(ws, axis=0))
        put(name, "BF16", am["shape"], out.tobytes())
        continue
    base = name[: -len(".weight")] if name.endswith(".weight") else None
    if base and base + ".weight_scale_2" in fp4.idx:  # NVFP4 -> bf16
        v, how = to_bf16(nvfp4(base + ".")), "nvfp4"
        take(name, base + ".weight_scale", base + ".weight_scale_2", base + ".input_scale")
    elif base and base + ".weight_scale" in fp4.idx:  # MXFP8 -> bf16
        v, how = to_bf16(mxfp8(base)), "mxfp8"
        take(name, base + ".weight_scale")
    else:
        r, rm = fp4.raw(name)
        take(name)
        if rm["dtype"] == am["dtype"]:
            v, how = np.asarray(r), "copy"
        elif rm["dtype"] == "F32" and am["dtype"] == "BF16":
            v, how = to_bf16(np.asarray(r)), "f32->bf16"
        else:
            raise SystemExit(f"{name}: {rm['dtype']} in the FP4 checkpoint, {am['dtype']} expected")
    assert list(v.shape) == list(am["shape"]), (name, v.shape, am["shape"])
    assert v.dtype.itemsize == {"BF16": 2, "F32": 4, "F16": 2, "I64": 8, "I32": 4}[am["dtype"]], (name, how)
    put(name, am["dtype"], am["shape"], np.ascontiguousarray(v).tobytes())
flush()

# --- the n-gram table: a PLE sidecar (one FP32 scale for the whole table, ngram_embedding.weight_scale_2)
ng = sorted((k for k in fp4.idx if ".ngram_embedding.shard_" in k and k.endswith(".weight")),
            key=lambda k: int(k.split(".shard_")[1].split(".")[0]))
g_name = ng[0].split(".shard_")[0] + ".weight_scale_2"
g = np.asarray(fp4.raw(g_name)[0]).astype(np.float32).reshape(1)
take(g_name)
rows = 0
for k in ng:
    i, b = int(k.split(".shard_")[1].split(".")[0]), k[: -len(".weight")]
    w, wm = fp4.raw(k)
    s, _ = fp4.raw(b + ".weight_scale")
    rows += wm["shape"][0]
    tensors = [("weight_f4", "U8", wm["shape"], np.asarray(w).tobytes()),
               ("weight_scale", "U8", s.shape, np.asarray(s).tobytes()), ("weight_global", "F32", (1,), g.tobytes())]
    hdr, off = {}, 0
    for n, dt, shape, by in tensors:
        hdr[n] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(by)]}
        off += len(by)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(f"{PLE_OUT}/shard_{i}.safetensors.tmp", "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for *_, by in tensors:
            f.write(by)
    os.replace(f"{PLE_OUT}/shard_{i}.safetensors.tmp", f"{PLE_OUT}/shard_{i}.safetensors")
    take(k, b + ".weight_scale")
json.dump({"layout": "group16_nvfp4_e4m3scale", "shards": len(ng), "rows": rows, "width": 160},
          open(PLE_OUT + "/META.json", "w"))
print(f"PLE: {len(ng)} shards, {rows} rows", flush=True)

if PLE_ONLY:
    print("done", flush=True)
    sys.exit(0)

# --- what was not used must be nothing but the activation scales
left = sorted(k for k in fp4.idx if k not in used and not k.endswith(".input_scale"))
assert not left, f"{len(left)} tensors of the FP4 checkpoint unused, e.g. {left[:5]}"

json.dump({"metadata": {}, "weight_map": weight_map}, open(OUT + "/model.safetensors.index.json", "w"), indent=1)
cfg = json.load(open(AWQ + "/config.json"))
cfg["quantization_config"] = {"quant_method": "qw-nvfp4", "source": "nvidia/Qwen3.8-Flash-Next-FP4-Local",
                              "config_groups": {"group_0": {"weights": {"num_bits": 4, "group_size": 16,
                                                                        "type": "nvfp4", "symmetric": True}}}}
json.dump(cfg, open(OUT + "/config.json", "w"), indent=2)
for f in os.listdir(AWQ):  # tokenizer, chat template, generation config, preprocessor
    if not f.endswith(".safetensors") and f not in ("config.json", "model.safetensors.index.json") and \
            os.path.isfile(f"{AWQ}/{f}"):
        shutil.copy(f"{AWQ}/{f}", f"{OUT}/{f}")
print("done", flush=True)
