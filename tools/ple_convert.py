"""From the raw bf16 n-gram shards (ple_dl.py) to the engine's PLE sidecars:
  OUT/bf16/shard_<i>.safetensors  weight bf16 [rows][160]                      (layout bf16)
  OUT/int8/shard_<i>.safetensors  weight_i8 i8 [rows][160], weight_scale f16 [rows][10]
                                  (layout group16_int8_fp16scale: max|x| / 127 per 16 values)
and checks shard 0 against the existing int4 sidecar (same rows, ~int4 error)."""
import json, os, struct, sys
import numpy as np
RAW, OUT, INT4 = sys.argv[1], sys.argv[2], sys.argv[3]
WORKER, WORKERS = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, 1)  # shards i % WORKERS == WORKER
SHARDS, ROWS, W = 128, 320001536, 160
PER = ROWS // SHARDS

def write_st(path, tensors):  # tensors: [(name, dtype str, shape, bytes)]
    hdr, off = {}, 0
    for name, dt, shape, b in tensors:
        hdr[name] = {"dtype": dt, "shape": list(shape), "data_offsets": [off, off + len(b)]}
        off += len(b)
    h = json.dumps(hdr).encode()
    h += b" " * (-len(h) % 8)
    with open(path + ".tmp", "wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for _, _, _, b in tensors:
            f.write(b)
    os.replace(path + ".tmp", path)

def read_st(path, name):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(n))[name]
        f.seek(8 + n + meta["data_offsets"][0])
        return np.frombuffer(f.read(meta["data_offsets"][1] - meta["data_offsets"][0]), dtype=np.uint8), meta["shape"]

for sub, layout in (("bf16", "bf16"), ("int8", "group16_int8_fp16scale")):
    os.makedirs(f"{OUT}/{sub}", exist_ok=True)
    json.dump({"layout": layout, "shards": SHARDS, "rows": ROWS, "width": W}, open(f"{OUT}/{sub}/META.json", "w"))
worst = 0.0
for i in range(WORKER, SHARDS, WORKERS):
    if os.path.exists(f"{OUT}/bf16/shard_{i}.safetensors") and os.path.exists(f"{OUT}/int8/shard_{i}.safetensors"):
        continue
    raw = np.fromfile(f"{RAW}/shard_{i}.bf16", dtype=np.uint16)
    assert raw.size == PER * W, (i, raw.size)
    if not os.path.exists(f"{OUT}/bf16/shard_{i}.safetensors"):
        write_st(f"{OUT}/bf16/shard_{i}.safetensors", [("weight", "BF16", (PER, W), raw.tobytes())])
    if not os.path.exists(f"{OUT}/int8/shard_{i}.safetensors"):
        x = (raw.astype(np.uint32) << 16).view(np.float32).reshape(PER, W // 16, 16)
        s = (np.abs(x).max(axis=2) / 127).astype(np.float16)
        sf = s.astype(np.float32)
        q = np.clip(np.rint(np.divide(x, sf[..., None], out=np.zeros_like(x), where=sf[..., None] > 0)), -127, 127).astype(np.int8)
        deq = q.astype(np.float32) * sf[..., None]
        err = float(np.abs(deq - x).max() / max(float(np.abs(x).max()), 1e-30))
        worst = max(worst, err)
        write_st(f"{OUT}/int8/shard_{i}.safetensors",
                 [("weight_i8", "I8", (PER, W), q.reshape(PER, W).tobytes()), ("weight_scale", "F16", (PER, W // 16), s.tobytes())])
    if i == 0 and WORKER == 0:  # the int4 sidecar holds the same rows
        q4, _ = read_st(f"{INT4}/shard_0.safetensors", "weight_i4")
        s4, _ = read_st(f"{INT4}/shard_0.safetensors", "weight_scale")
        q4 = q4.reshape(PER, W // 2)[:100000]
        s4 = s4.view(np.float16).astype(np.float32).reshape(PER, W // 16)[:100000]
        v4 = np.empty((100000, W), np.float32)
        v4[:, 0::2] = (q4 & 15).astype(np.float32) - 8
        v4[:, 1::2] = (q4 >> 4).astype(np.float32) - 8
        v4 = (v4.reshape(100000, W // 16, 16) * s4[..., None]).reshape(100000, W)
        xb = (raw.reshape(PER, W)[:100000].astype(np.uint32) << 16).view(np.float32)
        rel = np.linalg.norm(v4 - xb) / np.linalg.norm(xb)
        print(f"check vs int4 sidecar, shard 0 (100k rows): relative error {rel:.4f} (int4 rounding; ~1 means the rows do not line up)", flush=True)
    print(f"shard {i} done (int8 worst relative error so far {worst:.4f})", flush=True)
print("DONE", flush=True)
