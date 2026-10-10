"""Downloads the routed experts of some layers from a GGUF quantization of the model (split files on Hugging Face,
read by HTTP ranges), dequantizes them with gguf-py and writes them in tools/expert_download.py's layout, so that the
fp32 reference runs with them through QW_REF_EXPERTS_DIR (docs/DESIGN.md "A GGUF quantization of the experts"). GGUF
keeps the original parametrization (no rescaled side tensors), so the experts alone are a fair swap. With --orig (the
original experts) it prints the relative error of the weights. Needs numpy and gguf (pip install gguf).

  python3 tools/expert_gguf_download.py URL_PREFIX N_FILES OUT [--orig ORIG] LAYER [LAYER ...]
      URL_PREFIX e.g. https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/resolve/main/UD-Q4_K_XL/Qwen3.8-Flash-Next-UD-Q4_K_XL
      (the files are URL_PREFIX-0000i-of-0000N.gguf)
"""
import json, os, struct, sys, time, urllib.request
import numpy as np
from gguf import GGMLQuantizationType
from gguf.quants import dequantize
args = sys.argv[1:]
ORIG = args.pop(args.index("--orig") + 1) if "--orig" in args else None
if ORIG:
    args.remove("--orig")
PREFIX, NF, OUT, LAYERS = args[0], int(args[1]), args[2], [int(a) for a in args[3:]]
H, FFN, NEXP = 2560, 640, 512
os.makedirs(OUT, exist_ok=True)

def get(url, a, b, exact=True):  # exact=False: the range may run past the end of the file (reading a header)
    for attempt in range(50):
        try:
            buf = urllib.request.urlopen(urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}"}), timeout=300).read()
            if len(buf) == b - a or (buf and not exact):
                return buf
        except Exception as ex:
            print(f"  retry {attempt}: {ex}", flush=True)
        time.sleep(5)
    raise SystemExit("download failed")

class Header:  # the tensor table of one GGUF file: name -> (dims, type, absolute offset)
    def __init__(s, url):
        s.url, s.buf, s.pos = url, b"", 0
        assert s.rd("<I") == 0x46554747, "not GGUF"
        s.rd("<I")
        nt, nk = s.rd("<Q"), s.rd("<Q")
        align = 32
        for _ in range(nk):
            k = s.st()
            v = s.val(s.rd("<I"))
            if k == "general.alignment":
                align = v
        s.tensors = {}
        for _ in range(nt):
            name = s.st()
            dims = [s.rd("<Q") for _ in range(s.rd("<I"))]
            s.tensors[name] = (dims, s.rd("<I"), s.rd("<Q"))
        data = (s.pos + align - 1) // align * align
        s.tensors = {n: (d, t, data + o) for n, (d, t, o) in s.tensors.items()}
    def need(s, n):
        while s.pos + n > len(s.buf):
            s.buf += get(s.url, len(s.buf), len(s.buf) + max(1 << 22, n), exact=False)
    def rd(s, fmt):
        n = struct.calcsize(fmt); s.need(n)
        v = struct.unpack_from(fmt, s.buf, s.pos)[0]; s.pos += n
        return v
    def st(s):
        n = s.rd("<Q"); s.need(n)
        v = s.buf[s.pos:s.pos + n]; s.pos += n
        return v.decode("utf-8", "replace")
    def val(s, t):
        if t == 8:
            return s.st()
        if t == 9:
            et, n = s.rd("<I"), s.rd("<Q")
            for _ in range(n):
                s.val(et)
            return None
        return s.rd({0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}[t])

where = {}
for i in range(1, NF + 1):
    url = f"{PREFIX}-{i:05d}-of-{NF:05d}.gguf"
    for n, v in Header(url).tensors.items():
        where[n] = (url, *v)

def tensor(name, shape):  # dequantized fp32 in the checkpoint's (PyTorch) order; GGUF lists the dimensions reversed
    url, dims, t, off = where[name]
    assert dims[::-1] == list(shape), (name, dims)
    qt = GGMLQuantizationType(t)
    from gguf.constants import GGML_QUANT_SIZES
    block, size = GGML_QUANT_SIZES[qt]
    nbytes = int(np.prod(shape)) // block * size
    t0 = time.time()
    raw = np.frombuffer(get(url, off, off + nbytes), dtype=np.uint8)
    print(f"  {name} {qt.name} {nbytes >> 20} MB, {nbytes / 1e6 / (time.time() - t0):.0f} MB/s", flush=True)
    return dequantize(raw.reshape(-1, shape[-1] // block * size), qt).reshape(shape).astype(np.float32, copy=False), qt.name

def bf16(a):
    return (np.asarray(a).astype(np.uint32) << 16).view(np.float32)
def to_bf16(x):
    u = np.ascontiguousarray(x).view(np.uint32)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
rel = lambda a, b: float(np.linalg.norm(a - b) / np.linalg.norm(b))

metas = {}
for L in LAYERS:
    gu = np.memmap(f"{OUT}/L{L}.gate_up_proj.bf16", dtype=np.uint16, mode="w+", shape=(NEXP, 2 * FFN, H))
    dn = np.memmap(f"{OUT}/L{L}.down_proj.bf16", dtype=np.uint16, mode="w+", shape=(NEXP, H, FFN))
    for part, shape, dst in (("gate", (NEXP, FFN, H), lambda w: gu.__setitem__((slice(None), slice(0, FFN)), w)),
                             ("up", (NEXP, FFN, H), lambda w: gu.__setitem__((slice(None), slice(FFN, None)), w)),
                             ("down", (NEXP, H, FFN), lambda w: dn.__setitem__(slice(None), w))):
        w, q = tensor(f"blk.{L}.ffn_{part}_exps.weight", shape)
        dst(to_bf16(w))
        if ORIG:
            o = np.memmap(f"{ORIG}/L{L}.{'down_proj' if part == 'down' else 'gate_up_proj'}.bf16", dtype=np.uint16, mode="r",
                          shape=(NEXP, H, FFN) if part == "down" else (NEXP, 2 * FFN, H))
            o = o if part == "down" else (o[:, :FFN] if part == "gate" else o[:, FFN:])
            e = [rel(w[i], bf16(o[i])) for i in range(0, NEXP, 16)]
            print(f"  L{L} {part} ({q}): relative error against the original {np.mean(e):.4f} ({min(e):.4f}-{max(e):.4f})", flush=True)
        del w
    gu.flush(); dn.flush()
    metas[f"L{L}.gate_up_proj.bf16"], metas[f"L{L}.down_proj.bf16"] = list(gu.shape), list(dn.shape)
json.dump(metas, open(OUT + "/META.json", "w"))
print("done", flush=True)
