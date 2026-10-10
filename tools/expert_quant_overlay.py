"""Fetches the tensors around the routed experts (router, shared expert, the MLP's hyper-connection) of some layers
from another quantization of the model on Hugging Face into one safetensors file for the reference's QW_REF_OVERLAY.
A quantization that folds a scale per input channel into the experts changes these tensors to match, so its experts
(tools/expert_quant_download.py) are only right together with them.

  python3 tools/expert_quant_overlay.py REPO OUT.safetensors [--model DIR] LAYER [LAYER ...]
      --model: the checkpoint in use, to print which of the tensors differ from it
"""
import json, struct, sys, urllib.request
args = sys.argv[1:]
MODEL = args.pop(args.index("--model") + 1) if "--model" in args else None
if MODEL:
    args.remove("--model")
REPO, OUT, LAYERS = f"https://huggingface.co/{args[0]}/resolve/main/", args[1], [int(a) for a in args[2:]]

def get(f, a, b):
    return urllib.request.urlopen(urllib.request.Request(REPO + f, headers={"Range": f"bytes={a}-{b - 1}"}), timeout=120).read()

def local(name):  # the tensor's bytes in the checkpoint in use
    f = MODEL + "/" + local.idx[name]
    with open(f, "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        m = json.loads(fh.read(n))[name]
        fh.seek(8 + n + m["data_offsets"][0])
        return fh.read(m["data_offsets"][1] - m["data_offsets"][0])
if MODEL:
    local.idx = json.load(open(MODEL + "/model.safetensors.index.json"))["weight_map"]

idx = json.load(urllib.request.urlopen(REPO + "model.safetensors.index.json", timeout=60))["weight_map"]
hdrs, header, blobs, at = {}, {}, [], 0
for L in LAYERS:
    p = f"model.language_model.layers.{L}."
    for name in sorted(n for n in idx if n.startswith((p + "mlp.", p + "mlp_hyper_connection.")) and ".experts." not in n):
        f = idx[name]
        if f not in hdrs:
            n = struct.unpack("<Q", get(f, 0, 8))[0]
            hdrs[f] = (json.loads(get(f, 8, 8 + n)), 8 + n)
        h, base = hdrs[f]
        m = h[name]
        buf = get(f, base + m["data_offsets"][0], base + m["data_offsets"][1])
        assert len(buf) == m["data_offsets"][1] - m["data_offsets"][0], "short read"
        header[name] = {"dtype": m["dtype"], "shape": m["shape"], "data_offsets": [at, at + len(buf)]}
        blobs.append(buf)
        at += len(buf)
        print(f"{name} {m['dtype']} {m['shape']}" + (("  same" if local(name) == buf else "  DIFFERS") if MODEL else ""), flush=True)
hj = json.dumps(header, separators=(",", ":")).encode()
hj += b" " * (-len(hj) % 8)
with open(OUT, "wb") as out:
    out.write(struct.pack("<Q", len(hj)) + hj)
    for b in blobs:
        out.write(b)
print(f"{OUT}: {len(header)} tensors, {at >> 20} MB")
