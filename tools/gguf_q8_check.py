"""Checks that the reference's QW_REF_DENSE_Q8 rounding is what a GGUF's Q8_0 tensors hold: fetches some dense tensors
from the GGUF (HTTP ranges), dequantizes them with gguf-py and compares them with the same rounding of the checkpoint's
bf16 tensors. Needs numpy and gguf.

  python3 tools/gguf_q8_check.py URL_PREFIX N_FILES [--model DIR]
"""
import json, struct, sys, numpy as np
MODEL = sys.argv.pop(sys.argv.index("--model") + 1) if "--model" in sys.argv else "/mnt/llms/qwen3.8-flash-next-awq"
if "--model" in sys.argv:
    sys.argv.remove("--model")
sys.argv = [sys.argv[0], sys.argv[1], sys.argv[2], "/tmp/gguf_q8_check"]  # expert_gguf_download's arguments, no layers
import expert_gguf_download as g  # the header reader and the range download

idx = json.load(open(MODEL + "/model.safetensors.index.json"))["weight_map"]
def ours(name):
    with open(MODEL + "/" + idx[name], "rb") as fh:
        n = struct.unpack("<Q", fh.read(8))[0]
        m = json.loads(fh.read(n))[name]
        fh.seek(8 + n + m["data_offsets"][0])
        a = np.frombuffer(fh.read(m["data_offsets"][1] - m["data_offsets"][0]), dtype=np.uint16)
    return (a.astype(np.uint32) << 16).view(np.float32).reshape(m["shape"])
def q8_0(w):  # llama.cpp quantize_row_q8_0 + dequantize
    x = w.reshape(-1, 32)
    d = np.abs(x).max(axis=1, keepdims=True) / np.float32(127)
    idv = np.where(d > 0, np.float32(1) / np.where(d > 0, d, 1), 0).astype(np.float32)
    q = np.trunc(x * idv + np.copysign(np.float32(0.5), x))  # roundf: half away from zero
    return (q * d.astype(np.float16).astype(np.float32)).reshape(w.shape)
P = "model.language_model.layers."
for gname, name in (("blk.15.attn_k.weight", P + "15.self_attn.k_proj.weight"), ("blk.15.attn_output.weight", P + "15.self_attn.o_proj.weight"),
                    ("blk.2.attn_qkv.weight", P + "2.linear_attn.in_proj_qkv.weight"), ("blk.2.ssm_out.weight", P + "2.linear_attn.out_proj.weight"),
                    ("blk.30.hc_ffn_down.weight", P + "30.mlp_hyper_connection.input_mix_weight_down.weight"),
                    ("blk.30.ffn_up_shexp.weight", P + "30.mlp.shared_expert.up_proj.weight")):
    w = ours(name)
    got, q = g.tensor(gname, w.shape)
    e = q8_0(w)
    print(f"{gname} ({q}) {list(w.shape)}: identical to the emulation: {np.array_equal(got, e)}; max |diff| {np.abs(got - e).max():.3g}; "
          f"relative error against the original {np.linalg.norm(got - w) / np.linalg.norm(w):.5f}", flush=True)
