#!/usr/bin/env python3
"""GPTQ for the engine's int8 decode matrices (QW_INT8_DENSE=1): each dense
matrix (dumped by the engine with QW_INT8_DUMP) is quantized to int8 with one
fp32 scale per group of 32 along K, compensating each column's rounding error
on the columns not yet quantized through the inverse Hessian of the matrix's
input (tools/qw_calibrate). Writes <out>/r<rank>/<name>.q8 (int64 N, K, group;
int8 [N][K]; float32 scales [N][K/group]) for the engine's QW_INT8_DIR.

  python3 tools/gptq_int8.py --weights /tmp/w16 --hessians /tmp/hess --out /tmp/q8
"""
import argparse
import glob
import os

import numpy as np
import torch

GROUP = 32
BLOCK = 128


def load_f16(path):
    with open(path, "rb") as f:
        n, k = np.frombuffer(f.read(16), dtype=np.int64)
        w = np.frombuffer(f.read(), dtype=np.float16).reshape(n, k)
    return torch.from_numpy(w.astype(np.float32))


def load_h(path):
    with open(path, "rb") as f:
        k = int(np.frombuffer(f.read(8), dtype=np.int64)[0])
        rows = float(np.frombuffer(f.read(8), dtype=np.float64)[0])
        h = np.frombuffer(f.read(), dtype=np.float32).reshape(k, k)
    return torch.from_numpy(h.copy()), rows


def rtn(w):
    """Round-to-nearest with group scales (what the engine does without a GPTQ file)."""
    n, k = w.shape
    g = w.reshape(n, k // GROUP, GROUP)
    s = (g.abs().amax(dim=2, keepdim=True) / 127).clamp(min=1e-12)
    return (torch.clamp(torch.round(g / s), -127, 127) * s).reshape(n, k)


def gptq(w, h, damp=0.01):
    """GPTQ with group scales; returns (int8 values, scales [N][K/GROUP], dequantized)."""
    w = w.clone()
    n, k = w.shape
    h = h.clone()
    dead = torch.diag(h) == 0
    h[dead, dead] = 1
    w[:, dead] = 0
    h += torch.eye(k, device=h.device) * (damp * torch.mean(torch.diag(h)))
    hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(h)), upper=True)
    q = torch.zeros_like(w)
    scales = torch.zeros(n, k // GROUP, device=w.device)
    for i1 in range(0, k, BLOCK):
        i2 = min(i1 + BLOCK, k)
        w1 = w[:, i1:i2].clone()
        q1 = torch.zeros_like(w1)
        err1 = torch.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % GROUP == 0:  # the group's scale from its (error-updated) weights
                g_end = min(i + GROUP, i2 - i1)
                s = (w1[:, i:g_end].abs().amax(dim=1) / 127).clamp(min=1e-12)
                scales[:, col // GROUP] = s
            s = scales[:, col // GROUP]
            wc = w1[:, i]
            qc = torch.clamp(torch.round(wc / s), -127, 127)
            q1[:, i] = qc
            e = (wc - qc * s) / hinv1[i, i]
            w1[:, i:] -= e.unsqueeze(1) @ hinv1[i, i:].unsqueeze(0)
            err1[:, i] = e
        q[:, i1:i2] = q1
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    deq = (q.reshape(n, k // GROUP, GROUP) * scales.unsqueeze(2)).reshape(n, k)
    return q.to(torch.int8), scales, deq


def out_err(w, wq, h):
    """Relative output error on the calibration inputs: tr(dW H dW^T) / tr(W H W^T)."""
    d = w - wq
    return float(torch.sum((d @ h) * d) / torch.sum((w @ h) * w))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--hessians", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--group", type=int, default=32, help="scale group along K; the engine's QW_I8_GROUP")
    ap.add_argument("--rank", default=None, help="only this rank (r0..r3), e.g. one process per GPU")
    args = ap.parse_args()
    dev = args.device
    global GROUP
    GROUP = args.group
    for rdir in sorted(glob.glob(os.path.join(args.weights, "r*"))):
        r = os.path.basename(rdir)
        if args.rank and r != args.rank:
            continue
        os.makedirs(os.path.join(args.out, r), exist_ok=True)
        for wf in sorted(glob.glob(os.path.join(rdir, "*.f16"))):
            name = os.path.basename(wf)[:-4]
            hdir = os.path.join(args.hessians, r)
            w = load_f16(wf).to(dev)
            n, k = w.shape
            if name.endswith("_hc.down") or name == "final_hc.down":  # rows grouped by stream, one Hessian each
                parts = [(s, os.path.join(hdir, f"{name}.s{s}.h")) for s in range(4)]
                per = n // 4
            else:
                parts = [(0, os.path.join(hdir, name + ".h"))]
                per = n
            if not all(os.path.exists(p) for _, p in parts):
                print(f"{r} {name}: no Hessian, skipped (the engine rounds it to nearest)")
                continue
            qs, ss, e_rtn, e_gptq = [], [], [], []
            for s, hp in parts:
                h, rows = load_h(hp)
                h = (h / max(rows, 1)).to(dev)
                ws = w[s * per:(s + 1) * per]
                q, sc, deq = gptq(ws, h)
                qs.append(q)
                ss.append(sc)
                e_rtn.append(out_err(ws, rtn(ws), h))
                e_gptq.append(out_err(ws, deq, h))
            q = torch.cat(qs).cpu().numpy()
            sc = torch.cat(ss).cpu().numpy().astype(np.float32)
            with open(os.path.join(args.out, r, name + ".q8"), "wb") as f:
                f.write(np.array([n, k, GROUP], dtype=np.int64).tobytes())
                f.write(q.tobytes())
                f.write(sc.tobytes())
            print(f"{r} {name}: [{n}x{k}] output error rtn {np.mean(e_rtn):.2e} -> gptq {np.mean(e_gptq):.2e}",
                  flush=True)


if __name__ == "__main__":
    main()
