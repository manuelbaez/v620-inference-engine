#!/usr/bin/env python3
"""Reference dumps for the vision path (fp32, CPU), from HF transformers'
Qwen3VLProcessor and Qwen4ExpVisionModel with the checkpoint's model.visual.*
weights. For each test case it writes <out>/<name>.npz with:

  pixel_values [patches][1536], grid_thw [n][3], input_ids (the expanded
  prompt), embeds [tokens][2560] (the vision tower's output), and for the
  first image also block_0, block_13, block_26 hidden states [patches][1152].

Run in an environment with torch + transformers 5.16 (on llm-experiments:
HIP_VISIBLE_DEVICES=3 ~/vllm-rocm-venv/bin/python tools/vision_ref.py --out /tmp/vision_ref).
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
from PIL import Image, ImageDraw
from safetensors import safe_open
from transformers import AutoProcessor
from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel


def test_image(w, h, seed):
    """Text, lines and shapes: something with structure at several scales."""
    rng = np.random.default_rng(seed)
    img = Image.new("RGB", (w, h), tuple(int(x) for x in rng.integers(0, 255, 3)))
    d = ImageDraw.Draw(img)
    for i in range(12):
        x0, y0 = rng.integers(0, w), rng.integers(0, h)
        d.rectangle([x0, y0, x0 + rng.integers(5, w // 3 + 6), y0 + rng.integers(5, h // 3 + 6)],
                    fill=tuple(int(x) for x in rng.integers(0, 255, 3)))
    for i in range(0, h, max(h // 12, 12)):
        d.text((8, i), f"line {i}: the quick brown fox {seed}", fill=(0, 0, 0))
    return img


def test_video(frames, w, h):
    out = []
    for f in range(frames):
        img = Image.new("RGB", (w, h), (30, 30, 60))
        d = ImageDraw.Draw(img)
        x = int((w - 40) * f / max(frames - 1, 1))
        d.rectangle([x, h // 3, x + 40, h // 3 + 40], fill=(250, 200, 0))
        d.text((10, 10), f"frame {f}", fill=(255, 255, 255))
        out.append(np.asarray(img))
    return np.stack(out)


def load_visual(model_dir):
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    vcfg = Qwen4ExpVisionConfig(**cfg["vision_config"])
    vcfg._attn_implementation = "sdpa"  # the eager path runs single-threaded on the CPU
    model = Qwen4ExpVisionModel(vcfg).float().eval()
    state = {}
    for f in sorted(glob.glob(os.path.join(model_dir, "*.safetensors"))):
        with safe_open(f, "pt") as st:
            for k in st.keys():
                if k.startswith("model.visual."):
                    state[k[len("model.visual."):]] = st.get_tensor(k).float()
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert not unexpected, unexpected
    assert all("inv_freq" in m for m in missing), missing
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/mnt/llms/qwen3.8-flash-next-awq")
    ap.add_argument("--out", default="/tmp/vision_ref")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="fp32 either way; this torch build's CPU math is ~20 GFLOPS")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    torch.set_num_threads(os.cpu_count())
    proc = AutoProcessor.from_pretrained(args.model_dir)
    # the PIL backend (what the server mirrors) instead of the torchvision one: same steps, and
    # resizing by PIL makes the pixels comparable exactly
    from transformers.models.qwen2_vl.image_processing_pil_qwen2_vl import Qwen2VLImageProcessorPil
    proc.image_processor = Qwen2VLImageProcessorPil.from_pretrained(args.model_dir)
    model = load_visual(args.model_dir).to(args.device)

    cases = {
        "img_640x480": {"images": [test_image(640, 480, 1)]},
        "img_960x544": {"images": [test_image(960, 544, 2)]},
        "img_333x1000": {"images": [test_image(333, 1000, 3)]},
        "img_64x64": {"images": [test_image(64, 64, 4)]},
        "two_images": {"images": [test_image(400, 300, 5), test_image(300, 500, 6)]},
        "video_8f": {"videos": [test_video(8, 320, 240)]},
    }
    for i, (name, media) in enumerate(cases.items()):
        n_img = len(media.get("images", []))
        pads = "<|vision_start|><|image_pad|><|vision_end|>" * n_img
        if "videos" in media:
            pads += "<|vision_start|><|video_pad|><|vision_end|>"
        text = f"<|im_start|>user\n{pads}Describe this.<|im_end|>\n<|im_start|>assistant\n"
        kw = {}
        if "videos" in media:
            kw["videos"] = media["videos"]
            n_frames = len(media["videos"][0])
            kw["video_metadata"] = [{"fps": 2.0, "total_num_frames": n_frames, "frames_indices": list(range(n_frames))}]
            kw["do_sample_frames"] = False
        enc = proc(text=[text], images=media.get("images"), return_tensors="pt", **kw)
        pixels = enc["pixel_values"] if n_img else enc["pixel_values_videos"]
        grid = enc["image_grid_thw"] if n_img else enc["video_grid_thw"]
        hooks, blocks = [], {}
        if i == 0:
            for b in (0, 13, 26):
                hooks.append(model.blocks[b].register_forward_hook(
                    lambda m, a, o, b=b: blocks.__setitem__(f"block_{b}", o.detach().cpu().numpy().copy())))
        with torch.no_grad():
            out = model(pixels.float().to(args.device), grid_thw=grid.to(args.device))
        for h in hooks:
            h.remove()
        embeds = out.pooler_output.cpu().numpy()
        np.savez(os.path.join(args.out, name + ".npz"), pixel_values=pixels.float().numpy(),
                 grid_thw=grid.numpy(), input_ids=enc["input_ids"][0].numpy(), embeds=embeds, **blocks)
        act = max(float(np.abs(v).max()) for v in blocks.values()) if blocks else float("nan")
        print(f"{name}: patches {pixels.shape[0]}, grid {grid.tolist()}, tokens {embeds.shape[0]}, "
              f"ids {enc['input_ids'].shape[1]}, max |block act| {act:.1f}", flush=True)
    # the raw media, for the server-side preprocessing test
    for name, media in cases.items():
        if "images" in media:
            for j, img in enumerate(media["images"]):
                img.save(os.path.join(args.out, f"{name}_{j}.png"))
        else:
            np.save(os.path.join(args.out, f"{name}_frames.npy"), media["videos"][0])


if __name__ == "__main__":
    main()
