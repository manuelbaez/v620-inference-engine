"""Server-side image/video preprocessing against HF's Qwen3VLProcessor (PIL
backend), from the dumps of tools/vision_ref.py: identical grids and expanded
token ids, and pixel values within 1e-5 (videos: 2e-2, HF resizes video frames
with torchvision's bicubic, the server with PIL's).

  python server/tests/test_vision_preprocess.py /tmp/vision_ref [model_dir]
"""
import os
import sys

import numpy as np
from PIL import Image
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from qwserve import vision  # noqa: E402

CASES = {
    "img_640x480": ["img_640x480_0.png"],
    "img_960x544": ["img_960x544_0.png"],
    "img_333x1000": ["img_333x1000_0.png"],
    "img_64x64": ["img_64x64_0.png"],
    "two_images": ["two_images_0.png", "two_images_1.png"],
    "video_8f": "video_8f_frames.npy",
}


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vision_ref"
    model_dir = sys.argv[2] if len(sys.argv) > 2 else "/mnt/llms/qwen3.8-flash-next-awq"
    tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
    pre = vision.VisionPreprocessor(model_dir)
    fails = 0
    for name, src in CASES.items():
        ref = np.load(os.path.join(d, name + ".npz"))
        if isinstance(src, str):
            frames = np.load(os.path.join(d, src))
            media = [pre.video_frames(frames, 2.0)]
            pads = "<|vision_start|><|video_pad|><|vision_end|>"
        else:
            media = [pre.image(Image.open(os.path.join(d, f))) for f in src]
            pads = "<|vision_start|><|image_pad|><|vision_end|>" * len(src)
        text = f"<|im_start|>user\n{pads}Describe this.<|im_end|>\n<|im_start|>assistant\n"
        ids = vision.expand(tok.encode(text, add_special_tokens=False).ids, media, tok)
        pixels = np.concatenate([m.patches for m in media])
        grids = [list(m.grid) for m in media]
        err = float(np.abs(pixels - ref["pixel_values"]).max()) if pixels.shape == ref["pixel_values"].shape else -1
        ok_ids = ids == ref["input_ids"].tolist()
        tol = 2e-2 if isinstance(src, str) else 1e-5
        ok = grids == ref["grid_thw"].tolist() and ok_ids and 0 <= err < tol
        fails += not ok
        spans = [m.spans for m in media]
        print(f"{name}: grid {grids}, {len(ids)} ids {'same' if ok_ids else 'DIFFERENT'}, "
              f"max |dpixel| {err:.2e}, spans {spans} {'' if ok else 'FAIL'}")
        if not ok_ids:
            ref_ids = ref["input_ids"].tolist()
            k = next((i for i, (a, b) in enumerate(zip(ids, ref_ids)) if a != b), min(len(ids), len(ref_ids)))
            print("  first difference at", k, "ours", tok.decode(ids[max(0, k - 3):k + 6], skip_special_tokens=False),
                  "| ref", tok.decode(ref_ids[max(0, k - 3):k + 6], skip_special_tokens=False))
    print("FAILED" if fails else "all ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
