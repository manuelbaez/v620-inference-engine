#!/usr/bin/env python3
"""Image QA cases with known answers for tools/qw_vision_pos_eval (the
multimodal RoPE experiment). Writes <out>/cases.txt plus one .f32 patch file per
case. Each case line holds: name, answer start index, grid t h w, image token
start, patch file, then the token ids, then (per token) the vision-aware HF
positions t,h,w (3D M-RoPE, HF Qwen4Exp get_rope_index).

  ~/qwenv/bin/python tools/vision_pos_cases.py --out /tmp/vpos
"""
import argparse
import os
import random
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tokenizers import Tokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "server"))
from qwserve import vision  # noqa: E402

WORDS = "apple river stone cloud maple ember violet orbit canyon pixel harbor falcon lantern meadow copper".split()
COLORS = {"red": (220, 30, 30), "green": (30, 160, 60), "blue": (40, 70, 220), "yellow": (240, 210, 20),
          "purple": (140, 50, 170), "orange": (245, 140, 20), "black": (10, 10, 10), "white": (250, 250, 250)}
REGIONS = ["top-left", "top", "top-right", "left", "center", "right", "bottom-left", "bottom", "bottom-right"]


def font(size):
    try:
        return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def case_ocr(rng):
    lines = [" ".join(rng.sample(WORDS, 3)) for _ in range(4)]
    img = Image.new("RGB", (800, 400), "white")
    d = ImageDraw.Draw(img)
    for i, l in enumerate(lines):
        d.text((30, 30 + i * 90), l, fill="black", font=font(40))
    k = rng.randrange(4)
    return img, f"What is written on line {k + 1} of the image? Answer with the text only.", lines[k]


def case_grid(rng):
    names = list(COLORS)
    cells = [[rng.choice(names[:6]) for _ in range(3)] for _ in range(3)]
    img = Image.new("RGB", (600, 600), "white")
    d = ImageDraw.Draw(img)
    for r in range(3):
        for c in range(3):
            d.rectangle([c * 200 + 10, r * 200 + 10, c * 200 + 190, r * 200 + 190], fill=COLORS[cells[r][c]])
    r, c = rng.randrange(3), rng.randrange(3)
    where = {(0, 0): "top-left", (0, 2): "top-right", (2, 0): "bottom-left", (2, 2): "bottom-right",
             (1, 1): "center", (0, 1): "top-middle", (2, 1): "bottom-middle", (1, 0): "middle-left",
             (1, 2): "middle-right"}[(r, c)]
    return img, f"What color is the {where} square? Answer with one word.", cells[r][c].capitalize()


def case_table(rng):
    names = ["Alice", "Bob", "Carol", "Dave"]
    cols = ["Q1", "Q2", "Q3", "Q4"]
    vals = [[rng.randrange(10, 99) for _ in cols] for _ in names]
    img = Image.new("RGB", (700, 400), "white")
    d = ImageDraw.Draw(img)
    f = font(32)
    for j, c in enumerate(cols):
        d.text((180 + j * 120, 20), c, fill="black", font=f)
    for i, n in enumerate(names):
        d.text((20, 90 + i * 70), n, fill="black", font=f)
        for j in range(4):
            d.text((180 + j * 120, 90 + i * 70), str(vals[i][j]), fill="black", font=f)
    for i in range(6):
        d.line([(10, 70 + i * 70), (690, 70 + i * 70)], fill=(160, 160, 160))
    i, j = rng.randrange(4), rng.randrange(4)
    return img, f"In the table, what is the value for {names[i]} in column {cols[j]}? Answer with the number only.", \
        str(vals[i][j])


def case_where(rng):
    k = rng.randrange(9)
    img = Image.new("RGB", (900, 900), "white")
    d = ImageDraw.Draw(img)
    r, c = divmod(k, 3)
    cx, cy = c * 300 + 150, r * 300 + 150
    d.ellipse([cx - 80, cy - 80, cx + 80, cy + 80], fill=(200, 30, 30))
    return img, ("Where is the red circle in the image? Answer with one of: top-left, top, top-right, left, center, "
                 "right, bottom-left, bottom, bottom-right."), REGIONS[k]


def case_count(rng):
    n = rng.randrange(2, 8)
    img = Image.new("RGB", (800, 500), "white")
    d = ImageDraw.Draw(img)
    spots = rng.sample([(x, y) for x in range(8) for y in range(5)], n)
    for x, y in spots:
        d.rectangle([x * 100 + 20, y * 100 + 20, x * 100 + 80, y * 100 + 80], fill=(30, 60, 200))
    return img, "How many blue squares are in the image? Answer with a digit only.", str(n)


def hf_positions(ids, image_start, n_img, grid):
    """HF get_rope_index for one image: text positions count on; the image's tokens get
    (start, start + row, start + col) on the merged grid; the text after it resumes at
    start + max(rows, cols)."""
    _, gh, gw = grid
    rows, cols = gh // 2, gw // 2
    pos = []
    p = 0
    for i in range(len(ids)):
        if i < image_start:
            pos.append((i, i, i))
        elif i < image_start + n_img:
            j = i - image_start
            st = image_start
            pos.append((st, st + j // cols, st + j % cols))
        else:
            q = image_start + max(rows, cols) + (i - image_start - n_img)
            pos.append((q, q, q))
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="/mnt/llms/qwen3.8-flash-next-awq")
    ap.add_argument("--out", default="/tmp/vpos")
    ap.add_argument("--per-task", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    tok = Tokenizer.from_file(os.path.join(args.model_dir, "tokenizer.json"))
    pre = vision.VisionPreprocessor(args.model_dir)
    rng = random.Random(7)
    lines = []
    for task in (case_ocr, case_grid, case_table, case_where, case_count):
        for k in range(args.per_task):
            img, q, ans = task(rng)
            m = pre.image(img)
            text = (f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{q}<|im_end|>\n"
                    f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
            ids = vision.expand(tok.encode(text, add_special_tokens=False).ids, [m], tok)
            # the model answers words after a space (" Yellow", " apple river violet"), digits and
            # region names without one
            lead = " " if task in (case_ocr, case_grid) else ""
            ans_ids = tok.encode(lead + ans, add_special_tokens=False).ids + [tok.token_to_id("<|im_end|>")]
            start, n = m.spans[0]
            full = ids + ans_ids
            pos = hf_positions(full, start, n, m.grid)
            name = f"{task.__name__[5:]}_{k}"
            pf = os.path.join(args.out, name + ".f32")
            m.patches.astype(np.float32).tofile(pf)
            lines.append(" ".join([name, str(len(ids)), *map(str, m.grid), str(start), pf]) + "\n" +
                         ",".join(map(str, full)) + "\n" + ",".join(f"{a},{b},{c}" for a, b, c in pos) + "\n")
    with open(os.path.join(args.out, "cases.txt"), "w") as f:
        f.writelines(lines)
    print(len(lines), "cases in", args.out)


if __name__ == "__main__":
    main()
