"""Image and video inputs: decoding, resizing and patchifying as the model's
Qwen3-VL processor does (the PIL backend of transformers'
Qwen2VLImageProcessor and Qwen3VLVideoProcessor), and the prompt expansion of
each item's placeholder into its vision tokens.

A Media holds everything the engine needs for one item: its patches
[t*h*w][3*2*16*16] (fp32, merge-window order), its grid, a content hash (the
prefix cache's identity for its tokens) and where its tokens sit in the prompt.
"""

import base64
import hashlib
import io
import json
import math
import os
import subprocess
import urllib.request

import numpy as np
from PIL import Image

PATCH, MERGE, TEMPORAL = 16, 2, 2
FACTOR = PATCH * MERGE
MAX_FETCH_BYTES = 64 << 20


class Media:
    def __init__(self, kind, patches, grid, digest, timestamps=None):
        self.kind = kind            # "image" or "video"
        self.patches = patches      # float32 [t*h*w][1536]
        self.grid = grid            # (t, h, w) in patches
        self.hash = digest          # 8 bytes of sha256 over the patches and grid
        self.timestamps = timestamps or []
        self.spans = []             # (first token, count) in the prompt, set by expand(); videos
                                    # have one per temporal patch (timestamps sit between them)

    @property
    def tokens(self):
        t, h, w = self.grid
        return t * h * w // (MERGE * MERGE)


def smart_resize(height, width, min_pixels, max_pixels, factor=FACTOR):
    if max(height, width) / min(height, width) > 200:
        raise ValueError("image aspect ratio must be under 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def video_smart_resize(num_frames, height, width, min_pixels, max_pixels, factor=FACTOR):
    if height < factor or width < factor:
        scale = max(factor / height, factor / width)
        height, width = int(height * scale), int(width * scale)
    if max(height, width) / min(height, width) > 200:
        raise ValueError("video aspect ratio must be under 200")
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    t_bar = round(num_frames / TEMPORAL) * TEMPORAL
    if t_bar * h_bar * w_bar > max_pixels:
        beta = math.sqrt((num_frames * height * width) / max_pixels)
        h_bar = max(factor, math.floor(height / beta / factor) * factor)
        w_bar = max(factor, math.floor(width / beta / factor) * factor)
    elif t_bar * h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (num_frames * height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def _normalize(rgb, mean, std):
    """uint8 [H][W][3] -> float32 [3][H][W], rescaled to [0, 1] and normalized."""
    x = rgb.astype(np.float32) * (1.0 / 255.0)
    x = (x - np.asarray(mean, np.float32)) / np.asarray(std, np.float32)
    return np.ascontiguousarray(x.transpose(2, 0, 1))


def _patchify(frames):
    """float32 [T][3][H][W] (T a multiple of 2) -> [T/2 * H/16 * W/16][3*2*16*16], patches in
    merge-window order (temporal group, block row, block col, row in block, col in block) and
    each patch laid out (channel, frame, row, col)."""
    t, c, h, w = frames.shape
    gt, gh, gw = t // TEMPORAL, h // PATCH, w // PATCH
    p = frames.reshape(gt, TEMPORAL, c, gh // MERGE, MERGE, PATCH, gw // MERGE, MERGE, PATCH)
    p = p.transpose(0, 3, 6, 4, 7, 2, 1, 5, 8)
    return np.ascontiguousarray(p.reshape(gt * gh * gw, c * TEMPORAL * PATCH * PATCH)), (gt, gh, gw)


def _digest(patches, grid):
    h = hashlib.sha256()
    h.update(np.asarray(grid, np.int64).tobytes())
    h.update(patches.tobytes())
    return h.digest()[:8]


class VisionPreprocessor:
    def __init__(self, model_dir):
        def cfg(name):
            path = os.path.join(model_dir, name)
            return json.load(open(path)) if os.path.exists(path) else {}
        img, vid = cfg("preprocessor_config.json"), cfg("video_preprocessor_config.json")
        size = img.get("size", {})
        self.img_min = size.get("shortest_edge", 56 * 56)
        self.img_max = size.get("longest_edge", 28 * 28 * 1280)
        # The model's own budget (16.7 MP, ~16k tokens) costs ~20 s of vision tower for a 4K image;
        # by default images are scaled to at most ~1920x1088 (2,040 tokens, ~1.3 s on one card).
        self.img_max = min(self.img_max, int(os.environ.get("QW_VISION_MAX_PIXELS", 1920 * 1088)))
        self.img_mean, self.img_std = img.get("image_mean", [0.5] * 3), img.get("image_std", [0.5] * 3)
        vsize = vid.get("size", {})
        self.vid_min = vsize.get("shortest_edge", 128 * 32 * 32)
        self.vid_max = vsize.get("longest_edge", 32 * 32 * 768)
        vcap = os.environ.get("QW_VIDEO_MAX_PIXELS")
        if vcap:
            self.vid_max = min(self.vid_max, int(vcap))
        self.vid_mean, self.vid_std = vid.get("image_mean", [0.5] * 3), vid.get("image_std", [0.5] * 3)
        self.fps = float(vid.get("fps", 2.0))
        self.min_frames, self.max_frames = int(vid.get("min_frames", 4)), int(vid.get("max_frames", 768))

    # ---- images
    def image(self, pil):
        pil = pil.convert("RGB")
        h, w = smart_resize(pil.height, pil.width, self.img_min, self.img_max)
        if (w, h) != pil.size:
            pil = pil.resize((w, h), Image.BICUBIC)
        x = _normalize(np.asarray(pil), self.img_mean, self.img_std)
        frames = np.broadcast_to(x[None], (TEMPORAL,) + x.shape)
        patches, grid = _patchify(np.ascontiguousarray(frames))
        return Media("image", patches, grid, _digest(patches, grid))

    # ---- videos
    def video_frames(self, frames, fps):
        """frames: uint8 [N][H][W][3] decoded at `fps`. Samples at self.fps (Qwen3VLVideoProcessor)."""
        total = len(frames)
        n = int(total / fps * self.fps)
        n = min(max(n, self.min_frames), self.max_frames, total)
        idx = np.linspace(0, total - 1, n).round().astype(int)
        frames = frames[idx]
        _, h0, w0, _ = frames.shape
        h, w = video_smart_resize(len(frames), h0, w0, self.vid_min, self.vid_max)
        x = np.stack([_normalize(np.asarray(Image.fromarray(f).resize((w, h), Image.BICUBIC)) if (w, h) != (w0, h0)
                                 else f, self.vid_mean, self.vid_std) for f in frames])
        if pad := -len(x) % TEMPORAL:  # repeat the last frame to a whole temporal patch
            x = np.concatenate([x, np.repeat(x[-1:], pad, 0)])
            idx = np.concatenate([idx, np.repeat(idx[-1:], pad)])
        patches, grid = _patchify(x)
        # one timestamp per temporal patch: the mean of its frames' times
        times = idx / fps
        stamps = [(times[i] + times[i + TEMPORAL - 1]) / 2 for i in range(0, len(times), TEMPORAL)]
        return Media("video", patches, grid, _digest(patches, grid), stamps)

    def video_bytes(self, data):
        """Decodes a video file with ffmpeg at its native rate, then samples."""
        probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                "stream=width,height,avg_frame_rate", "-of", "json", "-"], input=data,
                               capture_output=True, timeout=60, check=True)
        s = json.loads(probe.stdout)["streams"][0]
        num, den = s["avg_frame_rate"].split("/")
        fps = float(num) / float(den) if float(den) else 24.0
        w, h = int(s["width"]), int(s["height"])
        raw = subprocess.run(["ffmpeg", "-v", "error", "-i", "-", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                             input=data, capture_output=True, timeout=300, check=True).stdout
        frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3)
        return self.video_frames(frames, fps)


def fetch(url):
    """Bytes of a data: URL or an http(s) URL."""
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    if url.startswith(("http://", "https://")):
        with urllib.request.urlopen(url, timeout=30) as r:
            data = r.read(MAX_FETCH_BYTES + 1)
        if len(data) > MAX_FETCH_BYTES:
            raise ValueError("media larger than 64 MB")
        return data
    raise ValueError("media URLs must be data: or http(s)")


def media_of_part(pre, part):
    """The Media of an OpenAI content part, or None for text parts."""
    if "video" in part or part.get("type") == "video":
        src = part.get("video") or {}
        url = src.get("url") if isinstance(src, dict) else src
        return pre.video_bytes(fetch(url))
    if "image_url" in part or "image" in part or part.get("type") == "image":
        src = part.get("image_url") or part.get("image") or {}
        url = src.get("url") if isinstance(src, dict) else src
        return pre.image(Image.open(io.BytesIO(fetch(url))))
    return None


def expand(ids, media, tok):
    """Replaces each item's placeholder token in `ids` by its vision tokens and sets each Media's
    spans. An image becomes t*h*w/4 pads. A video's pad becomes, per temporal patch,
    '<t seconds><|vision_start|>' + pads + '<|vision_end|>', inside the template's own
    start/end tokens (Qwen3VLProcessor.replace_video_token)."""
    image_pad, video_pad = tok.token_to_id("<|image_pad|>"), tok.token_to_id("<|video_pad|>")
    vstart, vend = tok.token_to_id("<|vision_start|>"), tok.token_to_id("<|vision_end|>")
    out, it, i = [], iter(media), 0
    while i < len(ids):
        t = ids[i]
        i += 1
        if t not in (image_pad, video_pad):
            out.append(t)
            continue
        m = next(it, None)
        if m is None or (t == image_pad) != (m.kind == "image"):
            raise ValueError("media do not match the prompt's placeholders")
        if m.kind == "image":
            m.spans = [(len(out), m.tokens)]
            out.extend([image_pad] * m.tokens)
            continue
        per = m.grid[1] * m.grid[2] // (MERGE * MERGE)
        for stamp in m.timestamps:
            out.extend(tok.encode(f"<{stamp:.1f} seconds>", add_special_tokens=False).ids)
            out.append(vstart)
            m.spans.append((len(out), per))
            out.extend([video_pad] * per)
            out.append(vend)
    if next(it, None) is not None:
        raise ValueError("more media than placeholders in the prompt")
    return out
