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
import ipaddress
import json
import math
import os
import socket
import subprocess
import tempfile
import threading
import urllib.request
from urllib.parse import urlsplit

import numpy as np
from PIL import Image

PATCH, MERGE, TEMPORAL = 16, 2, 2
FACTOR = PATCH * MERGE
MAX_FETCH_BYTES = 64 << 20
FFMPEG_SECONDS = 300  # per decoding pass
# Media by URL: the server fetches what a request names, so by default only public addresses
# (QW_MEDIA_ALLOW_PRIVATE=1 also lets it reach loopback and the LAN; QW_MEDIA_FETCH=0 allows data: URLs only)
ALLOW_REMOTE = os.environ.get("QW_MEDIA_FETCH", "1") != "0"
ALLOW_PRIVATE = os.environ.get("QW_MEDIA_ALLOW_PRIVATE", "0") == "1"


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
    def video_indices(self, total, fps):
        """The frames sampled from `total` decoded at `fps`: self.fps per second (Qwen3VLVideoProcessor)."""
        n = int(total / fps * self.fps)
        n = min(max(n, self.min_frames), self.max_frames, total)
        return np.linspace(0, total - 1, n).round().astype(int)

    def video_frames(self, frames, fps):
        """frames: uint8 [N][H][W][3] decoded at `fps`."""
        idx = self.video_indices(len(frames), fps)
        _, h0, w0, _ = frames.shape
        return self.video_sampled(iter(frames[idx]), idx, h0, w0, fps)

    def video_sampled(self, frames, idx, h0, w0, fps):
        """The Media of the frames at indices `idx` (of a video decoded at `fps`). `frames` yields them
        in that order, uint8 [h0][w0][3], and may be a generator: each is resized and normalized as it
        arrives, so only the small result is kept."""
        h, w = video_smart_resize(len(idx), h0, w0, self.vid_min, self.vid_max)
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
        """Decodes a video file with ffmpeg at its native rate, then samples. Decoding streams: one
        pass counts the frames, a second keeps the sampled ones (resized at once), so a long video
        never sits in memory (60 s of 1080p at 30 fps is 11 GB decoded; only ~2 frames a second are
        used). The video goes to a file first, not a pipe: ffmpeg can seek in it (an mp4 with its
        index at the end cannot be read from a pipe)."""
        with tempfile.TemporaryDirectory(prefix="qwvideo") as tmp:
            path = os.path.join(tmp, "video")
            with open(path, "wb") as f:
                f.write(data)
            try:
                probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                                        "stream=width,height,avg_frame_rate", "-of", "json", path],
                                       capture_output=True, timeout=60, check=True)
                s = json.loads(probe.stdout)["streams"][0]
                num, den = s["avg_frame_rate"].split("/")
                fps = float(num) / float(den) if float(den) else 24.0
                w, h = int(s["width"]), int(s["height"])
                total = _count_frames(path, w, h)
                if total == 0:
                    raise ValueError("the video has no frames")
                idx = self.video_indices(total, fps)
                return self.video_sampled(_sampled_frames(path, w, h, idx), idx, h, w, fps)
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired, KeyError, IndexError) as e:
                raise ValueError(f"cannot decode the video: {e}") from e


def _ffmpeg_raw(path):
    """ffmpeg decoding `path` to raw rgb24 frames on a pipe; its stderr goes to a file (a pipe
    nobody reads would stall it) that is read for the error message."""
    err = tempfile.TemporaryFile()
    proc = subprocess.Popen(["ffmpeg", "-v", "error", "-i", path, "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                            stdout=subprocess.PIPE, stderr=err)
    timer = threading.Timer(FFMPEG_SECONDS, proc.kill)
    timer.start()
    return proc, err, timer


def _ffmpeg_done(proc, err, timer, finished):
    """Cleans up after a pass; when it read the whole stream, ffmpeg must have succeeded."""
    timer.cancel()
    proc.stdout.close()
    if not finished:
        proc.kill()
    code = proc.wait()
    if finished and code != 0:
        err.seek(0)
        raise ValueError(f"ffmpeg failed ({code}): {err.read()[-300:].decode(errors='replace').strip()}")
    err.close()


def _read_exact(stream, buf):
    """Fills buf from the stream: the bytes read (less than len(buf) only at the end of the stream)."""
    view, got = memoryview(buf), 0
    while got < len(buf):
        n = stream.readinto(view[got:])
        if not n:
            break
        got += n
    return got


def _count_frames(path, w, h):
    """The number of frames ffmpeg decodes from the file, without keeping any."""
    size = w * h * 3
    proc, err, timer = _ffmpeg_raw(path)
    done, total_bytes = False, 0
    try:
        chunk = bytearray(1 << 22)
        while True:
            n = _read_exact(proc.stdout, chunk)
            total_bytes += n
            if n < len(chunk):
                break
        done = True
    finally:
        _ffmpeg_done(proc, err, timer, done)
    if total_bytes % size:
        raise ValueError("the decoded video does not match its reported size")
    return total_bytes // size


def _sampled_frames(path, w, h, idx):
    """Yields the frames at indices `idx` (ascending, repeats allowed) as uint8 [h][w][3] arrays;
    the others are read and dropped. Stops decoding after the last one."""
    buf = bytearray(w * h * 3)
    proc, err, timer = _ffmpeg_raw(path)
    i, j = 0, 0
    try:
        while j < len(idx):
            if _read_exact(proc.stdout, buf) < len(buf):
                raise ValueError("the video has fewer frames than it did on the first pass")
            frame = None
            while j < len(idx) and idx[j] == i:
                if frame is None:
                    frame = np.frombuffer(bytes(buf), np.uint8).reshape(h, w, 3)
                yield frame
                j += 1
            i += 1
    finally:
        _ffmpeg_done(proc, err, timer, False)  # the rest of the stream is not needed: stop ffmpeg


def _check_remote(url):
    """Raises ValueError unless the server may fetch `url`: remote fetching on, and (without
    QW_MEDIA_ALLOW_PRIVATE) every address its host resolves to public. The check is on the resolved
    addresses, so decimal or hex forms of 127.0.0.1 and names pointing inside the LAN are caught."""
    if not ALLOW_REMOTE:
        raise ValueError("fetching media by URL is off (QW_MEDIA_FETCH=0): send a data: URL")
    host = urlsplit(url).hostname
    if not host:
        raise ValueError("the media URL has no host")
    if ALLOW_PRIVATE:
        return
    try:
        infos = socket.getaddrinfo(host, None)
    except OSError as e:
        raise ValueError(f"cannot resolve {host}: {e}") from e
    for info in infos:
        ip = ipaddress.ip_address(info[4][0].split("%")[0])
        if ip.version == 6 and ip.ipv4_mapped:
            ip = ip.ipv4_mapped
        if not ip.is_global:
            raise ValueError(f"the media URL points to a private or local address ({host}); "
                             "QW_MEDIA_ALLOW_PRIVATE=1 allows it")


class _CheckedRedirects(urllib.request.HTTPRedirectHandler):
    """A redirect is a new URL to check: a public address can send the server inside."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        _check_remote(newurl)
        return super().redirect_request(req, fp, code, msg, headers, newurl)


_OPENER = urllib.request.build_opener(_CheckedRedirects)


def fetch(url):
    """Bytes of a data: URL or an http(s) URL (see _check_remote)."""
    if url.startswith("data:"):
        return base64.b64decode(url.split(",", 1)[1])
    if url.startswith(("http://", "https://")):
        _check_remote(url)
        with _OPENER.open(url, timeout=30) as r:
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
        try:
            return pre.image(Image.open(io.BytesIO(fetch(url))))
        except Image.DecompressionBombError as e:  # not an OSError or ValueError: it would be a 500
            raise ValueError(f"image too large: {e}") from e
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
