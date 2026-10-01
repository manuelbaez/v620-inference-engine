#!/usr/bin/env python3
"""Video decoding (no GPU, no checkpoint; needs ffmpeg and ffprobe on the path): the streaming
decoder returns exactly what the previous one did (which held every decoded frame in memory: 60 s
of 1080p at 30 fps is 11 GB), reads an mp4 whose index is at the end, maps ffmpeg failures to
ValueError (a 400), and keeps its memory flat.

  python3 server/tests/test_video.py
"""

import json
import os
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qwserve import vision  # noqa: E402


def legacy_video_frames(pre, frames, fps):
    """The previous VisionPreprocessor.video_frames."""
    total = len(frames)
    n = int(total / fps * pre.fps)
    n = min(max(n, pre.min_frames), pre.max_frames, total)
    idx = np.linspace(0, total - 1, n).round().astype(int)
    frames = frames[idx]
    _, h0, w0, _ = frames.shape
    h, w = vision.video_smart_resize(len(frames), h0, w0, pre.vid_min, pre.vid_max)
    x = np.stack([vision._normalize(np.asarray(vision.Image.fromarray(f).resize((w, h), vision.Image.BICUBIC))
                                    if (w, h) != (w0, h0) else f, pre.vid_mean, pre.vid_std) for f in frames])
    if pad := -len(x) % vision.TEMPORAL:
        x = np.concatenate([x, np.repeat(x[-1:], pad, 0)])
        idx = np.concatenate([idx, np.repeat(idx[-1:], pad)])
    patches, grid = vision._patchify(x)
    times = idx / fps
    stamps = [(times[i] + times[i + vision.TEMPORAL - 1]) / 2 for i in range(0, len(times), vision.TEMPORAL)]
    return vision.Media("video", patches, grid, vision._digest(patches, grid), stamps)


def legacy_video_bytes(pre, data):
    """The previous VisionPreprocessor.video_bytes: ffmpeg on a pipe, every frame in memory."""
    probe = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                            "stream=width,height,avg_frame_rate", "-of", "json", "-"], input=data,
                           capture_output=True, timeout=60, check=True)
    s = json.loads(probe.stdout)["streams"][0]
    num, den = s["avg_frame_rate"].split("/")
    fps = float(num) / float(den) if float(den) else 24.0
    w, h = int(s["width"]), int(s["height"])
    raw = subprocess.run(["ffmpeg", "-v", "error", "-i", "-", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"],
                         input=data, capture_output=True, timeout=300, check=True).stdout
    return legacy_video_frames(pre, np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3), fps)


def make_video(path, size, seconds, rate, faststart=True):
    cmd = ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", f"testsrc2=size={size}:rate={rate}",
           "-t", str(seconds), "-pix_fmt", "yuv420p"]
    subprocess.run(cmd + (["-movflags", "+faststart"] if faststart else []) + [path], check=True)
    with open(path, "rb") as f:
        return f.read()


def peak_rss_mb(code):
    """Peak RSS of a fresh interpreter running `code`: its VmHWM (ru_maxrss would include the parent's
    memory at fork time, which a child inherits across exec)."""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                         env={**os.environ, "PYTHONPATH": os.path.join(os.path.dirname(__file__), "..")})
    return float(out.stdout.strip().splitlines()[-1]) / 1024


def main():
    if not (shutil.which("ffmpeg") and shutil.which("ffprobe")):
        print("skipped: ffmpeg and ffprobe are not installed")
        return 0
    fails = 0

    def check(what, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{what}: {'ok' if ok else 'FAIL'} {detail}")

    pre = vision.VisionPreprocessor("/nonexistent")  # the processor's defaults
    with tempfile.TemporaryDirectory() as tmp:
        for size, seconds, rate in (("320x240", 3, 24), ("640x360", 10, 30), ("1280x720", 2, 10), ("322x242", 2.5, 25),
                                    ("160x120", 0.2, 25)):
            data = make_video(os.path.join(tmp, "v.mp4"), size, seconds, rate)
            old, new = legacy_video_bytes(pre, data), pre.video_bytes(data)
            same = (np.array_equal(old.patches, new.patches) and old.grid == new.grid and old.hash == new.hash
                    and old.timestamps == new.timestamps and old.tokens == new.tokens)
            check(f"{size} {seconds} s at {rate} fps: identical to the previous decoder", same,
                  f"(grid {new.grid}, {len(new.timestamps)} timestamps)")

        # an mp4 with its index at the end cannot be read from a pipe (the previous code failed on it)
        data = make_video(os.path.join(tmp, "tail.mp4"), "320x240", 2, 24, faststart=False)
        try:
            legacy_video_bytes(pre, data)
            legacy_ok = True
        except subprocess.CalledProcessError:
            legacy_ok = False
        head = make_video(os.path.join(tmp, "head.mp4"), "320x240", 2, 24, faststart=True)
        got = pre.video_bytes(data)
        check("an mp4 with the index at the end decodes (the pipe could not)", np.array_equal(got.patches,
              pre.video_bytes(head).patches), f"(the previous code {'could' if legacy_ok else 'failed on it'})")

        # failures are ValueError: the API answers 400, not 500
        for what, blob in (("garbage", os.urandom(5000)), ("empty", b"")):
            try:
                pre.video_bytes(blob)
                check(f"{what} is rejected", False)
            except ValueError as e:
                check(f"{what} is rejected with a ValueError", True, f"({str(e)[:60]!r})")

        # memory: 720p, 10 s at 30 fps is 830 MB decoded; the previous decoder holds all of it
        path = os.path.join(tmp, "big.mp4")
        make_video(path, "1280x720", 10, 30)
        harness = ("import resource, sys\nsys.path.insert(0, %r)\nfrom qwserve import vision\n"
                   "import test_video\npre = vision.VisionPreprocessor('/nonexistent')\ndata = open(%r, 'rb').read()\n"
                   "{call}\nprint(next(l for l in open('/proc/self/status') if l.startswith('VmHWM')).split()[1])\n"
                   % (os.path.dirname(os.path.abspath(__file__)), path))
        new_mb = peak_rss_mb(harness.format(call="pre.video_bytes(data)"))
        old_mb = peak_rss_mb(harness.format(call="test_video.legacy_video_bytes(pre, data)"))
        print(f"peak RSS decoding 300 frames of 720p: streaming {new_mb:.0f} MB, previous {old_mb:.0f} MB")
        check("the streaming decoder's memory stays flat (under a fifth of the previous one's)", new_mb * 5 < old_mb)

    print("FAILED" if fails else "all ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
