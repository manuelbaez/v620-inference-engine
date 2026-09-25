#!/usr/bin/env python3
"""Converts tools/vision_ref.py's .npz dumps to raw files for the C++ test:
<case>.grid.i64, <case>.pixels.f32, <case>.embeds.f32 (native endian)."""
import glob
import os
import sys

import numpy as np

d = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vision_ref"
for f in sorted(glob.glob(os.path.join(d, "*.npz"))):
    z = np.load(f)
    base = f[:-4]
    z["grid_thw"].astype(np.int64).tofile(base + ".grid.i64")
    z["pixel_values"].astype(np.float32).tofile(base + ".pixels.f32")
    z["embeds"].astype(np.float32).tofile(base + ".embeds.f32")
    print(os.path.basename(base), z["grid_thw"].tolist(), z["embeds"].shape)
