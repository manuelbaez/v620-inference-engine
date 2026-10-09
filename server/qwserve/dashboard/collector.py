"""Builds the dashboard's data on its own thread, every REFRESH_SECONDS: the
scheduler's metrics, KV slots, per-card VRAM and power (sysfs), host memory,
prefix cache and PLE table sizes. HTTP handlers only return the latest
snapshot, so page refreshes never touch the engine or the scheduler (the
engine is not thread-safe; cache statistics come from the scheduler's own
copy)."""

import glob
import json
import os
import threading
import time

REFRESH_SECONDS = 2.0
KV_BYTES_PER_TOKEN = 4 * 20800  # all 4 cards: K/V, raw and compressed indexer keys of 13 QSA layers


def gpus():
    """VRAM and power of the AMD cards from sysfs (readable without privileges)."""
    out = []
    for dev in sorted(glob.glob("/sys/class/drm/card*/device")):
        try:
            with open(os.path.join(dev, "vendor")) as f:
                if f.read().strip() != "0x1002":
                    continue
            with open(os.path.join(dev, "mem_info_vram_used")) as f:
                used = int(f.read())
            with open(os.path.join(dev, "mem_info_vram_total")) as f:
                total = int(f.read())
            power = None
            for h in glob.glob(os.path.join(dev, "hwmon", "hwmon*", "power1_average")):
                with open(h) as f:
                    power = round(int(f.read()) / 1e6, 1)
            out.append({"card": os.path.basename(os.path.dirname(dev)), "vram_used": used, "vram_total": total,
                        "power_w": power})
        except (OSError, ValueError):
            continue
    return out


def host_memory():
    try:
        with open("/proc/meminfo") as f:
            info = dict(line.split(":", 1) for line in f)
        return {k: int(info[k].split()[0]) * 1024 for k in ("MemTotal", "MemAvailable") if k in info}
    except OSError:
        return {}


def ple_table_bytes(ple_dir):
    try:
        with open(os.path.join(ple_dir, "META.json")) as f:
            meta = json.load(f)
        per_value = {"bf16": 2.0, "group16_int8_fp16scale": 1.125}.get(meta["layout"], 0.5625)
        return int(meta["rows"] * meta["width"] * per_value)
    except (OSError, ValueError, KeyError):
        return None


class Collector:
    def __init__(self, server):
        self.server = server  # api.Server: its scheduler, engine capacities and arguments
        self.ple_bytes = ple_table_bytes(server.args.ple_dir)
        self.snapshot = {}
        self._thread = threading.Thread(target=self._run, daemon=True, name="qw-dashboard")
        self._thread.start()

    def _run(self):
        while True:
            try:
                self.snapshot = self._build()
            except Exception as ex:  # noqa: BLE001  never let the dashboard take anything down
                self.snapshot = {"error": str(ex)}
            time.sleep(REFRESH_SECONDS)

    def _build(self):
        srv, sched, args = self.server, self.server.sched, self.server.args
        snap = sched.metrics.snapshot()
        cs = sched.cache_stats or {}
        mem = host_memory()
        capacity = list(srv.engine.capacity)
        snap["memory"] = {
            "kv_gpu_allocated": sum(capacity) * KV_BYTES_PER_TOKEN,
            "kv_gpu_used": snap["live"].get("kv_used_tokens", 0) * KV_BYTES_PER_TOKEN,
            "slots": capacity,
            "slot_vram": list(getattr(srv.engine, "vram_tokens", capacity)),
            "slot_state": snap["live"].get("slots", []),
            "host_cache_used": cs.get("ram_bytes"), "host_cache_budget": int(args.host_cache_gb * 1e9),
            "disk_cache_used": cs.get("disk_bytes"), "disk_cache_budget": int(args.disk_cache_gb * 1e9),
            "cache_blocks": cs.get("blocks"), "cache_snapshots": cs.get("snapshots"),
            "ple_table": self.ple_bytes, "ple_dir": os.path.basename(args.ple_dir.rstrip("/")),
            "host_total": mem.get("MemTotal"), "host_available": mem.get("MemAvailable"),
            "gpus": gpus(),
        }
        snap["model"] = srv.model_name
        snap["refreshed"] = round(time.time(), 1)
        return snap
