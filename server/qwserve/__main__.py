"""Entry point: python -m qwserve [options] (run from server/, or with server/ on PYTHONPATH)."""

import argparse
import os
import sys

from aiohttp import web

from . import __doc__ as DOC
from .api import Server


def main():
    ap = argparse.ArgumentParser(description=DOC.split("\n")[0])
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--model-dir", default="/mnt/llms/qwen3.8-flash-next-awq")
    ap.add_argument("--ple-dir", default="/mnt/llms/qwen3.8-flash-next-ple/ples_int4")
    ap.add_argument("--lib", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "build", "libqw_engine.so"))
    ap.add_argument("--served-model-name", default="qw/qwen3.8-flash-next")
    ap.add_argument("--slots", default=os.environ.get("QW_SLOTS", "262144,65536,32768,32768"),
                    help="KV capacity (tokens) of each sequence slot, comma-separated; one slot per concurrent "
                         "request (env QW_SLOTS)")
    ap.add_argument("--prefill-chunk", type=int, default=int(os.environ.get("QW_PREFILL_CHUNK", "8192")),
                    help="tokens per prefill pass; its buffers take VRAM in proportion (env QW_PREFILL_CHUNK)")
    ap.add_argument("--reasoning-effort", default=os.environ.get("QW_REASONING_EFFORT", "xhigh"),
                    choices=["none", "minimal", "low", "medium", "high", "xhigh", "max"],
                    help="default thinking effort when a request sets none (env QW_REASONING_EFFORT); none: "
                         "no thinking. Requests override it with reasoning_effort / chat_template_kwargs")
    ap.add_argument("--thinking-budget", type=int, default=int(os.environ.get("QW_THINKING_BUDGET", "-1")),
                    help="default cap on thinking tokens, -1 unlimited (env QW_THINKING_BUDGET); requests "
                         "override it with thinking_token_budget / thinking_budget_tokens / thinking.budget_tokens")
    ap.add_argument("--mtp", type=int, default=5,
                    help="most MTP draft tokens per step (0: no speculative decoding); each request drafts "
                         "adaptively up to it (5 measured best single-stream, docs/DESIGN.md)")
    ap.add_argument("--host-cache-gb", type=float, default=128,
                    help="pinned host RAM for conversations evicted from their slot (0: off)")
    ap.add_argument("--disk-cache-dir", default=os.path.expanduser("~/.cache/qw/prefix-cache"),
                    help="disk tier of the prefix cache, survives restarts ('' : off)")
    ap.add_argument("--disk-cache-gb", type=float, default=200)
    args = ap.parse_args()
    # the engine's session reads these when it starts
    os.environ["QW_HOST_CACHE_GB"] = str(args.host_cache_gb)
    os.environ["QW_DISK_CACHE_DIR"] = args.disk_cache_dir
    os.environ["QW_DISK_CACHE_GB"] = str(args.disk_cache_gb)
    srv = Server(args)
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.router.add_get("/", srv.dashboard)
    app.router.add_get("/dashboard", srv.dashboard)
    app.router.add_get("/metrics.json", srv.metrics_json)
    app.router.add_get("/health", srv.health)
    app.router.add_get("/v1/models", srv.models)
    app.router.add_post("/tokenize", srv.tokenize)
    app.router.add_post("/v1/chat/completions", srv.chat)
    app.router.add_post("/v1/completions", srv.completions)
    app.on_shutdown.append(srv.on_shutdown)
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    sys.exit(main())
