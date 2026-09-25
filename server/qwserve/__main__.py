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
    ap.add_argument("--slots", default="131072,65536,32768,32768",
                    help="context size of each sequence slot (concurrent requests); the largest is max_model_len")
    ap.add_argument("--prefill-chunk", type=int, default=8192)
    args = ap.parse_args()
    srv = Server(args)
    app = web.Application(client_max_size=256 * 1024 * 1024)
    app.router.add_get("/health", srv.health)
    app.router.add_get("/v1/models", srv.models)
    app.router.add_post("/tokenize", srv.tokenize)
    app.router.add_post("/v1/chat/completions", srv.chat)
    app.router.add_post("/v1/completions", srv.completions)
    web.run_app(app, host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    sys.exit(main())
