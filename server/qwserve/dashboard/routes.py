"""HTTP routes of the dashboard: the page and its assets, and the JSON it polls."""

import os

from aiohttp import web

STATIC = os.path.join(os.path.dirname(__file__), "static")


def add_routes(app, collector):
    async def index(_):
        return web.FileResponse(os.path.join(STATIC, "index.html"))

    async def metrics(_):
        return web.json_response(collector.snapshot)

    app.router.add_get("/", index)
    app.router.add_get("/dashboard", index)
    app.router.add_get("/metrics.json", metrics)
    app.router.add_static("/static/", STATIC)
