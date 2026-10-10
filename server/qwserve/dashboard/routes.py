"""HTTP routes of the dashboard: the page and its assets, and the JSON it polls."""

import os

from aiohttp import web

STATIC = os.path.join(os.path.dirname(__file__), "static")


def add_routes(app, collector):
    async def index(_):
        return web.FileResponse(os.path.join(STATIC, "index.html"))

    async def metrics(_):
        return web.json_response(collector.snapshot)

    async def revalidate(request, response):
        # The page and its scripts have no versioned names: without this a browser keeps old scripts for hours
        # after a deploy (a new index.html with the previous dashboard.js). no-cache = ask each time, 304 if same.
        if request.path in ("/", "/dashboard") or request.path.startswith("/static/"):
            response.headers["Cache-Control"] = "no-cache"

    app.on_response_prepare.append(revalidate)
    app.router.add_get("/", index)
    app.router.add_get("/dashboard", index)
    app.router.add_get("/metrics.json", metrics)
    app.router.add_static("/static/", STATIC)
