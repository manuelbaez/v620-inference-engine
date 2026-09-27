"""The engine's web dashboard (served at /, which llama-swap links as the
model's upstream page): serving metrics, their collection on a background
thread, and the static page."""

from .collector import Collector
from .metrics import Metrics
from .routes import add_routes


def setup(app, server):
    """Starts the collector for `server` and adds the dashboard's routes to `app`."""
    add_routes(app, Collector(server))


__all__ = ["Metrics", "setup"]
