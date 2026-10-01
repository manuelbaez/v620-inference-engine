#!/usr/bin/env python3
"""Media by URL (no GPU, no checkpoint, no network): the server fetches what a request names, so it
refuses loopback, LAN and link-local addresses (in any spelling) unless allowed, checks every
redirect, can be limited to data: URLs, and an oversize image is a ValueError (a 400) rather than
PIL's DecompressionBombError (a 500).

  python3 server/tests/test_fetch.py
"""

import base64
import io
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qwserve import vision  # noqa: E402


def png(w=40, h=30):
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (10, 120, 200)).save(buf, "PNG")
    return buf.getvalue()


def main():
    fails = 0

    def check(what, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{what}: {'ok' if ok else 'FAIL'} {detail}")

    hits = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            hits.append(self.path)
            body = png()
            self.send_response(200)
            self.send_header("Content-Type", "image/png")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *a):
            pass

    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    local = f"http://127.0.0.1:{srv.server_port}/img.png"

    def refused(url):
        try:
            vision.fetch(url)
            return False
        except ValueError:
            return True

    # private and local addresses are refused, whatever the spelling, and never contacted
    for url in (local, f"http://localhost:{srv.server_port}/x", f"http://[::1]:{srv.server_port}/x",
                "http://169.254.169.254/latest/meta-data/", "http://10.1.2.3/x", "http://192.168.0.5/x",
                "http://172.16.0.9/x", "http://100.64.0.1/x", "http://0.0.0.0/x",
                f"http://2130706433:{srv.server_port}/x", f"http://0x7f.0.0.1:{srv.server_port}/x",
                f"http://[::ffff:127.0.0.1]:{srv.server_port}/x", "http:///nohost"):
        check(f"refused: {url[:48]}", refused(url))
    check("the local server was never contacted", hits == [], f"({hits})")

    # public addresses pass the check (no connection is made by _check_remote)
    for url in ("http://93.184.216.34/x.png", "https://1.1.1.1/x.png", "http://[2606:4700:4700::1111]/x"):
        try:
            vision._check_remote(url)
            check(f"allowed: {url}", True)
        except ValueError as e:
            check(f"allowed: {url}", False, str(e))

    # explicitly allowed, it is fetched
    vision.ALLOW_PRIVATE = True
    try:
        data = vision.fetch(local)
        check("QW_MEDIA_ALLOW_PRIVATE lets it fetch the local server", data == png() and hits == ["/img.png"])
        # and the image goes through media_of_part
        pre = vision.VisionPreprocessor("/nonexistent")
        m = vision.media_of_part(pre, {"type": "image_url", "image_url": {"url": local}})
        check("an image by URL becomes a Media", m is not None and m.kind == "image")
    finally:
        vision.ALLOW_PRIVATE = False

    # redirects are checked: a public address cannot send the server inside
    handler = vision._CheckedRedirects()
    req = vision.urllib.request.Request("http://93.184.216.34/x")
    try:
        handler.redirect_request(req, None, 302, "Found", {}, "http://127.0.0.1:1/secret")
        check("a redirect to loopback is refused", False)
    except ValueError:
        check("a redirect to loopback is refused", True)
    try:
        handler.redirect_request(req, None, 302, "Found", {}, "http://93.184.216.35/next")
        check("a redirect to a public address is followed", True)
    except ValueError as e:
        check("a redirect to a public address is followed", False, str(e))

    # remote fetching off: data: URLs only
    vision.ALLOW_REMOTE = False
    try:
        data_url = "data:image/png;base64," + base64.b64encode(png()).decode()
        check("QW_MEDIA_FETCH=0 refuses http URLs", refused("http://93.184.216.34/x"))
        check("and still takes data: URLs", vision.fetch(data_url) == png())
    finally:
        vision.ALLOW_REMOTE = True
    check("other schemes are refused", refused("file:///etc/passwd") and refused("ftp://93.184.216.34/x"))

    # an image over PIL's pixel limit is a ValueError (the API answers 400), not a 500
    pre = vision.VisionPreprocessor("/nonexistent")
    part = {"type": "image", "image": {"url": "data:image/png;base64," + base64.b64encode(png(100, 100)).decode()}}
    old_limit = Image.MAX_IMAGE_PIXELS
    Image.MAX_IMAGE_PIXELS = 1000  # 100 x 100 is over twice that
    try:
        try:
            vision.media_of_part(pre, part)
            check("an oversize image is rejected", False)
        except ValueError as e:
            check("an oversize image is a ValueError", "too large" in str(e), f"({str(e)[:70]!r})")
        except Exception as e:  # noqa: BLE001
            check("an oversize image is a ValueError", False, repr(e))
    finally:
        Image.MAX_IMAGE_PIXELS = old_limit
    check("PIL's own error is not a ValueError (why the mapping is needed)",
          not issubclass(Image.DecompressionBombError, (ValueError, OSError)))

    srv.shutdown()
    print("FAILED" if fails else "all ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
