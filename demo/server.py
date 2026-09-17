"""Local dev server: the Vercel function's handler plus the static files
Vercel serves itself. Every engine here — the deployed one has no spaCy model.
"""

import os
import sys
from http.server import ThreadingHTTPServer
from pathlib import Path

HERE = Path(__file__).resolve().parent
os.environ.setdefault("TAXONOMY_ALL_ENGINES", "1")
sys.path.insert(0, str(HERE.parent / "api"))

from resolve import ENGINES, WARNING, handler  # noqa: E402 - after the env default

PORT = int(os.environ.get("PORT", "8000"))
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript",
    ".json": "application/json",
}


class Demo(handler):
    """POST is the deployed function's; only the static files are local."""

    def do_GET(self) -> None:
        name = self.path.lstrip("/").split("?")[0] or "index.html"
        served = (HERE / name).resolve()
        if served.parent != HERE or not served.is_file():
            self.send_error(404)
            return
        self._send(TYPES.get(served.suffix, "text/plain"), served.read_bytes())


if __name__ == "__main__":
    if WARNING:
        print(f"! {WARNING}")
    print(f"engines: {', '.join(ENGINES)}")
    print(f"query-taxonomy demo -> http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Demo).serve_forever()
