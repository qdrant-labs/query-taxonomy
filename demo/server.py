"""Live taxonomy annotator: type a query, watch every bank claim its spans.

Stdlib HTTP over the real `FeatureExtractor` — no build step, no new
dependency, and the demo can never drift from the banks it shows.
"""

import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from query_taxonomy.features import FeatureExtractor

HERE = Path(__file__).resolve().parent
PORT = int(os.environ.get("PORT", "8000"))
TYPES = {
    ".html": "text/html; charset=utf-8",
    ".js": "text/javascript",
    ".json": "application/json",
}


def _extractor() -> tuple[FeatureExtractor, str]:
    """All engines if the optional deps and the spaCy model are there, else
    regex-only — a missing `en_core_web_sm` should cost stats, not the demo."""
    try:
        every = FeatureExtractor(engines=None)
        every.resolve("warmup v1.0 auf 10.0.0.1")  # pays the model loads once
        return every, ""
    except Exception as exc:  # any import or model failure degrades, not fails
        return FeatureExtractor(), f"regex-only engine: {exc}"


EXTRACTOR, WARNING = _extractor()


class Demo(BaseHTTPRequestHandler):
    def _send(self, kind: str, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        name = self.path.lstrip("/").split("?")[0] or "index.html"
        served = (HERE / name).resolve()
        if served.parent != HERE or not served.is_file():
            self.send_error(404)
            return
        self._send(TYPES.get(served.suffix, "text/plain"), served.read_bytes())

    def do_POST(self) -> None:
        size = int(self.headers.get("Content-Length", "0"))
        text = json.loads(self.rfile.read(size) or b"{}").get("text", "")
        payload = EXTRACTOR.resolve(text).model_dump(mode="json")
        payload["warning"] = WARNING
        self._send("application/json", json.dumps(payload).encode())

    def log_message(self, *_args) -> None:
        """One line per keystroke is noise."""


if __name__ == "__main__":
    if WARNING:
        print(f"! {WARNING}")
    print(f"query-taxonomy demo -> http://localhost:{PORT}")
    ThreadingHTTPServer(("127.0.0.1", PORT), Demo).serve_forever()
