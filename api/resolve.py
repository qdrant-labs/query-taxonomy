"""Vercel Python function: the taxonomy over one query.

Regex + wordfreq + langid — every engine that needs no model download. The
local server subclasses this handler, so the deployed page and the local one
run the same code path.
"""

import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from query_taxonomy.core import Engine
from query_taxonomy.features import FeatureExtractor

PORTABLE = (Engine.REGEX, Engine.WORDFREQ, Engine.LANGID)
"""spaCy stats and subword fragmentation want a downloaded model, so they stay
where one is — `TAXONOMY_ALL_ENGINES` turns them on for the local server."""

MAX_CHARS = 2000
MAX_BODY = 64_000


def _extractor() -> tuple[FeatureExtractor, list[str], str]:
    wanted = None if os.environ.get("TAXONOMY_ALL_ENGINES") else PORTABLE
    try:
        extractor = FeatureExtractor(engines=wanted)
        extractor.resolve("warmup v1.0 auf 10.0.0.1")  # pays the table loads
        return extractor, [str(e) for e in (wanted or Engine)], ""
    except Exception as exc:  # a missing model costs stats, not the demo
        extractor = FeatureExtractor(engines=PORTABLE)
        extractor.resolve("warmup v1.0 auf 10.0.0.1")
        return extractor, [str(e) for e in PORTABLE], str(exc)


EXTRACTOR, ENGINES, WARNING = _extractor()


class handler(BaseHTTPRequestHandler):
    def _send(self, kind: str, body: bytes) -> None:
        self.send_response(200)
        self.send_header("Content-Type", kind)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        size = int(self.headers.get("Content-Length", "0"))
        if size > MAX_BODY:
            self.send_error(413)
            return
        try:
            text = str(json.loads(self.rfile.read(size) or b"{}").get("text", ""))
        except (ValueError, TypeError, AttributeError):
            self.send_error(400)
            return
        payload = EXTRACTOR.resolve(text[:MAX_CHARS]).model_dump(mode="json")
        payload["engines"] = ENGINES
        payload["warning"] = WARNING
        self._send("application/json", json.dumps(payload).encode())

    def log_message(self, *_args) -> None:
        """One line per keystroke is noise."""
