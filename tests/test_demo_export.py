"""The committed banks.json is what the static demo runs on — it must not
lag the banks it was exported from."""

import json
import sys
from pathlib import Path

DEMO = Path(__file__).resolve().parent.parent / "demo"
sys.path.insert(0, str(DEMO))

import export  # noqa: E402


def test_banks_json_is_current():
    committed = json.loads(export.OUT.read_text())
    assert committed["banks"] == export.banks(), "run: python demo/export.py"
    assert committed["stopwords"] == sorted(export.STOPWORDS)
