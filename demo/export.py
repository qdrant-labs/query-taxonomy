"""Freezes the regex layer to `banks.json` so the demo can run with no Python.

Patterns are read off the live bank objects, never retyped: the only thing
the browser reimplements is the claim-resolution loop.
"""

import json
import re
from pathlib import Path

from query_taxonomy import FEATURE_BANKS, split_bank
from query_taxonomy.core import RegexBank
from query_taxonomy.metrics.general import STOPWORDS, LengthBank

OUT = Path(__file__).with_name("banks.json")

# Python's `\w` is Unicode under re.UNICODE; JS `\w` is ASCII-only, which would
# score every Cyrillic query as zero words. The classes are the JS spelling of
# the same set — the one pattern worth restating, because length normalizes
# every share the stats emit.
TOKENS = r"[\p{L}\p{N}_]+"


def banks() -> list[dict]:
    out = []
    for group, specs in FEATURE_BANKS.items():
        for spec in specs:
            cls, kwargs = split_bank(spec)
            if not issubclass(cls, RegexBank):
                continue  # spaCy / wordfreq / tokenizer engines stay server-side
            bank = cls(**kwargs)
            out.append({
                "group": group.value,
                "name": str(bank.name),
                "tier": int(bank.ambiguity),
                "pattern": bank._regex.pattern,
                "flags": "gi" if bank._regex.flags & re.IGNORECASE else "g",
            })
    return out


if __name__ == "__main__":
    assert LengthBank()._tokens.pattern == r"\w+", "tokenizer changed, revisit TOKENS"
    payload = {
        "banks": banks(),
        "stopwords": sorted(STOPWORDS),
        "tokens": TOKENS,
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    print(f"{OUT}: {len(payload['banks'])} banks, {OUT.stat().st_size // 1024} KB"
          if OUT.exists() else "")
