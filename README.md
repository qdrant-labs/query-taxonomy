# Query Taxonomy

Feature extractors over queries and documents. Built for corpus profiling and the Strategy Router dataset — but the extractor is corpus-agnostic and reusable.

## What it does

Runs pluggable **banks** over text, sectioned by feature group, and returns claim-resolved spans and per-doc stats. Three engines cooperate under one API:

- **regex** — deterministic identifier vocabulary (79 types, 8 domains).
- **GLiNER2** — pinned schema, one shared forward pass, audited thresholds. Backstops `proper_noun` / `person` / `location` / `temporal`.
- **spaCy** — one pinned pipeline. POS profile, morphology, syntactic depth.

Within a group, banks compete for char ranges by ambiguity tier (RIGID first). Groups are independent layers.

## Install

```bash
poetry install                    # regex only, dependency-light
poetry install --with model       # + GLiNER2 + transformers
poetry run python -m spacy download en_core_web_sm   # required for spaCy banks
```

## Usage

```python
from query_taxonomy.core import Engine
from query_taxonomy.features import FeatureExtractor

# regex only (default) — no torch, no spaCy import at construction
extractor = FeatureExtractor()

# all engines — needs the `model` group + the spaCy model
extractor = FeatureExtractor(engines=None)

features = extractor.resolve("upgrade v2.1.0 on 192.168.0.1 before October")
print(features.spans)   # {FeatureGroup.STRUCTURED_IDENTIFIERS: {...}, ...}
print(features.stats)   # {FeatureGroup.STATISTICAL_METRICS: {...}}

corpus = extractor.extract(queries=[...])
print(corpus.summary())   # group-sectioned FP smell-test view
```

## Layout

```
src/query_taxonomy/
├── taxonomy.py          # code twin of query-taxonomy.csv (enums, no machinery)
├── core.py              # Engine, FeatureSpan, FeatureStat, GeneralBank, RegexBank, StatBank
├── features.py          # QueryFeatures, CorpusFeatures, FeatureExtractor
├── banks/               # regex banks per domain (structured identifiers)
├── entities/            # GLiNER2 banks
├── markers/             # sentence-marker banks
├── logical/             # logical-structure banks
├── corruption/          # corruption / noise banks
└── metrics/             # stat banks (length, stopwords, POS, morphology, depth)
```

## Writing banks

- `edify.RegexBuilder` is immutable — every call returns a clone. `define()` must return the completed chain.
- Alternation (`any_of`) is first-match. Order longest branch first.
- Ambiguity tier: keyword-gated / high-entropy → **RIGID**; known collisions → **MODERATE**; bare caps/digit shapes → **AMBIGUOUS**.
- Every bank needs ~2 positive / 2 negative cases in `tests/test_banks.py` (invariant enforced by `test_every_bank_has_cases`).

## Tests

```bash
poetry run pytest
poetry run ruff check src/query_taxonomy tests
```
