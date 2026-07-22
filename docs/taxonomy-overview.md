# Query Taxonomy — overview

The taxonomy describes **what kind of thing a query is**, not what it means. Each feature is a signal the Strategy Router uses to decide between dense, sparse, and hybrid retrieval — or to profile a corpus and drive dataset recipe design.

## The four groups

### Structured Identifiers
Exact-form tokens that only work with sparse retrieval: version strings, IP addresses, CVEs, legal citations, drug codes, tracking numbers, and 75+ more across eight domains.

The pattern is consistent across all domains: an identifier has a canonical form, and a mismatch of even one character returns the wrong thing. Dense embeddings cannot help — they paraphrase, which is exactly wrong. Presence of any identifier type is a hard signal toward sparse or hybrid.

**Eight domains:** general · network · tech · finance · legal · medical · logistics · media

**The assumptive banks** (suffix `-_like`): `stock_ticker_like`, `error_code_like`, `derivatives_symbol_like`, `ticket_like`, `booking_reference_like`, `aircraft_vessel_reg_like`, `game_notation_like`. These match by shape rather than keyword context — they will fire on gene names in scientific text, protein mutations in biomedical corpora, etc. The `-Like` suffix in the emitted identifier makes the assumption visible at the output boundary so callers can filter.

---

### Sentence Markers
Closed-list lexical cues about the **register** of the query — how it was written, not what it asks.

| Marker | Example | Router implication |
|--------|---------|-------------------|
| negation | *laptops without touchscreen* | Sparse can't capture it; dense required |
| acronym | *NASA, N.Y.* | Compressed identifier signal; lowercase forms undetectable by shape |
| comparative | *faster than Python, best database* | Ranking semantics a bag-of-words loses |
| greeting | *hi how do I set up Qdrant* | Conversational register → dense/hybrid |
| politeness | *please explain quantization* | Filler tokens hurt sparse precision |
| interjection | *ugh my container keeps crashing* | Noise dilutes sparse; conversational register |

---

### Logical Structures
Constructs that signal explicit **query logic** or **embedded formal grammars** — things that are not natural-language search at all.

| Feature | Example | Signal |
|---------|---------|--------|
| operator_syntax | *cats AND dogs, java NOT javascript* | Explicit boolean retrieval command → sparse |
| temporal | *bitcoin price today, 3 days ago* | Relative time expression; absolute dates live in Structured Identifiers |
| code_fragment | *why x != y in python, SELECT id FROM users* | Embedded code evidence tokens → strong sparse signal |
| math_expression | *solve x^2 + y^2 = 25, E = mc^2* | Equation grammar → exact symbols matter → sparse |

Operator syntax and code/math expressions are **not search dialect** — their presence in a query is itself the evidence. The banks claim evidence tokens only (not whole fragments); exact symbol matching is what makes sparse the right choice here.

---

### Statistical Metrics
Five router **signals** — scalars per query, no span claims. Each answers one question the router cares about. Named for what the scalar means, not how it's computed.

| Signal                      | Scalar(s)                          | What it measures                                            | Router implication                                                     |
| -----------------------------| ------------------------------------| -------------------------------------------------------------| ------------------------------------------------------------------------|
| **length**                  | `length_words`, `length_chars`     | How big is the query?                                       | Very short → sparse-safe; long → dense-friendly                        |
| **stopword_ratio**          | `stopword_ratio`, `stopword_count` | How natural-language-shaped? (REGEX fallback)               | High → natural phrasing → dense; near-zero → keyword telegram → sparse |
| **natural_language_signal** | `natural_language_share`           | Fraction of tokens that are function words (UD POS)         | 0.0 = keyword telegram; 0.4–0.5 = proper sentence                      |
| **morphology**              | `word_variation_share`             | Grammar-caused vocabulary mismatch risk                     | High → embeddings abstract over word forms; zero → BM25 safe           |
| **syntactic_depth**         | `nesting_depth`, `statement_count` | Compositional structure a bag-of-words loses                | Deep/multi-clause → dense or decompose+rerank                          |
| **coordination**            | `widest_list_size`                 | How many equal parts does the longest list string together? | Wide-but-flat enumeration nesting_depth cannot see                     |

**Joint reading caveat:** `nesting_depth`, `statement_count`, and `widest_list_size` are only meaningful where `natural_language_share` indicates natural language. A CVE telegram can out-depth a question grammatically — the parser hallucinates structure on non-sentences.

**Signals are coordinates, not labels.** Strata are boxes in signal space; synthetic queries are rejection-sampled against target signatures. Strategy labels always come from retrieval outcomes — never from signals directly. Labeling by signal would teach the router a heuristic it could never beat the production hard classifier with.

`stopword_ratio` is the REGEX fallback for `natural_language_share` — works without spaCy, used when the model group is unavailable or on non-English text.

---

## Engines

| Engine | Banks | Requires |
|--------|-------|---------|
| REGEX | All identifier banks, all sentence markers, all logical structures, length, stopword_ratio | Nothing (default install) |
| ALGO (spaCy) | natural_language_signal, morphology, syntactic_depth, coordination | `en_core_web_sm` download |

```python
# REGEX only — dependency-light, default
extractor = FeatureExtractor()

# All engines
extractor = FeatureExtractor(engines=None)

features = extractor.resolve("upgrade v2.1.0 on 192.168.0.1 before October")
# → spans: version_string, ip_address, temporal
# → stats: length_words=8, natural_language_share=0.25, ...
```

## Claim resolution

Within a group, banks compete for character ranges by **ambiguity tier** — RIGID claims first, AMBIGUOUS last. A lower-priority bank can never re-claim text already claimed by a higher-priority one (NUMBER can't steal `1.0` from a claimed `v1.0.0`).

Groups are **independent layers** — spans from different groups may overlap. Stats never enter the claim registry.

## What's not in the taxonomy (yet)

Deferred features intentionally excluded from the current implementation:
- **Multilingual / code-switching**: SemanticFeature enum defined, no bank registered (pending lingua-py, SPEC d18-19)
- **Corruption signals**: CorruptionKind enum defined, no bank registered (deferred — PMI, typo, noise)
- **Corpus-relative features** (vocabulary mismatch, ambiguity, query specificity): require a reference corpus, not query-only
