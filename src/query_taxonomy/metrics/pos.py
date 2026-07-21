"""spaCy-engine stat banks — three of the four router signals: NL-shape
(natural_language_share), vocabulary-mismatch risk (word_variation_share),
and compositional structure (nesting_depth, statement_count). One scalar
per question the router asks; anything that doesn't answer one was pruned
(the UD-17 histogram and its derived shares lived here once — recompute
from the shared doc if a corpus study ever needs them).
All banks share ONE cached pipeline (tagger + parser + lemmatizer, ner
disabled) — the pinned model is part of the determinism pin (SPEC d14/d17).
spaCy is a main dependency; the model needs a separate download
(`poetry run python -m spacy download en_core_web_sm`)."""

from abc import ABC
from functools import lru_cache
from typing import TYPE_CHECKING, override

from query_taxonomy.core import Engine, FeatureGroup, FeatureStat, StatBank
from query_taxonomy.metrics.config import CLAUSAL_DEPS, CLOSED_CLASS, DEFAULT_SPACY_MODEL
from query_taxonomy.taxonomy import StatisticalMetric

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc, Token

@lru_cache(maxsize=1)
def _pipeline(language: str = DEFAULT_SPACY_MODEL) -> "Language":
    # lazy: keeps regex-only extractor construction import-light
    import spacy

    return spacy.load(language, disable=["ner"])


@lru_cache(maxsize=4096)
def _doc(text: str) -> "Doc":
    """One parse per text, shared by every spaCy bank."""
    return _pipeline()(text)


class SpacyBank(StatBank["Language"], ABC):
    """Stat bank over the shared pinned spaCy pipeline."""

    engine = Engine.SPACY

    def __init__(self) -> None:
        super().__init__()
        self._nlp = self.define(_pipeline())

    @property
    @override
    def group(self) -> FeatureGroup:
        return FeatureGroup.STATISTICAL_METRICS

    @override
    def define(self, builder: "Language") -> "Language":
        """The definition artifact is the shared, component-trimmed
        pipeline; banks must not mutate it — the model is pinned."""
        return builder


class NaturalLanguageSignalBank(SpacyBank):
    """How natural-language-shaped is the query? (SPEC decision 14)
    natural_language_share = fraction of tokens that are function words
    (UD closed-class POS: the/of/to/is/when...). Keyword telegrams sit near
    0.0 -> sparse is safe; proper sentences sit near 0.4-0.5 -> dense wins.
    The stopword-ratio bank is its dependency-free REGEX fallback."""

    def __init__(self) -> None:
        super().__init__()
        self._closed_classes = CLOSED_CLASS

    @property
    @override
    def name(self) -> StatisticalMetric:
        return StatisticalMetric.NATURAL_LANGUAGE_SIGNAL

    @override
    def compute(self, text: str) -> list[FeatureStat]:
        tokens = [token for token in _doc(text) if not token.is_space]
        closed = sum(token.pos_ in self._closed_classes for token in tokens)
        share = closed / len(tokens) if tokens else 0.0
        return [FeatureStat("natural_language_share", share)]


class MorphologyBank(SpacyBank):
    """Vocabulary-mismatch risk from grammar: word_variation_share = share
    of alphabetic tokens not in their dictionary form (running -> run) —
    the grammar-caused half of vocabulary mismatch (CSV row 7). High share
    -> embeddings abstract over word forms; zero -> exact-match BM25 is
    not tripped up by conjugation."""

    @property
    @override
    def name(self) -> StatisticalMetric:
        return StatisticalMetric.MORPHOLOGY

    @override
    def compute(self, text: str) -> list[FeatureStat]:
        words = [token for token in _doc(text) if token.is_alpha]
        inflected = sum(
            token.lemma_.lower() != token.text.lower() for token in words
        )
        share = inflected / len(words) if words else 0.0
        return [FeatureStat("word_variation_share", share)]


def _depth(token: "Token") -> int:
    # compare by index: `token.head` builds a FRESH wrapper object on every
    # access, so an `is` check against the root never terminates
    depth = 0
    while token.head.i != token.i and depth < 20:
        token = token.head
        depth += 1
    return depth


class SyntacticDepthBank(SpacyBank):
    """Compositional structure: nesting_depth = how deeply sentence parts
    nest (max dependency-tree depth), statement_count = how many things the
    query asserts/asks (clause-headed relations, CSV row 18). Deep nesting
    = meaning a bag-of-words loses; multiple statements = rerank or
    decomposition matters. Read JOINTLY with natural_language_share: the
    parser hallucinates structure on non-sentences (a bare identifier
    telegram can out-depth a real question), so nesting_depth is only
    meaningful when the query is NL-shaped."""

    def __init__(self) -> None:
        super().__init__()
        self._clausal_deps = CLAUSAL_DEPS

    @property
    @override
    def name(self) -> StatisticalMetric:
        return StatisticalMetric.SYNTACTIC_DEPTH

    @override
    def compute(self, text: str) -> list[FeatureStat]:
        tokens = [token for token in _doc(text) if not token.is_space]
        depth = max((_depth(token) for token in tokens), default=0)
        clauses = sum(token.dep_ in self._clausal_deps for token in tokens)
        return [
            FeatureStat("nesting_depth", float(depth)),
            FeatureStat("statement_count", float(clauses)),
        ]
