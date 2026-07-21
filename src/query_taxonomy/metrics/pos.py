"""spaCy-engine stat banks — three of the four router signals: NL-shape
(closed_class_share), vocabulary-mismatch risk (inflected_share), and
compositional structure (parse_depth, clause_count). One scalar per
question the router asks; anything that doesn't answer one was pruned
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
from query_taxonomy.taxonomy import StatisticalMetric

if TYPE_CHECKING:
    from spacy.language import Language
    from spacy.tokens import Doc, Token

SPACY_MODEL = "en_core_web_sm"

# Universal Dependencies (UD) closed-class part-of-speech tags.
# Closed-class categories contain a small, relatively fixed inventory of words.
CLOSED_CLASS = frozenset(
    {
        "ADP",    # Adposition: prepositions/postpositions expressing grammatical relations (e.g. "in", "to", "with").
        "AUX",    # Auxiliary verb: grammatical verb marking tense, aspect, mood, voice, or polarity (e.g. "is", "have", "will").
        "CCONJ",  # Coordinating conjunction: links elements of equal syntactic status (e.g. "and", "or", "but").
        "DET",    # Determiner: specifies or limits a noun (e.g. "the", "this", "some", "each").
        "NUM",    # Numeral: cardinal or other numeric expression functioning as a number (e.g. "three", "42").
        "PART",   # Particle: function word not fitting other categories, often marking negation or infinitives (e.g. "not", "to").
        "PRON",   # Pronoun: substitutes for a noun phrase or refers to discourse participants (e.g. "he", "they", "who").
        "SCONJ",  # Subordinating conjunction: introduces a subordinate clause (e.g. "because", "if", "although").
    }
)

# Universal Dependencies relations headed by a clause.
CLAUSAL_DEPS = frozenset(
    {
        "ROOT",   # Root of the sentence: the main predicate of the entire utterance.
        "ccomp",  # Clausal complement: finite or non-finite clause functioning as an argument with its own subject.
        "xcomp",  # Open clausal complement: argument clause whose subject is controlled by another argument.
        "advcl",  # Adverbial clause modifier: subordinate clause expressing time, reason, condition, purpose, etc.
        "acl",    # Clausal modifier of a noun: clause modifying a noun (e.g. participial or infinitival modifier).
        "relcl",  # Relative clause modifier: clause modifying a noun through relativization.
        "csubj",  # Clausal subject: clause functioning as the syntactic subject of a predicate.
    }
)


@lru_cache(maxsize=1)
def _pipeline() -> "Language":
    # lazy: keeps regex-only extractor construction import-light
    import spacy

    return spacy.load(SPACY_MODEL, disable=["ner"])


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
    closed_class_share = fraction of tokens whose UD POS is a closed class
    (function words: the/of/to/is/when...). Keyword telegrams sit near 0.0
    -> sparse is safe; proper sentences sit near 0.4-0.5 -> dense wins.
    The stopword-ratio bank is its dependency-free REGEX fallback."""

    @property
    @override
    def name(self) -> StatisticalMetric:
        return StatisticalMetric.NATURAL_LANGUAGE_SIGNAL

    @override
    def compute(self, text: str) -> list[FeatureStat]:
        tokens = [token for token in _doc(text) if not token.is_space]
        closed = sum(token.pos_ in CLOSED_CLASS for token in tokens)
        share = closed / len(tokens) if tokens else 0.0
        return [FeatureStat("closed_class_share", share)]


class MorphologyBank(SpacyBank):
    """Vocabulary-mismatch risk from inflection: share of alphabetic tokens
    whose lemma differs from their surface form (running -> run) — the
    grammar-caused half of vocabulary mismatch (CSV row 7). High share ->
    embeddings abstract over morphology; zero -> exact-match BM25 is not
    tripped up by conjugation."""

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
        return [FeatureStat("inflected_share", share)]


def _depth(token: "Token") -> int:
    # compare by index: `token.head` builds a FRESH wrapper object on every
    # access, so an `is` check against the root never terminates
    depth = 0
    while token.head.i != token.i and depth < 20:
        token = token.head
        depth += 1
    return depth


class SyntacticDepthBank(SpacyBank):
    """Compositional structure: max dependency-tree depth and clause count
    (CSV row 18). Deep structure = meaning a bag-of-words loses; multiple
    clauses = multiple propositions, where rerank/decomposition matters.
    Read JOINTLY with closed_class_share: the parser hallucinates structure
    on non-sentences (a bare identifier telegram can out-depth a real
    question), so depth is only meaningful when the query is NL-shaped."""

    @property
    @override
    def name(self) -> StatisticalMetric:
        return StatisticalMetric.SYNTACTIC_DEPTH

    @override
    def compute(self, text: str) -> list[FeatureStat]:
        tokens = [token for token in _doc(text) if not token.is_space]
        depth = max((_depth(token) for token in tokens), default=0)
        clauses = sum(token.dep_ in CLAUSAL_DEPS for token in tokens)
        return [
            FeatureStat("parse_depth", float(depth)),
            FeatureStat("clause_count", float(clauses)),
        ]
