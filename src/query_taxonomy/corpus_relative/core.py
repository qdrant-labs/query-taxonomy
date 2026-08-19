from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import override

from query_taxonomy.core import FeatureGroup, FeatureStat, GeneralBank
from query_taxonomy.taxonomy import QueryCorpusFeature


@dataclass(frozen=True, slots=True)
class CorpusIndex:
    """The collection statistics every Query-Corpus bank reads. Stdlib only,
    by design: whoever owns the corpus (parquet, a live collection, a fixture
    dict) counts it and hands the counts over."""

    document_frequencies: Mapping[str, int]
    """Token -> number of documents containing it, in the SAME tokenization
    the query is handed in (the sparse route's, so the features and the
    retriever see one vocabulary)."""
    n_docs: int
    avgdl: float
    """Mean document length in tokens — BM25's length-normalization term."""
    pair_document_frequencies: Mapping[frozenset[str], int] = field(
        default_factory=dict
    )
    """{token_a, token_b} -> documents containing BOTH, for PMI. Precomputed
    by the corpus owner for the query set's pairs only (bounded by
    queries x pairs-per-query, never O(vocab^2)); empty when PMI is unused."""

    def df(self, token: str) -> int:
        return self.document_frequencies.get(token, 0)

    def pair_df(self, a: str, b: str) -> int:
        return self.pair_document_frequencies.get(frozenset((a, b)), 0)


class CorpusRelativeBank(GeneralBank[FeatureStat, CorpusIndex], ABC):
    """
    Stat bank over a query AND a collection. `compute` takes pre-tokenized
    input plus the index — the one deliberate break from the single-argument
    `compute(text)` family, because the tokenizer that must be used belongs
    to the retriever, not to the taxonomy (same reason spaCy banks read a
    shared pipeline they do not build).

    No `Engine`: these banks drag in stdlib arithmetic, and `Engine` answers
    "what heavy runtime does this bank import". Corpus presence, not engine
    filtering, is what selects them.

    The collection is bound at construction, like `RegexBank`'s compiled
    pattern or `SpacyBank`'s pipeline: `CORPUS_RELATIVE_BANKS` callers already
    dispatch these banks themselves (see corpus_relative/__init__.py), so
    `cls(corpus)` costs them nothing and a bank can never exist unbound.
    """

    def __init__(self, corpus: CorpusIndex) -> None:
        super().__init__()
        self._corpus = self.define(corpus)

    @property
    def corpus(self) -> CorpusIndex:
        return self._corpus

    @property
    @override
    def group(self) -> FeatureGroup:
        return FeatureGroup.QUERY_CORPUS

    @property
    @abstractmethod
    def name(self) -> QueryCorpusFeature:
        """
        Name of the current bank
        """

    @override
    def define(self, builder: CorpusIndex) -> CorpusIndex:
        """The definition artifact is the caller's index; banks read it and
        never own it, so there is nothing to build."""
        return builder

    @abstractmethod
    def compute(self, tokens: Sequence[str]) -> list[FeatureStat]:
        """Named scalars for one tokenized query against the bound collection."""

