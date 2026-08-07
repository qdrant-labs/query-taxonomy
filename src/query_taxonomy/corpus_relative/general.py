"""The six Query-Corpus scalars (SPEC d47b), each one scale-free so lanes
with 3.6K and 120K documents can be compared on the same axis. Two are pure
collection shape (size, doc length), four measure the query against the
collection's term distribution."""

import math
from collections.abc import Sequence
from typing import override

from query_taxonomy.core import FeatureStat
from query_taxonomy.corpus_relative.core import CorpusRelativeBank
from query_taxonomy.features import normalized_idf
from query_taxonomy.taxonomy import QueryCorpusFeature


class AvgIDFBank(CorpusRelativeBank):
    """Mean rarity of the query's terms in this collection. Normalized idf
    (share of the collection's own log(N+1) ceiling) rather than raw log-idf,
    which grows with collection size and cannot be compared across two."""

    @property
    @override
    def name(self) -> QueryCorpusFeature:
        return QueryCorpusFeature.AVG_IDF

    @override
    def compute(self, tokens: Sequence[str]) -> list[FeatureStat]:
        idfs = [
            normalized_idf(self.corpus.df(token), self.corpus.n_docs)
            for token in tokens
        ]
        mean = sum(idfs) / len(idfs) if idfs else 0.0
        return [FeatureStat("avg_idf", mean)]


class MaxIDFBank(CorpusRelativeBank):
    """Rarity of the query's rarest term — the exact-match anchor a sparse
    route can grip. Same normalized scale as avg_idf; the pair separates "one
    rare token in a common query" from "uniformly common"."""

    @property
    @override
    def name(self) -> QueryCorpusFeature:
        return QueryCorpusFeature.MAX_IDF

    @override
    def compute(self, tokens: Sequence[str]) -> list[FeatureStat]:
        idfs = [
            normalized_idf(self.corpus.df(token), self.corpus.n_docs)
            for token in tokens
        ]
        return [FeatureStat("max_idf", max(idfs, default=0.0))]


class OOVShareBank(CorpusRelativeBank):
    """Share of query terms the collection never saw (df == 0) — the terms
    BM25 structurally cannot score. A share, so corpus size drops out."""

    @property
    @override
    def name(self) -> QueryCorpusFeature:
        return QueryCorpusFeature.OOV_SHARE

    @override
    def compute(self, tokens: Sequence[str]) -> list[FeatureStat]:
        missing = sum(self.corpus.df(token) == 0 for token in tokens)
        share = missing / len(tokens) if tokens else 0.0
        return [FeatureStat("oov_share", share)]


class CollectionSizeBank(CorpusRelativeBank):
    """How many documents the query is competing against, in log10 — the
    collection-selection literature's first statistic (CORI/ReDDE), and the
    denominator every other number here is normalized by."""

    @property
    @override
    def name(self) -> QueryCorpusFeature:
        return QueryCorpusFeature.COLLECTION_SIZE

    @override
    def compute(self, tokens: Sequence[str]) -> list[FeatureStat]:
        return [
            FeatureStat("collection_size", math.log10(self.corpus.n_docs + 1))
        ]


class AvgDocLengthBank(CorpusRelativeBank):
    """Mean document length in BM25 tokens, log10 — where BM25's length
    normalization bites. A 60-token passage index and a 2000-token record
    index reward completely different query shapes."""

    @property
    @override
    def name(self) -> QueryCorpusFeature:
        return QueryCorpusFeature.AVG_DOC_LENGTH

    @override
    def compute(self, tokens: Sequence[str]) -> list[FeatureStat]:
        return [
            FeatureStat("avg_doc_length", math.log10(self.corpus.avgdl + 1.0))
        ]


class VocabOverlapBank(CorpusRelativeBank):
    """Mean share of the collection each query term touches (df / N) — the
    mass half of vocabulary mismatch. Linear where avg_idf is logarithmic, so
    it is driven by the query's COMMON terms and is not a restatement of
    oov_share, which only asks membership."""

    @property
    @override
    def name(self) -> QueryCorpusFeature:
        return QueryCorpusFeature.VOCAB_OVERLAP

    @override
    def compute(self, tokens: Sequence[str]) -> list[FeatureStat]:
        if not tokens or self.corpus.n_docs == 0:
            return [FeatureStat("vocab_overlap", 0.0)]
        shares = [self.corpus.df(token) / self.corpus.n_docs for token in tokens]
        return [FeatureStat("vocab_overlap", sum(shares) / len(shares))]
