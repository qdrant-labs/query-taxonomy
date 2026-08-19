"""Collocation over the query's term pairs: normalized PMI against the
collection's document-level co-occurrence (SPEC Q3). Separate from the d47b
six because it reads pair counts, not just per-token df."""

import math
from collections.abc import Sequence
from itertools import combinations
from typing import override

from query_taxonomy.core import FeatureStat
from query_taxonomy.corpus_relative.core import CorpusRelativeBank
from query_taxonomy.taxonomy import QueryCorpusFeature


class PMIBank(CorpusRelativeBank):
    """Do the query's terms belong together here? mean_pmi (average pairwise
    collocation) and min_pmi (the least-collocated pair — the unusual
    combination), as normalized PMI in [-1, 1]. Pairs with a term the
    collection never saw are skipped; that absence is oov_share's job. Emits
    nothing when the index carries no pair counts or the query has no scorable
    pair — unmeasured, not a fabricated 0.0 (which on this scale reads as
    'perfectly independent')."""

    @property
    @override
    def name(self) -> QueryCorpusFeature:
        return QueryCorpusFeature.PMI

    def _npmi(self, a: str, b: str) -> float | None:
        """Normalized PMI for one pair, or None when undefined (a term absent
        from the collection, or an empty collection)."""
        df_a, df_b, n = self.corpus.df(a), self.corpus.df(b), self.corpus.n_docs
        if df_a == 0 or df_b == 0 or n == 0:
            return None
        pair = self.corpus.pair_df(a, b)
        if pair == 0:  # never together -> the NPMI limit as co-occurrence -> 0
            return -1.0
        joint = min(pair, df_a, df_b)  # a bad producer cannot exceed a marginal
        p_joint = joint / n
        if p_joint >= 1.0:  # in every document together -> perfect collocation
            return 1.0
        pmi = math.log2(p_joint / ((df_a / n) * (df_b / n)))
        return pmi / -math.log2(p_joint)

    @override
    def compute(self, tokens: Sequence[str]) -> list[FeatureStat]:
        if not self.corpus.pair_document_frequencies:
            return []  # no pair counts supplied -> unmeasured
        scores = [
            npmi
            for a, b in combinations(sorted(set(tokens)), 2)
            if (npmi := self._npmi(a, b)) is not None
        ]
        if not scores:
            return []
        return [
            FeatureStat("mean_pmi", sum(scores) / len(scores)),
            FeatureStat("min_pmi", min(scores)),
        ]
