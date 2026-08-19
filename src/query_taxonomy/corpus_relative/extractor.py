"""Runs the Query-Corpus group against a bound collection. Separate from
FeatureExtractor because the corpus is a cross-cutting concern (a separate
layer, not a resolve(corpus=...) mode) and these banks take (tokens, index),
not the single-argument compute(text) the query-only extractor dispatches."""

from collections.abc import Iterable, Sequence

from query_taxonomy.core import FeatureStat
from query_taxonomy.corpus_relative.core import CorpusIndex, CorpusRelativeBank


class CorpusRelativeExtractor:
    """Binds the corpus-relative banks to one collection at construction (each
    bank reads the shared CorpusIndex) and runs them over pre-tokenized
    queries. Tokenization is the retriever's, so the caller hands tokens in."""

    def __init__(
        self,
        corpus: CorpusIndex,
        banks: Iterable[type[CorpusRelativeBank]] | None = None,
    ) -> None:
        # lazy: the default tuple lives in the package __init__, imported at
        # call time so the module import graph stays acyclic.
        if banks is None:
            from query_taxonomy.corpus_relative import CORPUS_RELATIVE_BANKS

            banks = CORPUS_RELATIVE_BANKS
        self._banks = tuple(cls(corpus) for cls in banks)

    def resolve(self, tokens: Sequence[str]) -> list[FeatureStat]:
        """Every corpus-relative scalar for one tokenized query."""
        return [stat for bank in self._banks for stat in bank.compute(tokens)]
