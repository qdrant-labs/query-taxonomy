"""Query-Corpus group. Deliberately absent from `FEATURE_BANKS`: these banks
need a `CorpusIndex` that `FeatureExtractor` has no way to supply, so callers
that own a corpus use `CorpusRelativeExtractor` (or import this tuple and
dispatch themselves)."""

from query_taxonomy.corpus_relative.collocation import PMIBank
from query_taxonomy.corpus_relative.core import CorpusIndex, CorpusRelativeBank
from query_taxonomy.corpus_relative.general import (
    AvgDocLengthBank,
    AvgIDFBank,
    CollectionSizeBank,
    MaxIDFBank,
    OOVShareBank,
    VocabOverlapBank,
)
from query_taxonomy.taxonomy import QueryCorpusFeature

CORPUS_RELATIVE_BANKS: tuple[type[CorpusRelativeBank], ...] = (
    AvgIDFBank,
    MaxIDFBank,
    OOVShareBank,
    CollectionSizeBank,
    AvgDocLengthBank,
    VocabOverlapBank,
    PMIBank,
)

# imported after the tuple: CorpusRelativeExtractor's default banks resolve to
# CORPUS_RELATIVE_BANKS lazily, so no import cycle.
from query_taxonomy.corpus_relative.extractor import (  # noqa: E402
    CorpusRelativeExtractor,
)

__all__ = [
    "CORPUS_RELATIVE_BANKS",
    "AvgDocLengthBank",
    "AvgIDFBank",
    "CollectionSizeBank",
    "CorpusIndex",
    "CorpusRelativeBank",
    "CorpusRelativeExtractor",
    "MaxIDFBank",
    "OOVShareBank",
    "PMIBank",
    "QueryCorpusFeature",
    "VocabOverlapBank",
]
