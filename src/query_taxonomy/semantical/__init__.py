"""Semantical group: word-level language segmentation (LANGID engine, wordfreq
+ unicodedata). One bank; language_set and code_switching are derived views over
its spans, computed in QueryFeatures, not banks."""

from query_taxonomy.semantical.core import (
    DEFAULT_CANDIDATES,
    LanguageBank,
    SegmentationBank,
)

SEMANTICAL_BANKS: tuple[type[LanguageBank], ...] = (SegmentationBank,)

__all__ = [
    "DEFAULT_CANDIDATES",
    "LanguageBank",
    "SEMANTICAL_BANKS",
    "SegmentationBank",
]
