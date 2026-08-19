"""Semantical group: language segmentation (LANGID/lingua). One bank;
language_set and code_switching are derived views over its spans, computed in
QueryFeatures, not banks (SPEC M2, docs/adr/0002)."""

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
