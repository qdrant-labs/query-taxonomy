"""Corruption group: text-damage detectors. Artifacts are REGEX span banks;
the lexicon signals are WORDFREQ (a stat and a span bank). Word-order noise
and identifier corruption have no detector by design (SPEC C6)."""

from query_taxonomy.corruption.artifacts import (
    EncodingArtifactBank,
    PasteResidueBank,
    TruncationBank,
)
from query_taxonomy.corruption.core import CorruptionRegexBank
from query_taxonomy.corruption.lexicon import TypoBank, UnknownTokenRateBank
from query_taxonomy.core import GeneralBank
from query_taxonomy.taxonomy import CorruptionKind

CORRUPTION_BANKS: tuple[type[GeneralBank], ...] = (
    EncodingArtifactBank,
    TruncationBank,
    PasteResidueBank,
    UnknownTokenRateBank,
    TypoBank,
)

__all__ = [
    "CORRUPTION_BANKS",
    "CorruptionKind",
    "CorruptionRegexBank",
    "EncodingArtifactBank",
    "PasteResidueBank",
    "TruncationBank",
    "TypoBank",
    "UnknownTokenRateBank",
]
