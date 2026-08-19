"""WORDFREQ-method corruption detectors: the share of word-shaped tokens
absent from the frequency table, and keyword-adjacent typos (absent AND one
edit from a frequent word). wordfreq ships its table in-package; imported
lazily so regex-only extraction stays import-light."""

import re
from functools import lru_cache
from typing import TYPE_CHECKING, override

from query_taxonomy.core import (
    AmbiguityTier,
    Engine,
    FeatureGroup,
    FeatureSpan,
    FeatureStat,
    GeneralBank,
    StatBank,
)
from query_taxonomy.taxonomy import CorruptionKind

if TYPE_CHECKING:
    from collections.abc import Callable

_LANG = "en"
_WORD = re.compile(r"[^\W\d_]+")
"""The word-shape guard: runs of letters only, so digits, identifiers, and
punctuation-bearing tokens never count as damaged words."""
_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
_MIN_TYPO_LEN = 3
"""Below this, a one-edit neighbour is meaningless (every letter is one edit
from the words 'a'/'i')."""
_MAX_TYPO_LEN = 25
"""Above this, no frequent word is one edit away — a long all-letter token is
a hash or slug, not a typo — so skip the wasted _edits1 candidate generation."""
FREQUENT_ZIPF = 3.0
"""A plausible typo target must be at least this common (~1 per million) —
placeholder pending calibration, see TODOS."""


@lru_cache(maxsize=1)
def _zipf() -> "Callable[..., float]":
    from wordfreq import zipf_frequency

    return zipf_frequency


def _edits1(word: str) -> set[str]:
    """Every string one insert/delete/substitute/transpose from `word`."""
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a, b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a, b in splits if len(b) > 1]
    replaces = [a + c + b[1:] for a, b in splits if b for c in _ALPHABET]
    inserts = [a + c + b for a, b in splits for c in _ALPHABET]
    return set(deletes + transposes + replaces + inserts)


class UnknownTokenRateBank(StatBank[str]):
    """Share of the query's word-shaped tokens absent from the frequency
    table. Distinct from term_rarity.rare_share (present but infrequent); this
    is entirely absent. Emits nothing when the query has no word-shaped token.
    The table is English: on a non-English lane this measures absence-from-EN,
    which is language, not corruption — annotate multilingual lanes when
    reading the census."""

    engine = Engine.WORDFREQ

    @property
    @override
    def group(self) -> FeatureGroup:
        return FeatureGroup.CORRUPTION

    @property
    @override
    def name(self) -> CorruptionKind:
        return CorruptionKind.UNKNOWN_TOKEN_RATE

    @override
    def define(self, builder: str) -> str:
        return builder

    @override
    def compute(self, text: str) -> list[FeatureStat]:
        zipf = _zipf()
        words = _WORD.findall(text)
        if not words:
            return []
        unknown = sum(zipf(word.lower(), _LANG) == 0.0 for word in words)
        return [FeatureStat("unknown_token_rate", unknown / len(words))]


class TypoBank(GeneralBank[FeatureSpan, str]):
    """Keyword-adjacent misspellings: a word-shaped token absent from the
    table yet within one edit of a frequent word. Spans, so each typo is
    located. Rare-real words and names are absent but NOT near a frequent
    word, so they are left alone."""

    engine = Engine.WORDFREQ

    @property
    @override
    def group(self) -> FeatureGroup:
        return FeatureGroup.CORRUPTION

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @property
    @override
    def name(self) -> CorruptionKind:
        return CorruptionKind.TYPO

    @override
    def define(self, builder: str) -> str:
        return builder

    def _is_typo(self, lower: str) -> bool:
        if not _MIN_TYPO_LEN <= len(lower) <= _MAX_TYPO_LEN:
            return False  # too short to judge, or too long to be a misspelling
        if _zipf()(lower, _LANG) > 0.0:
            return False  # a known word (incl. common misspellings in the table)
        return any(_zipf()(cand, _LANG) >= FREQUENT_ZIPF for cand in _edits1(lower))

    @override
    def compute(self, text: str) -> list[FeatureSpan]:
        return [
            FeatureSpan(match.group(0), match.start(), match.end())
            for match in _WORD.finditer(text)
            if self._is_typo(match.group(0).lower())
        ]
