"""LANGID-engine bank: language segmentation via lingua. One detector per
candidate-language set, built lazily so regex-only extraction never imports
lingua. LANGUAGE_SET and CODE_SWITCHING are derived views over the spans (in
QueryFeatures), not banks."""

import re
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING, override

from query_taxonomy.core import (
    Engine,
    FeatureGroup,
    GeneralBank,
    LanguageSpan,
)
from query_taxonomy.taxonomy import SemanticFeature

if TYPE_CHECKING:
    from lingua import LanguageDetector

DEFAULT_CANDIDATES: tuple[str, ...] = (
    "en", "ru", "zh", "ar", "ja", "ko", "hi",
    "fa", "th", "bn", "te", "el", "he", "uk",
)
"""English + non-Latin-script languages only. Measured decision, not taste: with
Latin-script neighbours (de/fr/es/it/nl/pt) in the set, lingua sprays them
across short English and code-heavy queries — 9-98% spurious is_code_switched
over the English lanes (bright-robotics 98%, freshstack 98%). Dropping them
leaves `en` as the only Latin option, so Latin text resolves to English (0% FP
on every English lane), while non-Latin languages stay detectable by script.
When real supply adds a Latin-script language, pin it via the candidate_languages
kwarg — the (type, kwargs) registry entry — for that run."""

_WORD = re.compile(r"[^\W\d_]+")
"""Letter runs. A query with no run of >=2 letters is numbers/identifiers, not
language — lingua would hallucinate one (12345 -> French), so it defaults to en."""


@lru_cache(maxsize=8)
def _detector(candidates: tuple[str, ...]) -> "LanguageDetector":
    from lingua import IsoCode639_1, Language, LanguageDetectorBuilder

    languages = [
        Language.from_iso_code_639_1(getattr(IsoCode639_1, code.upper()))
        for code in candidates
    ]
    return LanguageDetectorBuilder.from_languages(*languages).build()


class LanguageBank(GeneralBank[LanguageSpan, "LanguageDetector"], ABC):
    """LANGID-engine bank over a pinned candidate-language set. The definition
    artifact is the lingua detector, built lazily like WordFreqBank's table."""

    engine = Engine.LANGID

    def __init__(self, candidate_languages: tuple[str, ...] = DEFAULT_CANDIDATES) -> None:
        super().__init__()
        self._candidates = tuple(candidate_languages)

    @property
    @override
    def group(self) -> FeatureGroup:
        return FeatureGroup.SEMANTICAL

    @property
    @abstractmethod
    def name(self) -> SemanticFeature:
        """Feature-enum member this bank detects."""

    @override
    def define(self, builder: "LanguageDetector") -> "LanguageDetector":
        return builder


class SegmentationBank(LanguageBank):
    """Contiguous single-language spans over the query. A letterless query and a
    query lingua returns nothing for both default to one 'en' span (the corpus
    is English-dominant, so an unplaceable short query is almost always English)."""

    @property
    @override
    def name(self) -> SemanticFeature:
        return SemanticFeature.LANGUAGE_SEGMENTATION

    def _english_fallback(self, text: str) -> list[LanguageSpan]:
        return [LanguageSpan(text, 0, len(text), "en")]

    @override
    def compute(self, text: str) -> list[LanguageSpan]:
        if not text.strip():
            return []
        if not any(len(word) >= 2 for word in _WORD.findall(text)):
            return self._english_fallback(text)
        results = _detector(self._candidates).detect_multiple_languages_of(text)
        spans = [
            LanguageSpan(
                text[r.start_index : r.end_index],
                r.start_index,
                r.end_index,
                r.language.iso_code_639_1.name.lower(),
            )
            for r in results
        ]
        return spans or self._english_fallback(text)
