"""LANGID-engine word-level language segmentation. Unicode script decides
non-Latin tokens; per-word wordfreq frequency decides Latin tokens against the
query's carrier language. Deterministic and offline (wordfreq + stdlib); no
model, no network."""

import re
import unicodedata
from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING, override

from query_taxonomy.core import Engine, FeatureGroup, GeneralBank, LanguageSpan
from query_taxonomy.taxonomy import SemanticFeature

if TYPE_CHECKING:
    from collections.abc import Callable

# Candidate language -> its Unicode script. Latin languages share the
# wordfreq-argmax path; each non-Latin script is decided by the script itself.
# wordfreq-supported only (Thai has no wordfreq table, so it is omitted).
_SCRIPT_BY_LANG: dict[str, str] = {
    "en": "LATIN", "de": "LATIN", "fr": "LATIN", "es": "LATIN", "it": "LATIN",
    "nl": "LATIN", "pt": "LATIN", "sv": "LATIN", "pl": "LATIN", "id": "LATIN",
    "ru": "CYRILLIC", "uk": "CYRILLIC", "zh": "CJK", "ja": "CJK",
    "ar": "ARABIC", "fa": "ARABIC", "ko": "HANGUL", "hi": "DEVANAGARI",
    "el": "GREEK", "he": "HEBREW",
}
DEFAULT_CANDIDATES: tuple[str, ...] = tuple(_SCRIPT_BY_LANG)

# Non-Latin script -> its representative language. Within-script disambiguation
# (ru vs uk, zh vs ja-Han, ar vs fa) needs per-language tokenizers wordfreq can't
# run offline, and is low value, so the dominant language of each script wins.
_SCRIPT_TO_LANG: dict[str, str] = {
    "CYRILLIC": "ru", "CJK": "zh", "ARABIC": "ar", "HANGUL": "ko",
    "DEVANAGARI": "hi", "GREEK": "el", "HEBREW": "he",
    "HIRAGANA": "ja", "KATAKANA": "ja",
}

CARRIER_FLOOR = 3.0
"""A Latin word this common in the carrier is kept as carrier even if another
language also has it — stops a word shared by both from switching."""
TARGET_FLOOR = 4.0
"""The switch target must itself be this common in its language — stops an OOV
code token (rviz, zipf ~1 anywhere) from switching off a zero-freq carrier while
a real foreign word (fichier ~4.5 in fr) clears it. Calibrated to the
precision-first point (csP 0.94 / csR 0.83) on eval/codeswitch_queries.jsonl."""

_TOKEN = re.compile(r"\S+")
_CODE_CHARS = frozenset("<>{}[]()/\\|=;:\"`@#%^&*_") | frozenset("0123456789")
"""A token carrying any of these is code/markup/identifier, not prose — classify
it as the carrier rather than language-tagging a homograph (todo[], <lidar>)."""


@lru_cache(maxsize=1)
def _zipf() -> "Callable[..., float]":
    from wordfreq import zipf_frequency

    return zipf_frequency


@lru_cache(maxsize=1 << 16)
def _char_script(char: str) -> str | None:
    try:
        return unicodedata.name(char).split(" ", 1)[0]
    except ValueError:
        return None


def _token_script(token: str) -> str | None:
    """Dominant Unicode script over the token's letters, or None if letterless."""
    counts: dict[str, int] = {}
    for char in token:
        if char.isalpha():
            script = _char_script(char)
            if script:
                counts[script] = counts.get(script, 0) + 1
    return max(counts, key=lambda s: counts[s]) if counts else None


class LanguageBank(GeneralBank[LanguageSpan, tuple], ABC):
    """LANGID-engine bank emitting language-labeled spans. Deterministic; the
    definition artifact is the candidate-language set (read, never built)."""

    engine = Engine.LANGID

    @property
    @override
    def group(self) -> FeatureGroup:
        return FeatureGroup.SEMANTICAL

    @property
    @abstractmethod
    def name(self) -> SemanticFeature:
        """Feature-enum member this bank detects."""

    @override
    def define(self, builder: tuple) -> tuple:
        return builder


class SegmentationBank(LanguageBank):
    """Word-level language segmentation: script for non-Latin, carrier plus
    frequency floors for Latin. Carrier is the query's dominant Latin language
    (ties to English); OOV/code/letterless tokens stay carrier, so a monolingual
    query is one span."""

    def __init__(
        self,
        candidate_languages: tuple[str, ...] = DEFAULT_CANDIDATES,
        carrier_floor: float = CARRIER_FLOOR,
        target_floor: float = TARGET_FLOOR,
    ) -> None:
        super().__init__()
        self._carrier_floor = carrier_floor
        self._target_floor = target_floor
        self._candidates = frozenset(candidate_languages)
        self._latin = tuple(
            lang for lang in candidate_languages
            if _SCRIPT_BY_LANG.get(lang) == "LATIN"
        )

    @property
    @override
    def name(self) -> SemanticFeature:
        return SemanticFeature.LANGUAGE_SEGMENTATION

    def _carrier(self, tokens: list[str]) -> str:
        zipf = _zipf()
        mass = dict.fromkeys(self._latin, 0.0)
        for token in tokens:
            if _token_script(token) == "LATIN":
                lower = token.lower()
                for lang in self._latin:
                    mass[lang] += zipf(lower, lang, wordlist="small")
        if not self._latin or not any(mass.values()):
            return "en" if "en" in self._latin else (self._latin[0] if self._latin else "en")
        return max(self._latin, key=lambda lang: (mass[lang], lang == "en"))

    def _classify(self, token: str, carrier: str) -> str:
        zipf = _zipf()
        if not _CODE_CHARS.isdisjoint(token):
            return carrier  # code / markup / identifier, not prose
        script = _token_script(token)
        if script is None:
            return carrier  # punctuation-only
        if script != "LATIN":
            lang = _SCRIPT_TO_LANG.get(script)
            return lang if lang in self._candidates else carrier
        lower = token.lower()
        carrier_freq = zipf(lower, carrier, wordlist="small")
        others = [lang for lang in self._latin if lang != carrier]
        if not others:
            return carrier
        best = max(others, key=lambda lang: zipf(lower, lang, wordlist="small"))
        best_freq = zipf(lower, best, wordlist="small")
        switched = carrier_freq < self._carrier_floor and best_freq >= self._target_floor
        return best if switched else carrier

    @override
    def compute(self, text: str) -> list[LanguageSpan]:
        spans_in = [(m.group(0), m.start(), m.end()) for m in _TOKEN.finditer(text)]
        if not spans_in:
            return []
        tokens = [tok for tok, _, _ in spans_in]
        carrier = self._carrier(tokens)
        langs = [self._classify(tok, carrier) for tok in tokens]
        out: list[LanguageSpan] = []
        i = 0
        while i < len(spans_in):
            j = i
            while j + 1 < len(spans_in) and langs[j + 1] == langs[i]:
                j += 1
            start, end = spans_in[i][1], spans_in[j][2]
            out.append(LanguageSpan(text[start:end], start, end, langs[i]))
            i = j + 1
        return out
