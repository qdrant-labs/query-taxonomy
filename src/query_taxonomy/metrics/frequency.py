"""wordfreq-engine stat banks: global term rarity from the pinned 'en'
frequency table. wordfreq ships its data in-package (no separate download),
imported lazily so regex-only extraction stays import-light."""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING, override

from query_taxonomy.core import Engine, FeatureGroup, FeatureStat, StatBank
from query_taxonomy.taxonomy import StatisticalMetric

if TYPE_CHECKING:
    from collections.abc import Callable

RARE_ZIPF_MAX = 3.0
"""Zipf cutoff below which a known term counts as rare — a placeholder pending
calibration against the query-set Zipf distribution (see TODOS). Zipf 3.0 is
~one occurrence per million words."""


@lru_cache(maxsize=1)
def _wordfreq() -> "tuple[Callable[..., float], Callable[..., list[str]]]":
    from wordfreq import tokenize, zipf_frequency

    return zipf_frequency, tokenize


class WordFreqBank(StatBank[str], ABC):
    """Stat bank over the pinned wordfreq table. The definition artifact is the
    language code (read, never built), like CorpusRelativeBank's index."""

    engine = Engine.WORDFREQ

    def __init__(self, language: str = "en") -> None:
        super().__init__()
        self._language = self.define(language)

    @property
    @override
    def group(self) -> FeatureGroup:
        return FeatureGroup.STATISTICAL_METRICS

    @property
    @abstractmethod
    def name(self) -> StatisticalMetric:
        """Feature-enum member this bank detects."""

    @override
    def define(self, builder: str) -> str:
        return builder

    def _known_zipfs(self, text: str) -> list[float]:
        """Zipf value of each query token present in the table (absent tokens
        belong to unknown_token_rate, not rarity)."""
        zipf, tokenize = _wordfreq()
        values = (zipf(token, self._language) for token in tokenize(text, self._language))
        return [value for value in values if value > 0.0]


class RarityBank(WordFreqBank):
    """How globally rare are the query's terms? min_zipf (rarest term, the
    sparse exact-match anchor), mean_zipf (overall commonness), rare_share
    (fraction below the rarity cutoff). Global frequency, not this corpus.
    Emits nothing when no token is in the table (empty or all-OOV text): the
    rarity of no known word is undefined, and a 0.0 sentinel would read as
    maximally rare while poisoning the very distribution RARE_ZIPF_MAX is
    calibrated against. Digit strings get a synthetic wordfreq frequency, so
    numeric tokens count as known and never fall to the OOV side."""

    def __init__(
        self, rare_zipf_max: float = RARE_ZIPF_MAX, language: str = "en"
    ) -> None:
        super().__init__(language)
        self._rare_zipf_max = rare_zipf_max

    @property
    @override
    def name(self) -> StatisticalMetric:
        return StatisticalMetric.TERM_RARITY

    @override
    def compute(self, text: str) -> list[FeatureStat]:
        zipfs = self._known_zipfs(text)
        if not zipfs:
            return []
        rare = sum(value < self._rare_zipf_max for value in zipfs)
        return [
            FeatureStat("min_zipf", min(zipfs)),
            FeatureStat("mean_zipf", sum(zipfs) / len(zipfs)),
            FeatureStat("rare_share", rare / len(zipfs)),
        ]
