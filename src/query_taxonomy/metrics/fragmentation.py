"""tokenizer-engine stat banks: how far the query's words shatter under a
pinned WordPiece vocab. The tokenizer (bge-small-en-v1.5 is bert-base
WordPiece) is fetched once from the hub and cached; imported lazily so
regex-only extraction stays import-light."""

from abc import ABC, abstractmethod
from functools import lru_cache
from typing import TYPE_CHECKING, override

from query_taxonomy.core import Engine, FeatureGroup, FeatureStat, StatBank
from query_taxonomy.taxonomy import StatisticalMetric

if TYPE_CHECKING:
    from tokenizers import Tokenizer

DEFAULT_TOKENIZER = "bert-base-uncased"
"""The pinned WordPiece vocab — the tokenization bge-small-en-v1.5 uses."""


@lru_cache(maxsize=1)
def _tokenizer(model: str = DEFAULT_TOKENIZER) -> "Tokenizer":
    from tokenizers import Tokenizer

    return Tokenizer.from_pretrained(model)


class TokenizerBank(StatBank[str], ABC):
    """Stat bank over a pinned WordPiece tokenizer. The definition artifact is
    the model name (read, never built), like WordFreqBank's language."""

    engine = Engine.TOKENIZER

    def __init__(self, model: str = DEFAULT_TOKENIZER) -> None:
        super().__init__()
        self._model = self.define(model)

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

    def _pieces_per_word(self, text: str) -> list[int]:
        """Subword-piece count per word, grouped by the tokenizer's OWN word
        boundaries (`word_ids`), not `text.split()`: the pre-tokenizer makes
        punctuation and hyphens their own words, so `learning?` and
        `state-of-the-art` no longer read as one badly-shattered token."""
        encoding = _tokenizer(self._model).encode(text, add_special_tokens=False)
        counts: dict[int, int] = {}
        for word_id in encoding.word_ids:
            if word_id is not None:
                counts[word_id] = counts.get(word_id, 0) + 1
        return list(counts.values())


class FragmentationBank(TokenizerBank):
    """How badly do the query's words shatter into subword pieces?
    mean_pieces_per_word (overall) and max_pieces_per_word (the single
    most-shattered token, the rare-technical-term anchor). One piece = a clean
    in-vocabulary word; several = a rare or misspelled one. Emits nothing for
    text with no words — an absent feature, not a misleading zero (a real
    floor is 1.0, never 0)."""

    @property
    @override
    def name(self) -> StatisticalMetric:
        return StatisticalMetric.SUBWORD_FRAGMENTATION

    @override
    def compute(self, text: str) -> list[FeatureStat]:
        pieces = self._pieces_per_word(text)
        if not pieces:
            return []
        return [
            FeatureStat("mean_pieces_per_word", sum(pieces) / len(pieces)),
            FeatureStat("max_pieces_per_word", float(max(pieces))),
        ]
