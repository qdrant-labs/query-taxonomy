from abc import ABC, abstractmethod

from query_taxonomy.core import AmbiguityTier, FeatureGroup, RegexBank
from query_taxonomy.taxonomy import CorruptionKind


class CorruptionRegexBank(RegexBank, ABC):
    """One regex bank per REGEX-method CorruptionKind (artifacts). Same claim
    mechanics as the other span groups; corruption spans never compete with
    identifier or marker spans for text ranges."""

    @property
    def group(self) -> FeatureGroup:
        return FeatureGroup.CORRUPTION

    @property
    @abstractmethod
    def ambiguity(self) -> AmbiguityTier:
        """Re-abstracted: every span bank declares its tier explicitly."""

    @property
    @abstractmethod
    def name(self) -> CorruptionKind:
        """Feature-enum member this bank detects."""
