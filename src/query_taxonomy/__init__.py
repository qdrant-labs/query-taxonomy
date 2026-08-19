from typing import Any

from query_taxonomy.banks import BANKS
from query_taxonomy.core import Engine, GeneralBank
from query_taxonomy.corruption import CORRUPTION_BANKS
from query_taxonomy.logical import LOGICAL_BANKS
from query_taxonomy.markers import MARKER_BANKS
from query_taxonomy.metrics import METRIC_BANKS
from query_taxonomy.metrics.fragmentation import FragmentationBank
from query_taxonomy.metrics.frequency import RARE_ZIPF_MAX, RarityBank
from query_taxonomy.metrics.pos import (
    CoordinationBank,
    MorphologyBank,
    NaturalLanguageSignalBank,
    SyntacticDepthBank,
)
from query_taxonomy.semantical import SEMANTICAL_BANKS
from query_taxonomy.taxonomy import FeatureGroup


# GeneralBank is the common root: RegexBank, StatBank, and Gliner2Bank are
# siblings under it, so listing only two would miss the GLiNER family and
# the spaCy StatBank["Language"] specialization.
Bank = GeneralBank[Any, Any]
BankTypes = type[Bank]
# A registry entry is a bank type, or a (type, kwargs) pair for banks whose
# calibrated constants are injected at construction. Engine still reads off the
# type BEFORE instantiation, so regex-only stays import-light.
BankSpec = BankTypes | tuple[BankTypes, dict[str, Any]]


def split_bank(spec: BankSpec) -> tuple[BankTypes, dict[str, Any]]:
    """Normalize a registry entry to (type, kwargs) so callers instantiate uniformly."""
    return spec if isinstance(spec, tuple) else (spec, {})

# ALL banks of every engine live here — model-engine banks import their
# heavy dependencies lazily (at instantiation), so this registry stays
# import-light. Selection happens in FeatureExtractor via the `engines`
# filter, BEFORE instantiation.
#
# Must be defined before any `features` re-export: features.py imports this
# dict from the package, so it has to exist by the time that module loads.
FEATURE_BANKS = {
    FeatureGroup.STRUCTURED_IDENTIFIERS: BANKS,
    FeatureGroup.SENTENCE_MARKERS: MARKER_BANKS,
    FeatureGroup.LOGICAL_STRUCTURES: LOGICAL_BANKS ,
    FeatureGroup.STATISTICAL_METRICS: METRIC_BANKS
    + (
        NaturalLanguageSignalBank,
        MorphologyBank,
        SyntacticDepthBank,
        CoordinationBank,
        (RarityBank, {"rare_zipf_max": RARE_ZIPF_MAX}),
        FragmentationBank,
    ),
    FeatureGroup.CORRUPTION: CORRUPTION_BANKS,
    FeatureGroup.SEMANTICAL: SEMANTICAL_BANKS,
}

__all__ = ["FEATURE_BANKS", "Bank", "BankSpec", "BankTypes", "Engine", "split_bank"]
