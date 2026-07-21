"""Presentation layer over CorpusFeatures (SPEC d27).

`CorpusReport` renders the human report (`text()`) and emits chart-ready
rollup data (`domain_rollup()`, `domain_query_share()`). The rollup is a
view: profiles JSON and every audit surface stay bank-level. Charts are
drawn by the consumer — this module is stdlib-only by design.
"""

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from query_taxonomy.banks import domain_by_type
from query_taxonomy.features import CorpusFeatures, SpanProfile
from query_taxonomy.taxonomy import Domain, FeatureGroup

_LIKE_SUFFIX = "_like"
_GENERAL_REMAINDER = "general·other"


def _is_shape_guess(type_name: str) -> bool:
    """Assumptive banks self-mark at the emit boundary (`-Like` doctrine)."""
    return type_name.endswith(_LIKE_SUFFIX)


def _section_title(name: str) -> str:
    """Consistent section header across span groups and stats."""
    return name.replace("_", " ").upper()


@dataclass(frozen=True)
class DomainSlice:
    """One donut slice: inner ring sized by `spans`, outer ring split
    `certified` vs `assumed` (spans from -like banks — shape guesses).
    `group` names the taxonomy group the slice came from — identifier
    slices carry `structured_identifiers`, marker/logical slices carry
    their own group, so the chart layer can annotate mixed rollups."""

    label: str
    spans: int
    certified: int
    assumed: int
    queries: int
    group: FeatureGroup = FeatureGroup.STRUCTURED_IDENTIFIERS


class CorpusReport:
    """Human-facing view of one CorpusFeatures.

    SLICE_BUDGET: pies stop being readable past ~10 slices, so the
    `general` explosion shrinks from GENERAL_TOP member banks downward
    until the budget holds (d27 arch-validator guardrail). Consumers that
    still exceed the budget should fall back to stacked bars.
    """

    SLICE_BUDGET = 10
    GENERAL_TOP = 3

    def __init__(self, features: CorpusFeatures) -> None:
        self._features = features

    def __str__(self) -> str:
        return self.text()

    # -- chart-ready data ------------------------------------------------

    def domain_rollup(self) -> list[DomainSlice]:
        """Donut slices: domains by span mass, descending; `general`
        exploded in place into its top member banks + a remainder.

        Deliberately finer-grained than `domain_query_share()` — the
        donut is a part-of-whole chart where a raw `general` slice
        carries no signal, while the companion bar answers "how often
        does this domain show up at all?" and keeps the aggregate
        reading intact.
        """
        return [piece for piece, _ in self._rollup_pairs()]

    def feature_rollup(self) -> list[DomainSlice]:
        """All span-emitting features on one donut. Identifier banks are
        rolled up by Domain (per d27); each sentence_markers /
        logical_structures bank is its own slice (there are 6 and 2
        respectively — already at reasonable granularity). Sorted by
        span mass descending; the tiny-slice → legend rule in the chart
        layer keeps labels readable."""
        identifier_slices = [piece for piece, _ in self._rollup_pairs()]
        other_slices = self._non_identifier_slices()
        return sorted(
            identifier_slices + other_slices,
            key=lambda s: (-s.spans, s.label),
        )

    def _non_identifier_slices(self) -> list[DomainSlice]:
        """One slice per bank type outside structured_identifiers — banks
        in these groups are already the intended granularity."""
        slices: list[DomainSlice] = []
        for group, by_type in self._features.span_profiles.items():
            if group is FeatureGroup.STRUCTURED_IDENTIFIERS:
                continue
            for profile in by_type.values():
                slices.append(self._slice(profile.type, [profile], group))
        return slices

    def stat_means(self) -> dict[str, float]:
        """Per-stat corpus mean across every statistical-metrics bank.
        Flat by name because the stat vocabulary is globally unique
        (`length_words`, `natural_language_share`, ...) after d26."""
        means: dict[str, float] = {}
        for by_type in self._features.stat_profiles.values():
            for profile in by_type.values():
                for name, agg in profile.aggregates.items():
                    means[name] = float(agg["mean"])
        return means

    def domain_query_share(self) -> dict[str, float]:
        """Per domain: share of queries touching it (companion bar; NOT
        parts-of-a-whole — one query may touch several domains).

        Aggregates `general` even though `domain_rollup()` explodes it:
        "14% of queries carry a general identifier" is a legitimate
        rollup reading; the two methods answer different questions and
        render at different granularities on purpose.
        """
        total = len(self._features.queries)
        if total == 0:
            return {}
        shares = {
            label: len(self._doc_ids(profiles)) / total
            for label, profiles in self._profiles_by_domain().items()
        }
        return dict(sorted(shares.items(), key=lambda kv: (-kv[1], kv[0])))

    # -- human report ------------------------------------------------------

    def text(self) -> str:
        lines = [f"{len(self._features.queries)} queries profiled."]
        lines.extend(self._span_sections())
        lines.extend(self._stat_section())
        return "\n".join(lines)

    # -- rollup mechanics --------------------------------------------------

    def _identifier_profiles(self) -> list[SpanProfile]:
        by_type = self._features.span_profiles.get(
            FeatureGroup.STRUCTURED_IDENTIFIERS, {}
        )
        return list(by_type.values())

    def _profiles_by_domain(self) -> dict[str, list[SpanProfile]]:
        domains = domain_by_type()
        grouped: dict[str, list[SpanProfile]] = {}
        for profile in self._identifier_profiles():
            domain = domains.get(profile.type)
            label = domain.value if domain else "other"
            grouped.setdefault(label, []).append(profile)
        return grouped

    def _rollup_pairs(self) -> list[tuple[DomainSlice, list[SpanProfile]]]:
        ordered = sorted(
            self._profiles_by_domain().items(),
            key=lambda kv: (-self._mass(kv[1]), kv[0]),
        )
        top = self._general_top(ordered)
        return [
            pair
            for label, profiles in ordered
            for pair in self._domain_pairs(label, profiles, top)
        ]

    def _general_top(
        self, ordered: list[tuple[str, list[SpanProfile]]]
    ) -> int:
        """How deep to explode `general` so the whole donut stays under
        SLICE_BUDGET. -1 accounts for the remainder slice if it will
        exist; clamped to [1, GENERAL_TOP]."""
        non_general = sum(
            1 for label, _ in ordered if label != Domain.GENERAL.value
        )
        slack = self.SLICE_BUDGET - non_general
        return max(1, min(self.GENERAL_TOP, slack - 1))

    def _domain_pairs(
        self, label: str, profiles: list[SpanProfile], top: int
    ) -> list[tuple[DomainSlice, list[SpanProfile]]]:
        if label == Domain.GENERAL.value:
            return self._exploded_general(profiles, top)
        return [(self._slice(label, profiles), profiles)]

    def _exploded_general(
        self, profiles: list[SpanProfile], top: int
    ) -> list[tuple[DomainSlice, list[SpanProfile]]]:
        by_mass = sorted(
            profiles, key=lambda p: (-self._mass([p]), p.type)
        )
        pairs = [
            (self._slice(profile.type, [profile]), [profile])
            for profile in by_mass[:top]
        ]
        remainder = by_mass[top:]
        if remainder:
            pairs.append(
                (self._slice(_GENERAL_REMAINDER, remainder), remainder)
            )
        return pairs

    def _slice(
        self,
        label: str,
        profiles: Iterable[SpanProfile],
        group: FeatureGroup = FeatureGroup.STRUCTURED_IDENTIFIERS,
    ) -> DomainSlice:
        materialized = list(profiles)
        certified = self._mass(
            p for p in materialized if not _is_shape_guess(p.type)
        )
        assumed = self._mass(
            p for p in materialized if _is_shape_guess(p.type)
        )
        return DomainSlice(
            label=label,
            spans=certified + assumed,
            certified=certified,
            assumed=assumed,
            queries=len(self._doc_ids(materialized)),
            group=group,
        )

    @staticmethod
    def _mass(profiles: Iterable[SpanProfile]) -> int:
        return sum(
            len(spans)
            for profile in profiles
            for spans in profile.spans.values()
        )

    @staticmethod
    def _doc_ids(profiles: Iterable[SpanProfile]) -> set[str]:
        return {
            doc_id for profile in profiles for doc_id in profile.spans
        }

    # -- text sections -----------------------------------------------------

    def _span_sections(self) -> list[str]:
        total = len(self._features.queries)
        sections: list[str] = []
        groups = sorted(
            self._features.span_profiles.items(),
            key=lambda kv: -len(self._doc_ids(kv[1].values())),
        )
        for group, by_type in groups:
            profiles = list(by_type.values())
            tagged = len(self._doc_ids(profiles))
            share = 100 * tagged / total if total else 0.0
            sections.append(
                f"\n{_section_title(group.value)} — {tagged} of {total}"
                f" queries ({share:.1f}%) carry at least one."
            )
            if group is FeatureGroup.STRUCTURED_IDENTIFIERS:
                sections.extend(self._identifier_table())
            else:
                sections.extend(self._type_lines(profiles))
        return sections

    def _identifier_table(self) -> list[str]:
        pairs = self._rollup_pairs()
        if not pairs:
            return []
        pad = max(len(piece.label) for piece, _ in pairs)
        lines = [
            f"  {'':<{pad}}  spans  certified  shape-guesses  queries"
            "  top forms"
        ]
        for piece, profiles in pairs:
            lines.append(
                f"  {piece.label:<{pad}}  {piece.spans:5d}  "
                f"{piece.certified:9d}  {piece.assumed:13d}  "
                f"{piece.queries:7d}  {self._top_forms(profiles)}"
            )
        lines.append(
            "  (shape-guesses = spans from -like banks: the label is a"
            " pattern guess, not a certified match)"
        )
        return lines

    def _type_lines(self, profiles: Iterable[SpanProfile]) -> list[str]:
        materialized = sorted(
            profiles, key=lambda p: (-len(p.spans), p.type)
        )
        pad = max((len(p.type) for p in materialized), default=0)
        return [
            f"  {profile.type:<{pad}}  queries={len(profile.spans):3d}"
            f"  matches={self._mass([profile]):3d}"
            f"  top: {self._top_forms([profile])}"
            for profile in materialized
        ]

    @staticmethod
    def _top_forms(
        profiles: Iterable[SpanProfile], limit: int = 3
    ) -> str:
        dfs: Counter[str] = Counter()
        for profile in profiles:
            dfs.update(profile.dfs)
        top = sorted(dfs.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
        return ", ".join(f"{text} ({df})" for text, df in top)

    def _stat_section(self) -> list[str]:
        by_group = self._features.stat_profiles
        if not any(by_group.values()):
            return []
        total = len(self._features.queries)
        title = _section_title(FeatureGroup.STATISTICAL_METRICS.value)
        lines = [f"\n{title} — averages over all {total} queries."]
        aggregates: dict[str, dict[str, float]] = {}
        for by_type in by_group.values():
            for profile in by_type.values():
                aggregates.update(profile.aggregates)
        pad = max(len(name) for name in aggregates)
        for name, agg in sorted(aggregates.items()):
            lines.append(
                f"  {name:<{pad}}  mean={agg['mean']:.3f}"
                f"  min={agg['min']:.3f}  max={agg['max']:.3f}"
            )
        return lines
