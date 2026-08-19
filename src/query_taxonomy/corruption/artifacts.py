"""REGEX-method corruption detectors: encoding junk, truncation markers, and
copy-paste residue. Heuristic by nature — validated against natural-corpus
false-positive rates (the census), not asserted. High code points are written
as \\u escapes so the source carries no invisible control characters."""

from typing import override

from edify import RegexBuilder

from query_taxonomy.core import AmbiguityTier
from query_taxonomy.corruption.core import CorruptionRegexBank
from query_taxonomy.taxonomy import CorruptionKind


class EncodingArtifactBank(CorruptionRegexBank):
    """Mojibake and replacement characters: U+FFFD, `Ã` followed by a Latin-1
    continuation char (UTF-8 mis-decoded as Latin-1 — café -> cafÃ©), and the
    `â€` smart-quote lead (don't -> donâ€™t). Ignores word boundaries."""

    @property
    @override
    def name(self) -> CorruptionKind:
        return CorruptionKind.ENCODING_ARTIFACT

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder.any_of()
            .char("\ufffd")  # U+FFFD replacement char
            .group().char("\u00c3").range("\u0080", "\u00bf").end()  # Ã + Latin-1 cont.
            .string("\u00e2\u20ac")  # â€ smart-quote lead
            .end()
        )


class TruncationBank(CorruptionRegexBank):
    """A query cut off at the end: a trailing ellipsis (… or ...) glued to the
    word fragment it cut, claimed together so the fragment is not re-counted
    as a typo. The fragment is REQUIRED — a bare ellipsis carries no evidence
    of a cut and was 96% of bright-pony's hits, where it is a code template's
    placeholder body; a complete word before it ("well...") still fires, and
    gating that on vocabulary is the open precision upgrade (see TODOS)."""

    @property
    @override
    def name(self) -> CorruptionKind:
        return CorruptionKind.TRUNCATION

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder.one_or_more().word()
            .any_of().char("\u2026").string("...").end()
            .end_of_input()
        )


HTML_TAGS: tuple[str, ...] = (
    "blockquote", "button", "strong", "script", "style", "table", "span",
    "code", "pre", "div", "img", "ul", "ol", "li", "br", "hr", "em", "h1",
    "h2", "h3", "h4", "h5", "h6", "td", "tr", "th", "b", "i", "p", "a",
)
"""A closed list of real HTML tag names — the discriminator that separates
paste (<div>) from a generic type (<int>) or a comparison (a < b): both are
`<word>`-shaped, so only a known tag name can tell them apart."""


class PasteResidueBank(CorruptionRegexBank):
    """Copy-paste leftovers: real HTML tags (<div>, </a href=...>) and
    DETACHED numeric citation brackets ([12], not array[0]). The closed tag
    list rejects generics/comparisons, and the citation lookbehind rejects
    array indexing — both rampant in code queries."""

    @property
    @override
    def name(self) -> CorruptionKind:
        return CorruptionKind.PASTE_RESIDUE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        tag_name = builder.any_of().group().char("<").optional().char("/").any_of()
        for tag in sorted(HTML_TAGS, key=len, reverse=True):  # longest-first
            tag_name = tag_name.string(tag)
        return (
            tag_name.end().word_boundary()
            .zero_or_more().anything_but_chars(">").char(">").end()
            .group()
            .assert_not_behind().word().end()  # detached [12], not array[0]
            .char("[").one_or_more().digit().char("]").end()
            .end()
        )
