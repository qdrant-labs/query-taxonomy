from typing import override

from edify import RegexBuilder

from query_taxonomy.banks.core import (
    ALNUM_OR_DOT,
    HEX_DIGIT,
    IdentifierBank,
    UPPER_OR_UNDERSCORE,
)
from query_taxonomy.core import AmbiguityTier
from query_taxonomy.taxonomy import Domain, StructuralIdentifier


class CVEBank(IdentifierBank):
    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.CVE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return builder.string("CVE-").exactly(4).digit().char("-").at_least(4).digit()


_VERSION_KEYWORDS = ("version", "revision", "release", "build", "ver")

# semver forbids leading zeros — 04.06.010 is a statute citation, not a version
_SEMVER_PART = (
    RegexBuilder()
    .any_of()
        .group().char("0").end()
        .group().range("1", "9").zero_or_more().digit().end()
    .end()
)

# pre-release tags start with a letter (-beta.1, -rc2); -3.4 is a numeric range
_PRERELEASE = (
    RegexBuilder().char("-").range("a", "z").zero_or_more().subexpression(ALNUM_OR_DOT)
)


class VersionStringBank(IdentifierBank):
    """Three forms: canonical semver 1.2.3, v-prefixed v1.0 / v2.1.0-beta.1,
    and keyword-gated "version 1.4" / "Release 22".

    A bare two-part decimal is NOT a version: 0.5 is a p-value, a dose or an
    effect size, and nothing in the shape says otherwise — those were 92% of
    this bank's claims before the repair. Knowingly ceded with it: zero-padded
    parts (22.04.1, v-less 1.02), product-gated forms without a version
    keyword (Python 3.11), bare "v2" (ignore_case would eat clinical visit
    V2), and a three-part run inside a longer dotted run (1.2.3.4 — the IP
    bank owns those). Residual FPs the shape cannot rule out: US state statute
    citations (RCW 10.46.190), dotted dates (10.10.2020) and dotted phone
    numbers (800.729.4732) — a digit cap would cost real calendar versions
    (2024.5.3) and build numbers (10.0.1031), so they stay."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.VERSION_STRING

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        keywords = RegexBuilder().any_of()
        for keyword in sorted(_VERSION_KEYWORDS, key=len, reverse=True):
            keywords = keywords.string(keyword)
        keywords = keywords.end()

        return (
            builder
            .ignore_case()
            .word_boundary()
            .any_of()
                # keyword-gated: the gate is the evidence, so one part is enough
                .group()
                    .subexpression(keywords)
                    .optional().char(".")
                    # a literal space, not \s: the generator samples this
                    # pattern, and \s emits form feeds into surfaces
                    .optional().char(" ")
                    .one_or_more().digit()
                    .between(0, 3).group().char(".").one_or_more().digit().end()
                .end()
                .group()
                    .char("v")
                    .one_or_more().digit()
                    .between(1, 3).group().char(".").one_or_more().digit().end()
                    .optional().group().subexpression(_PRERELEASE).end()
                .end()
                # bare semver: exactly three parts, not a slice of a longer run
                .group()
                    .assert_not_behind().any_of().digit().char(".").end().end()
                    .subexpression(_SEMVER_PART).char(".")
                    .subexpression(_SEMVER_PART).char(".")
                    .subexpression(_SEMVER_PART)
                    .optional().group().subexpression(_PRERELEASE).end()
                    .assert_not_ahead().char(".").digit().end()
                .end()
            .end()
            .word_boundary()
        )


class FilePathBank(IdentifierBank):
    """Absolute unix paths (>=2 segments) and Windows drive paths."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.FILE_PATH

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                # unix: at least two /segment parts, so "and/or" and "/16" stay out
                .group()
                    .at_least(2).group()
                        .char("/")
                        .one_or_more().any_of()
                            .range("a", "z").range("A", "Z").range("0", "9")
                            .any_of_chars("._-")
                        .end()
                    .end()
                .end()
                # windows: C:\... up to whitespace
                .group()
                    .range("A", "Z")
                    .char(":")
                    .char("\\")
                    .one_or_more().non_whitespace_char()
                .end()
            .end()
        )


class UUIDBank(IdentifierBank):
    """Canonical 8-4-4-4-12 UUIDs, or bare 32-64 char hex digests (MD5/SHA-1/SHA-256);
    single-repeated-char runs (aaaa…) are padding artifacts, not digests, and rejected."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.UUID

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .assert_not_ahead()
                .capture().any_char().end()
                .zero_or_more().back_reference(1)
                .word_boundary()
            .end()
            .any_of()
                .group()
                    .exactly(8).subexpression(HEX_DIGIT).char("-")
                    .exactly(4).subexpression(HEX_DIGIT).char("-")
                    .exactly(4).subexpression(HEX_DIGIT).char("-")
                    .exactly(4).subexpression(HEX_DIGIT).char("-")
                    .exactly(12).subexpression(HEX_DIGIT)
                .end()
                .group().between(32, 64).subexpression(HEX_DIGIT).end()
            .end()
            .word_boundary()
        )


class URIBank(IdentifierBank):
    """scheme://... URIs (http, s3, file, ...). Claims full URLs before HostPortBank."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.URI

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .range("a", "z")
            .zero_or_more().any_of()
                .range("a", "z").range("0", "9").char("+").char(".").char("-")
            .end()
            .string("://")
            .one_or_more().non_whitespace_char()
        )


class ErrorCodeLikeBank(IdentifierBank):
    """Node ERR_* symbols and POSIX errno names. Assumptive (MODERATE, SPEC
    d22): the E-prefixed shape fires on medical acronyms (EPIC, ECMO) in
    scientific text. Known FP: the bare word ERROR."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.ERROR_CODE_LIKE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group().string("ERR_").one_or_more().subexpression(UPPER_OR_UNDERSCORE).end()
                .group().char("E").between(3, 9).range("A", "Z").end()
            .end()
            .word_boundary()
        )


class EnvVarBank(IdentifierBank):
    """$UPPER_SNAKE env vars and --long-flags. The $ branch requires an
    underscore or >=6 chars so short cashtags ($AAPL) fall to STOCK_TICKER_LIKE;
    the cost is that $HOME/$PATH-style short env vars are ceded too."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.ENV_VAR

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                # $WITH_UNDERSCORE first: alternation is first-match, and this
                # branch is the one that consumes the full underscored name
                .group()
                    .char("$")
                    .subexpression(UPPER_OR_UNDERSCORE)
                    .zero_or_more().any_of().range("A", "Z").range("0", "9").end()
                    .char("_")
                    .zero_or_more().any_of().range("A", "Z").range("0", "9").char("_").end()
                .end()
                # $LONGNAME: 6+ chars, no underscore needed
                .group()
                    .char("$")
                    .range("A", "Z")
                    .at_least(5).any_of().range("A", "Z").range("0", "9").end()
                .end()
                .group()
                    .string("--")
                    .range("a", "z")
                    .zero_or_more().any_of().range("a", "z").range("0", "9").char("-").end()
                .end()
            .end()
        )


class HexColorBank(IdentifierBank):
    """#RRGGBB and #RGB CSS colors."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.HEX_COLOR

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .char("#")
            .any_of()
                .group().exactly(6).subexpression(HEX_DIGIT).end()
                .group().exactly(3).subexpression(HEX_DIGIT).end()
            .end()
            .word_boundary()
        )


class CodeIdentifierBank(IdentifierBank):
    """camelCase, snake_case (lower/UPPER), dotted.paths (segments >= 2 chars,
    so `e.g` and `i.e` stay out). Known FP: prose camelCase brands (iPhone, eBay)."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.CODE_IDENTIFIER

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.AMBIGUOUS

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                # dotted.path — each segment at least 2 chars
                .group()
                    .range("a", "z")
                    .one_or_more().any_of().range("a", "z").range("0", "9").char("_").end()
                    .one_or_more().group()
                        .char(".")
                        .range("a", "z")
                        .one_or_more().any_of().range("a", "z").range("0", "9").char("_").end()
                    .end()
                .end()
                # camelCase
                .group()
                    .one_or_more().range("a", "z")
                    .one_or_more().group()
                        .range("A", "Z")
                        .zero_or_more().any_of().range("a", "z").range("0", "9").end()
                    .end()
                .end()
                # snake_case (lower)
                .group()
                    .range("a", "z")
                    .zero_or_more().any_of().range("a", "z").range("0", "9").end()
                    .one_or_more().group()
                        .char("_")
                        .one_or_more().any_of().range("a", "z").range("0", "9").end()
                    .end()
                .end()
                # SNAKE_CASE (upper)
                .group()
                    .range("A", "Z")
                    .zero_or_more().any_of().range("A", "Z").range("0", "9").end()
                    .one_or_more().group()
                        .char("_")
                        .one_or_more().any_of().range("A", "Z").range("0", "9").end()
                    .end()
                .end()
            .end()
            .word_boundary()
        )


class PackageCoordinateBank(IdentifierBank):
    """npm scoped packages (@scope/name) and Maven-style group:artifact."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.PACKAGE_COORDINATE

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .any_of()
                # @scope/name
                .group()
                    .char("@")
                    .one_or_more().any_of().range("a", "z").range("0", "9").char("-").end()
                    .char("/")
                    .one_or_more().any_of().range("a", "z").range("0", "9").any_of_chars("._-").end()
                .end()
                # group:artifact (letters after the colon, so host:port stays out)
                .group()
                    .word_boundary()
                    .range("a", "z")
                    .zero_or_more().any_of().range("a", "z").range("0", "9").any_of_chars("._-").end()
                    .char(":")
                    .range("a", "z")
                    .zero_or_more().any_of().range("a", "z").range("0", "9").any_of_chars("._-").end()
                    .word_boundary()
                .end()
            .end()
        )


_UNITS = (
    "GHz", "MHz", "kHz", "mAh", "fps", "dpi",
    "GB", "MB", "KB", "TB", "PB", "Hz",
    "mm", "cm", "km", "kg", "mg", "ml", "nm", "ms", "px",
    "kW", "mV", "mA", "W", "V",
)


class ValueWithUnitBank(IdentifierBank):
    """Quantity + whitelisted unit, optional space: 16GB, 3.5mm, 240 Hz."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.VALUE_WITH_UNIT

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        units = RegexBuilder().any_of()
        for unit in _UNITS:
            units = units.string(unit)
        units = units.end()

        return (
            builder
            .word_boundary()
            .one_or_more().digit()
            .optional().group().char(".").one_or_more().digit().end()
            .optional().whitespace_char()
            .subexpression(units)
            .word_boundary()
        )


class StandardsCitationBank(IdentifierBank):
    """RFC / ISO(-IEC) / IEEE citations: RFC 9110, ISO/IEC 27001, IEEE 802.11ax."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.STANDARDS_CITATION

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.RIGID

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .string("RFC")
                    .optional().whitespace_char()
                    .between(3, 5).digit()
                .end()
                .group()
                    .string("ISO")
                    .optional().group().string("/IEC").end()
                    .optional().whitespace_char()
                    .between(4, 5).digit()
                    .optional().group().char("-").one_or_more().digit().end()
                .end()
                .group()
                    .string("IEEE")
                    .optional().whitespace_char()
                    .between(3, 4).digit()
                    .optional().group().char(".").one_or_more().digit().end()
                    .between(0, 2).range("a", "z")
                .end()
            .end()
            .word_boundary()
        )


class CryptoAddressBank(IdentifierBank):
    """bech32 (bc1...), ETH (0x + 40 hex), legacy base58 BTC. The base58 form
    can collide with random long alnum tokens — hence MODERATE, not RIGID."""

    @property
    @override
    def name(self) -> StructuralIdentifier:
        return StructuralIdentifier.CRYPTO_ADDRESS

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @property
    @override
    def domain(self) -> Domain:
        return Domain.TECH

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of()
                .group()
                    .string("bc1")
                    .between(8, 87).any_of().range("a", "z").range("0", "9").end()
                .end()
                .group()
                    .string("0x")
                    .exactly(40).subexpression(HEX_DIGIT)
                .end()
                # legacy base58: [13] + 25-34 chars, no 0/O/I/l
                .group()
                    .any_of_chars("13")
                    .between(25, 34).any_of()
                        .range("1", "9").range("A", "H").range("J", "N")
                        .range("P", "Z").range("a", "k").range("m", "z")
                    .end()
                .end()
            .end()
            .word_boundary()
        )
