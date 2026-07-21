from typing import override

from edify import RegexBuilder

from query_taxonomy.core import AmbiguityTier
from query_taxonomy.logical.core import LogicalBank
from query_taxonomy.taxonomy import LogicalStructure


class OperatorSyntaxBank(LogicalBank):
    """Explicit boolean operators. Case-sensitive on purpose: lowercase
    and/or/not are ordinary function words, only the uppercase forms signal
    operator intent."""

    @property
    @override
    def name(self) -> LogicalStructure:
        return LogicalStructure.OPERATOR_SYNTAX

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder
            .word_boundary()
            .any_of().string("AND").string("NOT").string("OR").end()
            .word_boundary()
        )


TEMPORAL_PHRASES: tuple[str, ...] = (
    "today",
    "tonight",
    "yesterday",
    "tomorrow",
    "now",
    "currently",
    "recently",
    "latest",
    "right now",
    "last week",
    "last month",
    "last year",
    "next week",
    "next month",
    "next year",
    "this week",
    "this month",
    "this year",
)


class TemporalRelativeBank(LogicalBank):
    """Relative temporal expressions — the slice absolute identifier banks
    (DATETIME, BUSINESS_TEMPORAL) cannot express. Covers a closed vocabulary
    plus "N <unit>(s) ago". Layered feature: a GLiNER2 backstop at the same
    feature name lands with the Gliner2Bank wrapper (SPEC decision 16)."""

    @property
    @override
    def name(self) -> LogicalStructure:
        return LogicalStructure.TEMPORAL

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        chain = (
            builder.ignore_case().word_boundary().any_of()
            # the consuming "N <unit>(s) ago" branch goes first
            .group()
                .one_or_more().digit()
                .one_or_more().whitespace_char()
                .any_of()
                    .string("minute").string("second").string("month")
                    .string("hour").string("week").string("year").string("day")
                .end()
                .optional().char("s")
                .one_or_more().whitespace_char()
                .string("ago")
            .end()
        )
        for phrase in sorted(TEMPORAL_PHRASES, key=len, reverse=True):
            chain = chain.string(phrase)
        return chain.end().word_boundary()


_LETTER = RegexBuilder().any_of().range("a", "z").range("A", "Z").end()
_IDENTIFIER_START = (
    RegexBuilder().any_of().range("a", "z").range("A", "Z").char("_").end()
)

# One side of an equation: token, optionally chained by operators. Tokens
# keep ^ and . inside (mc^2, 3.14 stay one operand).
_EXPR_TOKEN = (
    RegexBuilder()
    .one_or_more()
    .any_of()
        .range("a", "z").range("A", "Z").range("0", "9")
        .char(".").char("^")
    .end()
)
_EXPRESSION = (
    RegexBuilder()
    .subexpression(_EXPR_TOKEN)
    .zero_or_more().group()
        .optional().whitespace_char()
        .any_of_chars("+-*/^")
        .optional().whitespace_char()
        .subexpression(_EXPR_TOKEN)
    .end()
)

# Operand of a symbolic run WITHOUT an `=` gate: a number or a lone letter
# (2, 3.14, x) — multi-letter words are excluded so "bed + breakfast"
# stays prose.
_RUN_OPERAND = (
    RegexBuilder()
    .any_of()
        .group()
            .one_or_more().digit()
            .optional().group().char(".").one_or_more().digit().end()
        .end()
        .group()
            .word_boundary().subexpression(_LETTER).word_boundary()
        .end()
    .end()
)


class CodeFragmentBank(LogicalBank):
    """Programming-language grammar, claimed by evidence tokens only
    (cue-claiming, SPEC d20): compound symbolic operators, `name(` call
    syntax, and SELECT gated on a FROM ahead. Symbolic operators are not
    attested search dialect — in a real query they mean embedded code.
    The full fragment's boundaries are never claimed."""

    @property
    @override
    def name(self) -> LogicalStructure:
        return LogicalStructure.CODE_FRAGMENT

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder.any_of()
            # longest compound first — alternation is first-match
            .group()
                .any_of()
                    .string("===").string("!=").string("<>").string("=>")
                    .string("->").string("::").string("&&").string("||")
                .end()
            .end()
            # call syntax `name(` — pluralization "file(s)" excluded
            .group()
                .word_boundary()
                .subexpression(_IDENTIFIER_START)
                .zero_or_more().word()
                .char("(")
                .assert_not_ahead().string("s)").end()
            .end()
            # keyword-gated bigram: claim SELECT alone, the gap and FROM
            # stay unclaimed (cue-claiming, not fragment segmentation)
            .group()
                .word_boundary().string("SELECT")
                .assert_ahead()
                    .between(1, 60).any_char()
                    .word_boundary().string("FROM").word_boundary()
                .end()
            .end()
            .end()
        )


class MathExpressionBank(LogicalBank):
    """Equation grammar (SPEC d20): `=` flanked by expressions ("E = mc^2",
    "x = 5") and operand-operator-operand runs ("2+2", "x^2 + y^2 < 25").
    Bare single-char =/+/-/</> are never claimed — operands are required on
    both sides. Runs exclude `-` and `/` as gates: "3-4 days" is a range
    and "24/7" an idiom, not math (both still count INSIDE an `=`-gated
    equation). Symbol-light math ("x squared plus y") is the deferred
    model layer."""

    @property
    @override
    def name(self) -> LogicalStructure:
        return LogicalStructure.MATH_EXPRESSION

    @property
    @override
    def ambiguity(self) -> AmbiguityTier:
        return AmbiguityTier.MODERATE

    @override
    def define(self, builder: RegexBuilder) -> RegexBuilder:
        return (
            builder.any_of()
            # equation branch first — it consumes the whole run. Guards
            # keep the `=` off code territory (==, ===, =>, <=, :=).
            .group()
                .word_boundary()
                .subexpression(_EXPRESSION)
                .optional().whitespace_char()
                .assert_not_behind().any_of_chars("=<>!:+-*/^").end()
                .char("=")
                .assert_not_ahead().any_of_chars("=<>").end()
                .optional().whitespace_char()
                .subexpression(_EXPRESSION)
            .end()
            # symbolic run without `=`: number / lone-letter operands only
            .group()
                .subexpression(_RUN_OPERAND)
                .one_or_more().group()
                    .optional().whitespace_char()
                    .any_of_chars("+*^<>")
                    .optional().whitespace_char()
                    .subexpression(_RUN_OPERAND)
                .end()
            .end()
            .end()
        )
