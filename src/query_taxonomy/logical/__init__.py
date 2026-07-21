from query_taxonomy.logical.core import LogicalBank
from query_taxonomy.logical.general import (
    CodeFragmentBank,
    MathExpressionBank,
    OperatorSyntaxBank,
    TemporalRelativeBank,
)

# All MODERATE, so tuple order = claim order within the tier. CodeFragment
# MUST precede MathExpression: ===/=>/->/!= are code evidence and have to
# be claimed before the equation grammar can touch their =/</> neighborhoods.
LOGICAL_BANKS: tuple[type[LogicalBank], ...] = (
    OperatorSyntaxBank,
    TemporalRelativeBank,
    CodeFragmentBank,
    MathExpressionBank,
)

__all__ = [
    "LOGICAL_BANKS",
    "CodeFragmentBank",
    "LogicalBank",
    "MathExpressionBank",
    "OperatorSyntaxBank",
    "TemporalRelativeBank",
]
