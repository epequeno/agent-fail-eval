"""agent-fail-eval: coding-agent failure evaluation toolkit."""

from agent_fail_eval.evaluator import (
    CompactionLossEvaluator,
    CompactionLossReport,
    Fact,
    FactCategory,
    FactStatus,
    Strategy,
)
from agent_fail_eval.models import CompactionBoundary, ParsedSession, SessionRecord
from agent_fail_eval.parser import SessionLoader

__all__ = [
    "CompactionBoundary",
    "CompactionLossEvaluator",
    "CompactionLossReport",
    "Fact",
    "FactCategory",
    "FactStatus",
    "ParsedSession",
    "SessionLoader",
    "SessionRecord",
    "Strategy",
]

