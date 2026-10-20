"""The answerability gate: ordered, named rules over explicit signals. The first rule that matches decides.

Every rule is one line in the README and appears in the answer as `confidence.rule`, so any decision can
be traced to exactly one reason. The gate runs before any generator (extractive or LLM).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class Signals:
    empty: bool = False
    account_intent: str | None = None
    topical: bool = True  # the question names something this KB covers specifically
    score: float | None = None  # the gate's score of the primary: the verifier's logit (or the ranker's without it)
    floor: float = 0.0  # answer threshold for the primary's unit type (prose / table row)
    high: float = 0.0  # "high" threshold for the primary's unit type
    too_many_rows: bool = False
    rows_matched: int = 0
    gaps: tuple[str, ...] = ()
    current: bool = True
    authority: int = 3
    conflict: bool = False
    outdated: bool = False
    outdated_only: bool = False  # a value was asked and only an outdated source has it
    # How strongly the evidence is *about* the question, on the ranker's scale. The gap rules need this, not
    # answerability: with "credit card" uncovered, the debit card's fee row rightly does not answer the question,
    # yet it is strong evidence for the rest of it. None means the gate's own score plays both roles.
    relevance: float | None = None
    relevance_high: float = 0.0


@dataclass(frozen=True)
class Verdict:
    decision: str  # "answer" | "partial" | "clarify" | "refuse"
    rule: str
    level: str  # "high" | "medium" | "low" | "none"
    action: str  # what the agent should do with it


def _relevant(s: Signals) -> bool:
    if s.relevance is None:
        return _strong(s)
    return s.relevance >= s.relevance_high


def _strong(s: Signals) -> bool:
    return s.score is not None and s.score >= s.high


RULES: tuple[tuple[str, Callable[[Signals], bool], str], ...] = (
    # Nothing to answer.
    ("empty_input", lambda s: s.empty, "refuse"),
    # Balances, statuses, "why was I charged": needs the account system. Share general policy if it is strong.
    ("account_lookup_with_policy", lambda s: s.account_intent is not None and _strong(s), "partial"),
    ("account_lookup", lambda s: s.account_intent is not None, "refuse"),
    # No word from any title, heading or row label: we cannot tell what the question is about.
    ("no_topical_term", lambda s: not s.topical, "clarify"),
    # A fee, time or limit was asked and only a source flagged as outdated has one: never quote a stale value.
    ("outdated_only", lambda s: s.outdated_only, "refuse"),
    # The best evidence is below the answer threshold for its unit type.
    ("no_relevant_evidence", lambda s: s.score is None or s.score < s.floor, "refuse"),
    # A key modifier is missing ("cashier's check") and the evidence is only moderately relevant:
    # the evidence is probably about something else, so do not offer it as the answer.
    ("missing_key_term", lambda s: bool(s.gaps) and not _relevant(s), "refuse"),
    # More matching table rows than an agent can scan on a call.
    ("too_many_matches", lambda s: s.too_many_rows, "clarify"),
    # Strong evidence for most of the question, but a modifier is not covered ("business credit card").
    ("uncovered_modifier", lambda s: bool(s.gaps), "partial"),
    ("answer", lambda s: True, "answer"),
)

ACTIONS = {
    ("answer", "high"): "Safe to read out: current, authoritative source, single value.",
    "answer": "Check the cited source before reading it out.",
    "partial": "Share only what is cited; escalate the uncovered part to Business Support Tier 2.",
    "clarify": "Ask the customer the clarifying question before answering.",
    "refuse": (
        "Not in the knowledge base: escalate to Business Support Tier 2. Do not tell the customer we don't offer it."
    ),
    "account": (
        "Needs an account lookup: verify the caller (Caller Identity Verification Procedure), "
        "then check the account system."
    ),
}


def _level(decision: str, s: Signals) -> str:
    if decision != "answer":
        return "none"  # make_verdict sets the level of anything but an answer
    if not _strong(s):
        return "low"
    if s.rows_matched <= 1 and s.current and s.authority >= 2 and not s.conflict and not s.outdated:
        return "high"
    return "medium"


def make_verdict(decision: str, rule: str, level: str = "low", action: str | None = None) -> Verdict:
    """The one place a decision becomes a level and an action: only an answer carries its own level
    (a partial is always "low", a clarification or refusal "none")."""
    if decision != "answer":
        level = "low" if decision == "partial" else "none"
    if action is None:
        action = (
            ACTIONS["account"]
            if rule.startswith("account_lookup")
            else ACTIONS.get((decision, level), ACTIONS[decision])
        )
    return Verdict(decision, rule, level, action)


def decide(s: Signals) -> Verdict:
    for name, predicate, decision in RULES:
        if predicate(s):
            return make_verdict(decision, name, _level(decision, s))
    raise AssertionError("the final rule always matches")


_RANK = {"none": 0, "low": 1, "medium": 2, "high": 3}
DECISION_RANK = {"refuse": 0, "clarify": 1, "partial": 2, "answer": 3}  # how much an answer commits to


def combine(verdicts: Sequence[Verdict]) -> Verdict:
    """A multi-part question is answered only if every part is; otherwise it is partial."""
    if len(verdicts) == 1:
        return verdicts[0]
    kinds = {v.decision for v in verdicts}
    if kinds == {"answer"}:
        decision = "answer"
    elif kinds == {"refuse"}:
        decision = "refuse"
    elif kinds <= {"clarify", "refuse"}:
        decision = "clarify"
    else:
        decision = "partial"
    level = min((v.level for v in verdicts), key=_RANK.__getitem__)
    rule = " + ".join(v.rule for v in verdicts)
    account = any(v.rule.startswith("account_lookup") for v in verdicts) and decision != "answer"
    return make_verdict(decision, rule, level, ACTIONS["account"] if account else None)
