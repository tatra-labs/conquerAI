"""The gate is a pure function of explicit signals, so every rule can be tested without a model."""

import pytest

from mbfs_rag.gate import Signals, Verdict, combine, decide, make_verdict

BASE = dict(score=6.0, floor=1.7, high=5.7)


@pytest.mark.parametrize(
    ("signals", "decision", "rule"),
    [
        (Signals(empty=True), "refuse", "empty_input"),
        (Signals(account_intent="my_balance", score=-6.0, floor=1.7, high=5.7), "refuse", "account_lookup"),
        (Signals(account_intent="status_of_mine", **BASE), "partial", "account_lookup_with_policy"),
        (Signals(topical=False, **BASE), "clarify", "no_topical_term"),
        (Signals(score=0.3, floor=1.7, high=5.7), "refuse", "no_relevant_evidence"),
        (Signals(score=3.0, floor=1.7, high=5.7, gaps=("cashier check",)), "refuse", "missing_key_term"),
        (Signals(too_many_rows=True, **BASE), "clarify", "too_many_matches"),
        (Signals(gaps=("credit card",), **BASE), "partial", "uncovered_modifier"),
        (Signals(**BASE), "answer", "answer"),
    ],
)
def test_first_matching_rule_decides(signals, decision, rule):
    verdict = decide(signals)
    assert (verdict.decision, verdict.rule) == (decision, rule)


def test_levels_are_agent_actions_not_probabilities():
    assert decide(Signals(**BASE)).level == "high"
    assert decide(Signals(**BASE, rows_matched=3)).level == "medium"  # several rows listed: check the source
    assert decide(Signals(**BASE, conflict=True)).level == "medium"
    assert decide(Signals(**BASE, current=False)).level == "medium"  # upcoming content as the answer
    assert decide(Signals(score=3.0, floor=1.7, high=5.7)).level == "low"
    assert "account lookup" in decide(Signals(account_intent="x", score=-6, floor=1.7, high=5.7)).action


def test_a_compound_question_is_answered_only_if_every_part_is():
    answer = Verdict("answer", "answer", "high", "")
    refuse = Verdict("refuse", "no_relevant_evidence", "none", "")
    assert combine([answer, answer]).decision == "answer"
    assert combine([answer, refuse]).decision == "partial"
    assert combine([refuse, refuse]).decision == "refuse"


def test_every_decision_and_level_has_an_action():
    for decision in ("answer", "partial", "clarify", "refuse"):
        for level in ("high", "medium", "low", "none"):
            verdict = make_verdict(decision, "some_rule", level)
            assert verdict.action, (decision, level)
            assert verdict.level == level if decision == "answer" else verdict.level in ("low", "none")
    assert "account lookup" in make_verdict("partial", "account_lookup_with_policy").action


def test_a_combined_answer_takes_the_action_of_its_weakest_part():
    strong, weak = decide(Signals(score=6.0, floor=1.7, high=5.7)), decide(Signals(score=3.0, floor=1.7, high=5.7))
    combined = combine([strong, weak])
    assert (combined.decision, combined.level) == ("answer", "low")
    assert combined.action.startswith("Check the cited source")  # not "Safe to read out"


def test_a_value_only_an_outdated_source_has_is_refused():
    verdict = decide(Signals(score=6.0, floor=1.7, high=5.7, outdated_only=True))
    assert (verdict.decision, verdict.rule) == ("refuse", "outdated_only")


def test_the_gap_rules_judge_relevance_and_the_floor_judges_answerability():
    """With 'credit card' uncovered, the verifier rightly says the debit card's fee row does not answer the question;
    whether to offer it as a partial answer depends on how strongly it is about the rest of the question."""
    base = dict(score=1.0, floor=-1.5, high=2.5, gaps=("credit card",), relevance_high=5.7)
    assert decide(Signals(**base, relevance=7.0)).rule == "uncovered_modifier"  # strongly about the rest: partial
    assert decide(Signals(**base, relevance=2.0)).rule == "missing_key_term"  # about something else: refuse
    assert decide(Signals(**{**base, "score": -3.0}, relevance=7.0)).rule == "no_relevant_evidence"  # no answer
