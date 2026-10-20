"""End-to-end behaviour with the real local models (skipped when they are not downloaded)."""

import json
import shutil
from datetime import date

import pytest

from mbfs_rag.evaluation import EVAL_DIR, load_items, score_answer

pytestmark = pytest.mark.models
DEV = load_items(EVAL_DIR / "dev.toml", "dev")
KNOWN_DECISION_MISSES = {"D23"}  # see the eval report: the gate refuses the changed-bank-details question


def facts_text(answer) -> str:
    return " ".join(s["text"] for s in answer.to_dict()["segments"] if s["kind"] in ("fact", "option"))


@pytest.mark.parametrize("item", DEV, ids=[i.id for i in DEV])
def test_answer_contract(service, item):
    """Every asserted segment is cited, and the eval's scorer finds no forbidden value, structural error
    (superseded, outdated-without-warning or wrong-plan citation, an amount in a refusal) or wrong decision."""
    answer = service.answer(item.query)
    data = json.loads(json.dumps(answer.to_dict(), default=str))  # JSON round-trip
    for seg in data["segments"]:
        if seg["kind"] in ("fact", "option"):
            assert seg["citations"], seg["text"]
    score = score_answer(item, data, service.corpus, 0)  # the eval's own, independent scorer
    assert not score.forbidden_hits and not score.structural, (item.id, score.forbidden_hits, score.structural)
    if item.id in KNOWN_DECISION_MISSES:  # strict: once it decides correctly, take it off the list
        assert not score.decision_ok, f"{item.id} now decides correctly"
    else:
        assert score.decision_ok, (item.id, score.decision, item.acceptable)


def test_point_in_time_answers_follow_the_validity_windows(service):
    then = service.answer("What is the chargeback fee?", date(2025, 6, 1))
    assert "$15.00" in facts_text(then) and all(c.doc_id != "fee-schedule-2026-07" for c in then.citations)
    later = service.answer("What's the monthly fee for Bridge Essentials?", date(2027, 2, 1))
    assert "$15.00" in facts_text(later) and any(c.doc_id == "notice-2027-fee-changes" for c in later.citations)
    now = service.answer("What's the monthly fee for Bridge Essentials?")
    assert "$12.00" in facts_text(now) and "$15.00" not in facts_text(now)
    assert any(c.doc_id == "notice-2027-fee-changes" for c in now.citations)  # the dated change is cited
    before = service.answer("What's the monthly fee for Bridge Essentials?", date(2026, 8, 15))
    assert "$12.00" in facts_text(before)
    assert all(c.doc_id != "notice-2027-fee-changes" for c in before.citations)  # published 2026-10-01
    old = service.answer("What's the monthly fee for Bridge Essentials?", date(2025, 6, 1))
    assert "$10.00" in facts_text(old) and not any("2027" in n for n in old.notes)  # it amends $12.00, not $10


def test_deleting_the_current_schedule_never_brings_back_the_old_fee(service):
    answer = service.answer("What is the chargeback fee?", exclude=frozenset({"fee-schedule-2026-07"}))
    assert "$15" not in facts_text(answer)


def test_answers_follow_the_corpus_not_memorised_values(service, tmp_path):
    """Value mutation: change WIR-03 Plus to $47.00 in a copy of the corpus; the answer must follow."""
    from mbfs_rag.ingest import load_corpus
    from mbfs_rag.service import KnowledgeService

    copy = tmp_path / "corpus"
    shutil.copytree(service.settings.corpus_dir, copy)
    csv = copy / "fees" / "fee-and-rate-schedule-2026-07.csv"
    text = csv.read_bytes().decode("utf-8")
    old = "WIR-03,Wire transfers,International wire – outgoing,INTL WIRE OUT FEE,$50.00,$45.00,$25.00"
    assert old in text
    csv.write_bytes(text.replace(old, old.replace("$45.00", "$47.00")).encode("utf-8"))
    mutated = KnowledgeService(
        load_corpus(copy), service.settings, service.retriever.embedder, service.reranker, verifier=service.verifier
    )
    answer = mutated.answer("What's the fee for an outgoing international wire on the Plus plan?")
    assert "$47.00" in facts_text(answer) and "$45.00" not in facts_text(answer)


def test_card_numbers_never_reach_the_trace(service):
    answer = service.answer("customer's card 5555 5555 5555 4444 was stolen, what now?", explain=True)
    assert "5555 5555" not in json.dumps(answer.to_dict(), default=str)


def test_internal_content_is_flagged(service):
    answer = service.answer("Can I waive this customer's NSF fee?")
    assert answer.internal_content and "[Internal" in answer.answer


def test_stale_faq_is_flagged_not_used(service):
    answer = service.answer("What time is the cut-off for international wires?")
    assert "2:00 PM" in facts_text(answer) and "1:00 PM" not in facts_text(answer)
    assert any("Wire Transfers FAQ" in n and "outdated" in n for n in answer.notes)


def test_concurrent_requests_get_the_answers_sequential_ones_get(service):
    """The API serves requests from a thread pool against one service, so no per-request state may be shared."""
    import sys
    from concurrent.futures import ThreadPoolExecutor

    questions = [
        "What is the cut-off time for domestic wires?",
        "What time is the cut-off for international wires?",
        "What is the chargeback fee?",
        "How long do I have to respond to a chargeback?",
    ]
    expected = {q: service.answer(q).answer for q in questions}
    interval = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)  # switch threads as often as possible
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            got = list(pool.map(lambda q: (q, service.answer(q).answer), questions * 4))
    finally:
        sys.setswitchinterval(interval)
    assert all(text == expected[q] for q, text in got)


def test_segments_carry_the_internal_flag(service):
    segments = service.answer("Can I waive this customer's NSF fee?").to_dict()["segments"]
    assert any(s["internal"] for s in segments if s["kind"] == "fact")


class StandInGenerator:
    """Stands in for Claude: records the verdicts that reach it and answers with the first extractive span."""

    def __init__(self, corpus):
        self.corpus, self.rules = corpus, []

    def generate(self, query, as_of, parts, verdict, today=None):
        from mbfs_rag.claude import Generated
        from mbfs_rag.evidence import Segment
        from mbfs_rag.gate import make_verdict

        self.rules.append(verdict.rule)
        sid = parts[0][0].facts[0].span_ids[0]
        segment = Segment("fact", self.corpus.spans[sid].text, [sid])
        return Generated([segment], make_verdict("answer", f"{verdict.rule} + llm:ANSWER", "medium"))


def test_the_generator_only_phrases_answers_the_gate_allowed_and_keeps_its_notes(service):
    from mbfs_rag.service import KnowledgeService

    stand_in = StandInGenerator(service.corpus)
    svc = KnowledgeService(
        service.corpus,
        service.settings,
        service.retriever.embedder,
        service.reranker,
        generator=stand_in,
        verifier=service.verifier,
    )
    assert svc.answer("What's the current balance on account ending 4417?").generator == "extractive"
    assert svc.answer("How much does it cost?").generator == "extractive"  # clarify: the gate's question stands
    assert stand_in.rules == []
    answer = svc.answer("What time is the cut-off for international wires?")
    assert answer.generator == "claude" and stand_in.rules
    assert any("Wire Transfers FAQ" in n for n in answer.notes)  # the gate's conflict note survives


def test_a_tax_year_in_the_question_does_not_hide_the_tax_article(service):
    answer = service.answer("When will I get my 1099-K for 2025?")
    assert answer.decision == "answer" and any(c.doc_id == "help-tax-forms" for c in answer.citations)


def test_a_form_code_alone_is_a_topic(service):
    assert service.answer("When will I get my 1099-K?").confidence["rule"] != "no_topical_term"


def test_an_amount_alone_does_not_turn_a_question_into_a_rate_lookup(service):
    answer = service.answer("Is my $400,000 in Reserve Savings FDIC insured?")
    assert "250,000" in facts_text(answer) and "APY" not in facts_text(answer)


def test_a_message_with_too_many_questions_is_a_partial_answer_that_names_the_rest(service):
    six = "What is the wire fee? What is the chargeback fee? How do I return a terminal? What is the ACH fee? "
    answer = service.answer(six + "What is the PCI fee? What is the stop payment fee?")
    assert answer.decision != "answer" and any("2 more question" in g for g in answer.gaps)


def test_an_api_key_without_the_sdk_falls_back_to_extractive_answers(service, monkeypatch):
    # `service` is unused: requesting it skips this test, like the rest of the module, when the models are missing.
    import sys

    from mbfs_rag.service import KnowledgeService

    monkeypatch.setenv("ANTHROPIC_API_KEY", "not-a-real-key")
    monkeypatch.setitem(sys.modules, "anthropic", None)  # "pip install" without the llm extra
    assert KnowledgeService.load().generator is None
    with pytest.raises(ImportError):
        KnowledgeService.load(use_llm=True)


def test_a_list_item_is_scored_on_its_own_words(service):
    """Each notice bullet is a different change: scoring one with the previous bullet prepended made the paper
    statement change answer an Instant Payout question."""
    answer = service.answer("How much will an Instant Payout cost on Essentials next year?")
    assert "1.75%" in facts_text(answer) and "Paper statements" not in facts_text(answer)


def test_a_fee_for_a_past_date_comes_only_from_sources_in_effect_then(service):
    then = service.answer("Was the chargeback fee refundable in 2025 and how much was it?")
    assert "$15.00" in facts_text(then) and "$25.00" not in facts_text(then)
    pci = service.answer("How much was the monthly PCI non-validation fee in 2025 if the SAQ was late?")
    assert "$14.95" in facts_text(pci) and "$19.95" not in facts_text(pci)
    before = service.answer("What was the chargeback fee in 2024?")  # no schedule in the KB covers 2024
    assert "$25.00" not in facts_text(before)


def test_the_generator_is_not_used_when_a_part_was_refused(service):
    from mbfs_rag.service import KnowledgeService

    stand_in = StandInGenerator(service.corpus)
    svc = KnowledgeService(
        service.corpus,
        service.settings,
        service.retriever.embedder,
        service.reranker,
        generator=stand_in,
        verifier=service.verifier,
    )
    answer = svc.answer("What is the stop payment fee? And how much does a cashier's check cost?")
    assert answer.generator == "extractive" and stand_in.rules == []


def test_list_items_read_with_their_lead_in(service):
    """Steps after "If you think your card was skimmed:" are scored with that lead-in, so the steps are quoted."""
    answer = facts_text(service.answer("my card got skimmed at a gas staton what do i do??"))
    assert "Lock the card" in answer and "Report it" in answer


def test_a_past_dated_fee_is_checked_on_what_the_answer_states(service):
    """No price word in the question, so only the answer-level check stops today's $0.50 minimum."""
    answer = facts_text(service.answer("What was the Instant Payout minimum in 2025?"))
    assert "$0.25" in answer and "$0.50" not in answer
