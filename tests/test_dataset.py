"""The eval set is only as good as its gold. These tests keep gold, traps and matchers honest."""

import csv
import io
from pathlib import Path

import pytest

from mbfs_rag.corpus import read_text
from mbfs_rag.evaluation import EVAL_DIR, load_items, matcher

ROOT = Path(__file__).resolve().parent.parent
CORPUS = ROOT / "corpus"
SETS = [p for p in (EVAL_DIR / "dev.toml", EVAL_DIR / "heldout_v1.toml", EVAL_DIR / "heldout_v2.toml") if p.exists()]


@pytest.fixture(scope="session")
def items():
    return [i for path in SETS for i in load_items(path, path.stem)]


def raw_row(doc_path: str, code: str) -> dict[str, str]:
    """Read a fee-schedule row with the stdlib csv module - deliberately NOT through the system's parser."""
    rows = list(csv.reader(io.StringIO(read_text(CORPUS / doc_path))))
    header = next(r for r in rows if r and r[0] == "Code")
    row = next(r for r in rows if r and r[0] == code)
    return dict(zip(header, row, strict=False))


def test_ids_unique_and_decisions_valid(items):
    ids = [i.id for i in items]
    assert len(ids) == len(set(ids))
    for item in items:
        assert item.acceptable and set(item.acceptable) <= {"answer", "partial", "clarify", "refuse"}, item.id


def test_every_gold_quote_exists_verbatim_and_every_row_exists(items, corpus):
    for item in items:
        for fact in item.facts:
            assert fact.support, f"{item.id}/{fact.id} has no support"
            for s in fact.support:
                assert s.doc in corpus.docs, f"{item.id}: unknown doc {s.doc}"
                if s.quote is not None:
                    assert s.quote in corpus.docs[s.doc].text, f"{item.id}/{fact.id}: quote missing from {s.doc}"
                else:
                    row = raw_row(corpus.meta(s.doc).path, s.row)
                    if s.column:
                        assert s.column in row, f"{item.id}: no column {s.column} in {s.row}"


def test_table_gold_values_match_the_raw_csv(items, corpus):
    """For facts supported by a single plan cell, the matcher must accept the raw CSV value."""
    for item in items:
        for fact in item.facts:
            for s in fact.support:
                if s.row and s.column and fact.match.startswith(("money:", "percent:")):
                    cell = raw_row(corpus.meta(s.doc).path, s.row)[s.column]
                    assert matcher(fact.match).search(cell), f"{item.id}/{fact.id}: {fact.match} vs CSV {cell!r}"


def test_matchers_are_typed_not_substrings():
    assert matcher("money:25").search("$25.00 per chargeback")
    assert not matcher("money:25").search("up to $25,000 per day")
    assert not matcher("money:5").search("$50.00")
    assert matcher("money:2500").search("at least $2,500")
    assert matcher("time:2pm").search("International wires: 2:00 PM ET")
    assert not matcher("time:2pm").search("by 12:00 PM ET")
    assert matcher("days:10").search("within 10 calendar days")
    assert matcher("percent:2.25").search("APY: 2.25%")
    assert not matcher("percent:1").search("1.75% of payout")
    assert matcher("percent:1").search("1.0% of payout") and not matcher("percent:1").search("10% of payout")
    assert matcher("percent:2.25").search("2.250%") and not matcher("percent:2").search("20%")
    assert matcher("percent:3.10").search("3.1% APY") and not matcher("percent:2.25").search("12.25%")
    # whole amounts only: $2 is not $2.50, but a sentence-ending "$2." still is $2
    assert not matcher("money:2").search("a $2.50 out-of-network ATM fee")
    assert matcher("money:2").search("the fee was $2.00") and matcher("money:2").search("It costs $2.")
    assert matcher("days:30").search("the same 30-day window")
    # text: word-bounded, case-insensitive, inflection on the last word only
    assert matcher("text:refund").search("Refunds are deducted from your next payout")
    assert matcher("text:employee card").search("order up to 10 employee cards")
    assert not matcher("text:refund").search("the fee is non-refundable")
    assert not matcher("text:never").search("nevertheless")


def test_forbidden_matchers_never_match_their_own_gold(items, corpus):
    for item in items:
        quotes = [s.quote for f in item.facts for s in f.support if s.quote]
        for bad in item.forbidden:
            pattern = matcher(bad["match"])
            assert not any(pattern.search(q) for q in quotes), f"{item.id}: forbidden {bad['match']} matches gold"


def test_traps_exist_where_intended(corpus):
    """An unanswerable item or a stale-value trap tests nothing if the trap is not really there."""
    faq = corpus.docs["faq-wire-transfers"].text
    assert "1:00 PM ET" in faq and "$40 on Plus" in faq
    assert "30 days" in corpus.docs["help-mobile-deposit"].text
    assert corpus.meta("faq-wire-transfers").outdated_by == ("wire-transfers",)
    old = raw_row("fees/fee-and-rate-schedule-2025-01.csv", "MER-05")
    assert old["Bridge Plus"].startswith("$15.00")
    assert "draft-2027-pricing-proposal" not in corpus.docs  # drafts are never indexed
    everything = " ".join(d.text for d in corpus.documents).lower()
    for absent in ("mortgage", "money order", "line of credit rate", "surcharge on card"):
        assert absent not in everything, absent
    assert everything.count("cashier") == 1  # only the overpayment-scam warning mentions it
    raw_payouts = read_text(CORPUS / "help" / "getting-paid.html")
    assert "1.25% of the payout amount" in raw_payouts  # the hidden A/B variant is in the raw file...
    assert "1.25%" not in corpus.docs["help-getting-paid"].text  # ...and never in the index


def test_model_written_questions_are_checked_for_forbidden_values(corpus):
    """A clarifying question the model wrote is shown to the agent, so it may not state a stale price either."""
    from mbfs_rag.evaluation import Item, score_answer

    item = Item(
        "X", "How much does a wire cost?", ["answer", "clarify"], forbidden=[{"match": "money:40", "why": "stale"}]
    )
    answer = {
        "segments": [{"kind": "route", "text": "Is it the $40 international wire on Plus?", "citations": []}],
        "citations": [],
        "as_of": corpus.snapshot.isoformat(),
        "decision": "clarify",
        "internal_content": False,
        "gaps": [],
        "generator": "claude",
        "confidence": {"score": None, "rule": "answer + llm:CLARIFY", "level": "none"},
        "answer": "",
    }
    score = score_answer(item, answer, corpus, 0)
    assert score.forbidden_hits == ["money:40"] and "abstention states an amount" in score.structural


def test_calibration_is_answerable_and_never_sees_an_eval_question(corpus, items):
    """Calibration sets the floors, so it must not contain a test question, and its positives must be answerable."""
    import tomllib

    from mbfs_rag.evaluation import calibration_questions

    calibration = calibration_questions(corpus)  # raises if a positive's quote is not verbatim in its document
    assert sum(q["gold"] is not None for q in calibration) >= 100 and sum(q["gold"] is None for q in calibration) >= 50
    tested = {" ".join(i.query.lower().split()) for i in items}
    pairs = tomllib.loads((EVAL_DIR / "invariance.toml").read_text(encoding="utf-8"))["pair"]
    tested |= {" ".join(p["query"].lower().split()) for p in pairs}
    assert not any(" ".join(q["question"].lower().split()) in tested for q in calibration)
