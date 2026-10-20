"""Lexical retrieval and the validity mask, without models (BM25 mode)."""

from datetime import date

from mbfs_rag.config import Settings
from mbfs_rag.retrieval import Candidate, LexicalIndex, Retriever, reciprocal_rank_fusion


def test_bm25_idf_never_goes_negative_and_exact_terms_rank_first():
    index = LexicalIndex(["fee fee wire", "fee chargeback", "fee interest", "fee wire cut-off"])
    assert index.idf("fee") > 0  # Lucene IDF: positive even for a term in every document
    scores = index.scores(["chargeback"])
    assert scores.argmax() == 1


def test_rrf_ties_are_stable():
    fused = reciprocal_rank_fusion([[3, 1, 2], [1, 3, 2]], k=60)
    assert fused[1] == fused[3] > fused[2]


def test_superseded_content_is_never_retrieved_for_the_current_date(corpus):
    retriever = Retriever(corpus, Settings())  # no embedder or reranker: BM25 mode only
    now, _ = retriever.retrieve("chargeback fee per chargeback", corpus.snapshot, mode="bm25")
    assert now and all(c.meta.doc_id != "fee-schedule-2025-01" for c in now)
    then, _ = retriever.retrieve("chargeback fee per chargeback", date(2026, 3, 1), mode="bm25")
    assert any(c.meta.doc_id == "fee-schedule-2025-01" for c in then)
    assert all(c.meta.doc_id != "fee-schedule-2026-07" for c in then)  # not yet published on 2026-03-01


def test_deleted_documents_cannot_resurrect_the_old_schedule(corpus):
    """Excluding the current schedule must not make the 2025 one current: supersession comes from the catalog."""
    retriever = Retriever(corpus, Settings())
    cands, _ = retriever.retrieve(
        "chargeback fee", corpus.snapshot, mode="bm25", exclude=frozenset({"fee-schedule-2026-07"})
    )
    assert all(c.meta.doc_id not in ("fee-schedule-2026-07", "fee-schedule-2025-01") for c in cands)


def test_upcoming_content_is_retrievable_but_flagged(corpus):
    retriever = Retriever(corpus, Settings())
    cands, _ = retriever.retrieve("Essentials monthly maintenance fee increase", corpus.snapshot, mode="bm25")
    notice = [c for c in cands if c.meta.doc_id == "notice-2027-fee-changes"]
    assert notice and all(c.state == "upcoming" for c in notice)


def test_a_date_from_the_question_chooses_versions_and_leaves_other_documents_visible(corpus):
    retriever = Retriever(corpus, Settings())
    year_end = date(2025, 12, 31)
    dated, _ = retriever.retrieve("1099-K tax form", year_end, mode="bm25", kb_date=corpus.snapshot)
    assert any(c.meta.doc_id == "help-tax-forms" for c in dated)  # published 2026, still the answer
    full, _ = retriever.retrieve("1099-K tax form", year_end, mode="bm25")  # explicit as_of: point in time
    assert all(c.meta.doc_id != "help-tax-forms" for c in full)
    fees, _ = retriever.retrieve("chargeback fee", date(2025, 6, 1), mode="bm25", kb_date=corpus.snapshot)
    schedules = {c.meta.doc_id for c in fees if c.meta.doc_type == "fee_schedule"}
    assert schedules == {"fee-schedule-2025-01"}


def test_an_outdated_source_never_supplies_a_value(corpus):
    """The stale FAQ may explain a process as a last resort, but its fees are exactly what went stale."""
    from mbfs_rag.evidence import EvidenceBuilder
    from mbfs_rag.query import Facet

    faq = next(p for p in corpus.passages.values() if p.doc_id == "faq-wire-transfers" and "$40" in p.text)
    meta = corpus.meta(faq.doc_id)
    stale = Candidate(faq, meta, "current", "", logit=5.0)
    builder = EvidenceBuilder(corpus, Settings(), None)
    asked = "How much is an outgoing international wire on Plus"
    ev = builder.build(Facet(asked, asked, ("Bridge Plus",)), [stale], corpus.snapshot, ())
    assert ev.withheld == [stale] and ev.primary is None and not ev.facts
