"""The real corpus: exact spans, validity in time, and knowledge derived from KB tables rather than code."""

import re
from datetime import date

from mbfs_rag.corpus import DocMeta


def test_spans_tile_their_passages_and_table_cells_quote_their_record(corpus):
    """Span.text is sliced from doc.text by construction, so comparing the two proves nothing. These checks
    do not depend on the offsets being right, and fail if a span is cut short or runs into its neighbour."""
    assert len(corpus.spans) > 1000
    for passage in corpus.passages.values():
        text = corpus.docs[passage.doc_id].text
        cuts = [passage.start, *(x for s in passage.spans for x in (s.start, s.end)), passage.end]
        for a, b in zip(cuts[::2], cuts[1::2], strict=True):  # only whitespace and "; " may sit between spans
            assert a <= b and re.fullmatch(r"[\s;]*", text[a:b]), (passage.passage_id, text[a:b])
        for span in passage.spans:
            assert passage.start <= span.start < span.end <= passage.end, span.span_id
            assert span.text and span.text == span.text.strip() and "\n\n" not in span.text, span.span_id
            assert span.line_start >= 1 and span.line_end >= span.line_start
            if passage.record and span.label in passage.record:  # a cell quote is "Header: value", in full
                assert span.text == f"{span.label}: {passage.record[span.label]}", span.span_id


def test_drafts_are_never_indexed_and_superseded_content_expires(corpus):
    assert "draft-2027-pricing-proposal" not in corpus.docs
    old = corpus.meta("fee-schedule-2025-01")
    assert old.effective_to == date(2026, 7, 1)  # from the catalog's "supersedes", not from what parsed
    assert old.state(date(2026, 6, 30)) == "current"
    assert old.state(corpus.snapshot) == "superseded"


def test_notice_is_upcoming_until_its_effective_date(corpus):
    notice = corpus.meta("notice-2027-fee-changes")
    assert notice.state(date(2026, 9, 30)) is None  # not yet published
    assert notice.state(corpus.snapshot) == "upcoming"
    assert notice.state(date(2027, 1, 1)) == "current"


def test_staleness_is_derived_from_catalog_metadata(corpus):
    # The FAQ (reviewed 2025-02-10) predates the wire policy (2026-03-01) on a shared topic.
    assert corpus.meta("faq-wire-transfers").outdated_by == ("wire-transfers",)
    # The newer help article that drifted from policy is NOT outdated by date: authority handles that case.
    assert corpus.meta("help-mobile-deposit").outdated_by == ()


def test_glossary_and_scheduled_changes_come_from_kb_tables(corpus):
    terms = {g.term for g in corpus.glossary}
    assert {"CB", "IP", "ADB", "Overseas wire"} <= terms
    change = next(c for c in corpus.changes if c.code == "ACC-01" and c.plan == "Bridge Essentials")
    assert (change.old_value, change.new_value, change.effective) == ("$12.00", "$15.00", date(2027, 1, 1))


def test_facet_words_distinguish_sibling_rows(corpus):
    wires = corpus.facets[("fee-schedule-2026-07", ("Wire transfers",))]
    assert {"domest", "intern", "incom", "outgo"} <= wires
    assert "fee" not in wires and "wire" not in wires  # shared by every sibling, so not a facet


def test_validity_states_for_a_meta_without_dates():
    meta = DocMeta("d", "d.md", "T", "policy", "customer", "draft", 3, None)
    assert meta.state(date(2026, 1, 1)) is None
