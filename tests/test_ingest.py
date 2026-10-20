"""Ingestion: what reaches the index decides what can ever be cited."""

from datetime import date

from mbfs_rag.corpus import DocMeta, DocumentBuilder
from mbfs_rag.ingest.html import parse_html
from mbfs_rag.ingest.markdown import parse_markdown
from mbfs_rag.ingest.tables import parse_csv


def build(blocks, doc_id="doc"):
    meta = DocMeta(doc_id, f"{doc_id}.x", "Title", "policy", "customer", "published", 3, date(2026, 1, 1))
    builder = DocumentBuilder(meta)
    for block in blocks:
        builder.add(block)
    return builder.build()


HTML = """<!DOCTYPE html><html><head><title>t</title><script>var fee = "$99";</script></head><body>
<div class="cookie-banner" role="dialog"><p>We use cookies.</p></div>
<header><nav><a href="/fees">Fees &amp; Pricing</a></nav></header>
<main><article>
  <h1>Getting paid</h1>
  <p class="article-meta">Updated July 5, 2026</p>
  <!-- TODO: confirm fee before publishing: $30? -->
  <h2>Instant Payouts</h2>
  <p>Instant Payouts cost 1.5% of the payout amount.</p>
  <div class="exp-variant" hidden><p>Instant Payouts cost 1% of the payout amount.</p></div>
  <p style="display: none">Legacy text.</p>
  <details><summary>Can I cancel an Instant Payout?</summary><p>No. Instant Payouts are final.</p></details>
  <table><tr><th>Plan</th><th>Payout timing</th></tr><tr><td>Bridge Plus</td><td>Next business day</td></tr></table>
</article>
<aside class="related-articles"><h2>Related articles</h2><ul><li><a>Wire fees</a></li></ul></aside>
<section class="article-feedback"><p>Was this article helpful?</p></section>
</main><footer><p>Member FDIC</p></footer></body></html>"""


def test_html_keeps_article_and_drops_boilerplate_hidden_and_comments():
    doc = build(parse_html(HTML))
    assert "Instant Payouts cost 1.5% of the payout amount." in doc.text
    for leaked in (
        "cookies",
        "Fees & Pricing",
        "Member FDIC",
        "Related articles",
        "helpful",
        "1% of the payout",
        "Legacy text",
        "TODO",
        "$99",
        "Updated July",
    ):
        assert leaked not in doc.text, leaked


def test_html_faq_accordion_question_becomes_the_section():
    doc = build(parse_html(HTML))
    answer = next(p for p in doc.passages if "final" in p.text)
    assert answer.section == ("Instant Payouts", "Can I cancel an Instant Payout?")


def test_html_table_rows_are_self_describing_passages_with_cell_spans():
    doc = build(parse_html(HTML))
    row = next(p for p in doc.passages if p.kind == "table_row")
    assert row.text == "Plan: Bridge Plus; Payout timing: Next business day"
    assert [s.label for s in row.spans] == ["Plan", "Payout timing"]


MARKDOWN = """# Chargebacks and Disputes Policy

Effective: July 1, 2026

## Responding to a chargeback

You have **10 calendar days** from the notification date to respond. If you do not respond, the
chargeback is accepted.

1. Open the [Merchant Portal](https://portal.merchantbridge.example).
2. Upload your evidence.

| Stage | Typical timing |
|---|---|
| Representment | 30 to 75 days |

## Revision history

- v2.2 2025-01-01: Chargeback fee updated to $15.00.
"""


def test_markdown_drops_revision_history_but_keeps_content():
    doc = build(parse_markdown(MARKDOWN))
    assert "$15.00" not in doc.text
    assert "You have 10 calendar days from the notification date to respond." in doc.text
    assert "1. Open the Merchant Portal." in doc.text  # inline markup stripped, step number kept


def test_markdown_sentences_lists_and_tables_become_exact_spans():
    doc = build(parse_markdown(MARKDOWN))
    spans = [s for p in doc.passages for s in p.spans]
    assert all(doc.text[s.start : s.end] == s.text for s in spans)
    sentence = next(s for s in spans if s.text.startswith("You have 10"))
    assert sentence.text.endswith("to respond.")
    assert sentence.line_start == 7
    row = next(p for p in doc.passages if p.kind == "table_row")
    assert row.section == ("Responding to a chargeback",)
    assert row.text == "Stage: Representment; Typical timing: 30 to 75 days"


CSV = (
    "﻿MBFS Business Fee & Rate Schedule,,,,\n"
    '"Effective July 1, 2026.",,,,\n'
    ",,,,\n"
    "Code,Category,Service,Bridge Essentials,Bridge Premier\n"
    "ACC-01,Account,Monthly maintenance fee,$12.00 [1],$45.00 [1]\n"
    "MER-05,Card processing,Chargeback,$25.00,$25.00\n"
    ",,,,\n"
    "Footnotes,,,,\n"
    "[1] Waivable — see Business Checking Plans.,,,,\n"
)


def test_csv_header_detection_footnote_resolution_and_line_numbers():
    doc = build(parse_csv(CSV.lstrip("﻿")))
    rows = [p for p in doc.passages if p.kind == "table_row"]
    assert [r.record["Code"] for r in rows] == ["ACC-01", "MER-05"]
    fee = rows[0]
    assert fee.section == ("Account",)
    assert fee.text.endswith("Footnote [1]: Waivable — see Business Checking Plans.")
    assert fee.line_start == 5
    assert "Footnote" not in rows[1].text  # footnotes attach only to rows that reference them
    assert "Effective July 1, 2026." in doc.text  # caption rows are kept as content
