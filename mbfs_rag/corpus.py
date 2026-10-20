"""Corpus data model: documents, passages (retrieval units) and spans (citation units).

Every document is reduced to one canonical text. Passages and spans are character ranges into it,
so `doc.text[span.start:span.end] == span.text` holds for every citation the system can return
(asserted for every span in the tests). Spans also carry the line range in the raw file so a
person can open the original.

Governance metadata (type, audience, dates, supersession, topics) comes from catalog.json, the way
it would come from a CMS. Three kinds of knowledge are derived from KB *tables* rather than from
code: the glossary (abbreviations and customer phrasings), the dated fee changes announced in
notices, and the facet vocabulary that tells sibling table rows apart.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from .text import code_tokens, sentence_spans, terms


@dataclass(frozen=True)
class Span:
    """A citable unit: one sentence of prose, or one "Header: value" cell of a table row."""

    span_id: str
    passage_id: str
    doc_id: str
    start: int
    end: int
    text: str
    line_start: int
    line_end: int
    label: str | None = None  # column header for table cells, e.g. "Bridge Plus"


@dataclass(frozen=True)
class Passage:
    """A retrieval unit: consecutive blocks from one section (prose), or exactly one table row."""

    passage_id: str
    doc_id: str
    section: tuple[str, ...]
    start: int
    end: int
    text: str
    spans: tuple[Span, ...]
    line_start: int
    line_end: int
    kind: str = "prose"  # "prose" | "table_row"
    record: dict[str, str] | None = None  # table rows: header -> cell value (markers kept)

    def cell(self, header: str) -> Span | None:
        return next((s for s in self.spans if s.label == header), None)

    def footnotes_for(self, value: str) -> list[Span]:
        """Footnote spans referenced by one cell value ("$2.50 [4][5]" -> footnotes 4 and 5)."""
        wanted = {f"Footnote [{n}]" for n in re.findall(r"\[(\d+)\]", value)}
        return [s for s in self.spans if s.label in wanted]


VERSIONED_TYPES = frozenset({"fee_schedule", "rate_sheet", "notice"})


@dataclass(frozen=True)
class DocMeta:
    doc_id: str
    path: str
    title: str
    doc_type: str
    audience: str
    status: str
    authority: int
    effective_from: date | None
    published: date | None = None  # defaults to effective_from; notices are published before they take effect
    effective_to: date | None = None  # from the catalog's supersession chain
    topics: tuple[str, ...] = ()
    owner: str | None = None
    outdated_by: tuple[str, ...] = ()  # newer authoritative docs on a shared topic
    supersedes: str | None = None

    @property
    def versioned(self) -> bool:
        """Dated price lists, notices and anything in a supersession chain: the documents a date in the
        question ("the chargeback fee in 2025") chooses between. Every other document is read as of today."""
        return self.doc_type in VERSIONED_TYPES or self.effective_to is not None or self.supersedes is not None

    def state(self, as_of: date) -> str | None:
        """'current', 'upcoming' (published, not yet effective), 'superseded', or None (not visible)."""
        if self.status != "published" or self.effective_from is None:
            return None
        if self.published is not None and self.published > as_of:
            return None
        if self.effective_to is not None and self.effective_to <= as_of:
            return "superseded"
        return "upcoming" if self.effective_from > as_of else "current"


def state_on(meta: DocMeta, as_of: date, kb_date: date | None = None) -> str | None:
    """Validity when the knowledge base is read as it stood on `kb_date` and dated facts on `as_of`.

    A document must be published by `kb_date` to exist at all; a versioned one (price list, notice, supersession
    chain) then takes its state on `as_of`, the date the question is about. Without a `kb_date` both are `as_of`:
    an explicit point-in-time view."""
    if kb_date is None:
        return meta.state(as_of)
    state = meta.state(kb_date)
    return meta.state(as_of) if state is not None and meta.versioned else state


@dataclass
class Document:
    meta: DocMeta
    text: str
    passages: list[Passage] = field(default_factory=list)

    @property
    def doc_id(self) -> str:
        return self.meta.doc_id


# ---------------------------------------------------------------- canonical text assembly


@dataclass
class Block:
    """A parser's output unit, before offsets are assigned."""

    kind: str  # "heading" | "text" | "table_row"
    text: str
    section: tuple[str, ...]
    line_start: int
    line_end: int
    cells: list[tuple[str, str]] | None = None  # table rows: (header, value), footnotes appended
    record: dict[str, str] | None = None


class DocumentBuilder:
    """The only code that assigns offsets and IDs: parsers emit Blocks, this joins them."""

    SEPARATOR = "\n\n"
    MAX_PASSAGE_WORDS = 160

    def __init__(self, meta: DocMeta):
        self.meta = meta
        self._parts: list[str] = []
        self._length = 0
        self._placed: list[tuple[Block, int, int]] = []

    def add(self, block: Block) -> None:
        text = block.text.strip()
        if not text:
            return
        if self._parts:
            self._parts.append(self.SEPARATOR)
            self._length += len(self.SEPARATOR)
        start = self._length
        self._parts.append(text)
        self._length += len(text)
        block.text = text
        self._placed.append((block, start, self._length))

    def build(self) -> Document:
        doc = Document(self.meta, "".join(self._parts))
        groups: list[list[tuple[Block, int, int]]] = []
        current: list[tuple[Block, int, int]] = []
        words = 0
        for placed in self._placed:
            block = placed[0]
            if block.kind in ("heading", "table_row") or (
                current
                and (current[0][0].section != block.section or words + len(block.text.split()) > self.MAX_PASSAGE_WORDS)
            ):
                if current:
                    groups.append(current)
                current, words = [], 0
            if block.kind == "table_row":
                groups.append([placed])
            elif block.kind != "heading":
                current.append(placed)
                words += len(block.text.split())
        if current:
            groups.append(current)

        span_count = 0
        for n, group in enumerate(groups):
            passage_id = f"{self.meta.doc_id}:p{n}"
            spans: list[Span] = []
            for block, start, _end in group:
                for s, e, label in self._span_ranges(block, start):
                    spans.append(
                        Span(
                            f"{self.meta.doc_id}:s{span_count}",
                            passage_id,
                            self.meta.doc_id,
                            s,
                            e,
                            doc.text[s:e],
                            block.line_start,
                            block.line_end,
                            label,
                        )
                    )
                    span_count += 1
            first, last = group[0], group[-1]
            doc.passages.append(
                Passage(
                    passage_id=passage_id,
                    doc_id=self.meta.doc_id,
                    section=first[0].section,
                    start=first[1],
                    end=last[2],
                    text=doc.text[first[1] : last[2]],
                    spans=tuple(spans),
                    line_start=first[0].line_start,
                    line_end=last[0].line_end,
                    kind="table_row" if first[0].kind == "table_row" else "prose",
                    record=first[0].record if first[0].kind == "table_row" else None,
                )
            )
        return doc

    @staticmethod
    def _span_ranges(block: Block, start: int) -> Iterator[tuple[int, int, str | None]]:
        if block.kind == "table_row" and block.cells:
            cursor = 0
            for header, value in block.cells:
                piece = f"{header}: {value}"
                at = block.text.index(piece, cursor)
                yield start + at, start + at + len(piece), header
                cursor = at + len(piece)
        else:
            for s, e in sentence_spans(block.text, start):
                yield s, e, None


# ---------------------------------------------------------------- catalog


def read_text(path: Path) -> str:
    """Decode as UTF-8 (tolerating a spreadsheet BOM) and normalise line endings and NBSP before
    any offset is computed, so offsets do not depend on the platform a file was checked out on."""
    text = path.read_bytes().decode("utf-8-sig")
    return text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")


def _date(value: str | None) -> date | None:
    return date.fromisoformat(value) if value else None


def load_catalog(corpus_dir: Path) -> tuple[list[DocMeta], date]:
    """Read catalog.json. Validity windows come from the catalog's supersession chain, never from
    what happened to parse, so a broken new schedule cannot silently resurrect the old one."""
    raw = json.loads((corpus_dir / "catalog.json").read_text(encoding="utf-8"))
    authority: dict[str, int] = raw["authority"]
    entries = raw["documents"]
    superseded_on = {e["supersedes"]: _date(e["effective_from"]) for e in entries if e.get("supersedes")}
    metas = [
        DocMeta(
            doc_id=e["doc_id"],
            path=e["path"],
            title=e["title"],
            doc_type=e["doc_type"],
            audience=e.get("audience", "customer"),
            status=e["status"],
            authority=authority[e["doc_type"]],
            effective_from=_date(e.get("effective_from")),
            published=_date(e.get("published")) or _date(e.get("effective_from")),
            effective_to=superseded_on.get(e["doc_id"]),
            topics=tuple(e.get("topics", ())),
            owner=e.get("owner"),
            supersedes=e.get("supersedes"),
        )
        for e in entries
    ]
    return _with_staleness(metas), date.fromisoformat(raw["snapshot_date"])


def _with_staleness(metas: list[DocMeta]) -> list[DocMeta]:
    """A help article or FAQ is 'possibly outdated' when an authoritative document on a shared topic
    took effect after it was last reviewed (e.g. the wire FAQ predates the March 2026 wire policy)."""
    authoritative = [m for m in metas if m.authority >= 3 and m.status == "published" and m.effective_from]
    out = []
    for m in metas:
        newer = ()
        if m.authority < 3 and m.effective_from:
            newer = tuple(
                a.doc_id for a in authoritative if set(a.topics) & set(m.topics) and a.effective_from > m.effective_from
            )
        out.append(DocMeta(**{**m.__dict__, "outdated_by": newer}) if newer else m)
    return out


# ---------------------------------------------------------------- derived knowledge


@dataclass(frozen=True)
class GlossaryEntry:
    term: str
    meaning: str
    span_id: str


@dataclass(frozen=True)
class FeeChange:
    """An approved future change announced in a notice, keyed to a fee-schedule row and plan."""

    code: str
    plan: str
    old_value: str
    new_value: str
    effective: date
    published: date | None
    passage_id: str


@dataclass
class Corpus:
    documents: list[Document]
    snapshot: date
    glossary: list[GlossaryEntry] = field(default_factory=list)
    changes: list[FeeChange] = field(default_factory=list)
    facets: dict[tuple[str, tuple[str, ...]], frozenset[str]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.docs = {d.doc_id: d for d in self.documents}
        self.passages = {p.passage_id: p for d in self.documents for p in d.passages}
        self.spans = {s.span_id: s for p in self.passages.values() for s in p.spans}
        self.glossary = list(self._glossary())
        self.changes = list(self._changes())
        self.facets = self._facets()
        self.topical_terms = self._topical_terms()
        self.codes = self._codes()
        self.vocabulary = frozenset(t for p in self.passages.values() for t in terms(p.text)) | self.topical_terms

    def meta(self, doc_id: str) -> DocMeta:
        return self.docs[doc_id].meta

    def _table_rows(self, doc_types: set[str]) -> Iterator[tuple[Document, Passage]]:
        for doc in self.documents:
            if doc.meta.doc_type in doc_types:
                for p in doc.passages:
                    if p.kind == "table_row" and p.record:
                        yield doc, p

    def _glossary(self) -> Iterator[GlossaryEntry]:
        for _, row in self._table_rows({"glossary"}):
            if "Term" in row.record and "Meaning" in row.record:
                cell = row.cell("Meaning")
                yield GlossaryEntry(
                    row.record["Term"], row.record["Meaning"], cell.span_id if cell else row.spans[0].span_id
                )

    def _changes(self) -> Iterator[FeeChange]:
        for doc, row in self._table_rows({"notice"}):
            new_key = next((h for h in row.record if h.lower().startswith("new fee")), None)
            if new_key and {"Code", "Plan"} <= row.record.keys() and doc.meta.effective_from:
                old = next((v for h, v in row.record.items() if h.lower().startswith("current")), "")
                yield FeeChange(
                    row.record["Code"],
                    row.record["Plan"],
                    old,
                    row.record[new_key],
                    doc.meta.effective_from,
                    doc.meta.published,
                    row.passage_id,
                )

    def _facets(self) -> dict[tuple[str, tuple[str, ...]], frozenset[str]]:
        """Tokens that distinguish sibling rows within one table section, e.g. for wires
        {domestic, international, incoming, outgoing, recall, ...}. Derived, not hand-written."""
        labels: dict[tuple[str, tuple[str, ...]], list[set[str]]] = defaultdict(list)
        for doc in self.documents:
            for p in doc.passages:
                if p.kind == "table_row" and p.record:
                    labels[(doc.doc_id, p.section)].append(set(terms(row_label(p.record))))
        generic = {t for w in ("fee", "fees", "charge", "cost", "price", "rate") for t in terms(w)}
        facets = {}
        for key, sets in labels.items():
            if len(sets) > 1:
                every = set.intersection(*sets)
                facets[key] = frozenset(t for t in set.union(*sets) - every if t not in generic and not t[:1].isdigit())
        return facets

    def _topical_terms(self) -> frozenset[str]:
        """Every term in a title, heading or row label: a question with none of these is not about
        anything this knowledge base covers specifically."""
        vocab: set[str] = set()
        for doc in self.documents:
            vocab.update(terms(doc.meta.title))
            for p in doc.passages:
                for heading in p.section:
                    vocab.update(terms(heading))
                if p.record:
                    vocab.update(terms(row_label(p.record)))
        return frozenset(vocab)

    def _codes(self) -> frozenset[str]:
        """Form and fee codes the KB names in a title, heading or row code: "1099k", "wir03", "p2pe"."""
        found: set[str] = set()
        for doc in self.documents:
            found |= code_tokens(doc.meta.title)
            for p in doc.passages:
                found |= code_tokens(" ".join(p.section))
                if p.record:
                    found |= code_tokens(p.record.get("Code", ""))
        return frozenset(found)


def row_label(record: dict[str, str]) -> str:
    """The human name of a table row: the fee's Service, a glossary Term, a tier, or the first cell."""
    for key in ("Service", "Fee", "Term", "Plan", "Tier", "Stage"):
        if key in record:
            return record[key] if key != "Tier" else f"Tier {record[key]}"
    return next(iter(record.values()))
