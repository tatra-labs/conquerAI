"""Tables — the one place where naive chunking silently produces wrong answers.

Decision: retrieve at ROW level, cite at CELL level.

* A row only means something together with its header, so every row becomes one passage rendered as
  "Header: value; Header: value; ...". BM25, the embedding model and the reranker all see the fee
  name, the plan names and the amounts together; a lone "$45.00" cell is never a retrieval unit.
* Each "Header: value" pair is its own citation span, so an answer about the Plus plan can cite exactly
  "Bridge Plus: $45.00" rather than the whole row.
* Footnote markers ("$12.00 [1]") are resolved into the row. A footnote retrieved without its row is
  useless, and a row retrieved without its footnote can be wrong ("waivable", "max 3 per day").
* The structured record is kept, so answers render exact values instead of paraphrasing numbers.

The same row renderer is used for CSV files, markdown tables and HTML tables.
"""

from __future__ import annotations

import csv
import io
import re

from ..corpus import Block

FOOTNOTE_MARK = re.compile(r"\[(\d+)\]")
FOOTNOTE_ROW = re.compile(r"^\[(\d+)\]\s*(.+)$")


def row_block(
    headers: list[str],
    values: list[str],
    section: tuple[str, ...],
    lines: tuple[int, int],
    footnotes: dict[str, str] | None = None,
) -> Block | None:
    """Render one table row as a self-describing block; empty cells are dropped."""
    cells = [(h.strip(), v.strip()) for h, v in zip(headers, values, strict=False) if h.strip() and v.strip()]
    if not cells:
        return None
    record = dict(cells)
    for number in sorted({m for _, v in cells for m in FOOTNOTE_MARK.findall(v)}, key=int):
        if footnotes and number in footnotes:
            cells.append((f"Footnote [{number}]", footnotes[number]))
    text = "; ".join(f"{h}: {v}" for h, v in cells)
    return Block("table_row", text, section, *lines, cells=cells, record=record)


def parse_csv(raw: str) -> list[Block]:
    """Parse an exported spreadsheet: title rows, a header row, data rows, then footnote/notes rows.

    The header is the first row with at least three non-empty cells; single-cell rows before it are the
    caption, single-cell rows after the data are footnotes ("[n] ...") or free-text notes.
    """
    reader = csv.reader(io.StringIO(raw))
    rows: list[tuple[int, list[str]]] = []
    for values in reader:
        rows.append((reader.line_num, [v.strip() for v in values]))

    def filled(values: list[str]) -> int:
        return sum(1 for v in values if v)

    header_at = next(i for i, (_, values) in enumerate(rows) if filled(values) >= 3)
    headers = rows[header_at][1]
    caption = [(line, values) for line, values in rows[:header_at] if filled(values)]
    body = [(line, values) for line, values in rows[header_at + 1 :] if filled(values)]

    footnotes: dict[str, str] = {}
    notes: list[tuple[int, str]] = []
    data: list[tuple[int, list[str]]] = []
    for line, values in body:
        if filled(values) >= 3:
            data.append((line, values))
            continue
        text = next(v for v in values if v)
        if match := FOOTNOTE_ROW.match(text):
            footnotes[match.group(1)] = match.group(2)
        elif text.lower().rstrip(":") not in {"footnotes", "notes"}:
            notes.append((line, text))

    blocks = [Block("text", " ".join(v for v in values if v), (), line, line) for line, values in caption]
    category_col = headers.index("Category") if "Category" in headers else None
    for line, values in data:
        section = (values[category_col],) if category_col is not None else ()
        if block := row_block(headers, values, section, (line, line), footnotes):
            blocks.append(block)
    if notes:
        blocks.append(Block("heading", "Notes", ("Notes",), notes[0][0], notes[0][0]))
        blocks.extend(Block("text", text, ("Notes",), line, line) for line, text in notes)
    return blocks
