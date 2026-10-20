"""Markdown policies: headings give the section path, tables use the shared row renderer.

Change-log style sections ("Revision history") are dropped. They are metadata about the document,
and they are exactly where superseded values live ("Chargeback fee updated to $15.00").
"""

from __future__ import annotations

import re

from ..corpus import Block
from .tables import row_block

NON_CONTENT_SECTIONS = frozenset({"revision history", "change log", "changelog", "document history", "version history"})

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*#*\s*$")
_LIST_ITEM = re.compile(r"^\s*([-*+]|\d+[.)])\s+(.*)$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_FENCE = re.compile(r"^\s*(```|~~~)")


def clean_inline(text: str) -> str:
    """Strip inline markdown (links, emphasis, code) and collapse whitespace."""
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)
    text = re.sub(r"(\*\*|__)(.+?)\1", r"\2", text)
    text = re.sub(r"(?<![\w*])\*(?!\s)(.+?)(?<!\s)\*(?![\w*])", r"\1", text)
    text = re.sub(r"`([^`]+)`", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def _cells(line: str) -> list[str]:
    return [clean_inline(c) for c in line.strip().strip("|").split("|")]


def parse_markdown(raw: str) -> list[Block]:
    lines = raw.splitlines()
    blocks: list[Block] = []
    path: list[tuple[int, str]] = []  # heading stack below the H1 (the H1 is the document title)
    skip_below: int | None = None  # heading level of a non-content section being skipped
    paragraph: list[str] = []
    paragraph_start = 0

    def section() -> tuple[str, ...]:
        return tuple(text for _, text in path)

    def flush(end_line: int) -> None:
        if paragraph:
            blocks.append(Block("text", clean_inline(" ".join(paragraph)), section(), paragraph_start, end_line))
            paragraph.clear()

    i = 0
    while i < len(lines):
        line, number = lines[i], i + 1
        if heading := _HEADING.match(line):
            flush(number - 1)
            level, text = len(heading.group(1)), clean_inline(heading.group(2))
            if skip_below is not None and level <= skip_below:
                skip_below = None
            if level == 1:
                path.clear()
            else:
                while path and path[-1][0] >= level:
                    path.pop()
                path.append((level, text))
                if text.lower() in NON_CONTENT_SECTIONS:
                    skip_below = level
                elif skip_below is None:
                    blocks.append(Block("heading", text, section(), number, number))
            i += 1
            continue
        if skip_below is not None:
            i += 1
            continue
        if _FENCE.match(line):
            flush(number - 1)
            i += 1
            while i < len(lines) and not _FENCE.match(lines[i]):
                i += 1
            i += 1
            continue
        if _TABLE_ROW.match(line) and i + 1 < len(lines) and _TABLE_SEP.match(lines[i + 1]):
            flush(number - 1)
            headers, j = _cells(line), i + 2
            while j < len(lines) and _TABLE_ROW.match(lines[j]):
                if block := row_block(headers, _cells(lines[j]), section(), (j + 1, j + 1)):
                    blocks.append(block)
                j += 1
            i = j
            continue
        if item := _LIST_ITEM.match(line):
            flush(number - 1)
            marker, parts, j = item.group(1), [item.group(2)], i + 1
            while (
                j < len(lines)
                and lines[j].startswith((" ", "\t"))
                and lines[j].strip()
                and not _LIST_ITEM.match(lines[j])
            ):
                parts.append(lines[j].strip())
                j += 1
            prefix = f"{marker} " if marker[0].isdigit() else ""  # keep step numbers, drop bullets
            blocks.append(Block("text", prefix + clean_inline(" ".join(parts)), section(), number, j))
            i = j
            continue
        if line.lstrip().startswith(">"):
            flush(number - 1)
            quoted, j = [], i
            while j < len(lines) and lines[j].lstrip().startswith(">"):
                quoted.append(lines[j].lstrip()[1:].strip())
                j += 1
            blocks.append(Block("text", clean_inline(" ".join(quoted)), section(), number, j))
            i = j
            continue
        if not line.strip():
            flush(number - 1)
        else:
            if not paragraph:
                paragraph_start = number
            paragraph.append(line.strip())
        i += 1
    flush(len(lines))
    return blocks
