"""Load the catalogued corpus into canonical documents."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path

from ..corpus import Block, Corpus, DocumentBuilder, load_catalog, read_text
from .html import parse_html
from .markdown import parse_markdown
from .tables import parse_csv

log = logging.getLogger(__name__)

PARSERS: dict[str, Callable[[str], list[Block]]] = {".md": parse_markdown, ".html": parse_html, ".csv": parse_csv}


def load_corpus(corpus_dir: Path) -> Corpus:
    """Parse every published document in the catalog. Drafts are never ingested; a document that
    fails to parse raises instead of silently disappearing."""
    metas, snapshot = load_catalog(corpus_dir)
    documents = []
    for meta in metas:
        if meta.status != "published":
            log.info("not indexing %s (status=%s)", meta.doc_id, meta.status)
            continue
        path = corpus_dir / meta.path
        builder = DocumentBuilder(meta)
        try:
            for block in PARSERS[path.suffix](read_text(path)):
                builder.add(block)
        except Exception as exc:  # pragma: no cover - surfaced to the operator
            raise RuntimeError(f"failed to parse {meta.path}: {exc}") from exc
        documents.append(builder.build())
    catalogued = {m.path for m in metas}
    for path in sorted(p for p in corpus_dir.rglob("*") if p.suffix in PARSERS):
        if path.relative_to(corpus_dir).as_posix() not in catalogued:
            log.warning("%s is not in catalog.json and was not indexed", path)
    return Corpus(documents, snapshot)
