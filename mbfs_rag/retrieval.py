"""Hybrid retrieval: BM25 and a dense retriever each nominate candidates; a cross-encoder scores the union.

* BM25 — fee names, plan names, codes and amounts must match exactly ("Plus", "WIR-03", "1099-K").
  Title and section terms form a second, lower-weight field (a light BM25F), so heading context helps
  without letting a long title dominate a 25-word table row.
* Dense (bge-small, ONNX) — paraphrases and typos the lexical side cannot see.
* Cross-encoder — the most precise ranker in the stack and the only *absolute* relevance signal. With
  it downstream, fusion only decides who gets scored, so the pool is simply the union of both
  nominations (RRF would demote an exact BM25 hit that the dense side missed). RRF is kept for the
  no-reranker ablation.
* Validity — computed per `as_of` *before* candidates are generated: superseded content can never
  be retrieved, and upcoming content (an approved notice of future fees) is retrievable but flagged.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import date

import numpy as np

from .config import Settings
from .corpus import Corpus, DocMeta, Passage, state_on
from .encoders import Embedder, Reranker
from .text import terms

MODES = ("bm25", "dense", "hybrid_rrf", "full", "rerank_all")


class LexicalIndex:
    """Okapi BM25 with Lucene's always-positive IDF, log(1 + (N - df + 0.5) / (df + 0.5))."""

    def __init__(self, texts: Sequence[str], k1: float = 1.2, b: float = 0.75):
        self.k1, self.b = k1, b
        counts = [Counter(terms(t)) for t in texts]
        self.lengths = np.array([sum(c.values()) for c in counts], dtype=float)
        self.avg_length = float(self.lengths.mean()) if len(texts) else 0.0
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for i, c in enumerate(counts):
            for term, tf in c.items():
                self.postings[term].append((i, tf))
        self.size = len(texts)

    def idf(self, term: str) -> float:
        df = len(self.postings.get(term, ()))
        return math.log(1 + (self.size - df + 0.5) / (df + 0.5))

    def scores(self, query_terms: Sequence[str]) -> np.ndarray:
        scores = np.zeros(self.size)
        if not self.avg_length:
            return scores
        for term in set(query_terms):
            idf = self.idf(term)
            for i, tf in self.postings.get(term, ()):
                norm = tf + self.k1 * (1 - self.b + self.b * self.lengths[i] / self.avg_length)
                scores[i] += idf * tf * (self.k1 + 1) / norm
        return scores


def reciprocal_rank_fusion(rankings: Sequence[Sequence[int]], k: int = 60) -> dict[int, float]:
    """score(d) = sum over rankings of 1 / (k + rank), ranks starting at 1 (Cormack et al., 2009)."""
    fused: dict[int, float] = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(ranking, 1):
            fused[item] += 1.0 / (k + rank)
    return dict(fused)


def header_text(passage: Passage, meta: DocMeta) -> str:
    return " > ".join((meta.title, *passage.section))


@dataclass
class Candidate:
    passage: Passage
    meta: DocMeta
    state: str  # "current" | "upcoming": superseded content is never retrieved (the no-validity ablation relabels it)
    text: str  # "<title> > <section>: <passage>" — what the cross-encoder reads
    bm25_rank: int | None = None
    dense_rank: int | None = None
    logit: float | None = None

    @property
    def passage_id(self) -> str:
        return self.passage.passage_id

    @property
    def is_table(self) -> bool:
        return self.passage.kind == "table_row"


@dataclass
class RetrievalTrace:
    query: str
    mode: str
    hidden_by_date: int = 0
    bm25_top: list[tuple[str, float]] = field(default_factory=list)
    dense_top: list[tuple[str, float]] = field(default_factory=list)
    pool_size: int = 0


class Retriever:
    def __init__(
        self,
        corpus: Corpus,
        settings: Settings,
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
        *,
        contextual: bool = True,
    ):
        self.corpus, self.settings = corpus, settings
        self.embedder, self.reranker = embedder, reranker
        self.items: list[tuple[Passage, DocMeta]] = [(p, d.meta) for d in corpus.documents for p in d.passages]
        headers = [header_text(p, m) for p, m in self.items]
        self.texts = [f"{h}: {p.text}" if contextual else p.text for h, (p, _) in zip(headers, self.items, strict=True)]
        self.body = LexicalIndex([p.text for p, _ in self.items])
        self.header = LexicalIndex(headers if contextual else [""] * len(headers))
        self.vectors = embedder.embed(self.texts) if embedder else None

    def _states(self, as_of: date, exclude: frozenset[str], kb_date: date | None) -> list[str | None]:
        """Each unit's validity (corpus.state_on); None means it cannot be retrieved: superseded, not yet
        published, or excluded. So "my 1099-K for 2025" still sees the tax article published in 2026."""
        no_validity = "validity" in self.settings.ablate
        states = []
        for _, meta in self.items:
            state = None if meta.doc_id in exclude else state_on(meta, as_of, kb_date)
            if state == "superseded":
                state = "current" if no_validity else None
            states.append(state)
        return states

    def retrieve(
        self,
        query: str,
        as_of: date,
        *,
        mode: str = "full",
        exclude: frozenset[str] = frozenset(),
        kb_date: date | None = None,
    ) -> tuple[list[Candidate], RetrievalTrace]:
        """`as_of` decides every document's validity, unless `kb_date` is given: then the knowledge base is read
        as it stood on `kb_date` and only versioned documents follow `as_of`, a date taken from the question."""
        if mode not in MODES:
            raise ValueError(f"unknown retrieval mode {mode!r}; expected one of {MODES}")
        s = self.settings
        trace = RetrievalTrace(query, mode)
        states = self._states(as_of, exclude, kb_date)
        valid = np.array([st is not None for st in states])
        trace.hidden_by_date = int((~valid).sum())

        bm25_order: list[int] = []
        dense_order: list[int] = []
        if mode in ("bm25", "hybrid_rrf", "full"):
            q = terms(query)
            scores = self.body.scores(q) + s.header_weight * self.header.scores(q)
            scores = np.where(valid & (scores > 0), scores, -np.inf)
            bm25_order = [int(i) for i in np.argsort(-scores, kind="stable") if np.isfinite(scores[i])]
            trace.bm25_top = [(self.items[i][0].passage_id, round(float(scores[i]), 2)) for i in bm25_order[:8]]
        if mode in ("dense", "hybrid_rrf", "full"):
            if self.vectors is None or self.embedder is None:
                raise ValueError(f"mode {mode!r} needs the embedding model")
            scores = np.where(valid, self.vectors @ self.embedder.embed_query(query), -np.inf)
            dense_order = [int(i) for i in np.argsort(-scores, kind="stable") if np.isfinite(scores[i])]
            trace.dense_top = [(self.items[i][0].passage_id, round(float(scores[i]), 3)) for i in dense_order[:8]]

        k = s.pool_per_retriever
        if mode == "rerank_all":
            pool = [i for i in range(len(self.items)) if valid[i]]
        elif mode == "full":
            pool = list(dict.fromkeys(bm25_order[:k] + dense_order[:k]))
        else:  # ranking-only ablations: no cross-encoder, order is the ranking itself
            rankings = [r for r in (bm25_order, dense_order) if r]
            fused = reciprocal_rank_fusion([r[: s.rrf_k] for r in rankings], s.rrf_k)
            pool = sorted(fused, key=lambda i: (-fused[i], i))[: 2 * k]
        trace.pool_size = len(pool)

        bm25_rank = {i: r for r, i in enumerate(bm25_order, 1)}
        dense_rank = {i: r for r, i in enumerate(dense_order, 1)}
        candidates = [
            Candidate(
                self.items[i][0],
                self.items[i][1],
                states[i] or "current",
                self.texts[i],
                bm25_rank.get(i),
                dense_rank.get(i),
            )
            for i in pool
        ]
        if mode in ("full", "rerank_all"):
            if self.reranker is None:
                raise ValueError(f"mode {mode!r} needs the cross-encoder")
            for candidate, logit in zip(
                candidates, self.reranker.logits(query, [c.text for c in candidates]), strict=True
            ):
                candidate.logit = float(logit)
            candidates.sort(key=lambda c: (-(c.logit or 0.0), c.passage_id))
        return candidates, trace
