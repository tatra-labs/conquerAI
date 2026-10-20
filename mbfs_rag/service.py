"""KnowledgeService.answer(): question -> decision, cited answer, confidence, and a decision trace."""

from __future__ import annotations

import logging
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import date

from .config import Settings
from .corpus import Corpus, row_label
from .encoders import Embedder, Reranker
from .evidence import EvidenceBuilder, FacetEvidence, Segment
from .gate import Signals, Verdict, combine, decide, make_verdict
from .query import QueryAnalysis, analyze, content_terms
from .retrieval import Retriever
from .text import code_tokens

log = logging.getLogger(__name__)


@dataclass
class Citation:
    n: int
    span_id: str
    doc_id: str
    title: str
    section: str
    doc_type: str
    audience: str
    source_path: str
    lines: tuple[int, int]
    char_range: tuple[int, int]
    quote: str
    effective_from: str | None
    validity: str
    cell: dict | None = None  # {"row": ..., "column": ...} for table citations


@dataclass
class Answer:
    query: str
    as_of: str
    decision: str
    answer: str
    citations: list[Citation]
    confidence: dict
    segments: list[dict]
    gaps: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    internal_content: bool = False
    generator: str = "extractive"
    trace: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class KnowledgeService:
    def __init__(
        self,
        corpus: Corpus,
        settings: Settings,
        embedder: Embedder | None,
        reranker: Reranker | None,
        *,
        generator=None,
        contextual: bool = True,
        verifier=None,
    ):
        self.corpus, self.settings = corpus, settings
        self.reranker = reranker
        self.verifier = None if "verifier" in settings.ablate else verifier
        self.retriever = Retriever(corpus, settings, embedder, reranker, contextual=contextual)
        self.evidence = EvidenceBuilder(corpus, settings, reranker, verifier=self.verifier)
        self.generator = generator

    @classmethod
    def load(cls, settings: Settings | None = None, *, use_llm: bool | None = None) -> KnowledgeService:
        """Build the service from disk. Models load from the local cache only (see `download-models`).
        The Claude generator is used only when an API key is configured and `use_llm` is not False."""
        from .encoders import FastEmbedEmbedder, FastEmbedReranker, FastEmbedVerifier
        from .ingest import load_corpus

        settings = settings or Settings()
        corpus = load_corpus(settings.corpus_dir)
        generator = None
        if use_llm or (use_llm is None and os.environ.get("ANTHROPIC_API_KEY")):
            from .claude import ClaudeGenerator

            try:
                generator = ClaudeGenerator.from_env(corpus, settings)
            except ImportError:
                if use_llm:  # asked for explicitly: say so
                    raise
                log.warning(
                    "ANTHROPIC_API_KEY is set but the anthropic SDK cannot be imported; answering extractively "
                    "(pip install -e .[llm] to use Claude)"
                )
        return cls(
            corpus,
            settings,
            FastEmbedEmbedder(settings),
            FastEmbedReranker(settings),
            generator=generator,
            verifier=FastEmbedVerifier(settings),
        )

    # ------------------------------------------------------------------ main entry point

    def answer(
        self, query: str, as_of: date | None = None, *, explain: bool = False, exclude: frozenset[str] = frozenset()
    ) -> Answer:
        """Thread-safe: all per-request state lives in locals, so one service can serve concurrent requests."""
        started = time.perf_counter()
        ablate = self.settings.ablate
        glossary = [] if "glossary" in ablate else self.corpus.glossary
        analysis = analyze(query or "", glossary, self.corpus.snapshot, strip="filler" not in ablate)
        # An explicit as_of (API, CLI, eval) is a full point-in-time view. A date in the question ("the chargeback
        # fee in 2025", "my 1099-K for 2025") only chooses between versions of price lists and notices: the
        # knowledge base itself is read as it stands on the snapshot date (corpus.state_on).
        kb_date = None if as_of else self.corpus.snapshot
        as_of = as_of or analysis.as_of_override or self.corpus.snapshot
        trace: dict = {
            "analysis": _analysis_trace(analysis),
            "as_of": as_of.isoformat(),
            "kb_date": (kb_date or as_of).isoformat(),
            "facets": [],
        }

        if not analysis.text.strip():
            verdict = decide(Signals(empty=True))
            return self._finish(as_of, kb_date, verdict, [], analysis, trace, started, explain)

        topical = self._topical(analysis)
        evidences: list[FacetEvidence] = []
        verdicts: list[Verdict] = []
        for facet in analysis.facets:
            t0 = time.perf_counter()
            candidates, rtrace = self.retriever.retrieve(facet.expanded, as_of, exclude=exclude, kb_date=kb_date)
            ev = self.evidence.build(facet, candidates, as_of, facet.amounts, kb_date=kb_date, exclude=exclude)
            if self.verifier is not None and ev.primary is not None:
                # The verifier judges whether the unit about to be quoted answers the question (scored already if it
                # was on the shortlist the primary was chosen from).
                verified = ev.verifier_scores.get(ev.primary.passage_id)
                if verified is None:
                    verified = self.verifier.score(facet.expanded, ev.primary.text)
                ev.ranker_score, ev.score = ev.score, verified
            signals = self._signals(analysis, ev, topical)
            verdict = decide(signals)
            if "gate" in ablate:  # ablation: answer from whatever the best evidence is
                verdict = Verdict("answer", "no_gate", "low", "Ablation: gate disabled.")
            evidences.append(ev)
            verdicts.append(verdict)
            trace["facets"].append(
                {
                    "facet": facet.text,
                    "expanded": facet.expanded,
                    "plans": list(facet.plans),
                    "retrieval": asdict(rtrace),
                    "evidence": ev.trace,
                    "ranker_score": ev.ranker_score,
                    "signals": asdict(signals),
                    "rule": verdict.rule,
                    "decision": verdict.decision,
                    "ms": round(1000 * (time.perf_counter() - t0)),
                }
            )
        verdict = combine(verdicts)
        if analysis.unanswered and verdict.decision == "answer":  # a question was left out: not a full answer
            verdict = make_verdict("partial", f"{verdict.rule} + too_many_parts")
        parts = list(zip(evidences, verdicts, strict=True))
        return self._finish(as_of, kb_date, verdict, parts, analysis, trace, started, explain)

    # ------------------------------------------------------------------ signals

    def _topical(self, analysis: QueryAnalysis) -> bool:
        """The question names something the KB covers: a title, heading or row-label word, or a form or fee
        code the KB uses ("1099-K", "WIR-03"), which the tokenizer would split into meaningless halves."""
        if analysis.expansions or code_tokens(analysis.text) & self.corpus.codes:
            return True
        return any(t in self.corpus.topical_terms for t in content_terms(analysis.text))

    def _signals(self, analysis: QueryAnalysis, ev: FacetEvidence, topical: bool) -> Signals:
        p = ev.primary
        verified = ev.ranker_score is not None
        floor = self.evidence.answer_floor(p, verified) if p else 0.0
        high = self.evidence.answer_high(p, verified) if p else 0.0
        return Signals(
            account_intent=analysis.account_intent,
            topical=topical,
            score=ev.score,
            floor=floor,
            high=high,
            relevance=ev.ranker_score,
            relevance_high=self.evidence.strong(p) if p else 0.0,
            too_many_rows=bool(ev.options),
            rows_matched=ev.rows_matched,
            gaps=tuple(ev.gaps),
            current=p is not None and p.state == "current",
            authority=p.meta.authority if p else 0,
            conflict=ev.conflict,
            outdated=bool(p and p.meta.outdated_by),
            outdated_only=bool(ev.withheld),
        )

    # ------------------------------------------------------------------ rendering

    def _finish(self, as_of, kb_date, verdict, parts, analysis, trace, started, explain) -> Answer:
        segments: list[Segment] = []
        gaps: list[str] = []
        multi = len(parts) > 1
        for ev, v in parts:
            if multi:  # a heading of its own, so the sub-question's words are never mistaken for the answer's
                segments.append(Segment("part", ev.facet.text, []))
            segments.extend(self._facet_segments(ev, v))
            gaps.extend(ev.gaps)
        if not parts:
            segments.append(Segment("route", "Please enter a question.", []))
        if parts and kb_date is not None and as_of != kb_date:
            when = f"{as_of:%B} {as_of.day}, {as_of.year}"
            segments.append(Segment("note", f"Fees, rates and notices are as of {when}, the date in the question.", []))
        if analysis.unanswered:
            skipped = f"{len(analysis.unanswered)} more question(s) in this message"
            segments.append(Segment("gap", f"Not looked up: {skipped}. Ask them one at a time.", []))
            gaps.append(skipped)

        generator = "extractive"
        # The model phrases answers the gate has already allowed; account lookups keep their fixed routing.
        if (
            self.generator is not None
            and parts
            and verdict.decision in ("answer", "partial")
            and all(v.decision in ("answer", "partial") for _, v in parts)
            and "account_lookup" not in verdict.rule
        ):
            # analysis.original is masked: a card number never reaches the model
            generated = self.generator.generate(analysis.original, as_of, parts, verdict, today=kb_date or as_of)
            if generated is not None:
                # Dated changes, conflicting sources and gaps are the gate's findings: they stay with the answer.
                answered = generated.verdict.decision in ("answer", "partial")
                kept = [s for s in segments if s.kind in ("note", "gap")] if answered else []
                segments, verdict, generator = [*generated.segments, *kept], generated.verdict, "claude"
                trace["llm"] = generated.trace
        if analysis.redacted:
            segments.append(Segment("note", "A long number that could be a card number was masked and not used.", []))

        citations, text, numbering = self._cite(segments)
        internal = any(c.audience == "internal" for c in citations)
        primary_scores = [ev.score for ev, _ in parts if ev.score is not None]
        confidence = {
            "level": verdict.level,
            "rule": verdict.rule,
            "action": verdict.action,
            "score": round(min(primary_scores), 2) if primary_scores else None,
            "reasons": _reasons(parts),
        }
        trace["ms_total"] = round(1000 * (time.perf_counter() - started))
        return Answer(
            query=analysis.original,  # card numbers masked: the raw question is never kept
            as_of=as_of.isoformat(),
            decision=verdict.decision,
            answer=text,
            citations=citations,
            confidence=confidence,
            segments=[
                {
                    "kind": s.kind,
                    "text": s.text,
                    "citations": list(dict.fromkeys(numbering[c] for c in s.span_ids)),
                    "internal": self._internal(s),
                }
                for s in segments
            ],
            gaps=gaps,
            notes=[s.text for s in segments if s.kind == "note"],
            internal_content=internal,
            generator=generator,
            trace=trace if explain else None,
        )

    def _facet_segments(self, ev: FacetEvidence, v: Verdict) -> list[Segment]:
        if v.rule.startswith("account_lookup"):
            lead = Segment(
                "route",
                "I can't see customer account data; this needs an account lookup after verifying the caller.",
                [],
            )
            return [lead, *ev.facts[:1]] if v.decision == "partial" else [lead]
        if v.rule == "outdated_only" and ev.withheld:
            old = ev.withheld[0].meta
            newer = ", ".join(self.corpus.meta(d).title for d in old.outdated_by)
            reviewed = f"{old.effective_from:%B} {old.effective_from.day}, {old.effective_from.year}"
            text = (
                f"The only source found is the {old.title} (last reviewed {reviewed}), which predates {newer}. "
                f"Its values may be out of date, so none are quoted: check the {newer}."
            )
            return [Segment("route", text, [])]
        if v.decision == "refuse":
            closest = self._closest_titles(ev)
            text = "I couldn't find this in the knowledge base."
            if ev.gaps:
                text = f"I couldn't find anything about {', '.join(repr(g) for g in ev.gaps)} in the knowledge base."
            if closest:
                text += f" Closest articles: {closest}."
            return [Segment("route", text, [])]
        if v.decision == "clarify":
            if ev.options:
                return [Segment("route", "Several fees match. Which one do you mean?", []), *ev.options]
            return [
                Segment(
                    "route",
                    "Could you say which product or process this is about (for example payouts, wires, "
                    "deposits, or card disputes)?",
                    [],
                )
            ]
        out = [*ev.facts, *ev.notes]
        for gap in ev.gaps:
            out.append(Segment("gap", f"Not covered by the knowledge base: {gap!r}.", []))
        return out

    def _closest_titles(self, ev: FacetEvidence) -> str:
        titles = list(dict.fromkeys(c.meta.title for c in ev.candidates[:5]))
        return "; ".join(titles[:2])

    def _internal(self, seg: Segment) -> bool:
        return any(self.corpus.meta(self.corpus.spans[s].doc_id).audience == "internal" for s in seg.span_ids)

    def _cite(self, segments: list[Segment]) -> tuple[list[Citation], str, dict[str, int]]:
        """Number citations in order of first use; returns the numbering rather than keeping it on self."""
        numbering: dict[str, int] = {}
        citations: list[Citation] = []
        lines = []
        for seg in segments:
            for sid in seg.span_ids:
                if sid not in numbering:
                    numbering[sid] = len(numbering) + 1
                    citations.append(self._citation(numbering[sid], sid))
            if seg.kind == "part":
                lines.append(f'"{seg.text}":')
                continue
            marker = "".join(f"[{n}]" for n in dict.fromkeys(numbering[sid] for sid in seg.span_ids))
            prefix = {"note": "Note: ", "option": "- "}.get(seg.kind, "")
            flag = "[Internal: do not read verbatim] " if self._internal(seg) else ""
            lines.append(f"{prefix}{flag}{seg.text}{(' ' + marker) if marker else ''}")
        return citations, "\n".join(lines), numbering

    def _citation(self, n: int, span_id: str) -> Citation:
        span = self.corpus.spans[span_id]
        passage = self.corpus.passages[span.passage_id]
        meta = self.corpus.meta(span.doc_id)
        cell = None
        if passage.kind == "table_row" and passage.record and span.label:
            cell = {"row": row_label(passage.record), "column": span.label, "code": passage.record.get("Code")}
        return Citation(
            n=n,
            span_id=span_id,
            doc_id=span.doc_id,
            title=meta.title,
            section=" > ".join(passage.section),
            doc_type=meta.doc_type,
            audience=meta.audience,
            source_path=f"corpus/{meta.path}",
            lines=(span.line_start, span.line_end),
            char_range=(span.start, span.end),
            quote=span.text,
            effective_from=meta.effective_from.isoformat() if meta.effective_from else None,
            validity=meta.state(self.corpus.snapshot) or "unknown",
            cell=cell,
        )


def _analysis_trace(a: QueryAnalysis) -> dict:
    return {
        "cleaned": a.text,
        "expansions": [list(e) for e in a.expansions],
        "facets": [f.text for f in a.facets],
        "plans": list(a.plans),
        "amounts": list(a.amounts),
        "account_intent": a.account_intent,
        "as_of_override": a.as_of_override.isoformat() if a.as_of_override else None,
        "card_number_masked": a.redacted,
    }


def _reasons(parts: list[tuple[FacetEvidence, Verdict]]) -> list[str]:
    reasons = []
    for ev, v in parts:
        p = ev.primary
        if p is not None and ev.score is not None:
            kind = "table row" if p.is_table else "passage"
            how = (
                f"verifier {ev.score:.1f}, ranker {ev.ranker_score:.1f}"
                if ev.ranker_score is not None
                else f"ranker {ev.score:.1f}"
            )
            reasons.append(f"Best {kind}: {p.meta.title} ({how}, {p.state}).")
        if ev.rows_matched > 1:
            reasons.append(f"{ev.rows_matched} table rows match; all are listed.")
        if ev.gaps:
            reasons.append(f"Not covered by the cited text: {', '.join(ev.gaps)}.")
        if ev.conflict:
            reasons.append("A lower-authority source states a different value.")
        reasons.append(f"Rule: {v.rule}.")
    return reasons
