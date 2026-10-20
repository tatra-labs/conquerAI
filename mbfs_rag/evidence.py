"""Turn reranked candidates into cited answer segments for one sub-question.

The cross-encoder is good at "is this passage about the question?" and bad at everything that
needs structure: it cannot tell "incoming" from "outgoing" rows apart, compare $30,000 with a tier
range, or know that an approved notice is not yet in effect. So it picks the evidence; structure
picks the value:

* primary source — among candidates within `authority_margin` logits of the best, prefer current over
  upcoming, then authority (policy > help article > FAQ), then sources not flagged outdated;
* tables — rows narrowed by the facet words that distinguish sibling rows, plan columns by the plans
  the question names, tiers by interval lookup; dated notice changes are overlaid on the cell;
* prose — the best contiguous run of sentences, each scored with its heading as context;
* gaps — a question word that the cited spans do not cover, sitting right before one they do
  ("credit card", "cashier's check"), means part of the question is not answered.
"""

from __future__ import annotations

import calendar
import copy
import re
from dataclasses import dataclass, field
from datetime import date

from .config import Settings
from .corpus import Corpus, FeeChange, Passage, Span, row_label, state_on
from .encoders import Reranker
from .query import PLANS as PLAN_COLUMNS
from .query import Facet, content_terms
from .retrieval import Candidate, header_text
from .text import covered, parse_money, plan_values, stem, terms, value_kind, values, words

FEE_INTENT = frozenset(terms("fee"))  # "how much / cost / price / charge" asks for a fee
_NUMBERED = re.compile(r"\d+[.)]\s")
PLAN_TERMS = frozenset(terms(" ".join(PLAN_COLUMNS)))
# Product nouns: a modifier in front of one of these picks a *different product* ("credit card" vs debit card,
# "cashier's check" vs a check you wrote, "mortgage rate" vs savings rate), so leaving it uncovered is a real gap.
HEAD_NOUNS = frozenset(
    terms(
        "card cards check checks account accounts loan loans rate rates wire wires transfer transfers payment payments "
        "deposit deposits fee fees terminal terminals statement statements payout payouts line lines order orders"
    )
)


def _disagree(ours: set[str], our_plans: dict[str, str], theirs: set[str], their_plans: dict[str, str]) -> bool:
    """Two sources disagree if they tie different values to the same plan, or state values of the same kind
    (money, percent, time, duration) that do not overlap at all."""
    if any(plan in our_plans and our_plans[plan] != value for plan, value in their_plans.items()):
        return True
    kinds = {value_kind(v) for v in ours} & {value_kind(v) for v in theirs}
    return any(not ({v for v in ours if value_kind(v) == k} & {v for v in theirs if value_kind(v) == k}) for k in kinds)


def expected_kinds(text: str) -> set[str]:
    """The kind of value a question asks for: "how much / fee / APY" a price or rate, "how long" a duration,
    "what time / cut-off" a time of day. A unit that states no value of that kind cannot be the answer."""
    t = text.lower()
    if re.search(r"\bwhat time\b|\bcut-?off\b|\bby when\b", t):
        return {"time"}
    if re.search(r"\bhow long\b|\bhow many (?:days|months|years|hours)\b|\bdeadline\b", t):
        return {"duration"}
    if _asks_price(t) or re.search(r"\bapy\b|\binterest rate\b|\bhow much\b", t):
        return {"money", "percent"}
    return set()


def _took_effect_after(meta, as_of: date) -> bool:
    """A policy or article read as of today that took effect after the date the question is about."""
    return not meta.versioned and meta.effective_from is not None and meta.effective_from > as_of


def _states_fee(text: str) -> bool:
    """A sentence that states a price or a rate: a $ or % value in a sentence about a fee, cost, rate or minimum."""
    kinds = {value_kind(v) for v in values(text)}
    return bool(kinds & {"money", "percent"}) and (_asks_price(text) or _asks_rate(text) or "minimum" in text.lower())


def _asks_rate(text: str) -> bool:
    """APY, interest, yield: the questions a tiered rate table answers. "Is my $400,000 FDIC insured?" is not one."""
    return bool(re.search(r"\b(?:apy|apr|interest|rates?|yield|earn(?:s|ing|ed)?|tiers?)\b", text, re.IGNORECASE))


def _asks_price(text: str) -> bool:
    return bool(
        re.search(
            r"\bhow much\b|\bcosts?\b|\bprices?\b|\bpricing\b|\bcharges?\b|\bcharged\b|\bfees?\b", text, re.IGNORECASE
        )
    )


# Listing every reading is the better answer for "how much is a wire?" (AmbigQA-style); only a question
# that matches more rows than an agent can scan ("what are your fees?") gets a clarifying question, which
# offers at most MAX_OPTIONS of them to choose from.
MAX_LISTED_ROWS = 6
MAX_OPTIONS = 8
# A unit that states no value of the kind asked for ("how much" with no price) counts as this many missing words.
WRONG_KIND_MISSES = 2


@dataclass
class Segment:
    kind: str  # "fact" | "option" | "note" | "gap"
    text: str
    span_ids: list[str] = field(default_factory=list)


@dataclass
class FacetEvidence:
    facet: Facet
    primary: Candidate | None = None
    score: float | None = None  # the gate's score of the primary: the verifier's logit, or the ranker's without it
    ranker_score: float | None = None  # the ranker's logit of the primary, when the verifier scored it
    verifier_scores: dict[str, float] = field(default_factory=dict)  # verifier logits of the window's shortlist
    is_table: bool = False
    facts: list[Segment] = field(default_factory=list)
    notes: list[Segment] = field(default_factory=list)
    options: list[Segment] = field(default_factory=list)  # filled when too many rows match
    rows_matched: int = 0
    gaps: list[str] = field(default_factory=list)  # uncovered "modifier head" phrases
    uncovered: list[str] = field(default_factory=list)  # other uncovered content words (soft)
    conflict: bool = False
    plan_values: dict[str, str] = field(default_factory=dict)  # plan -> canonical value of the rendered cell
    candidates: list[Candidate] = field(default_factory=list)  # every scored candidate, best first
    window: list[Candidate] = field(default_factory=list)  # the sources the answer was chosen from, notes included
    withheld: list[Candidate] = field(default_factory=list)  # outdated sources that would have supplied a value
    trace: dict = field(default_factory=dict)

    @property
    def span_ids(self) -> list[str]:
        return [sid for seg in self.facts + self.options for sid in seg.span_ids]

    def candidate(self, passage_id: str) -> Candidate | None:
        return next((c for c in self.candidates if c.passage_id == passage_id), None)


def _fmt(d: date) -> str:
    return f"{d:%B} {d.day}, {d.year}"


def _strip_markers(value: str) -> str:
    return re.sub(r"\s*\[\d+\](?:\[\d+\])*", "", value).strip()


class EvidenceBuilder:
    def __init__(self, corpus: Corpus, settings: Settings, reranker: Reranker | None, *, verifier=None):
        self.corpus, self.settings, self.reranker = corpus, settings, reranker
        self.t = settings.thresholds
        self.verifier = verifier  # judges the units it may quote, so evidence is assembled even when the ranker is weak
        self.verified = verifier is not None
        # Per-call state (sentence scores, the as-of date, the notice changes visible then) lives only on the
        # copy that build() makes.
        self._sentences: dict[str, list[tuple[Span, float]]] = {}
        self._as_of: date | None = None
        self._kb_date: date | None = None
        self._changes: list[FeeChange] = []

    def relevant(self, c: Candidate) -> float:
        """The ranker logit at which a candidate counts as relevant while evidence is assembled."""
        return self.t.relevance_table if c.is_table else self.t.relevance_prose

    def strong(self, c: Candidate) -> float:
        return self.relevant(c) + self.t.high_margin

    def answer_floor(self, c: Candidate, verified: bool) -> float:
        """The gate's floor for the primary's score: on the verifier's scale when it judged the unit."""
        if not verified:
            return self.relevant(c)
        return self.t.answer_table if c.is_table else self.t.answer_prose

    def answer_high(self, c: Candidate, verified: bool) -> float:
        return self.answer_floor(c, verified) + self.t.high_margin

    def _misses(self, c: Candidate, asked: set[str], kinds: set[str] = frozenset()) -> int:
        """Question terms missing from what the unit *is about*: a table row's label, section, title and
        plan columns (not its unit or notes, so "per month" does not make the PCI fee a monthly fee),
        or a prose passage's text, section and title. A unit that states no value of the kind the question
        asks for (a price, a duration, a time) counts one more miss."""
        if c.is_table and c.passage.record:
            record = c.passage.record
            if any(h in PLAN_COLUMNS for h in record):  # a priced row is about its label, not its unit or notes
                about = [row_label(record), *c.passage.section, c.meta.title, *(h for h in record if h in PLAN_COLUMNS)]
            else:  # a descriptive row (dispute stages, glossary) is about everything it says
                about = [c.passage.text, *c.passage.section, c.meta.title]
        else:
            about = [c.passage.text, *c.passage.section, c.meta.title]
        have = set(terms(" ".join(about)))
        misses = sum(1 for t in asked if not covered(t, have, self.corpus.vocabulary))
        if kinds and not {value_kind(v) for v in values(c.passage.text)} & kinds:
            misses += WRONG_KIND_MISSES  # it cannot answer "how much" without a price: worse than a missing word
        return misses

    def _score_sentences(self, query: str, candidates: list[Candidate]) -> None:
        """Score every sentence of the passages we may quote in ONE cross-encoder call, each with its heading and
        its context: the previous sentence of the same paragraph or item ("It is non-refundable." keeps its
        subject), or, for a list item, the list's lead-in ("If you think your card was skimmed:"). A list item never
        borrows the previous item's words: they are a different item, and borrowing them let a notice's paper-
        statement bullet answer an Instant Payout question."""
        jobs: list[tuple[str, Span, str]] = []
        for c in candidates:
            if c.is_table or c.passage_id in self._sentences:
                continue
            where = header_text(c.passage, c.meta)
            spans = list(c.passage.spans)
            lead: Span | None = None
            for i, s in enumerate(spans):
                prev = spans[i - 1] if i else None
                if prev is not None and (prev.line_start, prev.line_end) == (s.line_start, s.line_end):
                    context = prev.text
                elif lead is not None and (prev is lead or _NUMBERED.match(s.text)):
                    context = lead.text
                else:
                    context = ""
                jobs.append((c.passage_id, s, f"{where}: {(context + ' ') if context else ''}{s.text}"))
                if s.text.rstrip().endswith(":"):
                    lead = s
        if not jobs:
            return
        if self.reranker is None:
            logits = [0.0] * len(jobs)
        else:
            logits = [float(x) for x in self.reranker.logits(query, [text for _, _, text in jobs])]
        for (pid, span, _), logit in zip(jobs, logits, strict=True):
            self._sentences.setdefault(pid, []).append((span, logit))

    # ------------------------------------------------------------ entry point

    def build(
        self,
        facet: Facet,
        candidates: list[Candidate],
        as_of: date,
        amounts: tuple[float, ...],
        *,
        kb_date: date | None = None,
        exclude: frozenset[str] = frozenset(),
    ) -> FacetEvidence:
        """Thread-safe: the work runs on a shallow per-call copy, so concurrent requests sharing one builder
        never see each other's sentence scores or dates. Notice changes follow the retriever's validity rule
        (corpus.state_on), so a notice that was not yet published, or was excluded, overlays nothing."""
        call = copy.copy(self)
        call._sentences, call._as_of, call._kb_date = {}, as_of, kb_date
        call._changes = [
            c
            for c in self.corpus.changes
            if (doc_id := self.corpus.passages[c.passage_id].doc_id) not in exclude
            and state_on(self.corpus.meta(doc_id), as_of, kb_date) in ("current", "upcoming")
        ]
        ev = call._build(facet, candidates, as_of, amounts)
        past = kb_date is not None and as_of < kb_date
        p = ev.primary
        if past and "after_the_date" not in ev.trace and p is not None and _took_effect_after(p.meta, as_of):
            if any(_states_fee(f.text) for f in ev.facts):  # today's fee, from a source written after that date
                kept = [c for c in candidates if not _took_effect_after(c.meta, as_of)]
                ev = call._build(facet, kept, as_of, amounts)
                ev.trace["after_the_date"] = [c.passage_id for c in candidates if c not in kept]
        return ev

    def _build(
        self, facet: Facet, candidates: list[Candidate], as_of: date, amounts: tuple[float, ...]
    ) -> FacetEvidence:
        ev = FacetEvidence(facet)
        scored = [c for c in candidates if c.logit is not None]
        if self._kb_date is not None and as_of < self._kb_date and expected_kinds(facet.text) & {"money", "percent"}:
            # "The chargeback fee in 2025": a policy that took effect in 2026 states today's fee, not 2025's. A fee
            # or rate for a past date comes only from sources in effect then; with none, the question is refused.
            later = [c for c in scored if _took_effect_after(c.meta, as_of)]
            scored = [c for c in scored if c not in later]
            ev.trace["after_the_date"] = [c.passage_id for c in later]
        ev.candidates = scored
        ev.trace["reranked"] = [
            {
                "passage": c.passage_id,
                "logit": round(c.logit, 2),
                "state": c.state,
                "authority": c.meta.authority,
                "outdated": bool(c.meta.outdated_by),
            }
            for c in scored[:10]
        ]
        if not scored:
            return ev
        top = scored[0]
        ablate = self.settings.ablate
        above = [c for c in scored if c.logit >= self.relevant(c) or "gate" in ablate]
        if not above and self.verified:
            # Nothing the ranker calls relevant, which is common for real phrasing: assemble from the best candidates
            # anyway and let the verifier judge whether the unit answers the question.
            above = scored
            ev.trace["assembled_below_relevance"] = True
        # Tiers of trust: current and not flagged outdated, then upcoming (approved, not yet effective),
        # then outdated. A lower tier supplies the answer only when no higher tier has relevant evidence;
        # otherwise it can only appear as a note. The window is anchored on the best trusted candidate.
        tiers = (
            [c for c in above if c.state == "current" and not c.meta.outdated_by],
            [c for c in above if c.state == "upcoming"],
            [c for c in above if c.meta.outdated_by],
        )
        if "trust" in ablate:
            tiers = (above,)
        pool = next((t for t in tiers if t), [])
        if pool and len(tiers) == 3 and pool is tiers[2] and expected_kinds(facet.text):
            # Only an outdated source clears the floor. It may explain a process as a last resort, but it never
            # supplies a fee, time or limit: those are what went stale ("$40 on Plus" from a 2025 FAQ).
            ev.withheld = pool[:2]
            ev.trace["withheld_outdated"] = [c.passage_id for c in pool]
            return ev
        if not pool:
            ev.primary, ev.score, ev.is_table = top, top.logit, top.is_table
            return ev
        anchor = pool[0].logit
        window = [c for c in pool if c.logit >= anchor - self.t.authority_margin]
        asked = set(content_terms(facet.expanded)) | (FEE_INTENT if _asks_price(facet.text) else set())
        kinds = expected_kinds(facet.text)
        misses = {c.passage_id: self._misses(c, asked, kinds) for c in window}
        # The reranker ranks topical relevance; structure checks the fit: each question term missing from what a
        # unit is about (a row's label, a passage's text) costs `miss_penalty` logits. Authority breaks ties.
        penalty = 0.0 if "structure" in ablate else self.t.miss_penalty
        primary = max(window, key=lambda c: (c.logit - penalty * misses[c.passage_id], c.meta.authority))
        if self.verifier is not None and self.settings.verify_top > 1:
            # The verifier re-scores the strongest few units of the window, and the primary is chosen among them on
            # its scale, with the same fit penalty: the ranker's top choice is right only about half the time.
            shortlist = sorted(window, key=lambda c: -c.logit)[: self.settings.verify_top]
            scores = self.verifier.scores(facet.expanded, [c.text for c in shortlist])
            ev.verifier_scores = {c.passage_id: v for c, v in zip(shortlist, scores, strict=True)}
            primary = max(
                shortlist,
                key=lambda c: (ev.verifier_scores[c.passage_id] - penalty * misses[c.passage_id], c.meta.authority),
            )
            ev.trace["verifier_shortlist"] = {pid: round(v, 2) for pid, v in ev.verifier_scores.items()}
        # A rate question with an amount ("APY on $30,000 in savings") is answered from the tiered table - never from
        # a prose worked example about a different amount - provided it names the table's product. An amount alone
        # is not enough: "is my $400,000 in savings FDIC insured?" is about insurance, not the tier it falls in.
        if amounts and "structure" not in ablate and _asks_rate(facet.text):
            query_terms = set(terms(facet.expanded))
            tiered = next(  # from the trusted pool: an upcoming or outdated rate sheet cannot jump the tiers
                (c for c in pool if c.is_table and self._range_rows(c.meta.doc_id, query_terms, amounts)), None
            )
            if tiered is not None:
                primary = tiered
                if tiered not in window:
                    window = [tiered, *window]
        ev.primary, ev.score, ev.is_table = primary, primary.logit, primary.is_table
        ev.trace["window"] = [c.passage_id for c in window]
        ev.trace["misses"] = misses
        ev.trace["primary"] = primary.passage_id
        others = [
            c
            for c in above
            if c.passage_id not in {w.passage_id for w in window} and c.logit >= top.logit - self.t.authority_margin
        ]  # upcoming / outdated near the top
        window = window + others
        ev.window = window
        quotable = [c for c in window if c is not primary and not c.is_table][: self.settings.sentence_pool]
        self._score_sentences(facet.expanded, [primary, *quotable])

        if primary.is_table:
            self._table(ev, primary, scored, as_of, amounts)
        else:
            ev.facts.append(self._prose_segment(facet.expanded, primary))
        self._companions(ev, primary, window)
        self._gaps(ev, scored)
        if primary.state == "upcoming":
            when = _fmt(primary.meta.effective_from)
            ev.notes.insert(0, Segment("note", f"Upcoming — takes effect {when}; not in effect yet.", []))
        if primary.meta.outdated_by:
            ev.notes.append(self._outdated_note(primary))
        return ev

    # ------------------------------------------------------------ tables

    def _table(
        self, ev: FacetEvidence, primary: Candidate, scored: list[Candidate], as_of: date, amounts: tuple[float, ...]
    ) -> None:
        doc_id = primary.meta.doc_id
        # Sibling rows the reranker cannot tell apart (it scores "incoming" and "outgoing" within a fraction
        # of a logit of each other) are candidates for listing; anything clearly weaker is not.
        near = max(self.relevant(primary), primary.logit - self.t.row_margin)
        rows = [c for c in scored if c.is_table and c.meta.doc_id == doc_id and c.logit >= near]
        rows = list({c.passage_id: c for c in [primary, *rows]}.values())
        query_terms = set(terms(ev.facet.expanded)) - PLAN_TERMS  # plan names pick columns, never rows
        trace: dict = {"rows_in_pool": [c.passage_id for c in rows]}

        structure = "structure" not in self.settings.ablate
        ranged = self._range_rows(doc_id, query_terms, amounts) if structure else None
        if not structure:
            chosen = [c.passage for c in rows]
        elif ranged is not None:
            chosen = ranged
            trace["range_lookup"] = [p.passage_id for p in chosen]
        else:
            facet_words = set()
            for c in rows:
                facet_words |= self.corpus.facets.get((doc_id, c.passage.section), frozenset())
            asked = query_terms & facet_words
            if asked:

                def hits(c: Candidate) -> int:
                    own = asked & self.corpus.facets.get((doc_id, c.passage.section), frozenset())
                    return len(own & set(terms(row_label(c.passage.record or {}))))

                # Keep the rows whose labels cover the most facet words the question used: "outgoing international"
                # keeps one row, "incoming" keeps both incoming rows, "returned" keeps every returned-item fee.
                best = max(hits(c) for c in rows)
                rows = [c for c in rows if hits(c) == best] if best else rows
                trace["facet_words"] = sorted(asked)
            chosen = [c.passage for c in rows]
        ev.rows_matched = len(chosen)
        ev.trace["table"] = trace

        if len(chosen) > MAX_LISTED_ROWS:
            for passage in chosen[:MAX_OPTIONS]:
                label = row_label(passage.record or {})
                span = next((s for s in passage.spans if s.text.endswith(label)), passage.spans[0])
                ev.options.append(Segment("option", label, [span.span_id]))
            return
        prefix = ""
        if ranged is not None and len(amounts) == 1 and len(chosen) == 1:
            prefix = f"A balance of ${amounts[0]:,.2f} falls in "
        for passage in chosen:
            ev.facts.append(self._row_segment(passage, ev.facet.plans, as_of, ev.notes, prefix))
        if len(chosen) == 1 and chosen[0].record:
            for plan in [p for p in PLAN_COLUMNS if p in chosen[0].record]:
                found = values(_strip_markers(chosen[0].record[plan]))
                if found:
                    ev.plan_values[plan] = next(iter(found))

    def _range_rows(self, doc_id: str, query_terms: set[str], amounts: tuple[float, ...]) -> list[Passage] | None:
        """Interval lookup for tiered tables (minimum/maximum columns), only when the question names the
        table's product (so '$30,000 in my checking' never gets a savings tier)."""
        if not amounts:
            return None
        doc = self.corpus.docs[doc_id]
        tiers = [p for p in doc.passages if p.record and any(h.lower().startswith("minimum") for h in p.record)]
        if not tiers:
            return None
        product = (
            set(content_terms(doc.meta.title))
            - set(terms("rate sheet schedule fee effective"))
            - set(terms(" ".join(calendar.month_name[1:])))
        )
        if not product & query_terms:
            return None
        if len(amounts) > 1:
            return tiers
        amount = amounts[0]

        def bounds(p: Passage) -> tuple[float, float]:
            low_key = next(h for h in p.record if h.lower().startswith("minimum"))
            high_key = next((h for h in p.record if h.lower().startswith("maximum")), None)
            low = parse_money(p.record[low_key]) or 0.0
            high = parse_money(p.record.get(high_key, "")) if high_key else None
            return low, high if high is not None else float("inf")

        matched = [p for p in tiers if bounds(p)[0] <= amount <= bounds(p)[1]]
        return matched or tiers

    def _row_segment(
        self, passage: Passage, plans: tuple[str, ...], as_of: date, notes: list[Segment], prefix: str = ""
    ) -> Segment:
        record = passage.record or {}
        label = row_label(record)
        label_key = next((k for k, v in record.items() if v == label or f"Tier {v}" == label), None)
        spans: list[Span] = []
        if label_key and (s := passage.cell(label_key)):
            spans.append(s)
        plan_cols = [p for p in PLAN_COLUMNS if p in record]
        if plan_cols:
            shown = [p for p in plans if p in record] or plan_cols
            unit = record.get("Unit", "")
            parts = []
            for plan in shown:
                value = _strip_markers(record[plan])
                cell = passage.cell(plan)
                change = self._change(record.get("Code"), plan, value)
                if change and change.effective <= as_of:
                    notice = self.corpus.passages[change.passage_id]
                    value = f"{change.new_value} (changed from {change.old_value} on {_fmt(change.effective)})"
                    spans.extend(s for s in notice.spans if s.label and s.label.lower().startswith("new fee"))
                elif change:
                    notice = self.corpus.passages[change.passage_id]
                    notes.append(
                        Segment(
                            "note",
                            f"Scheduled change: {label} on {plan} goes from {change.old_value} to {change.new_value} "
                            f"on {_fmt(change.effective)} "
                            f"(notice published {_fmt(change.published or change.effective)}).",
                            [
                                s.span_id
                                for s in notice.spans
                                if s.label and (s.label.lower().startswith("new fee") or s.label == "Plan")
                            ],
                        )
                    )
                parts.append((plan, value))
                if cell:
                    spans.append(cell)
                    spans.extend(passage.footnotes_for(record[plan]))
            if len(shown) == 1:
                text = f"{label} on {shown[0]}: {parts[0][1]} {unit}".rstrip()
            elif len(shown) == len(plan_cols) and len({v for _, v in parts}) == 1:
                text = f"{label}: {parts[0][1]} {unit} on every plan".replace("  ", " ")
            else:
                text = f"{label}: " + "; ".join(f"{p} {v}" for p, v in parts) + (f" ({unit})" if unit else "")
            if unit and (s := passage.cell("Unit")):
                spans.append(s)
            if record.get("Notes") and (s := passage.cell("Notes")):
                text += f". {_strip_markers(record['Notes'])}"
                spans.append(s)
                spans.extend(passage.footnotes_for(record["Notes"]))
        else:
            others = [(h, v) for h, v in record.items() if h != label_key]
            text = f"{label}: " + "; ".join(f"{h} {_strip_markers(v)}" for h, v in others)
            spans.extend(s for s in passage.spans if s.label in {h for h, _ in others})
        footnote_text = " ".join(
            s.text.split(": ", 1)[1] for s in dict.fromkeys(spans) if s.label and s.label.startswith("Footnote")
        )
        text = f"{prefix}{text}." + (f" {footnote_text}" if footnote_text else "")
        return Segment("fact", text.replace("..", "."), list(dict.fromkeys(s.span_id for s in spans)))

    def _change(self, code: str | None, plan: str, value: str) -> FeeChange | None:
        """The visible notice change for this cell, provided it amends the value shown: the 2027 notice raises
        $12.00, so it says nothing about a 2025 schedule's $10.00."""
        for change in self._changes:
            if code and change.code == code and change.plan == plan and values(change.old_value) == values(value):
                return change
        return None

    # ------------------------------------------------------------ prose

    def _sentence_scores(self, query: str, passage: Passage) -> list[tuple[Span, float]]:
        cached = self._sentences.get(passage.passage_id)
        if cached is None:
            meta = self.corpus.meta(passage.doc_id)
            self._score_sentences(query, [Candidate(passage, meta, meta.state(self.corpus.snapshot) or "current", "")])
            cached = self._sentences.get(passage.passage_id, [(s, 0.0) for s in passage.spans])
        return cached

    def _prose_segment(self, query: str, candidate: Candidate, max_sentences: int = 3) -> Segment:
        """Best sentence plus contiguous neighbours that score within `sentence_margin` logits of it."""
        scored = self._sentence_scores(query, candidate.passage)
        best = max(range(len(scored)), key=lambda i: scored[i][1])
        keep = {best}
        lo = hi = best
        floor = scored[best][1] - self.t.sentence_margin
        while len(keep) < max_sentences:
            options = [i for i in (lo - 1, hi + 1) if 0 <= i < len(scored) and scored[i][1] >= floor]
            if not options:
                break
            nxt = max(options, key=lambda i: scored[i][1])
            keep.add(nxt)
            lo, hi = min(keep), max(keep)
        # A procedure is only useful whole: if part of a numbered list was chosen, include the whole list. Spans of one
        # list item share its block's line range, so a multi-sentence step stays together.
        blocks: list[list[int]] = []
        for i, (s, _) in enumerate(scored):
            if blocks and (scored[blocks[-1][0]][0].line_start, scored[blocks[-1][0]][0].line_end) == (
                s.line_start,
                s.line_end,
            ):
                blocks[-1].append(i)
            else:
                blocks.append([i])
        is_step = [bool(re.match(r"\d+\.\s", scored[b[0]][0].text)) for b in blocks]
        hit = [n for n, b in enumerate(blocks) if is_step[n] and keep & set(b)]
        if hit:
            lo_b = hi_b = hit[0]
            while lo_b > 0 and is_step[lo_b - 1]:
                lo_b -= 1
            while hi_b + 1 < len(blocks) and is_step[hi_b + 1] and hi_b - lo_b < 9:
                hi_b += 1
            for b in blocks[lo_b : hi_b + 1]:
                keep |= set(b)
        # A list item means little without its lead-in ("…may be approved only after enhanced due diligence:").
        first = min(keep)
        if first > 0 and scored[first - 1][0].text.rstrip().endswith(":"):
            keep.add(first - 1)
        chosen = [scored[i][0] for i in sorted(keep)]
        text = ""
        for span in chosen:  # list items carry no closing punctuation; keep them apart with "; "
            sep = "" if not text else (" " if text.rstrip()[-1:] in ".!?:" else "; ")
            text += sep + span.text
        return Segment("fact", text, [s.span_id for s in chosen])

    # ------------------------------------------------------------ other sources in the window

    def _companions(self, ev: FacetEvidence, primary: Candidate, window: list[Candidate]) -> None:
        """Other strong sources near the top: an upcoming change becomes a dated note; an outdated source, or a
        lower-authority one on the same topic, that states a different value of the same kind becomes a
        conflict note; a strong source on a different topic is offered as "also relevant"."""
        seen = {primary.meta.doc_id}
        notice_noted = any(n.text.startswith("Scheduled change") for n in ev.notes)
        primary_text = " ".join(f.text for f in ev.facts)
        primary_values = values(primary_text)
        primary_plans = ev.plan_values or plan_values(primary_text)
        for c in window:
            if c.meta.doc_id in seen or len(seen) > 3:
                continue
            seen.add(c.meta.doc_id)
            seg = self._unit_segment(ev, c, 2 if c.state == "upcoming" else 1)
            theirs = values(seg.text)
            if c.state == "upcoming":
                # Table answers get keyed "Scheduled change" notes; prose answers get an upcoming passage only if it is
                # about as relevant as the answer itself and says something different.
                if (
                    not notice_noted
                    and primary.state != "upcoming"
                    and not primary.is_table
                    and c.logit >= primary.logit - self.t.row_margin
                    and not theirs <= primary_values
                ):
                    ev.notes.append(
                        Segment("note", f"Upcoming (effective {_fmt(c.meta.effective_from)}): {seg.text}", seg.span_ids)
                    )
                    notice_noted = True
                continue
            if c.meta.outdated_by or (
                set(c.meta.topics) & set(primary.meta.topics) and c.meta.authority < primary.meta.authority
            ):
                seg = self._value_sentence(ev, c) or seg  # compare what the other source *states*, not its best filler
                theirs = values(seg.text)
                if _disagree(primary_values, primary_plans, theirs, plan_values(seg.text)):
                    ev.conflict = True
                    note = (
                        f'Conflicting source: {c.meta.title} says "{seg.text}" '
                        f"This answer follows {primary.meta.title} "
                        f"({primary.meta.doc_type.replace('_', ' ')}, effective {_fmt(primary.meta.effective_from)})."
                    )
                    if c.meta.outdated_by:
                        kind, reviewed = c.meta.doc_type.replace("_", " "), _fmt(c.meta.effective_from)
                        note += f" The {kind} was last reviewed {reviewed} and may be outdated."
                    ev.notes.append(Segment("note", note, seg.span_ids))
                continue
            if (
                c.is_table
                or c.logit < self.strong(c)
                or len(ev.facts) >= 3
                or set(c.meta.topics) & set(primary.meta.topics)
            ):
                continue  # same topic: the primary source already covers it (a worked example is not an alternative)
            # A second source earns a place only if it answers part of the question the first did not, or
            # if it is a competing answer (nearly as relevant, a different value of the same kind) —
            # e.g. card disputes (60 days) vs unauthorized ACH debits (1 business day).
            alt = self._prose_segment(ev.facet.expanded, c, 2)
            adds = self._uncovered(ev.facet, [*ev.facts]) - self._uncovered(ev.facet, [*ev.facts, alt])
            competing = c.logit >= primary.logit - self.t.competing_margin and _disagree(
                primary_values, primary_plans, values(alt.text), {}
            )
            if adds or competing:
                alt.text = f"Also relevant ({c.meta.title}): {alt.text}"
                ev.facts.append(alt)

    def _uncovered(self, facet: Facet, facts: list[Segment]) -> set[str]:
        evidence = self._evidence_terms([sid for seg in facts for sid in seg.span_ids])
        glossary = self._glossary_cover(facet, evidence)
        return {t for t in content_terms(facet.text) if not self._is_covered(t, evidence, glossary)}

    def _value_sentence(self, ev: FacetEvidence, c: Candidate) -> Segment | None:
        """The best-scoring sentence of a prose candidate that states a value (within `sentence_margin` of its best)."""
        if c.is_table:
            return None
        scored = self._sentence_scores(ev.facet.expanded, c.passage)
        best = max(score for _, score in scored)
        with_values = [(s, score) for s, score in scored if values(s.text) and score >= best - self.t.sentence_margin]
        if not with_values:
            return None
        span = max(with_values, key=lambda x: x[1])[0]
        return Segment("note", span.text, [span.span_id])

    def _unit_segment(self, ev: FacetEvidence, c: Candidate, max_sentences: int) -> Segment:
        """A short quotable segment from any candidate: a rendered row, or its best sentence(s)."""
        if c.is_table:
            return self._row_segment(c.passage, ev.facet.plans, self._as_of, [], "")
        return self._prose_segment(ev.facet.expanded, c, max_sentences)

    def _outdated_note(self, c: Candidate) -> Segment:
        newer = ", ".join(self.corpus.meta(d).title for d in c.meta.outdated_by)
        return Segment(
            "note",
            f"This comes from the {c.meta.title} (last reviewed {_fmt(c.meta.effective_from)}), "
            f"which predates {newer}; confirm before quoting.",
            [],
        )

    # ------------------------------------------------------------ coverage

    def _evidence_terms(self, span_ids: list[str]) -> set[str]:
        """What cited spans are about: their words, plus where they sit ("Wire Transfers FAQ" covers "transfer",
        and every row of the "Fee & Rate Schedule" is a fee)."""
        cited = [self.corpus.spans[sid] for sid in span_ids]
        evidence = set(terms(" ".join(s.text for s in cited)))
        for pid in {s.passage_id for s in cited}:
            passage = self.corpus.passages[pid]
            evidence |= set(terms(" ".join(passage.section))) | set(terms(self.corpus.meta(passage.doc_id).title))
        return evidence

    def _is_covered(self, term: str, evidence: set[str], glossary: set[str]) -> bool:
        """Plan names choose columns, so they are never the missing part of a question; a glossary term counts
        as covered when the evidence covers its meaning."""
        return term in PLAN_TERMS or term in glossary or covered(term, evidence, self.corpus.vocabulary)

    def _gaps(self, ev: FacetEvidence, scored: list[Candidate]) -> None:
        evidence = self._evidence_terms(ev.span_ids)
        glossary = self._glossary_cover(ev.facet, evidence)
        content = set(content_terms(ev.facet.text))
        surface = words(ev.facet.text)
        gaps, uncovered = [], []
        for i, word in enumerate(surface):
            t = stem(word)
            if t not in content or self._is_covered(t, evidence, glossary):
                continue
            nxt = stem(surface[i + 1]) if i + 1 < len(surface) else None
            if nxt and nxt in HEAD_NOUNS and self._is_covered(nxt, evidence, glossary):
                gaps.append(f"{word} {surface[i + 1]}")
            else:
                uncovered.append(word)
        if gaps:
            self._fill_gaps(ev, gaps, scored)
        if "gaps" in self.settings.ablate:
            uncovered.extend(gaps)
            gaps = []
        ev.gaps = gaps
        ev.uncovered = list(dict.fromkeys(uncovered))
        ev.trace["coverage"] = {"gaps": ev.gaps, "uncovered": ev.uncovered}

    def _glossary_cover(self, facet: Facet, evidence: set[str]) -> set[str]:
        out: set[str] = set()
        for entry in self.corpus.glossary:
            if re.search(rf"(?<![\w+]){re.escape(entry.term)}(?![\w+])", facet.text, re.IGNORECASE):
                if all(
                    covered(t, evidence, self.corpus.vocabulary) for t in content_terms(entry.meaning.split(" — ")[0])
                ):
                    out |= set(terms(entry.term))
        return out

    def _fill_gaps(self, ev: FacetEvidence, gaps: list[str], scored: list[Candidate]) -> None:
        """Before reporting a gap, look for a sentence that names the missing phrase and is relevant to the
        *whole* question ("We do not offer ... credit cards"). Scoring against the whole question, not the
        bare phrase, keeps an unrelated mention (a scam warning that names cashier's checks) out."""
        remaining = []
        primary_id = ev.primary.passage_id if ev.primary else None
        for gap in gaps:
            need = {stem(w) for w in gap.split()}
            best: tuple[Span, float] | None = None
            options = [
                c
                for c in scored
                if not c.is_table
                and c.passage_id != primary_id
                and c.state == "current"
                and not c.meta.outdated_by
                and need <= set(terms(c.passage.text))
            ]
            for c in options[:3]:
                for span, logit in self._sentence_scores(ev.facet.expanded, c.passage):
                    if (
                        need <= set(terms(span.text))
                        and logit >= self.t.relevance_prose
                        and (best is None or logit > best[1])
                    ):
                        best = (span, logit)
            if best:
                ev.facts.append(Segment("fact", best[0].text, [best[0].span_id]))
            else:
                remaining.append(gap)
        gaps[:] = remaining
