"""Evaluation harness: python -m mbfs_rag eval [--set dev|heldout_v1|heldout_v2|all] [--ablations]

Gold matchers are typed regexes written independently of the system's tokenizer, so a normaliser bug
cannot hide itself (money matchers refuse "$25" inside "$25,000"; refusals must state no amounts).
Every number is reported as k/n with a Wilson interval: at n of about 25, one query is 4 points.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import statistics
import sys
import time
import tomllib
from dataclasses import asdict, dataclass, field, replace
from datetime import date
from pathlib import Path

from .config import PROJECT_ROOT, Settings

EVAL_DIR = PROJECT_ROOT / "eval"
RESULTS_DIR = EVAL_DIR / "results"
ABLATIONS = ("validity", "trust", "structure", "gaps", "glossary", "filler", "gate", "verifier")


# ---------------------------------------------------------------- dataset


@dataclass
class Support:
    doc: str
    quote: str | None = None  # prose: a verbatim quote from the canonical text
    row: str | None = None  # table: the row code (e.g. "WIR-03"), matched with the cell's column
    column: str | None = None


@dataclass
class Fact:
    id: str
    match: str  # typed matcher, see `matcher()`
    support: list[Support]
    required: bool = True


@dataclass
class Item:
    id: str
    query: str
    acceptable: list[str]
    tags: list[str] = field(default_factory=list)
    facts: list[Fact] = field(default_factory=list)
    forbidden: list[dict] = field(default_factory=list)  # {"match": ..., "why": ...}, checked on asserted text only
    notes: str = ""
    predicted_failure: str = ""  # written before the first full run; reported, not hidden
    as_of: str | None = None
    expect: list[str] = field(default_factory=list)  # "internal": must be flagged; "gap": must name what's missing
    split: str = "dev"


def load_items(path: Path, split: str) -> list[Item]:
    raw = tomllib.loads(path.read_text(encoding="utf-8"))
    items = []
    for q in raw["query"]:
        facts = [
            Fact(f["id"], f["match"], [Support(**s) for s in f.get("support", [])], f.get("required", True))
            for f in q.get("facts", [])
        ]
        items.append(
            Item(
                id=q["id"],
                query=q["query"],
                acceptable=q["acceptable"],
                tags=q.get("tags", []),
                facts=facts,
                forbidden=q.get("forbidden", []),
                notes=q.get("notes", ""),
                predicted_failure=q.get("predicted_failure", ""),
                as_of=q.get("as_of"),
                expect=q.get("expect", []),
                split=split,
            )
        )
    return items


def matcher(spec: str) -> re.Pattern[str]:
    """money:45 | percent:2.25 | time:4pm | days:10 | text:some words | re:<regex>

    money matches the whole amount ($45 or $45.00, never $45.50 or $45,000); days also accepts "10-day";
    text matches the words in order, case-insensitively, allowing a plural or -ed/-ing ending on the last word."""
    kind, _, arg = spec.partition(":")
    if kind == "money":
        whole, _, cents = arg.partition(".")
        whole = f"{int(whole):,}".replace(",", ",?")
        cents_re = rf"\.{cents}" if cents and cents != "00" else r"(?:\.00)?"
        return re.compile(rf"\${whole}{cents_re}(?![\d,]*\d)(?!\.\d)")
    if kind == "percent":  # 2.25 matches 2.250%, 3.10 matches 3.1%, 1 matches 1.0%; never 10% or 12.25%
        whole, _, frac = arg.partition(".")
        frac = frac.rstrip("0")
        number = rf"{int(whole)}\.{frac}0*" if frac else rf"{int(whole)}(?:\.0+)?"
        return re.compile(rf"(?<![\d.]){number}\s?%")
    if kind == "time":
        m = re.fullmatch(r"(\d{1,2})(?::(\d{2}))?\s*([ap])m", arg)
        hour, minute, ap = m.groups()
        minute_re = f":{minute}" if minute and minute != "00" else "(?::00)?"
        return re.compile(rf"\b{hour}{minute_re}\s?{ap}\.?m\.?", re.IGNORECASE)
    if kind == "days":
        return re.compile(rf"\b{arg}[\s-]+(?:(?:calendar|business|banking)[\s-]+)?days?\b", re.IGNORECASE)
    if kind == "text":
        return re.compile(r"\b" + r"\s+".join(map(re.escape, arg.split())) + r"(?:s|es|ed|ing)?\b", re.IGNORECASE)
    if kind == "re":
        return re.compile(arg)
    raise ValueError(f"unknown matcher {spec!r}")


_AMOUNT = re.compile(r"\$\d|\d%")


# ---------------------------------------------------------------- scoring one answer


@dataclass
class Score:
    item: str
    decision: str
    decision_ok: bool
    decision_exact: bool
    facts_matched: list[str]
    facts_missing: list[str]
    supported: list[str]  # required facts with at least one citation on an acceptable support
    forbidden_hits: list[str]
    structural: list[str]  # citations to superseded, outdated-as-evidence, or wrong-version content
    passed: bool
    stage: str  # where it went wrong, or "ok"
    score: float | None
    rule: str
    level: str
    ms: int
    answer: str


def _support_spans(item: Item, corpus) -> dict[str, list[tuple[str, int, int] | tuple[str, str, str | None]]]:
    out: dict[str, list] = {}
    for fact in item.facts:
        spans = []
        for s in fact.support:
            if s.quote is not None:
                doc = corpus.docs[s.doc]
                at = doc.text.find(s.quote)
                if at < 0:
                    raise ValueError(f"{item.id}/{fact.id}: quote not found in {s.doc}: {s.quote!r}")
                spans.append(("prose", s.doc, at, at + len(s.quote)))
            else:
                spans.append(("row", s.doc, s.row, s.column))
        out[fact.id] = spans
    return out


def _citation_hits(citation: dict, supports: list) -> bool:
    for sup in supports:
        if sup[0] == "prose":
            _, doc, start, end = sup
            if citation["doc_id"] == doc and citation["char_range"][0] < end and start < citation["char_range"][1]:
                return True
        else:
            _, doc, row, column = sup
            cell = citation.get("cell") or {}
            if citation["doc_id"] != doc or cell.get("code") != row:
                continue
            cited_column = cell.get("column") or ""
            # The row's label, unit, notes and footnotes support it; another plan's column does not.
            if column is None or cited_column == column or not cited_column.startswith("Bridge "):
                return True
    return False


def _wrong_plans(citations: list[dict], supports: list) -> list[str]:
    """A plan-specific gold row cited only in other plans' columns: the answer gives another plan's price.
    Listing every plan, the asked one included, is not an error."""
    gold: dict[tuple[str, str], set[str]] = {}
    for sup in supports:
        if sup[0] == "row" and sup[3]:
            gold.setdefault((sup[1], sup[2]), set()).add(sup[3])
    cited: dict[tuple[str, str], set[str]] = {}
    for c in citations:
        cell = c.get("cell") or {}
        column = cell.get("column") or ""
        key = (c["doc_id"], cell.get("code"))
        if column.startswith("Bridge ") and key in gold:
            cited.setdefault(key, set()).add(column)
    return [
        f"cites {', '.join(sorted(columns))} instead of {', '.join(sorted(gold[key]))} for {key[1]}"
        for key, columns in cited.items()
        if not columns & gold[key]
    ]


def score_answer(item: Item, answer: dict, corpus, ms: int) -> Score:
    segments = answer["segments"]
    asserted = [s for s in segments if s["kind"] in ("fact", "option")]
    body = " ".join(s["text"] for s in asserted)
    by_n = {c["n"]: c for c in answer["citations"]}
    fact_citations = [by_n[n] for s in asserted for n in s["citations"]]
    supports = _support_spans(item, corpus)

    # A fact counts only where it is stated: some asserted segment must both state it and cite a span that
    # supports it. A value that merely appears somewhere (a tier boundary of $250,000 in an answer about APY)
    # is "unsupported", not a pass.
    matched, supported = [], []
    for fact in (f for f in item.facts if f.required):
        stating = [s for s in asserted if matcher(fact.match).search(s["text"])]
        if stating:
            matched.append(fact.id)
        if any(_citation_hits(by_n[n], supports[fact.id]) for s in stating for n in s["citations"]):
            supported.append(fact.id)
    missing = [f.id for f in item.facts if f.required and f.id not in matched]
    unsupported = [f for f in matched if f not in supported]
    said = body
    if answer.get("generator") == "claude":  # the model writes route text (a clarifying question) as well
        said = " ".join(s["text"] for s in segments if s["kind"] in ("fact", "option", "route"))
    forbidden = [f["match"] for f in item.forbidden if matcher(f["match"]).search(said)]
    structural = []
    as_of = date.fromisoformat(answer["as_of"])
    for c in fact_citations:
        meta = corpus.meta(c["doc_id"])
        if meta.state(as_of) == "superseded":
            structural.append(f"cites superseded {c['doc_id']}")
        if meta.outdated_by and not any("outdated" in s["text"] or "predates" in s["text"] for s in segments):
            # A last resort is allowed; citing it silently is not.
            structural.append(f"cites outdated {c['doc_id']} without a warning")
    structural += _wrong_plans(fact_citations, [s for spans in supports.values() for s in spans])
    decision = answer["decision"]
    decision_ok = decision in item.acceptable
    if decision in ("refuse", "clarify") and _AMOUNT.search(said):
        structural.append("abstention states an amount")
    answered = decision in ("answer", "partial")
    if "internal" in item.expect and answered and not answer["internal_content"]:
        structural.append("internal content not flagged")
    if (
        "gap" in item.expect
        and answered
        and not answer["gaps"]
        and not re.search(r"(?i)\b(?:do not|don't|does not) offer", body)
    ):
        missing.append("gap named")
    facts_ok = not missing and not unsupported if answered else True
    passed = decision_ok and facts_ok and not forbidden and not structural
    if passed:
        stage = "ok"
    elif not decision_ok:
        stage = f"decision ({decision}, wanted {'/'.join(item.acceptable)})"
    elif forbidden or structural:
        stage = "wrong or stale value"
    elif missing:
        stage = "missing fact"
    else:
        stage = "unsupported fact"
    conf = answer["confidence"]
    return Score(
        item.id,
        decision,
        decision_ok,
        decision == item.acceptable[0],
        matched,
        missing,
        supported,
        forbidden,
        sorted(set(structural)),
        passed,
        stage,
        conf.get("score"),
        conf.get("rule", ""),
        conf.get("level", ""),
        ms,
        answer["answer"],
    )


# ---------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for b vs c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(0, min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def fmt_rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k}/{n} ({100 * k / n:.0f}%, 95% CI {100 * lo:.0f}-{100 * hi:.0f}%)" if n else "0/0"


# ---------------------------------------------------------------- runs


def _service(settings: Settings, corpus, embedder, reranker, **kw):
    from .service import KnowledgeService

    return KnowledgeService(corpus, settings, embedder, reranker, **kw)


def run_items(service, items: list[Item]) -> list[tuple[Item, dict, Score]]:
    out = []
    for item in items:
        t0 = time.perf_counter()
        as_of = date.fromisoformat(item.as_of) if item.as_of else None
        answer = service.answer(item.query, as_of, explain=True).to_dict()
        ms = round(1000 * (time.perf_counter() - t0))
        out.append((item, answer, score_answer(item, answer, service.corpus, ms)))
    return out


def retrieval_ranks(service, items: list[Item]) -> dict[str, dict[str, int | None]]:
    """Rank of the first unit carrying an acceptable support, per retrieval mode (None = not retrieved)."""
    ranks: dict[str, dict[str, int | None]] = {}
    corpus = service.corpus
    retr = service.retriever
    for item in items:
        sup = [s for spans in _support_spans(item, corpus).values() for s in spans]
        if not sup:
            continue
        as_of = date.fromisoformat(item.as_of) if item.as_of else corpus.snapshot
        row = {}
        for mode in ("bm25", "dense", "hybrid_rrf", "full", "rerank_all"):
            cands, _ = retr.retrieve(item.query, as_of, mode=mode)
            rank = None
            for i, c in enumerate(cands, 1):
                if _unit_hits(c.passage, sup):
                    rank = i
                    break
            row[mode] = rank
        ranks[item.id] = row
    return ranks


def _unit_hits(passage, supports) -> bool:
    for sup in supports:
        if sup[0] == "prose":
            _, doc, start, end = sup
            if passage.doc_id == doc and passage.start < end and start < passage.end:
                return True
        elif passage.doc_id == sup[1] and passage.record and passage.record.get("Code") == sup[2]:
            return True
    return False


def calibration_questions(corpus) -> list[dict]:
    """The calibration set, eval/calibration.toml. It is not part of the evaluation. Positives are realistic
    questions (agent shorthand, customer words, a situation, typos) written from the corpus alone, each with a
    verbatim quote that answers it; negatives are off-topic and not-covered questions in the same styles. The
    first floors were fitted on questions templated from the documents' own headings, which score far higher than
    real questions do, and held-out v1 refused right answers for it."""
    raw = tomllib.loads((EVAL_DIR / "calibration.toml").read_text(encoding="utf-8"))
    out = []
    for q in raw.get("positive", []):
        doc = corpus.docs[q["doc"]]
        at = doc.text.find(q["quote"])
        if at < 0:
            raise ValueError(f"calibration quote not found in {q['doc']}: {q['quote']!r}")
        gold = frozenset(p.passage_id for p in doc.passages if p.start <= at < p.end)
        out.append({"question": q["query"], "gold": gold, "style": q.get("style", "")})
    for q in raw.get("negative", []):
        out.append({"question": q["query"], "gold": None, "style": q.get("style", ""), "kind": q.get("kind", "")})
    return out


def calibrate(service) -> list[dict]:
    """One sample per calibration question that reaches the floor rule: the logit of the unit the gate tests against
    the floor (the primary, chosen with the gate off so that the floor cannot shape its own evidence), whether that
    unit is the question's gold, and the unit type. A question an earlier rule decides (account data, no KB topic)
    never meets the floor, so it does not pull the floor either way."""
    nogate = _service(
        replace(service.settings, ablate=frozenset({"gate"})),
        service.corpus,
        service.retriever.embedder,
        service.reranker,
        verifier=service.verifier,
    )
    samples = []
    for q in calibration_questions(service.corpus):
        answer = nogate.answer(q["question"], explain=True)
        facets = (answer.trace or {}).get("facets") or []
        if not facets:
            continue
        first = facets[0]  # a calibration question is one question; one that splits is judged on its first part
        signals, primary = first["signals"], first["evidence"].get("primary")
        if signals["score"] is None or primary is None:
            continue
        samples.append(
            {
                "question": q["question"],
                "style": q["style"],
                "label": "positive" if q["gold"] else "negative",
                "kind": "table" if service.corpus.passages[primary].kind == "table_row" else "prose",
                "score": round(signals["score"], 3),
                "reaches_floor": signals["account_intent"] is None and signals["topical"],
                "top_is_gold": bool(q["gold"] and primary in q["gold"]),
            }
        )
    return samples


def suggest_thresholds(samples: list[dict], cost_wrong: float = 10.0, high_precision: float = 0.95) -> dict:
    """Per unit type, over the samples that reach the floor rule. Floor: minimise 10 x negatives answered + 1 x
    positives refused; where several cuts tie, take the middle one, not the most conservative edge. High: the
    lowest score at or above the floor where at least 95% of the questions scoring that high get their own gold
    unit as the answer (None when no cut reaches that precision)."""
    out = {}
    for kind in ("prose", "table"):
        rows = [s for s in samples if s["kind"] == kind and s.get("reaches_floor", True)]
        pos = [s["score"] for s in rows if s["label"] == "positive"]
        neg = [s["score"] for s in rows if s["label"] == "negative"]
        if not pos:
            out[kind] = None
            continue
        values = sorted({s["score"] for s in rows})
        cuts = [values[0] - 0.5] + [(a + b) / 2 for a, b in zip(values, values[1:], strict=False)] + [values[-1] + 0.5]

        costs = [cost_wrong * sum(n >= t for n in neg) + sum(p < t for p in pos) for t in cuts]
        tied = [t for t, c in zip(cuts, costs, strict=True) if c == min(costs)]
        floor = tied[len(tied) // 2]
        high = None
        for t in [c for c in cuts if c >= floor]:
            above = [s for s in rows if s["score"] >= t]
            if above and sum(s["top_is_gold"] for s in above) / len(above) >= high_precision:
                high = t
                break
        out[kind] = {
            "answer": round(floor, 2),
            "high": round(high, 2) if high is not None else None,
            "n_positive": len(pos),
            "n_negative": len(neg),
            "negatives_above_floor": sum(n >= floor for n in neg),
            "positives_below_floor": sum(p < floor for p in pos),
        }
    return out


def distractor_samples(service, items: list[Item]) -> list[dict]:
    """What the score can and cannot see, on the real dev questions: the gate's score of the best unit carrying gold
    support, versus its score of the top unit after deleting every gold document (the best on-topic distractor).
    The gate's score is the verifier's when there is one, else the ranker's."""
    corpus = service.corpus
    samples = []

    def gate_score(question: str, c) -> float:
        return service.verifier.score(question, c.text) if service.verifier is not None else c.logit

    for item in items:
        as_of = date.fromisoformat(item.as_of) if item.as_of else corpus.snapshot
        sup = [s for spans in _support_spans(item, corpus).values() for s in spans]
        answerable = any(d in ("answer", "partial") for d in item.acceptable[:1])
        cands, _ = service.retriever.retrieve(item.query, as_of)
        if answerable and sup:
            hit = next((c for c in cands if _unit_hits(c.passage, sup)), None)
            if hit is not None:
                samples.append(
                    {
                        "item": item.id,
                        "label": "gold present",
                        "kind": "table" if hit.is_table else "prose",
                        "score": round(gate_score(item.query, hit), 3),
                    }
                )
            gold_docs = frozenset(s[1] for s in sup)
            cands, _ = service.retriever.retrieve(item.query, as_of, exclude=gold_docs)
            label = "gold deleted"
        else:
            label = "not answerable"
        if cands:
            top = cands[0]
            samples.append(
                {
                    "item": item.id,
                    "label": label,
                    "kind": "table" if top.is_table else "prose",
                    "score": round(gate_score(item.query, top), 3),
                }
            )
    return samples


def invariance(service, base_answers: dict[str, dict]) -> list[dict]:
    """CheckList-style invariance: a paraphrase must get the same decision and cite the same documents."""
    path = EVAL_DIR / "invariance.toml"
    if not path.exists():
        return []
    rows = []
    for pair in tomllib.loads(path.read_text(encoding="utf-8"))["pair"]:
        base = base_answers.get(pair["base"])
        if base is None:
            continue
        para = service.answer(pair["query"]).to_dict()
        base_docs = {c["doc_id"] for c in base["citations"]}
        para_docs = {c["doc_id"] for c in para["citations"]}
        same_decision = para["decision"] == base["decision"]
        same_sources = (not base_docs and not para_docs) or bool(base_docs & para_docs)
        rows.append(
            {
                "base": pair["base"],
                "query": pair["query"],
                "base_decision": base["decision"],
                "decision": para["decision"],
                "same_decision": same_decision,
                "same_sources": same_sources,
                "passed": same_decision and same_sources,
                "answer": para["answer"],
            }
        )
    return rows


def risk_coverage(scores: list[Score], answerable: dict[str, bool]) -> list[dict]:
    """From the no-gate run: sort by score, answer the top k, measure risk (wrong among answered)."""
    ranked = sorted((s for s in scores if s.score is not None), key=lambda s: -s.score)
    curve, wrong = [], 0
    for k, s in enumerate(ranked, 1):
        supported = set(s.facts_matched) <= set(s.supported)
        correct = answerable[s.item] and not s.facts_missing and supported and not s.forbidden_hits and not s.structural
        wrong += 0 if correct else 1
        curve.append(
            {"coverage": round(k / len(ranked), 3), "risk": round(wrong / k, 3), "score": s.score, "item": s.item}
        )
    return curve


# ---------------------------------------------------------------- entry point


def run(settings: Settings, which: str = "all", *, ablations: bool = False) -> int:
    from .encoders import FastEmbedEmbedder, FastEmbedReranker, FastEmbedVerifier
    from .ingest import load_corpus

    sets = {
        "dev": EVAL_DIR / "dev.toml",
        "heldout_v1": EVAL_DIR / "heldout_v1.toml",
        "heldout_v2": EVAL_DIR / "heldout_v2.toml",
    }
    if which != "all" and not sets[which].exists():
        print(f"{sets[which]} not found; nothing was run and eval/results is unchanged", file=sys.stderr)
        return 2
    items = []
    for name, path in sets.items():
        if which in (name, "all") and path.exists():
            items += load_items(path, name)
    if not items:
        print("no eval items found; eval/results is unchanged", file=sys.stderr)
        return 2

    t0 = time.perf_counter()
    corpus = load_corpus(settings.corpus_dir)
    embedder, reranker, verifier = FastEmbedEmbedder(settings), FastEmbedReranker(settings), FastEmbedVerifier(settings)
    service = _service(settings, corpus, embedder, reranker, verifier=verifier)
    cold_ms = round(1000 * (time.perf_counter() - t0))

    runs = run_items(service, items)
    results: dict = {
        "generated_for_as_of": corpus.snapshot.isoformat(),
        "settings": {
            k: str(v) for k, v in asdict(settings).items() if k not in ("corpus_dir", "model_dir", "cache_dir")
        },
        "frozen": {
            name: hashlib.sha256(path.read_bytes()).hexdigest()[:16] for name, path in sets.items() if path.exists()
        },
        "cold_start_ms": cold_ms,
        "items": [
            {
                "id": i.id,
                "split": i.split,
                "query": i.query,
                "tags": i.tags,
                "acceptable": i.acceptable,
                "predicted_failure": i.predicted_failure,
                "notes": i.notes,
                "score": asdict(s),
                "trace": a.get("trace"),
                "citations": a["citations"],
                "segments": a["segments"],
            }
            for i, a, s in runs
        ],
    }
    by_split: dict[str, list[Score]] = {}
    for item, _, s in runs:
        by_split.setdefault(item.split, []).append(s)
    results["summary"] = {
        split: _summary(scores, [i for i, _, _ in runs if i.split == split]) for split, scores in by_split.items()
    }
    results["retrieval"] = retrieval_ranks(service, items)
    dev_items = [i for i in items if i.split == "dev"]
    calibration = calibrate(service)
    results["calibration"] = {
        "samples": calibration,
        "suggested": suggest_thresholds(calibration),
        "in_use": asdict(settings.thresholds),
    }
    results["distractors"] = distractor_samples(service, dev_items)

    answerable = {i.id: i.acceptable[0] in ("answer", "partial") for i in items}
    answered = [s for _, _, s in runs if s.decision in ("answer", "partial")]
    results["operating_point"] = {
        "coverage": round(len(answered) / len(runs), 3) if runs else 0,
        "risk": round(sum(not s.passed for s in answered) / len(answered), 3) if answered else 0,
    }
    results["invariance"] = invariance(service, {i.id: a for i, a, _ in runs})
    if ablations:
        matrix = {"full": {s.item: s.passed for _, _, s in runs}}
        for flag in ABLATIONS:
            ab = _service(replace(settings, ablate=frozenset({flag})), corpus, embedder, reranker, verifier=verifier)
            scores = [s for _, _, s in run_items(ab, items)]
            matrix[flag] = {s.item: s.passed for s in scores}
            if flag == "gate":  # the no-gate run answers everything: its scores give the risk-coverage curve
                results["risk_coverage"] = risk_coverage(scores, answerable)
        headers = _service(settings, corpus, embedder, reranker, contextual=False, verifier=verifier)
        matrix["headers"] = {s.item: s.passed for _, _, s in run_items(headers, items)}
        results["ablations"] = {
            "matrix": matrix,
            "mcnemar": {flag: _mcnemar(matrix["full"], col) for flag, col in matrix.items() if flag != "full"},
        }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(
        json.dumps(results, indent=1, ensure_ascii=False, default=str), encoding="utf-8", newline="\n"
    )
    from .report import write_reports

    write_reports(results, RESULTS_DIR)
    for split, summary in results["summary"].items():
        print(
            f"{split}: passed {summary['passed']}  |  decisions {summary['decision_ok']}  "
            f"|  unsafe answers {summary['unsafe']}"
        )
    print(f"wrote {RESULTS_DIR / 'results.json'}, report.md, report.html")
    return 0


def _mcnemar(full: dict[str, bool], other: dict[str, bool]) -> dict:
    b = sum(1 for k in full if full[k] and not other.get(k, False))
    c = sum(1 for k in full if not full[k] and other.get(k, False))
    return {
        "full_only": b,
        "ablation_only": c,
        "p_exact": round(mcnemar_exact(b, c), 4),
        "flipped": sorted(k for k in full if full[k] != other.get(k, False)),
    }


def _summary(scores: list[Score], items: list[Item]) -> dict:
    n = len(scores)
    kinds = {i.id: i.acceptable[0] for i in items}
    answerable = [s for s in scores if kinds[s.item] in ("answer", "partial")]
    abstain = [s for s in scores if kinds[s.item] in ("refuse", "clarify")]
    unsafe = [s for s in abstain if s.decision in ("answer", "partial")] + [
        s for s in scores if s.forbidden_hits or s.structural
    ]
    latencies = [s.ms for s in scores]
    return {
        "n": n,
        "passed": fmt_rate(sum(s.passed for s in scores), n),
        "decision_ok": fmt_rate(sum(s.decision_ok for s in scores), n),
        "decision_exact": fmt_rate(sum(s.decision_exact for s in scores), n),
        "false_refusals": fmt_rate(sum(s.decision in ("refuse", "clarify") for s in answerable), len(answerable)),
        "false_answers": fmt_rate(sum(s.decision in ("answer", "partial") for s in abstain), len(abstain)),
        "unsafe": len({s.item for s in unsafe}),
        "fact_recall": fmt_rate(
            sum(len(s.facts_matched) for s in answerable),
            sum(len(s.facts_matched) + len(s.facts_missing) for s in answerable),
        ),
        "citation_support": fmt_rate(
            sum(len(s.supported) for s in answerable),
            sum(len(s.facts_matched) + len(s.facts_missing) for s in answerable),
        ),
        "latency_ms": {
            "median": statistics.median(latencies) if latencies else None,
            "max": max(latencies) if latencies else None,
        },
        "confusion": _confusion(scores, kinds),
        "stages": dict(sorted({s.stage: sum(x.stage == s.stage for x in scores) for s in scores}.items())),
        "rules": dict(sorted({s.rule: sum(x.rule == s.rule for x in scores) for s in scores}.items())),
    }


def _confusion(scores: list[Score], kinds: dict[str, str]) -> dict[str, dict[str, int]]:
    labels = ("answer", "partial", "clarify", "refuse")
    table = {g: dict.fromkeys(labels, 0) for g in labels}
    for s in scores:
        table[kinds[s.item]][s.decision] += 1
    return table
