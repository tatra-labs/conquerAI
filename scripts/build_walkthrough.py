"""Build the walkthrough page (walkthrough.html) from eval/results/results.json and the corpus.

    python scripts/build_walkthrough.py            # writes eval/results/walkthrough.html

Everything on the page is read from results.json or the corpus at build time; nothing is typed in by hand
except the narrative, so the page cannot drift from the numbers.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mbfs_rag.config import Settings  # noqa: E402
from mbfs_rag.evaluation import wilson  # noqa: E402
from mbfs_rag.ingest import load_corpus  # noqa: E402

RESULTS = ROOT / "eval" / "results" / "results.json"
V1_BLIND = ROOT / "eval" / "results" / "heldout_v1_blind" / "results.json"  # the single blind run of v1, kept as is
OUT = ROOT / "eval" / "results" / "walkthrough.html"
TEMPLATE = Path(__file__).with_name("walkthrough_template.html")

# Questions whose full decision path the page walks through, in this order (dev items: their traces were
# studied during development, so showing them reveals nothing about the held-out set).
SHOWCASE = ["D02", "D03", "D13", "D15", "D12", "D04", "D22", "D07"]

# What went wrong on each failing question, from reading its trace. Held-out entries are written after the
# single held-out run, from its traces; they explain a failure and never change the system.
DIAGNOSIS = {
    "D01": "The answer gives the 7-day retrieval-request window instead of the 10-day chargeback deadline. Both sit in "
    "the same section, and the cross-encoder ranks the retrieval-request sentence first (6.85) in every model tried "
    "(MiniLM-L6, jina-turbo, bge-reranker-v2-m3).",
    "D06": "The 'paid faster' part is refused. The question says 'upgrade' and 'paid faster' where the policy says "
    "'move from Bridge Essentials to Bridge Plus' and 'standard payouts'. Its best passage scores 1.37 against the "
    "prose floor of 1.73: a paraphrase gap the small cross-encoder does not bridge. The fee part is answered.",
    "D23": "Refused. 'My supplier emailed new bank details' needs the business-email-compromise warning in the wire "
    "policy (call a number you already know), but the unit chosen scores -4.7 on the verifier. The right behaviour is a "
    "partial answer that leads with the verification warning.",
    "D26": "The reranker prefers the notice's 'how to avoid the increase' paragraph (3.07) to its change table, even "
    "after 'next year' is rewritten to 'in 2027'. The answer is about the right notice, but not the list of increases.",
    "D27": "Aggregation: 'which fees are included or free on Premier' needs every row whose Premier cell says "
    "Included. Top-k retrieval plus row rendering returns the two PCI rows. This needs a table query path.",
    # held-out v2, from the single blind run's traces
    "V02": "Refused. A narrative about one payout ($300 short since Saturday): the KB explains weekend batch timing and "
    "what a payout deducts, across several documents, but no unit reads as its answer (verifier -6.5).",
    "V04": "Answered from the wrong sentence ('If you've lost your return label, call Business Support') for 'do I get "
    "my money back?'. The label wanted the bought-or-rented question, or the partial rules.",
    "V05": "Half right. The internal caller-verification table says what a view-only user may do, but not how to add one "
    "(an owner or admin, in the app); that sits in the online-banking article, which was not quoted.",
    "V07": "Right deadline (10 calendar days, weekends included), but not where to respond (the Merchant Portal's "
    "Disputes tab), the third fact the label requires.",
    "V08": "Weak: quoted the internal SOP's 'MBFS will never ask for these' instead of the steps in the security "
    "articles (call the fraud line, reset the password, lock the card).",
    "V09": "Refused both parts. A plan change takes effect from the next statement cycle and switching is free (Business "
    "Checking Plans), but the best unit scored -3.1.",
    "V11": "Right fee ($25.00), but not the waiver threshold ($10,000 average daily balance) the label requires: the fee "
    "row only points to Business Checking Plans.",
    "V12": "Refused: 'card machines' never meets 'BridgeTerminal'. A vocabulary gap the KB glossary does not cover.",
    "V13": "Answered what the PCI non-validation fee is, not whether it can be refunded once the SAQ is done: the answer "
    "is in the internal agent guidelines (agents cannot reverse it; Merchant Risk can).",
    "V14": "Refused. Tracing one wire needs the account, but the KB covers how to ask for a trace and its $35 fee; the "
    "label wanted that partial answer.",
    "V15": "Right batch close (8:00 PM ET), but not which plans pay out the next business day (Plus and Premier).",
    "V16": "Gave the fraud line and the phishing address, but not that MBFS never asks for a code, or to change the "
    "password.",
    "V17": "Picked the wrong sentence: 'A check without this endorsement may be rejected' instead of the wording just "
    "above it. Sentence choice inside a passage is still the small ranker's.",
    "V18": "Refused both parts: the same scenario as held-out v1's H11. The online-banking article (an owner or admin "
    "adds users in Settings) and the debit-card article (employee cards) score below the floor.",
    "V19": "Answered with the debit card's daily limits, correct for that reading. The label says clarify (which limit: "
    "card, ATM, deposits, wires?), so it counts as the one unsafe answer.",
}

# The blind v1 run, failure by failure, and what changed because of it (general fixes, not per-question ones).
V1_DIAGNOSIS = {
    "H01": "Refused. A narrative question (sales versus payout amounts): every passage scores about -6, though the KB "
    "explains what a payout deducts across several documents. No fix: no single passage reads as its answer.",
    "H03": "Answered from the 'we never ask for your passcode' passage. The label also wants the advice in another "
    "passage of the same article (change your password; report the text to phishing@), which ranked outside the window.",
    "H04": "Refused, though the right article ranked first (0.22): below the old prose floor of 1.73. Fixed by the "
    "realistic calibration.",
    "H06": "'merchant just got the notice' became a second question and was refused. Fixed: a statement after a "
    "question is its context. The first part also gave the dispute steps rather than the 10-day deadline.",
    "H08": "Refused. With 'savigns' misspelled the right passage scored -3.3, far below the old floor. Addressed by the "
    "realistic calibration, which includes typos.",
    "H09": "Unsafe. Nothing current cleared the floor, so the outdated FAQ answered as a last resort with '$40 on Plus', "
    "a stale value, under a warning. Fixed: an outdated source never supplies a fee, time or limit.",
    "H10": "Asked to clarify instead of routing to an account lookup: 'this cust's application' did not match rules "
    "written for 'my application'. Fixed: the account rules also read the agent's voice.",
    "H11": "Refused both parts, though the right articles ranked first for each (-4.3 and -1.1). Addressed by the "
    "realistic calibration.",
    "H12": "Answered that every merchant must complete the SAQ, but not the P2PE nuance the label asks for, which sits "
    "in another passage of the PCI policy.",
}

INVARIANCE_NOTE = (
    "Paraphrases in agent and customer shorthand, written by an agent that saw only the dev questions. Two causes "
    "account for the failures. Unknown shorthand: 'intl' is not a word the KB uses, so 'intl wire' reads as an "
    "uncovered modifier of 'wire', the same rule that rightly flags 'mortgage rate'. Casual phrasing: the "
    "cross-encoder scores 'got $400k sitting in Reserve Savings, is that covered by the FDIC?' below the floor that "
    "the formal question clears. Both are reported as measured: a shorthand list or a re-fit floor written after "
    "seeing these paraphrases would turn a blind test into a dev set."
)

QA = [
    (
        "Why not hand the retrieved passages to an LLM and let it answer?",
        "The failure modes in this brief are time (superseded and upcoming fees), authority (a stale FAQ "
        "outranking the policy), table structure (incoming versus outgoing rows) and knowing when not to answer. A "
        "generator fixes none of these, because they happen before it reads anything. So the default answer is "
        "extractive and verbatim. The Claude path is optional: it only phrases answers the gate allowed, every "
        "sentence must be cited, and every stated value must appear in what it cites.",
    ),
    (
        "How is the confidence signal designed?",
        "It is a level that tells the agent what to do: high means read it out, medium means check the source, low "
        "means share only what is cited, and none is a question or a route. Ordered, named rules set it from "
        "explicit signals: the verifier's score of the quoted unit against a per-unit-type floor, uncovered "
        "modifiers, rows matched, current or upcoming, authority, conflicts. The rule's name is in every answer. "
        "The number beside it is a raw logit, never a percentage, because it is not a probability.",
    ),
    (
        "Where do the thresholds come from, and why are they not tuned on the eval?",
        "From a realistic calibration set: about 200 questions written from the corpus alone in agent shorthand, "
        "customer words, situations and with typos, each verified against a verbatim quote, against 100 off-topic "
        "and not-covered ones, with nothing overlapping a test question. The fit minimises 10 times the off-topic "
        "answers plus the on-topic refusals, on the score the gate actually tests. The first floors were fitted on "
        "questions templated from the documents' headings, which score far higher than real ones, and the blind "
        "held-out v1 caught it (3 of 12). That is also why there is a verifier: on realistic questions no floor "
        "could make the small ranker's score work.",
    ),
    (
        "Why hybrid retrieval, and why no fusion?",
        "BM25 catches exact fee names, plan names and codes. The dense model catches paraphrase and typos. With a "
        "cross-encoder downstream, fusion only decides who gets scored, and RRF demoted exact BM25 hits the dense "
        "side missed. So both nominate 12 and the small cross-encoder ranks the union. The retrieval chart shows "
        "that this recovers almost everything that scoring every unit would. A second, stronger cross-encoder then "
        "verifies only the unit about to be quoted: ranking and answerability are different jobs.",
    ),
    (
        "What did you do about tables?",
        "Each row is a passage and each cell is a citable span, with footnotes resolved onto the cells that "
        "reference them. The cross-encoder picks the table but cannot pick the row, so structure does. Facet words "
        "derived from the rows narrow them, plan names pick the column, and interval lookup picks a savings tier, "
        "but only for a rate question that names the product. A notice's approved change is overlaid on the cell "
        "as a dated note, and only if the notice was published by then and amends the value shown.",
    ),
    (
        "How do you know the evaluation is honest?",
        "The matchers are typed and independent of the system's tokenizer: money:25 never matches $25,000 or "
        "$25.50. Gold is anchored to verbatim quotes, or to a fee-schedule row and column checked against the raw "
        "CSV with the standard csv module. A fact counts only if a sentence both states it and cites a span that "
        "supports it. A code reviewer found the loophole before this rule, where a wrong FDIC answer passed. The "
        "held-out questions were written and labeled blind, then frozen by hash and run once. Every rate carries a "
        "Wilson interval, and failure predictions were written before the first run.",
    ),
    (
        "What does it get wrong?",
        "Ranking limits of a small cross-encoder: the 7-day versus 10-day chargeback distractor, the notice's advice "
        "paragraph outranking its change table, and casual paraphrases scored below the floor. Aggregation "
        "questions need a table query path. Unknown shorthand such as 'intl' is read as a missing modifier. Every "
        "failure is listed with its trace above.",
    ),
    (
        "Is it safe to put in front of agents?",
        "Unsafe answers, meaning a wrong or stale value or an answer where the label says refuse, are counted "
        "separately, and the count is on this page. Card numbers are masked however they are pasted. Internal "
        "procedures are flagged on every segment and citation. Account questions route to verification and are "
        "never sent to the model. The service is safe under concurrent requests, which a threaded test checks. A "
        "customer-facing deployment would also need to filter by audience before retrieval.",
    ),
    (
        "How would this scale to the real knowledge base?",
        "The in-process indices are fine to around 100k passages, with embeddings cached by content hash. Beyond "
        "that, a vector store replaces the dense index without touching the evidence or gate logic, which is "
        "per-query. Ingestion is driven by a CMS-style catalog: validity windows, supersession and audience come "
        "from metadata, so a re-ingest is incremental. Latency is about half a second warm on CPU, with ONNX "
        "capped at 4 threads.",
    ),
    (
        "What would you do with another week?",
        "Live-test the Claude path on dev and held-out, with risk-coverage per generator. Add a reader model with "
        "a null answer for the distractor cases. Add a table query path for aggregation. Mine shorthand from agent "
        "logs into the KB glossary the contact-center team already owns. Replace the synthetic calibration with "
        "logged agent feedback. Then the score could honestly become a probability.",
    ),
    (
        "How did you use AI tools, and what did you check yourself?",
        "Claude Code did the design critique before building (six lenses), drafted and fact-checked the corpus, "
        "ran two rounds of code review where a separate agent tried to refute every finding, and wrote and "
        "blind-labeled the held-out set. I checked library behaviour against the installed sources, read traces "
        "for every failure, and verified every number in the README against results.json. Some of what the tools "
        "got wrong is on this page, such as the stale FAQ used to fill a gap and the calibration that refused "
        "everything.",
    ),
]


SPLIT_NAMES = {
    "dev": "dev",
    "heldout_v1_blind": "held-out v1, blind, before the fixes",
    "heldout_v1": "held-out v1 again, after the fixes (not blind)",
    "heldout_v2": "held-out v2, blind, after the fixes",
}


def _results_intro(results: dict, v1_blind: dict | None) -> str:
    s = results.get("summary", {})
    parts = []
    if v1_blind and "heldout" in v1_blind.get("summary", {}):
        parts.append(f"held-out v1, blind: {v1_blind['summary']['heldout']['passed']} passed")
    for split in ("heldout_v2", "heldout_v1", "dev"):
        if split in s:
            parts.append(f"{SPLIT_NAMES[split]}: {s[split]['passed']} passed, {s[split]['unsafe']} unsafe")
    if not parts:
        return ""
    return (
        "; ".join(parts)
        + ". The first blind set exposed a miscalibrated floor and four other general faults; they were fixed, and a "
        "fresh blind set measured the result. Dev was used throughout development, so its numbers are optimistic by "
        "construction. Hover any mark for its value; each chart has its data underneath."
    )


def _rate(k: int, n: int) -> dict:
    lo, hi = wilson(k, n)
    return {"k": k, "n": n, "p": k / n if n else 0.0, "lo": lo, "hi": hi}


def _split_rates(items: list[dict], split: str) -> dict:
    rows = [i for i in items if i["split"] == split]
    if not rows:
        return {}
    answerable = [i for i in rows if i["acceptable"][0] in ("answer", "partial")]
    facts = sum(len(i["score"]["facts_matched"]) + len(i["score"]["facts_missing"]) for i in answerable)
    return {
        "n": len(rows),
        "passed": _rate(sum(i["score"]["passed"] for i in rows), len(rows)),
        "decision": _rate(sum(i["score"]["decision_ok"] for i in rows), len(rows)),
        "support": _rate(sum(len(i["score"]["supported"]) for i in answerable), facts),
    }


def _passage_info(corpus, pid: str) -> dict:
    p = corpus.passages.get(pid)
    if p is None:
        return {"title": pid, "section": "", "snippet": "", "kind": "prose"}
    meta = corpus.meta(p.doc_id)
    text = p.text if len(p.text) <= 220 else p.text[:217].rstrip() + "…"
    return {
        "title": meta.title,
        "section": " > ".join(p.section),
        "snippet": text,
        "kind": p.kind,
        "doc_type": meta.doc_type,
        "audience": meta.audience,
    }


def _trim_trace(trace: dict | None) -> dict | None:
    if not trace:
        return None
    facets = []
    for f in trace.get("facets", []):
        ev = f.get("evidence", {})
        rt = f.get("retrieval", {})
        facets.append(
            {
                "facet": f.get("facet"),
                "expanded": f.get("expanded"),
                "plans": f.get("plans"),
                "bm25_top": rt.get("bm25_top", [])[:5],
                "dense_top": rt.get("dense_top", [])[:5],
                "hidden_by_date": rt.get("hidden_by_date"),
                "pool_size": rt.get("pool_size"),
                "reranked": ev.get("reranked", [])[:8],
                "window": ev.get("window", []),
                "misses": ev.get("misses", {}),
                "primary": ev.get("primary"),
                "table": ev.get("table"),
                "coverage": ev.get("coverage"),
                "signals": f.get("signals"),
                "ranker_score": f.get("ranker_score"),
                "rule": f.get("rule"),
                "decision": f.get("decision"),
                "ms": f.get("ms"),
            }
        )
    return {"analysis": trace.get("analysis"), "as_of": trace.get("as_of"), "facets": facets}


def build() -> Path:
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    v1_blind = json.loads(V1_BLIND.read_text(encoding="utf-8")) if V1_BLIND.exists() else None
    settings = Settings()
    corpus = load_corpus(settings.corpus_dir)
    items = results["items"]

    passages: dict[str, dict] = {}
    slim_items = []
    for it in items:
        trace = _trim_trace(it.get("trace")) if it["id"] in SHOWCASE else None
        if trace:
            for f in trace["facets"]:
                ids = [r["passage"] for r in f["reranked"]] + f["window"] + [f["primary"] or ""]
                ids += [pid for pid, _ in f["bm25_top"]] + [pid for pid, _ in f["dense_top"]]
                for pid in ids:
                    if pid and pid not in passages:
                        passages[pid] = _passage_info(corpus, pid)
        s = it["score"]
        slim_items.append(
            {
                "id": it["id"],
                "split": it["split"],
                "query": it["query"],
                "tags": it["tags"],
                "acceptable": it["acceptable"],
                "predicted": it.get("predicted_failure", ""),
                "notes": it.get("notes", ""),
                "decision": s["decision"],
                "rule": s["rule"],
                "level": s["level"],
                "score": s["score"],
                "passed": s["passed"],
                "stage": s["stage"],
                "missing": s["facts_missing"],
                "forbidden": s["forbidden_hits"],
                "structural": s["structural"],
                "ms": s["ms"],
                "answer": s["answer"],
                "segments": it["segments"],
                "citations": [
                    {
                        "n": c["n"],
                        "span_id": c["span_id"],
                        "doc_id": c["doc_id"],
                        "title": c["title"],
                        "section": c["section"],
                        "doc_type": c["doc_type"],
                        "audience": c["audience"],
                        "source_path": c["source_path"],
                        "lines": c["lines"],
                        "char_range": c["char_range"],
                        "quote": c["quote"],
                        "validity": c["validity"],
                        "cell": c.get("cell"),
                    }
                    for c in it["citations"]
                ],
                "trace": trace,
            }
        )

    abl = results.get("ablations") or {}
    data = {
        "snapshot": results["generated_for_as_of"],
        "frozen": results["frozen"],
        "cold_ms": results["cold_start_ms"],
        "thresholds": results["calibration"]["in_use"],
        "summary": results["summary"],
        "rates": {
            **{split: _split_rates(items, split) for split in ("dev", "heldout_v1", "heldout_v2")},
            "heldout_v1_blind": _split_rates(v1_blind["items"], "heldout") if v1_blind else {},
        },
        "split_names": SPLIT_NAMES,
        "v1_blind": [
            {
                "id": i["id"],
                "query": i["query"],
                "acceptable": i["acceptable"],
                "decision": i["score"]["decision"],
                "rule": i["score"]["rule"],
                "passed": i["score"]["passed"],
                "stage": i["score"]["stage"],
                "diagnosis": V1_DIAGNOSIS.get(i["id"], ""),
            }
            for i in (v1_blind["items"] if v1_blind else [])
            if i["split"] == "heldout"
        ],
        "v1_blind_summary": (v1_blind or {}).get("summary", {}).get("heldout"),
        "items": slim_items,
        "passages": passages,
        "retrieval": results.get("retrieval", {}),
        "ablations": abl,
        "risk_coverage": results.get("risk_coverage", []),
        "operating_point": results.get("operating_point"),
        "distractors": results.get("distractors", []),
        "calibration": [
            {"label": s["label"], "kind": s["kind"], "score": s["score"]} for s in results["calibration"]["samples"]
        ],
        "invariance": results.get("invariance", []),
        "showcase": [i for i in SHOWCASE if any(x["id"] == i for x in items)],
        "diagnosis": DIAGNOSIS,
        "invariance_note": INVARIANCE_NOTE,
        "qa": QA,
        "results_intro": _results_intro(results, v1_blind),
    }
    html = TEMPLATE.read_text(encoding="utf-8")
    blob = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")
    OUT.write_text(html.replace("__DATA__", blob), encoding="utf-8", newline="\n")
    return OUT


if __name__ == "__main__":
    path = build()
    print(f"wrote {path} ({path.stat().st_size // 1024} KB)")
