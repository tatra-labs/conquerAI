"""Build RESULTS.md: the results as plain text, readable in any editor without a browser.

    python scripts/build_results_md.py             # writes RESULTS.md at the project root

The results sections are copied from README.md, so the two cannot drift. One dev question is traced end to end
from eval/results/results.json; its gate step re-runs the gate's own rules on the recorded signals and fails the
build unless they reach the recorded rule and level. The risk-coverage chart of the report becomes a table.
"""

from __future__ import annotations

import json
import re
import sys
import tomllib
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from mbfs_rag.corpus import state_on  # noqa: E402
from mbfs_rag.evaluation import matcher  # noqa: E402
from mbfs_rag.gate import RULES, Signals, _level, make_verdict  # noqa: E402
from mbfs_rag.ingest import load_corpus  # noqa: E402

RESULTS = ROOT / "eval" / "results" / "results.json"
README = ROOT / "README.md"
OUT = ROOT / "RESULTS.md"
TRACED = "D02"  # a dev question: its trace was studied during development, so it reveals nothing held out

CHECKS = {  # what each gate rule tests, in the gate's order (mbfs_rag/gate.py RULES)
    "empty_input": "the question is empty",
    "account_lookup_with_policy": "it asks about the customer's own account, and policy evidence is strong",
    "account_lookup": "it asks about the customer's own account",
    "no_topical_term": "no word names anything the KB covers",
    "outdated_only": "a fee, time or limit was asked and only an outdated source has one",
    "no_relevant_evidence": "the verifier's score is below the floor for the unit type",
    "missing_key_term": "a key modifier is uncovered and the evidence is only moderately relevant",
    "too_many_matches": "more matching table rows than an agent can scan",
    "uncovered_modifier": "a modifier in the question is not covered by the evidence",
    "answer": "none of the above",
}


def section(readme: list[str], start: str, end: str, *, include_end: bool = False) -> str:
    """README lines from the first starting with `start` to the next starting with `end`, minus trailing rules."""
    i = next(n for n, line in enumerate(readme) if line.startswith(start))
    j = next(n for n in range(i + 1, len(readme)) if readme[n].startswith(end))
    lines = readme[i : j + 1 if include_end else j]
    while lines and lines[-1].strip() in ("", "---"):
        lines.pop()
    return "\n".join(lines)


def esc(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def yn(flag: bool) -> str:
    return "yes" if flag else "no"


def row_name(p) -> str:
    rec = p.record or {}
    return " ".join(x for x in (rec.get("Code"), rec.get("Service")) if x) or next(iter(rec.values()), "")


def label(corpus, pid: str) -> str:
    """A unit's document and section, plus its row for table rows (several rows share one section)."""
    p = corpus.passages.get(pid)
    if p is None:
        return pid
    name = " > ".join((corpus.meta(p.doc_id).title, *p.section))
    return f"{name}, row {row_name(p)}" if p.kind == "table_row" else name


def status(corpus, entry: dict) -> str:
    meta = corpus.meta(corpus.passages[entry["passage"]].doc_id)
    parts = ["FAQ" if meta.doc_type == "faq" else meta.doc_type.replace("_", " "), entry["state"]]
    if entry.get("outdated"):
        parts.append("flagged possibly outdated")
    return ", ".join(parts)


def hidden(corpus, as_of: date, kb_date: date | None, expected: int) -> str:
    """Which units the date filter removed (as retrieval._states: superseded or not visible)."""
    per_doc = Counter()
    for p in corpus.passages.values():
        if state_on(corpus.meta(p.doc_id), as_of, kb_date) in (None, "superseded"):
            per_doc[p.doc_id] += 1
    if sum(per_doc.values()) != expected:
        return f"{expected} units were not valid on {as_of} and could not be retrieved."
    parts = []
    for doc_id, n in per_doc.items():
        meta = corpus.meta(doc_id)
        total = sum(1 for p in corpus.passages.values() if p.doc_id == doc_id)
        why = "superseded" if meta.state(as_of) == "superseded" else "not yet published"
        parts.append(f"{'every unit' if n == total else f'{n} units'} of {meta.title}, which is {why}")
    return f"{expected} units could not be retrieved on {as_of}: {'; '.join(parts)}."


def observed(name: str, s: Signals) -> str:
    score = "none" if s.score is None else f"{s.score:+.2f}"
    gaps = ", ".join(s.gaps) or "none"
    return {
        "empty_input": f"empty: {yn(s.empty)}",
        "account_lookup_with_policy": f"account data asked: {s.account_intent or 'no'}",
        "account_lookup": f"account data asked: {s.account_intent or 'no'}",
        "no_topical_term": f"names a KB topic: {yn(s.topical)}",
        "outdated_only": f"value only in an outdated source: {yn(s.outdated_only)}",
        "no_relevant_evidence": f"score {score}, floor {s.floor:+.2f}",
        "missing_key_term": f"uncovered modifiers: {gaps}",
        "too_many_matches": f"rows matched: {s.rows_matched}",
        "uncovered_modifier": f"uncovered modifiers: {gaps}",
        "answer": "",
    }[name]


def level_reason(level: str, s: Signals) -> str:
    """Why an answer got its level, by the same tests as gate._level."""
    if level == "high":
        return "strong, one row, current, authoritative and uncontested"
    if level == "low":
        return f"the score is below the strong threshold ({s.high:+.2f})"
    reasons = []
    if s.rows_matched > 1:
        reasons.append(f"{s.rows_matched} table rows matched")
    if not s.current:
        reasons.append("the source is not current")
    if s.authority < 2:
        reasons.append("the source is an FAQ")
    if s.conflict:
        reasons.append("another source gives a conflicting value")
    if s.outdated:
        reasons.append("the source is flagged as possibly outdated")
    return f"the score clears the strong threshold ({s.high:+.2f}), but {' and '.join(reasons)}"


def gold(item: dict) -> dict:
    sets = tomllib.loads((ROOT / "eval" / f"{item['split']}.toml").read_text(encoding="utf-8"))
    return next(q for q in sets["query"] if q["id"] == item["id"])


def grading(it: dict) -> list[str]:
    """How the answer met its gold label, fact by fact (the scorer is evaluation.score_answer)."""
    g, sc = gold(it), it["score"]
    asserted = " ".join(x["text"] for x in it["segments"] if x["kind"] in ("fact", "option"))
    elsewhere = " ".join(x["text"] for x in it["segments"] if x["kind"] not in ("fact", "option"))
    out = [f"Decision: {sc['decision']}; acceptable: {', '.join(g['acceptable'])}."]
    for fact in g.get("facts", []):
        sup = fact.get("support", [])
        where = "; ".join(x["doc"] + (f" row {x['row']}, column {x['column']}" if "row" in x else "") for x in sup)
        cited = [
            str(c["n"])
            for c in it["citations"]
            for x in sup
            if c.get("cell") and c["cell"]["code"] == x.get("row") and c["cell"]["column"] == x.get("column")
        ]
        if fact["id"] not in sc["supported"]:
            result = "missing."
        elif cited:
            result = f"stated and cited ([{'], ['.join(cited)}])."
        else:
            result = "stated and cited."
        out.append(f"Required fact `{fact['id']}` (`{fact['match']}`, supported by {where}): {result}")
    for bad in g.get("forbidden", []):
        if bad["match"] in sc["forbidden_hits"]:
            result = "stated."
        elif matcher(bad["match"]).search(elsewhere) and not matcher(bad["match"]).search(asserted):
            result = (
                "not stated; it appears only quoted inside the conflict note, and forbidden values are checked "
                "only on what the answer asserts."
            )
        else:
            result = "not stated."
        out.append(f"Forbidden `{bad['match']}` ({bad['why']}): {result}")
    out.append(
        "Nothing it cites is superseded."
        if not sc["structural"]
        else f"Structural problems: {', '.join(sc['structural'])}."
    )
    return out


def trace_section(r: dict, corpus, item_id: str) -> str:
    it = next(i for i in r["items"] if i["id"] == item_id)
    tr = it["trace"]
    a = tr["analysis"]
    assert len(tr["facets"]) == 1, "the walkthrough below assumes a one-part question"
    f = tr["facets"][0]
    ev = f["evidence"]
    s = Signals(**{k: tuple(v) if isinstance(v, list) else v for k, v in f["signals"].items()})
    thresholds = r["calibration"]["in_use"]
    pen = thresholds["miss_penalty"]
    logit = {e["passage"]: e["logit"] for e in ev.get("reranked", [])}
    primary = corpus.passages[ev["primary"]]
    as_of = date.fromisoformat(tr["as_of"])
    kb_date = date.fromisoformat(tr["kb_date"]) if tr.get("kb_date") else None
    expansions = ", ".join(f"{k} → {v}" for k, v in dict(a["expansions"]).items()) or "none"

    lines = [
        f"## One question, end to end: {item_id}",
        "",
        f"> {it['query']}",
        "",
        "The real trace of this dev question from the final run, stage by stage: what each component saw and",
        "decided. Logits are raw cross-encoder scores, not probabilities.",
        "",
        "**1. Read the question.** "
        f'After cleaning: "{a["cleaned"]}". Parts: {len(a["facets"])}. '
        f"Plans named: {', '.join(a['plans']) or 'none'}. "
        f"Amounts: {', '.join(str(x) for x in a['amounts']) or 'none'}. "
        f"Glossary expansions: {expansions}. Account data asked: {a['account_intent'] or 'no'}. "
        f"Card number masked: {yn(a['card_number_masked'])}. "
        f"Date: {tr['as_of']}{' (from the question)' if a['as_of_override'] else ' (the corpus snapshot)'}.",
        "",
        "**2. Retrieve.** BM25 (body plus a heading field) and dense retrieval (bge-small) each nominate 12 "
        f"candidates; their union is {f['retrieval']['pool_size']}. "
        + hidden(corpus, as_of, kb_date, f["retrieval"]["hidden_by_date"]),
        "",
        "| BM25 top 5 | score | dense top 5 | cosine |",
        "|---|---|---|---|",
    ]
    for (b, bs), (d, ds) in zip(f["retrieval"]["bm25_top"][:5], f["retrieval"]["dense_top"][:5], strict=False):
        lines.append(f"| {esc(label(corpus, b))} | {bs:.2f} | {esc(label(corpus, d))} | {ds:.3f} |")

    window = list(ev.get("window", []))
    lines += [
        "",
        "**3. Rerank.** The small cross-encoder (MiniLM-L6) scores every candidate in the union. The top 8:",
        "",
        "| candidate | ranker logit | source, state | in the trusted window |",
        "|---|---|---|---|",
    ]
    for e in ev.get("reranked", [])[:8]:
        chosen = " **(chosen)**" if e["passage"] == ev.get("primary") else ""
        lines.append(
            f"| {esc(label(corpus, e['passage']))}{chosen} | {e['logit']:+.2f} | {status(corpus, e)} "
            f"| {yn(e['passage'] in window)} |"
        )
    lines += [
        "",
        "**4. Choose the evidence.** Trust comes before rank. The window holds the current sources not flagged as "
        f"outdated, within {thresholds['authority_margin']:g} logits of the best of them; an upcoming or outdated "
        "source near the top (here the FAQ that ranks first) can only appear as a note. Inside the window each "
        f"candidate pays {pen:g} logits per question term it does not cover; authority breaks ties.",
        "",
        f"| candidate in the window | ranker logit | question terms not covered | logit − {pen:g} × not covered |",
        "|---|---|---|---|",
    ]
    for pid in window:
        m = ev.get("misses", {}).get(pid, 0)
        lg = logit.get(pid)
        mark = " **(chosen)**" if pid == ev.get("primary") else ""
        shown = "" if lg is None else f"{lg:+.2f}"
        fit = "" if lg is None else f"{lg - pen * m:+.2f}"
        lines.append(f"| {esc(label(corpus, pid))}{mark} | {shown} | {m} | {fit} |")
    table = ev.get("table") or {}
    if table and primary.kind == "table_row":
        rows = [row_name(corpus.passages[pid]) for pid in table.get("rows_in_pool", [])]
        cells = [f"{plan} → {primary.record[plan]}" for plan in a["plans"] if plan in (primary.record or {})]
        stems = ", ".join(f"`{w}`" for w in table.get("facet_words", [])) or "none"
        lines += [
            "",
            "The chosen unit is a table row, so structure narrows the rows and picks the cell. Rows of this table "
            f"in the pool: {'; '.join(rows)}. Question words that narrow them (as stems): {stems}. "
            f"Rows left: {s.rows_matched}, {row_name(primary)}."
            + (f" The plan named picks the column: {', '.join(cells)}." if cells else ""),
        ]

    unit = "table row" if primary.kind == "table_row" else "prose unit"
    lines += [
        "",
        "**5. Verify.** A stronger cross-encoder (bge-reranker-v2-m3) scores only the unit about to be quoted: "
        f'{s.score:+.2f}, against the answer floor of {s.floor:+.2f} for a {unit} (and {s.high:+.2f} for "strong"). '
        f"The small ranker gave the same unit {f['ranker_score']:+.2f}; the two scales differ, so each has its own "
        "thresholds, fitted on the calibration set.",
        "",
        "**6. Gate.** Ordered rules over these signals; the first that fires decides. Re-run here with the gate's own "
        "code on the recorded signals:",
        "",
        "| # | rule | fires when | this question | fires |",
        "|---|---|---|---|---|",
    ]
    fired = None
    for n, (name, pred, decision) in enumerate(RULES, 1):
        hit = pred(s)
        verdict_cell = f"**yes: {decision}**" if hit else "no"
        lines.append(f"| {n} | `{name}` | {CHECKS[name]} | {observed(name, s)} | {verdict_cell} |")
        if hit:
            fired = (name, decision)
            break
    assert fired == (f["rule"], f["decision"]), (fired, f["rule"], f["decision"])
    level = _level(f["decision"], s)
    verdict = make_verdict(f["decision"], f["rule"], level)
    assert verdict.level == it["score"]["level"], (verdict.level, it["score"]["level"])

    lines += [
        "",
        f"**7. Answer.** Decision `{verdict.decision}`, level `{verdict.level}`: {level_reason(level, s)}. "
        f'What the agent is told to do: "{verdict.action}"',
        "",
        "```text",
        it["score"]["answer"],
        "```",
        "",
        "| # | source | cited text | where |",
        "|---|---|---|---|",
    ]
    for c in it["citations"]:
        cell = f" [{c['cell']['code']} / {c['cell']['column']}]" if c.get("cell") else ""
        lines.append(
            f'| {c["n"]} | {esc(c["title"])}{esc(cell)} | "{esc(c["quote"])}" | `{c["source_path"]}` line '
            f"{c['lines'][0]}, chars {c['char_range'][0]}-{c['char_range'][1]}, span `{c['span_id']}` |"
        )
    sc = it["score"]
    lines += [
        "",
        f"**Graded against its gold label: {'pass' if sc['passed'] else 'fail, ' + sc['stage']}.**",
        "",
        *[f"- {x}" for x in grading(it)],
    ]
    return "\n".join(lines)


def risk_table(r: dict) -> str:
    """The risk-coverage curve at about every tenth of coverage (the report draws it as a chart)."""
    pts = r["risk_coverage"]
    n = len(pts)
    lines = [
        "",
        "The curve in numbers. With the gate off, answer the questions in descending order of the gate's score and "
        "stop at each cut:",
        "",
        "| questions answered | gate score at the cut | failing among those answered |",
        "|---|---|---|",
    ]
    for k in sorted({max(1, round(n * q / 10)) for q in range(1, 11)}):
        p = pts[k - 1]
        lines.append(f"| {k} of {n} ({p['coverage']:.0%}) | {p['score']:+.2f} | {p['risk']:.0%} |")
    op = r["operating_point"]
    answered = [i for i in r["items"] if i["score"]["decision"] in ("answer", "partial")]  # as in evaluation.run
    failing = [i for i in answered if not i["score"]["passed"]]
    missing = [i for i in failing if i["score"]["stage"] == "missing fact"]
    decided = [i["id"] for i in failing if i["score"]["stage"].startswith("decision")]
    other = len(failing) - len(missing) - len(decided)
    assert len(answered) == round(op["coverage"] * n), (len(answered), op)
    low = max((k for k, p in enumerate(pts, 1) if p["risk"] <= 0.10), default=0)
    lines += [
        "",
        f"For comparison, the gate's rules (not a single cut) answer {len(answered)} of {n} ({op['coverage']:.0%}), "
        f"with {op['risk']:.0%} of those failing. Of those {len(failing)} failures, {len(missing)} have an "
        f"acceptable decision but miss a required fact, and {len(decided)} are wrong decisions "
        f"({', '.join(decided)})" + (f"; {other} other" if other else "") + ". So at this coverage the failure "
        "rate is set by answer completeness, and a stricter floor would only trade those failures for coverage: "
        f"a single cut stays at or under 10% failing only up to {low} of {n} questions answered "
        f"({low / n:.0%}). What the gate adds is in which questions it does not answer: see the no-gate ablation "
        "above.",
    ]
    return "\n".join(lines)


def slug(heading: str) -> str:
    """GitHub-style anchor of a heading."""
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def main() -> None:
    r = json.loads(RESULTS.read_text(encoding="utf-8"))
    corpus = load_corpus(ROOT / "corpus")
    readme = README.read_text(encoding="utf-8").splitlines()

    frozen = ", ".join(f"{k} `{v}`" for k, v in r["frozen"].items())
    headline = section(readme, "| | **held-out v2**", "Cold start", include_end=True)
    detail = section(readme, "## Results in detail", "## What I would have built in one hour")
    assert detail.splitlines()[-1].startswith("With the gate off"), "the risk table must follow Risk and coverage"
    detail += "\n" + risk_table(r)
    next_steps = section(readme, "## What I would do next", "## How I used AI tools")
    trace = trace_section(r, corpus, TRACED)

    doc = f"""# MBFS Knowledge Service: results

The final evaluation run, written out in full as plain text. The results sections are
copied from `README.md`, so the two always agree; the worked example and the risk-coverage table are generated
from `eval/results/results.json` (`python scripts/build_results_md.py` rebuilds this file). Answers are as of the
corpus snapshot, {r["generated_for_as_of"]}. Frozen eval files (first 16 hex digits of each file's SHA-256):
{frozen}. Every rate is k/n with a 95% Wilson interval: at n = 20 one question moves a rate by 5 points, at
n = 12 by 8.

Contents: [The headline](#the-headline) · [One question, end to end](#{slug(trace.splitlines()[0][3:])}) ·
[Results in detail](#results-in-detail) · [What I would do next](#what-i-would-do-next) ·
[Where to look next](#where-to-look-next)

## The headline

{headline}

{trace}

{detail}

{next_steps}

## Where to look next

- `README.md`: the whole write-up: how to run it, how it works and why, the dataset, what was cut, how AI tools
  were used.
- `eval/results/report.md`: the summary tables and every question's decision, rule, score and result, in plain
  text.
- `eval/results/report.html`: the same with charts, plus every question's answer, citations and decision trace.
  `eval/results/walkthrough.html`: a guided tour of decision traces, charts and failures. Both are self-contained
  and work offline; GitHub shows `.html` files as source, so clone the repo or download them to view them.
- `eval/results/results.json`: everything above, as data. The blind held-out v1 run is kept unchanged in
  `eval/results/heldout_v1_blind/`, and `eval/results/reranker_selection.json` holds the model comparison.
- To reproduce: `python -m mbfs_rag download-models` once, then `python -m mbfs_rag eval --ablations`.
  `python scripts/package_submission.py` builds the submission zip in `dist/`, which is not committed.
"""
    anchors = {slug(line.lstrip("#")) for line in doc.splitlines() if line.startswith("#")}
    broken = sorted({m for m in re.findall(r"\]\(#([^)]+)\)", doc) if m not in anchors})
    assert not broken, f"links to missing headings: {broken}"
    OUT.write_text(doc, encoding="utf-8", newline="\n")
    print(f"wrote {OUT} ({len(doc.splitlines())} lines; internal links resolve)")


if __name__ == "__main__":
    main()
