"""Render eval/results/results.json as report.md and a self-contained report.html (inline SVG, no libraries)."""

from __future__ import annotations

import html
import json
from pathlib import Path

VARIANTS = ("full", "validity", "trust", "structure", "gaps", "glossary", "filler", "gate", "verifier", "headers")
VARIANT_LABEL = {
    "full": "full system",
    "validity": "no validity windows",
    "trust": "no trust tiers",
    "structure": "no structure",
    "gaps": "no gap check",
    "glossary": "no glossary",
    "filler": "no filler strip",
    "gate": "no gate",
    "verifier": "no verifier",
    "headers": "no headings",
}


def _clip(text: str, limit: int = 100) -> str:
    """Shorten a table cell at a word boundary."""
    return text if len(text) <= limit else text[:limit].rsplit(" ", 1)[0].rstrip(",;:") + "…"


def write_reports(results: dict, out_dir: Path) -> None:
    (out_dir / "report.md").write_text(_markdown(results), encoding="utf-8", newline="\n")
    (out_dir / "report.html").write_text(_html(results), encoding="utf-8", newline="\n")


# ---------------------------------------------------------------- markdown


def _markdown(r: dict) -> str:
    lines = [
        "# MBFS RAG - evaluation report",
        "",
        f"Answers as of {r['generated_for_as_of']} (corpus snapshot). "
        f"Frozen inputs: {', '.join(f'{k} {v}' for k, v in r['frozen'].items())}.",
        "",
    ]
    for split, s in r["summary"].items():
        lines += [f"## {split} (n={s['n']})", "", "| measure | result |", "|---|---|"]
        for key in (
            "passed",
            "decision_ok",
            "decision_exact",
            "false_answers",
            "false_refusals",
            "fact_recall",
            "citation_support",
        ):
            lines.append(f"| {key.replace('_', ' ')} | {s[key]} |")
        lines += [
            f"| unsafe answers | {s['unsafe']} |",
            f"| latency (warm, median / max) | {s['latency_ms']['median']} ms / {s['latency_ms']['max']} ms |",
            "",
        ]
    lines += [
        "## Per question",
        "",
        "| id | split | decision | rule | score | result | stage | predicted failure |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for it in r["items"]:
        s = it["score"]
        lines.append(
            f"| {it['id']} | {it['split']} | {s['decision']} | {s['rule']} | {s['score']} | "
            f"{'pass' if s['passed'] else 'FAIL'} | {s['stage']} | {_clip(it['predicted_failure'])} |"
        )
    if "ablations" in r:
        lines += [
            "",
            "## Ablations (pass/fail per item; McNemar exact test vs full)",
            "",
            "| variant | full-only passes | variant-only passes | p | items that flipped |",
            "|---|---|---|---|---|",
        ]
        for flag, m in r["ablations"]["mcnemar"].items():
            lines.append(
                f"| {VARIANT_LABEL.get(flag, flag)} | {m['full_only']} | {m['ablation_only']} | {m['p_exact']} | {', '.join(m['flipped'])} |"
            )
    cal = r["calibration"]["suggested"]
    lines += [
        "",
        "## Threshold choice",
        "",
        "Floors from the realistic calibration set (written from the corpus alone; 10:1 cost of a wrong answer "
        "to a refusal; fitted on the score the gate tests, for questions that reach the floor rule):",
        "",
    ]
    for kind, v in cal.items():
        if v:
            lines.append(
                f"- {kind}: floor {v['answer']} ({v['n_positive']} on-topic / {v['n_negative']} off-topic; "
                f"{v['negatives_above_floor']} off-topic above, {v['positives_below_floor']} on-topic below)"
            )
    lines += [f"- in use: {r['calibration']['in_use']}", ""]
    inv = r.get("invariance") or []
    if inv:
        lines += [
            "## Invariance (paraphrases of dev questions)",
            "",
            f"{sum(p['passed'] for p in inv)}/{len(inv)} keep the same decision and cite the same documents.",
            "",
            "| base | paraphrase | base decision | paraphrase decision | same sources |",
            "|---|---|---|---|---|",
        ]
        lines += [
            f"| {p['base']} | {p['query']} | {p['base_decision']} | {p['decision']} | {p['same_sources']} |"
            for p in inv
        ]
    return "\n".join(lines) + "\n"


def _invariance(r: dict) -> str:
    inv = r.get("invariance") or []
    if not inv:
        return "<p>No invariance set.</p>"
    rows = "".join(
        f"<tr><td>{_e(p['base'])}</td><td>{_e(p['query'])}</td><td>{_e(p['base_decision'])}</td><td>{_e(p['decision'])}</td>"
        f'<td class="{"ok" if p["passed"] else "no"}">{"&#10003; same" if p["passed"] else "&#10007; changed"}</td></tr>'
        for p in inv
    )
    head = f"<p>{sum(p['passed'] for p in inv)}/{len(inv)} paraphrases keep the decision and the cited documents.</p>"
    return f"{head}<table><tr><th>base</th><th>paraphrase</th><th>base</th><th>paraphrase</th><th>result</th></tr>{rows}</table>"


# ---------------------------------------------------------------- html

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--grid:#e1e0d9;
--axis:#c3c2b7;--s1:#2a78d6;--s2:#eb6834;--good:#0ca30c;--bad:#d03b3b;--ring:rgba(11,11,11,.10)}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;
--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;--ring:rgba(255,255,255,.10)}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--muted:#898781;
--grid:#2c2c2a;--axis:#383835;--s1:#3987e5;--s2:#d95926;--ring:rgba(255,255,255,.10)}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:1080px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 8px}
p.sub{color:var(--ink2);margin:0 0 16px}.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.tile{background:var(--surface);border:1px solid var(--ring);border-radius:10px;padding:12px 14px}.tile .l{color:var(--ink2);font-size:13px}
.tile .v{font-size:26px;font-weight:600}.tile .d{color:var(--muted);font-size:12px}.card{background:var(--surface);border:1px solid var(--ring);
border-radius:10px;padding:14px;margin-top:8px;overflow-x:auto}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}
th,td{text-align:left;padding:5px 8px;border-bottom:1px solid var(--grid);vertical-align:top}th{color:var(--ink2);font-weight:600}
.ok{color:var(--good)}.no{color:var(--bad)}.legend{display:flex;gap:16px;flex-wrap:wrap;color:var(--ink2);font-size:13px;margin:6px 0}
.key{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:6px;vertical-align:-1px}details{border-bottom:1px solid var(--grid);padding:8px 0}
summary{cursor:pointer}pre{white-space:pre-wrap;font-size:12px;color:var(--ink2);margin:6px 0}.ans{white-space:pre-wrap;margin:6px 0}
#tip{position:fixed;pointer-events:none;background:var(--surface);border:1px solid var(--ring);border-radius:8px;padding:6px 8px;font-size:12px;
display:none;box-shadow:0 2px 8px rgba(0,0,0,.15);max-width:360px}svg text{fill:var(--muted);font-size:11px}
"""

JS = """
const tip=document.getElementById('tip');
document.querySelectorAll('[data-tip]').forEach(el=>{
 const show=e=>{tip.textContent=el.getAttribute('data-tip');tip.style.display='block';
  tip.style.left=Math.min(e.clientX+12,innerWidth-370)+'px';tip.style.top=(e.clientY+12)+'px'};
 el.addEventListener('pointermove',show);el.addEventListener('pointerleave',()=>tip.style.display='none');
 el.addEventListener('focus',e=>{const b=el.getBoundingClientRect();show({clientX:b.x,clientY:b.y})});
 el.addEventListener('blur',()=>tip.style.display='none');});
"""


def _e(s) -> str:
    return html.escape(str(s), quote=True)


def _tiles(r: dict) -> str:
    out = []
    for split, s in r["summary"].items():
        out.append(
            f'<div class="tile"><div class="l">{_e(split)}: passed</div><div class="v">{_e(s["passed"].split(" ")[0])}</div>'
            f'<div class="d">{_e(s["passed"].split(" ", 1)[1])}</div></div>'
        )
        out.append(
            f'<div class="tile"><div class="l">{_e(split)}: unsafe answers</div><div class="v">{s["unsafe"]}</div>'
            f'<div class="d">answered an unanswerable, or stated a wrong/stale value</div></div>'
        )
        out.append(
            f'<div class="tile"><div class="l">{_e(split)}: false refusals</div><div class="v">{_e(s["false_refusals"].split(" ")[0])}</div>'
            f'<div class="d">answerable questions refused or bounced</div></div>'
        )
        lat = s["latency_ms"]
        out.append(
            f'<div class="tile"><div class="l">{_e(split)}: latency</div><div class="v">{lat["median"]} ms</div>'
            f'<div class="d">median; max {lat["max"]} ms (CPU, warm)</div></div>'
        )
    return f'<div class="tiles">{"".join(out)}</div>'


def _strip(r: dict) -> str:
    """Rerank score strip plot: what the floor separates (calibration) and what it cannot (distractors)."""
    rows = [
        (
            "calibration: on-topic",
            [s for s in r["calibration"]["samples"] if s["label"] == "positive" and s.get("reaches_floor", True)],
            "s1",
        ),
        (
            "calibration: off-topic",
            [s for s in r["calibration"]["samples"] if s["label"] == "negative" and s.get("reaches_floor", True)],
            "s2",
        ),
        ("dev: gold passage", [s for s in r["distractors"] if s["label"] == "gold present"], "s1"),
        ("dev: best distractor, gold deleted", [s for s in r["distractors"] if s["label"] == "gold deleted"], "s2"),
    ]
    lo, hi, w, left, row_h = -12.0, 12.0, 980, 230, 34
    x = lambda v: left + (max(lo, min(hi, v)) - lo) / (hi - lo) * (w - left - 16)  # noqa: E731
    h = row_h * len(rows) + 40
    parts = [f'<svg viewBox="0 0 {w} {h}" width="100%" role="img" aria-label="Reranker score by evidence class">']
    for t in range(-12, 13, 4):
        parts.append(
            f'<line x1="{x(t):.1f}" x2="{x(t):.1f}" y1="8" y2="{h - 26}" stroke="var(--grid)"/>'
            f'<text x="{x(t):.1f}" y="{h - 10}" text-anchor="middle">{t}</text>'
        )
    floors = r["calibration"]["in_use"]
    for key, label in (("answer_prose", "prose floor"), ("answer_table", "table floor")):
        fx = x(floors[key])
        parts.append(
            f'<line x1="{fx:.1f}" x2="{fx:.1f}" y1="8" y2="{h - 26}" stroke="var(--ink2)" stroke-width="1"/>'
            f'<text x="{fx + 4:.1f}" y="18" style="fill:var(--ink2)">{label} {floors[key]}</text>'
        )
    for i, (label, samples, color) in enumerate(rows):
        y = 34 + i * row_h
        parts.append(
            f'<text x="{left - 10}" y="{y + 4}" text-anchor="end" style="fill:var(--ink2);font-size:12px">{_e(label)}</text>'
        )
        for j, s in enumerate(samples):
            jitter = ((j * 7) % 11 - 5) * 1.4
            what = s.get("question") or s.get("item")
            tip = _e(f"{what} | score {s['score']} | {s['kind']}")
            if s["kind"] == "table":
                parts.append(
                    f'<rect x="{x(s["score"]) - 4:.1f}" y="{y + jitter - 4:.1f}" width="8" height="8" rx="1" fill="var(--{color})" '
                    f'stroke="var(--surface)" stroke-width="2" tabindex="0" data-tip="{tip}"/>'
                )
            else:
                parts.append(
                    f'<circle cx="{x(s["score"]):.1f}" cy="{y + jitter:.1f}" r="4.5" fill="var(--{color})" '
                    f'stroke="var(--surface)" stroke-width="2" tabindex="0" data-tip="{tip}"/>'
                )
    parts.append("</svg>")
    legend = (
        '<div class="legend"><span><span class="key" style="background:var(--s1)"></span>answer-bearing evidence</span>'
        '<span><span class="key" style="background:var(--s2)"></span>no answer present</span>'
        "<span>circle = prose passage, square = table row</span><span>x = cross-encoder logit</span></div>"
    )
    return legend + "".join(parts)


def _risk_coverage(r: dict) -> str:
    curve = r.get("risk_coverage")
    if not curve:
        return "<p>Run with <code>--ablations</code> to compute the no-gate curve.</p>"
    w, h, left, bottom = 520, 260, 44, 30
    x = lambda c: left + c * (w - left - 12)  # noqa: E731
    y = lambda rk: 10 + (1 - rk) * (h - bottom - 10)  # noqa: E731
    parts = [
        f'<svg viewBox="0 0 {w} {h}" width="100%" style="max-width:560px" role="img" aria-label="Risk versus coverage">'
    ]
    for t in (0, 0.25, 0.5, 0.75, 1):
        parts.append(
            f'<line x1="{left}" x2="{w - 12}" y1="{y(t):.1f}" y2="{y(t):.1f}" stroke="var(--grid)"/>'
            f'<text x="{left - 6}" y="{y(t) + 4:.1f}" text-anchor="end">{int(t * 100)}%</text>'
            f'<text x="{x(t):.1f}" y="{h - 10}" text-anchor="middle">{int(t * 100)}%</text>'
        )
    pts = " ".join(f"{x(p['coverage']):.1f},{y(p['risk']):.1f}" for p in curve)
    parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--s1)" stroke-width="2" stroke-linejoin="round"/>')
    for p in curve:
        tip = _e(
            "answer the {:.0%} with score >= {}: {:.0%} of those wrong".format(p["coverage"], p["score"], p["risk"])
        )
        parts.append(
            f'<circle cx="{x(p["coverage"]):.1f}" cy="{y(p["risk"]):.1f}" r="9" fill="transparent" tabindex="0" '
            f'data-tip="{tip}"/>'
        )
    op = r.get("operating_point")
    if op:
        tip = _e("deployed gate: answers {:.0%}, {:.0%} of those wrong".format(op["coverage"], op["risk"]))
        cx, cy = x(op["coverage"]), y(op["risk"])
        parts.append(
            f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="6" fill="var(--s2)" stroke="var(--surface)" stroke-width="2" '
            f'tabindex="0" data-tip="{tip}"/><text x="{cx + 9:.1f}" y="{cy - 8:.1f}" style="fill:var(--ink2)">deployed gate</text>'
        )
    parts.append("</svg>")
    return (
        '<div class="legend"><span><span class="key" style="background:var(--s1)"></span>score only (answer the top-k by rerank score)</span>'
        '<span><span class="key" style="background:var(--s2)"></span>deployed gate (rules + score)</span>'
        "<span>x = share answered, y = share of answered that were wrong</span></div>" + "".join(parts)
    )


def _matrix(r: dict) -> str:
    if "ablations" not in r:
        return "<p>Run with <code>--ablations</code>.</p>"
    m = r["ablations"]["matrix"]
    head = "".join(f"<th>{_e(VARIANT_LABEL[v])}</th>" for v in VARIANTS if v in m)
    rows = []
    for it in r["items"]:
        cells = []
        for v in VARIANTS:
            if v in m:
                ok = m[v].get(it["id"])
                cells.append(f'<td class="{"ok" if ok else "no"}">{"&#10003; pass" if ok else "&#10007; fail"}</td>')
        rows.append(f"<tr><td>{_e(it['id'])}</td><td>{_e(it['query'][:60])}</td>{''.join(cells)}</tr>")
    mc = r["ablations"]["mcnemar"]
    foot = "".join(
        f"<td>{mc[v]['full_only']}&ndash;{mc[v]['ablation_only']}<br>p={mc[v]['p_exact']}</td>"
        if v in mc
        else "<td></td>"
        for v in VARIANTS
        if v in m
    )
    return (
        f"<table><tr><th>id</th><th>question</th>{head}</tr>{''.join(rows)}"
        f"<tr><th colspan=2>items only the full system passes &ndash; only the variant passes (exact McNemar)</th>{foot}</tr></table>"
    )


def _confusion(r: dict) -> str:
    labels = ("answer", "partial", "clarify", "refuse")
    out = []
    for split, s in r["summary"].items():
        body = ""
        for g in labels:
            cells = "".join("<td>{}</td>".format(s["confusion"][g][a]) for a in labels)
            body += f"<tr><th>{g}</th>{cells}</tr>"
        header = "".join(f"<th>{a}</th>" for a in labels)
        out.append(
            f"<h3 style='font-size:14px;margin:8px 0'>{_e(split)}</h3>"
            f"<table><tr><th>preferred \\ actual</th>{header}</tr>{body}</table>"
        )
    return "".join(out)


def _retrieval(r: dict) -> str:
    modes = ("bm25", "dense", "hybrid_rrf", "full", "rerank_all")
    ranks = r.get("retrieval", {})
    if not ranks:
        return ""
    summary = []
    for mode in modes:
        vals = [row.get(mode) for row in ranks.values()]
        at1 = sum(1 for v in vals if v == 1)
        at5 = sum(1 for v in vals if v is not None and v <= 5)
        summary.append(f"<td>{at1}/{len(vals)} at 1<br>{at5}/{len(vals)} in top 5</td>")
    rows = ""
    for item_id, row in ranks.items():
        cells = "".join("<td>{}</td>".format(row.get(m) or "&mdash;") for m in modes)
        rows += f"<tr><td>{_e(item_id)}</td>{cells}</tr>"
    header = "".join(f"<th>{m}</th>" for m in modes)
    return f"<table><tr><th>item</th>{header}</tr><tr><th>summary</th>{''.join(summary)}</tr>{rows}</table>"


def _details(r: dict) -> str:
    out = []
    for it in r["items"]:
        s = it["score"]
        mark = (
            '<span class="ok">&#10003; pass</span>'
            if s["passed"]
            else f'<span class="no">&#10007; {_e(s["stage"])}</span>'
        )
        cits = "".join(
            f"<li>[{c['n']}] {_e(c['title'])} &rsaquo; {_e(c['section'])} &mdash; <i>{_e(c['quote'][:160])}</i> "
            f"<span style='color:var(--muted)'>({_e(c['source_path'])} L{c['lines'][0]}, {_e(c['validity'])})</span></li>"
            for c in it["citations"]
        )
        pred = (
            f"<p><b>Predicted before the first run:</b> {_e(it['predicted_failure'])}</p>"
            if it["predicted_failure"]
            else ""
        )
        out.append(
            f"<details><summary><b>{_e(it['id'])}</b> [{_e(it['split'])}] {_e(it['query'])} &mdash; {mark} &middot; "
            f"{_e(s['decision'])} ({_e(s['level'])}) &middot; rule <code>{_e(s['rule'])}</code> &middot; score {_e(s['score'])}</summary>"
            f"<div class='ans'>{_e(s['answer'])}</div><ul>{cits}</ul>{pred}<p><b>Good answer:</b> {_e(it['notes'])}</p>"
            f"<p>missing: {_e(s['facts_missing'])} &middot; forbidden hits: {_e(s['forbidden_hits'])} &middot; structural: {_e(s['structural'])}</p>"
            f"<details><summary>decision trace</summary><pre>{_e(json.dumps(it['trace'], indent=1, ensure_ascii=False))}</pre></details></details>"
        )
    return "".join(out)


def _html(r: dict) -> str:
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>MBFS RAG evaluation</title><style>{CSS}</style></head><body><main>
<h1>MBFS RAG evaluation</h1>
<p class="sub">Answers as of {_e(r["generated_for_as_of"])} &middot; models: {_e(r["settings"].get("embedding_model"))} + {_e(r["settings"].get("reranker_model"))}
&middot; frozen inputs {_e(r["frozen"])} &middot; every rate is k/n with a 95% Wilson interval (n is small: one question is about 4 points).</p>
{_tiles(r)}
<h2>What the reranker score can and cannot see</h2>
<p class="sub">The floor separates on-topic from off-topic questions (top two rows). It cannot tell the right passage from an on-topic
distractor: with every gold document deleted, the best remaining passage still often scores high (bottom row). That is why near-misses are
decided by structure (validity, authority, row facets, the gap check) and the score is only a floor.</p>
<div class="card">{_strip(r)}</div>
<h2>Risk versus coverage</h2>
<p class="sub">Each step answers one more question, in order of rerank score, with the gate disabled. Lower and further right is better.</p>
<div class="card">{_risk_coverage(r)}</div>
<h2>What each defence buys: items &times; ablations</h2>
<p class="sub">Each column removes one defence. A trap is only a trap if some column fails it.</p>
<div class="card">{_matrix(r)}</div>
<h2>Decisions</h2><div class="card">{_confusion(r)}</div>
<h2>Invariance: same question, different words</h2><div class="card">{_invariance(r)}</div>
<h2>Retrieval: rank of the first unit carrying gold support</h2>
<p class="sub">BM25, dense and RRF are ranking-only ablations; "full" is the cross-encoder over the union of both retrievers' candidates;
"rerank_all" scores every valid unit (ceiling).</p><div class="card">{_retrieval(r)}</div>
<h2>Every question, with its decision trace</h2><div class="card">{_details(r)}</div>
</main><div id="tip" role="tooltip"></div><script>{JS}</script></body></html>"""
