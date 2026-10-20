"""Compare cross-encoders for answerability, on the CALIBRATION set only (no eval set is touched).

    python scripts/select_reranker.py              # writes eval/results/reranker_selection.json (~30 min on CPU)

For each calibration question the system's own candidate pool (the BM25 + dense union, 12 + 12) is computed once,
and each model scores the same pool. Reported per model: the AUC of its top logit, on-topic against not-covered
questions (overall and per question style); the share of on-topic questions answered at the 10:1 cost-optimal
threshold, and how many not-covered ones that threshold lets through; top-1 gold accuracy; seconds per question;
and the model revision used. This is the evidence for the README's choice of MiniLM-L6 as the ranker and
bge-reranker-v2-m3 as the verifier. Models other than the three the service uses are fetched only if missing.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402
from fastembed.common.model_description import ModelSource  # noqa: E402
from fastembed.rerank.cross_encoder import TextCrossEncoder  # noqa: E402

from mbfs_rag.config import Settings  # noqa: E402
from mbfs_rag.encoders import FastEmbedEmbedder  # noqa: E402
from mbfs_rag.evaluation import calibration_questions  # noqa: E402
from mbfs_rag.ingest import load_corpus  # noqa: E402
from mbfs_rag.retrieval import Retriever  # noqa: E402

OUT = ROOT / "eval" / "results" / "reranker_selection.json"
MODELS = (
    "Xenova/ms-marco-MiniLM-L-6-v2",
    "Xenova/ms-marco-MiniLM-L-12-v2",
    "jinaai/jina-reranker-v1-turbo-en",
    "onnx-community/bge-reranker-v2-m3-ONNX",
)
STYLES = ("shorthand", "customer", "narrative", "typo")


class Zero:
    """A stand-in reranker: retrieval only needs the pool, not its order."""

    name = "zero"

    def logits(self, query, texts):
        return np.zeros(len(texts))


def auc(pos: list[float], neg: list[float]) -> float:
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0 for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def cost_optimal(pos: list[float], neg: list[float], ratio: float = 10.0) -> tuple[float, float, int]:
    """The threshold minimising ratio x (not-covered answered) + (on-topic refused); the highest such cut."""
    vals = sorted(set(pos + neg))
    cuts = [vals[0] - 1] + [(a + b) / 2 for a, b in zip(vals, vals[1:], strict=False)] + [vals[-1] + 1]
    best = min(cuts, key=lambda t: (ratio * sum(n >= t for n in neg) + sum(p < t for p in pos), -t))
    return best, sum(p >= best for p in pos) / len(pos), sum(n >= best for n in neg)


def revision(settings: Settings, name: str) -> str | None:
    snapshots = settings.model_dir / f"models--{name.replace('/', '--')}" / "snapshots"
    found = sorted(p.name for p in snapshots.iterdir()) if snapshots.exists() else []
    return ", ".join(found) or None


def main() -> None:
    settings = Settings()
    corpus = load_corpus(settings.corpus_dir)
    retriever = Retriever(corpus, settings, FastEmbedEmbedder(settings), Zero())
    questions = calibration_questions(corpus)
    pools = []
    for q in questions:
        cands, trace = retriever.retrieve(q["question"], corpus.snapshot)
        pools.append((q, cands, trace.dense_top[0][1] if trace.dense_top else 0.0))
    n_pos = sum(q["gold"] is not None for q in questions)
    print(f"{len(pools)} pools ({n_pos} on-topic), mean size {sum(len(c) for _, c, _ in pools) / len(pools):.1f}")

    try:
        TextCrossEncoder.add_custom_model(
            model=settings.verifier_model,
            sources=ModelSource(hf=settings.verifier_model),
            model_file=settings.verifier_file,
            size_in_gb=0.57,
        )
    except ValueError:  # already registered
        pass

    out = {
        "about": (
            "Cross-encoders compared on the calibration set (eval/calibration.toml), scoring the system's own "
            "candidate pool; AUC of the top logit, on-topic vs not-covered. Timings are per question on a shared "
            f"desktop CPU, ONNX Runtime {settings.onnx_threads} threads."
        ),
        "questions": {"on_topic": n_pos, "not_covered": len(questions) - n_pos},
        "models": {},
    }
    for name in MODELS:
        local = revision(settings, name) is not None
        model = TextCrossEncoder(
            name, cache_dir=str(settings.model_dir), threads=settings.onnx_threads, local_files_only=local
        )
        model.rerank("warm-up", ["warm-up"])
        t0 = time.perf_counter()
        rows = []
        for q, cands, _ in pools:
            scores = list(model.rerank(q["question"], [c.text for c in cands])) if cands else []
            top = int(np.argmax(scores)) if scores else -1
            rows.append(
                {
                    "pos": q["gold"] is not None,
                    "style": q["style"],
                    "top": max(scores) if scores else -99.0,
                    "gold1": bool(q["gold"] and scores and cands[top].passage_id in q["gold"]),
                }
            )
        secs = (time.perf_counter() - t0) / len(pools)
        pos = [r["top"] for r in rows if r["pos"]]
        neg = [r["top"] for r in rows if not r["pos"]]
        thr, answered, let_through = cost_optimal(pos, neg)
        out["models"][name] = {
            "revision": revision(settings, name),
            "auc": round(auc(pos, neg), 3),
            "auc_by_style": {
                st: round(auc([r["top"] for r in rows if r["pos"] and r["style"] == st], neg), 3) for st in STYLES
            },
            "threshold_10to1": round(thr, 2),
            "on_topic_answered": round(answered, 3),
            "not_covered_answered": let_through,
            "top1_gold": round(sum(r["gold1"] for r in rows if r["pos"]) / len(pos), 3),
            "sec_per_question": round(secs, 3),
        }
        print(name, json.dumps(out["models"][name]), flush=True)

    dpos = [d for q, _, d in pools if q["gold"] is not None]
    dneg = [d for q, _, d in pools if q["gold"] is None]
    thr, answered, let_through = cost_optimal(dpos, dneg)
    out["dense_cosine_alone"] = {
        "auc": round(auc(dpos, dneg), 3),
        "on_topic_answered": round(answered, 3),
        "not_covered_answered": let_through,
    }
    OUT.write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
