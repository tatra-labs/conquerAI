"""Every tunable in one frozen place, each with where its value came from."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class Thresholds:
    """Logit cut-offs on two scales, each per unit type: table rows score systematically lower than prose, so one
    number would either refuse table answers or accept prose near-misses. None is a probability.

    Relevance, on the ranking cross-encoder (MS MARCO MiniLM-L6): which candidates count while the evidence is
    assembled (which sources compete, which sibling rows are listed, which second source is strong enough to add).

    Answerability, on the verifier (bge-reranker-v2-m3) scoring the one unit about to be quoted: whether to answer
    at all. These floors come from the realistic calibration set (eval/calibration.toml: questions written from
    the corpus alone in agent shorthand, customer words, situations and with typos, against off-topic and
    not-covered ones), minimising 10 x off-topic answered + 1 x on-topic refused over the questions that reach the
    floor rule; they were frozen before held-out v2 was run. A floor only separates "the KB answers this" from "it
    does not": the score cannot tell the right passage from an on-topic distractor (the report's strip plot), which
    is why near-misses are handled by structure. "High" adds a stated margin to the floor; it is not tuned.

    Without the verifier (its ablation), the gate falls back to the relevance values, which were its floors before:
    fitted on questions templated from the documents' headings, they refused most real phrasing in held-out v1."""

    relevance_prose: float = 1.73
    relevance_table: float = -1.74
    # Fitted on the realistic calibration set (evaluation.calibrate / suggest_thresholds), frozen before held-out v2:
    # 64% of its on-topic questions answered, 1 of 77 off-topic. The ranker's score, fitted the same way, answered 36%.
    answer_prose: float = -1.47
    answer_table: float = -3.09
    # "High" is the floor plus a stated margin. At that point 74% (prose) and 80% (table) of calibration answers come
    # from the exact gold unit, a lower bound since other units often answer too.
    high_margin: float = 4.0

    @property
    def high_prose(self) -> float:
        return self.answer_prose + self.high_margin

    @property
    def high_table(self) -> float:
        return self.answer_table + self.high_margin

    authority_margin: float = 3.0  # candidates this close to the best trusted one compete on fit and authority
    row_margin: float = 1.0  # sibling rows within this of the chosen row are listed (the reranker can't separate them)
    sentence_margin: float = 2.5  # neighbouring sentences this close to the best one are quoted with it
    competing_margin: float = 1.5  # another source this close to the primary, stating a different value, is shown too
    # Logits charged per question term a unit does not cover when choosing the primary. Set at the order of the
    # reranker's paraphrase noise (the same passage moves by ~2 logits under rewording), so a missing concept
    # outweighs noise but not a clear relevance gap. Not tuned on the eval set.
    miss_penalty: float = 2.0


@dataclass(frozen=True)
class Settings:
    corpus_dir: Path = PROJECT_ROOT / "corpus"
    model_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("MBFS_MODEL_DIR", PROJECT_ROOT / ".cache" / "models"))
    )
    cache_dir: Path = PROJECT_ROOT / ".cache"

    # Retrieval
    embedding_model: str = "BAAI/bge-small-en-v1.5"  # 67 MB ONNX; strong small English retriever
    query_prefix: str = "Represent this sentence for searching relevant passages: "  # BGE's retrieval instruction
    reranker_model: str = "Xenova/ms-marco-MiniLM-L-6-v2"  # 80 MB; L-12 is no better on MS MARCO and 2x slower
    # Answerability verifier: scores only the unit about to be quoted. On the realistic calibration set it separates
    # answerable from not-covered questions far better than the ranker (AUC 0.95 vs 0.87) at ~10x the cost per pair.
    verifier_model: str = "onnx-community/bge-reranker-v2-m3-ONNX"
    verifier_file: str = "onnx/model_int8.onnx"  # 570 MB int8
    pool_per_retriever: int = 12  # each retriever nominates this many; the cross-encoder scores the union
    header_weight: float = 0.4  # BM25F-lite: title/section terms count, but less than body terms (not tuned)
    rrf_k: int = 60  # Cormack et al. (2009); used only by the no-reranker ablation
    sentence_pool: int = 4  # passages whose sentences are scored individually for span selection
    # The verifier re-scores this many of the strongest candidates in the trusted window, and the primary is chosen
    # among them on its scale (the fit penalty still applies). 1 scores only the unit structure chose. On the
    # calibration set, 4 answered 60% of on-topic questions against 64% for 1, with the same precision, so 1.
    verify_top: int = int(os.environ.get("MBFS_VERIFY_TOP", "1"))
    # ONNX Runtime defaults to one spinning thread per core. On a shared agent desktop that starves other apps
    # and, once anything else is busy, slows the service itself; two small models on short batches need few.
    onnx_threads: int = int(os.environ.get("MBFS_ONNX_THREADS", "4"))

    thresholds: Thresholds = Thresholds()

    # Ablation switches, used only by the evaluation to show what each defence buys. Each name removes one:
    #   validity    - superseded documents are treated as current
    #   trust       - upcoming and outdated sources compete with current ones on score alone
    #   structure   - the top-scoring unit is the answer (no fit check, no row facets, no tier lookup)
    #   gaps        - uncovered modifiers ("credit card") are ignored
    #   glossary    - no expansion of shorthand and customer phrasing
    #   filler      - greetings and courtesy words are not stripped
    #   gate        - always answer from the best evidence; never refuse or clarify
    #   verifier    - the gate judges the ranker's score instead of the verifier's
    ablate: frozenset[str] = frozenset()

    # Generation (the LLM path is used only when an API key is present)
    llm_model: str = os.environ.get("MBFS_LLM_MODEL", "claude-opus-5-5")
    llm_effort: str = os.environ.get(
        "MBFS_LLM_EFFORT", "low"
    )  # the agent is waiting on the call; raise only if measured
    llm_max_tokens: int = 4096  # Opus 5.5 always thinks; leave room for thinking plus a short reply
