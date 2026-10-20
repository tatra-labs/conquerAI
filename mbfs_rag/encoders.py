"""fastembed (ONNX Runtime, no torch) adapters for the embedding model and the cross-encoder.

fastembed is imported lazily (it costs seconds to import on Windows), runtime constructors never touch
the network (`local_files_only=True`), and only `python -m mbfs_rag download-models` downloads.
"""

from __future__ import annotations

import contextlib
import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

import numpy as np

from .config import Settings


class ModelsMissing(RuntimeError):
    def __init__(self, settings: Settings, cause: Exception):
        super().__init__(
            f"Local models not found in {settings.model_dir} ({cause.__class__.__name__}: {str(cause)[:200]}). "
            "Run `python -m mbfs_rag download-models` once (~730 MB)."
        )


class DownloadFailed(RuntimeError):
    """download-models could not fetch or load a model."""


class Embedder(Protocol):
    name: str

    def embed(self, texts: Sequence[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class Reranker(Protocol):
    name: str

    def logits(self, query: str, texts: Sequence[str]) -> np.ndarray: ...


def _unit_rows(matrix: np.ndarray) -> np.ndarray:
    return matrix / np.clip(np.linalg.norm(matrix, axis=-1, keepdims=True), 1e-12, None)


class FastEmbedEmbedder:
    def __init__(self, settings: Settings, *, allow_download: bool = False):
        from fastembed import TextEmbedding  # an import failure (say, onnxruntime's DLL) surfaces as itself

        try:
            self._model = TextEmbedding(
                settings.embedding_model,
                cache_dir=str(settings.model_dir),
                threads=settings.onnx_threads,
                local_files_only=not allow_download,
            )
        except Exception as exc:  # fastembed raises ValueError/OSError variants for a missing model
            raise ModelsMissing(settings, exc) from exc
        self.name = settings.embedding_model
        self._prefix = settings.query_prefix
        self._cache_dir = settings.cache_dir / "index"

    def embed(self, texts: Sequence[str]) -> np.ndarray:
        """Passage embeddings, cached on disk by a hash of the model id and the exact texts."""
        key = hashlib.sha256("\x1f".join([self.name, *texts]).encode("utf-8")).hexdigest()[:24]
        path = self._cache_dir / f"{key}.npy"
        if path.exists():
            return np.load(path)
        matrix = _unit_rows(np.array(list(self._model.embed(list(texts))), dtype=np.float32))
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(path, matrix)
        return matrix

    def embed_query(self, text: str) -> np.ndarray:
        return _unit_rows(np.array(list(self._model.query_embed(self._prefix + text)), dtype=np.float32))[0]


class FastEmbedReranker:
    """MS MARCO MiniLM cross-encoder. Returns raw logits: they rank well within a query, but they are
    not probabilities and are thresholded per unit type (see config.Thresholds)."""

    def __init__(self, settings: Settings, *, allow_download: bool = False):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        try:
            self._model = TextCrossEncoder(
                settings.reranker_model,
                cache_dir=str(settings.model_dir),
                threads=settings.onnx_threads,
                local_files_only=not allow_download,
            )
        except Exception as exc:
            raise ModelsMissing(settings, exc) from exc
        self.name = settings.reranker_model
        self._model.rerank("warm-up", ["warm-up"])  # first call pays a one-off graph initialisation

    def logits(self, query: str, texts: Sequence[str]) -> np.ndarray:
        if not texts:
            return np.zeros(0)
        return np.array(list(self._model.rerank(query, list(texts))), dtype=float)


class FastEmbedVerifier:
    """bge-reranker-v2-m3 (int8 ONNX), asked one thing: does this unit answer this question? It scores the single
    unit about to be quoted, never a pool, so its cost is one pair per sub-question."""

    def __init__(self, settings: Settings, *, allow_download: bool = False):
        from fastembed.common.model_description import ModelSource
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        with contextlib.suppress(ValueError):  # registered once per process
            TextCrossEncoder.add_custom_model(
                model=settings.verifier_model,
                sources=ModelSource(hf=settings.verifier_model),
                model_file=settings.verifier_file,
                size_in_gb=0.57,
            )
        try:
            self._model = TextCrossEncoder(
                settings.verifier_model,
                cache_dir=str(settings.model_dir),
                threads=settings.onnx_threads,
                local_files_only=not allow_download,
            )
        except Exception as exc:
            raise ModelsMissing(settings, exc) from exc
        self.name = settings.verifier_model
        self._model.rerank("warm-up", ["warm-up"])

    def score(self, query: str, text: str) -> float:
        return self.scores(query, [text])[0]

    def scores(self, query: str, texts: Sequence[str]) -> list[float]:
        return [float(x) for x in self._model.rerank(query, list(texts))] if texts else []


def download_models(settings: Settings) -> list[tuple[str, Path]]:
    """Fetch both models into the project-local cache and prove they run."""
    try:
        embedder = FastEmbedEmbedder(settings, allow_download=True)
        reranker = FastEmbedReranker(settings, allow_download=True)
        verifier = FastEmbedVerifier(settings, allow_download=True)
    except ModelsMissing as exc:  # "run download-models" would be circular advice here
        raise DownloadFailed(f"Could not download the models into {settings.model_dir}: {exc.__cause__}") from exc
    assert embedder.embed_query("chargeback fee").shape[0] > 0
    assert reranker.logits("chargeback fee", ["The chargeback fee is $25.00."]).shape == (1,)
    assert isinstance(verifier.score("chargeback fee", "The chargeback fee is $25.00."), float)
    return [(m.name, settings.model_dir) for m in (embedder, reranker, verifier)]
