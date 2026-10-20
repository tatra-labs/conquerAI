from pathlib import Path

import pytest

from mbfs_rag.config import Settings
from mbfs_rag.ingest import load_corpus

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def corpus():
    return load_corpus(ROOT / "corpus")


def _models_present() -> bool:
    return any(Settings().model_dir.rglob("*.onnx")) if Settings().model_dir.exists() else False


@pytest.fixture(scope="session")
def service():
    """The real pipeline (local models). Tests using it are marked `models` and skipped without them."""
    if not _models_present():
        pytest.skip("models not downloaded: run `python -m mbfs_rag download-models`")
    from mbfs_rag.service import KnowledgeService

    return KnowledgeService.load(use_llm=False)
