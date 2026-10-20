"""Zip the submission: source, corpus, eval sets and results, tests, README. No virtualenv, models or caches.

python scripts/package_submission.py          # writes dist/mbfs_rag_submission.zip
"""

from __future__ import annotations

import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "dist" / "mbfs_rag_submission.zip"
SKIP_DIRS = {".venv", ".cache", "__pycache__", ".pytest_cache", ".ruff_cache", "dist", ".git", ".idea", ".vscode"}


def files() -> list[Path]:
    out = []
    for path in sorted(ROOT.rglob("*")):
        rel = path.relative_to(ROOT)
        if path.is_file() and not any(part in SKIP_DIRS for part in rel.parts) and path.suffix != ".pyc":
            out.append(rel)
    return out


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    chosen = files()
    with zipfile.ZipFile(OUT, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for rel in chosen:
            zf.write(ROOT / rel, Path("mbfs_rag_submission") / rel)
    print(f"wrote {OUT} ({len(chosen)} files, {OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
