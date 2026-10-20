"""Command line.

  python -m mbfs_rag download-models          one-time (~730 MB), the only command that uses the network
  python -m mbfs_rag ask "question" [--explain] [--json] [--as-of YYYY-MM-DD]
  python -m mbfs_rag show <doc_id> [--span <span_id>] [--lines A-B]
  python -m mbfs_rag eval [--set dev|heldout_v1|heldout_v2|all] [--ablations]
  python -m mbfs_rag serve [--port 8000]

Exit codes: 0 for every decision (refusals included), 2 for usage errors, 3 when models are missing.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date

from .config import Settings
from .encoders import DownloadFailed, ModelsMissing


def _print_answer(answer, explain: bool) -> None:
    c = answer.confidence
    print(f"Decision: {answer.decision.upper()} ({c['level']}) - {c['action']}")
    print(f"Rule: {c['rule']}   Score: {c['score']}   As of: {answer.as_of}   Generator: {answer.generator}")
    if answer.internal_content:
        print("Contains INTERNAL content: for agents, do not read verbatim to the customer.")
    print()
    print(answer.answer)
    if answer.citations:
        print("\nSources:")
        for cit in answer.citations:
            where = f"{cit.source_path} L{cit.lines[0]}" + (f"-{cit.lines[1]}" if cit.lines[1] != cit.lines[0] else "")
            cell = f" [{cit.cell['row']} / {cit.cell['column']}]" if cit.cell else ""
            print(
                f" [{cit.n}] {cit.title} > {cit.section or '-'}{cell}  ({cit.doc_type}, {cit.validity}, {cit.audience})"
            )
            print(f"     {where}  chars {cit.char_range[0]}-{cit.char_range[1]}  span {cit.span_id}")
            print(f'     "{cit.quote}"')
    if explain and answer.trace:
        print("\nDecision trace:")
        print(json.dumps(answer.trace, indent=2, ensure_ascii=False))


def _show(settings: Settings, doc_id: str, span_id: str | None, lines: str | None) -> int:
    from .corpus import read_text
    from .ingest import load_corpus

    corpus = load_corpus(settings.corpus_dir)
    if doc_id not in corpus.docs:
        print(f"unknown doc_id {doc_id!r}; known: {', '.join(sorted(corpus.docs))}", file=sys.stderr)
        return 2
    doc = corpus.docs[doc_id]
    if span_id:
        span = corpus.spans.get(span_id)
        if span is None or span.doc_id != doc_id:
            print(f"unknown span {span_id!r} for {doc_id}", file=sys.stderr)
            return 2
        ok = doc.text[span.start : span.end] == span.text
        print(
            f"{doc.meta.title} | {doc.meta.path} lines {span.line_start}-{span.line_end} "
            f"| chars {span.start}-{span.end}"
        )
        print(f"canonical text slice matches quote: {ok}")
        before, after = doc.text[max(0, span.start - 200) : span.start], doc.text[span.end : span.end + 200]
        print(f"...{before}>>>{span.text}<<<{after}...")
        return 0 if ok else 1
    if lines:
        start, _, end = lines.partition("-")
        raw = read_text(settings.corpus_dir / doc.meta.path).split("\n")
        for n in range(int(start), int(end or start) + 1):
            print(f"{n:5d}  {raw[n - 1]}")
        return 0
    print(
        json.dumps(
            {"meta": {k: str(v) for k, v in doc.meta.__dict__.items()}, "text": doc.text}, indent=2, ensure_ascii=False
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")  # Windows consoles and redirects default to cp1252
    parser = argparse.ArgumentParser(prog="python -m mbfs_rag", description="MBFS customer knowledge RAG service")
    sub = parser.add_subparsers(dest="command", required=True)
    ask = sub.add_parser("ask", help="answer one question")
    ask.add_argument("question")
    ask.add_argument("--as-of", type=date.fromisoformat, help="answer as of this date (default: corpus snapshot)")
    ask.add_argument("--json", action="store_true", help="print the full answer object as JSON")
    ask.add_argument("--explain", action="store_true", help="include the decision trace")
    ask.add_argument("--no-llm", action="store_true", help="force the extractive generator")
    show = sub.add_parser("show", help="verify a citation against the source")
    show.add_argument("doc_id")
    show.add_argument("--span")
    show.add_argument("--lines")
    ev = sub.add_parser("eval", help="run the evaluation and write eval/results")
    ev.add_argument("--set", choices=("dev", "heldout_v1", "heldout_v2", "all"), default="all")
    ev.add_argument("--ablations", action="store_true", help="also run the retrieval and gate ablations")
    sub.add_parser("download-models", help="fetch the embedding and reranker models (one time)")
    serve = sub.add_parser("serve", help="run the HTTP API")
    serve.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    settings = Settings()

    try:
        if args.command == "download-models":
            from .encoders import download_models

            for name, path in download_models(settings):
                print(f"ok  {name}  ->  {path}")
            return 0
        if args.command == "show":
            return _show(settings, args.doc_id, args.span, args.lines)
        if args.command == "ask":
            from .service import KnowledgeService

            service = KnowledgeService.load(settings, use_llm=False if args.no_llm else None)
            answer = service.answer(args.question, args.as_of, explain=args.explain)
            if args.json:
                print(json.dumps(answer.to_dict(), indent=2, ensure_ascii=False, default=str))
            else:
                _print_answer(answer, args.explain)
            return 0
        if args.command == "eval":
            from .evaluation import run

            return run(settings, args.set, ablations=args.ablations)
        if args.command == "serve":
            import uvicorn

            uvicorn.run("mbfs_rag.api:app", port=args.port)
            return 0
    except (ModelsMissing, DownloadFailed) as exc:
        print(str(exc), file=sys.stderr)
        return 3
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
