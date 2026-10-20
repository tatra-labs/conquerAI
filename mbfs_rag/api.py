"""Thin HTTP wrapper (optional `api` extra): POST /answer, GET /documents/{doc_id}, GET /health."""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import date

from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .service import KnowledgeService


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    as_of: date | None = None
    explain: bool = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    if getattr(app.state, "service", None) is None:  # tests may attach an already-loaded service
        app.state.service = KnowledgeService.load()
    yield


app = FastAPI(title="MBFS Knowledge Service", lifespan=lifespan)


@app.exception_handler(RequestValidationError)
async def invalid_request(_: Request, exc: RequestValidationError) -> JSONResponse:
    """Say what was wrong without echoing what was sent: FastAPI's default 422 body repeats the input, and a
    rejected question (say, a pasted e-mail thread over the length limit) may contain a card number."""
    errors = [{k: v for k, v in e.items() if k not in ("input", "ctx")} for e in exc.errors()]
    return JSONResponse(status_code=422, content={"detail": errors})


@app.post("/answer")
def answer(request: AnswerRequest) -> dict:
    return app.state.service.answer(request.query, request.as_of, explain=request.explain).to_dict()


@app.get("/documents/{doc_id}")
def document(doc_id: str) -> dict:
    """Canonical text and metadata, so any citation's char_range can be verified independently."""
    corpus = app.state.service.corpus
    if doc_id not in corpus.docs:
        raise HTTPException(404, f"unknown document {doc_id!r}")
    doc = corpus.docs[doc_id]
    meta = doc.meta
    return {
        "doc_id": doc_id,
        "title": meta.title,
        "doc_type": meta.doc_type,
        "audience": meta.audience,  # "internal": agent guidance, not to be read to a customer verbatim
        "validity": meta.state(corpus.snapshot),
        "effective_from": meta.effective_from.isoformat() if meta.effective_from else None,
        "path": f"corpus/{meta.path}",
        "text": doc.text,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "documents": len(app.state.service.corpus.docs)}
