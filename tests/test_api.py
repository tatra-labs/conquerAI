"""HTTP contract (models required; FastAPI runs the lifespan only inside the TestClient context manager)."""

import pytest

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def client(service):
    from fastapi.testclient import TestClient

    from mbfs_rag import api

    api.app.state.service = service  # reuse the session's loaded models
    with TestClient(api.app) as c:
        yield c


def test_answer_endpoint_returns_decision_citations_and_confidence(client):
    body = client.post("/answer", json={"query": "What is the chargeback fee?"}).json()
    assert (
        body["decision"] == "answer" and body["citations"] and body["confidence"]["level"] in ("high", "medium", "low")
    )
    cit = body["citations"][0]
    doc = client.get(f"/documents/{cit['doc_id']}").json()
    assert doc["text"][cit["char_range"][0] : cit["char_range"][1]] == cit["quote"]


def test_validation_and_unknown_documents(client):
    assert client.post("/answer", json={"query": ""}).status_code == 422
    assert client.get("/documents/not-a-doc").status_code == 404
    assert client.get("/health").json()["status"] == "ok"


def test_documents_say_who_they_are_for_and_whether_they_are_current(client):
    doc = client.get("/documents/caller-verification-sop").json()
    assert doc["audience"] == "internal" and doc["validity"] == "current" and doc["doc_type"]


def test_a_rejected_question_is_not_echoed_back(client):
    """FastAPI's default 422 repeats the input; a pasted thread over the length limit may hold a card number."""
    pasted = (
        "Hi team, " + "please see below. " * 120 + "My card number is 4111 1111 1111 1111. Why was I charged twice?"
    )
    response = client.post("/answer", json={"query": pasted})
    assert response.status_code == 422 and "4111" not in response.text
    wrong_field = client.post("/answer", json={"question": "card 4111 1111 1111 1111 was stolen"})
    assert wrong_field.status_code == 422 and "4111" not in wrong_field.text  # the whole body was echoed before
