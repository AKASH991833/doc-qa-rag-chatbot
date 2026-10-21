# Doc QA RAG Chatbot - Python (FastAPI + scikit-learn)

> **Live browser demo:** https://akash991833.github.io/doc-qa-rag-chatbot/ - zero-install version. The app in this folder is the pro-grade rebuild; run steps below.
> **Repo:** https://github.com/AKASH991833/doc-qa-rag-chatbot

![doc-qa-rag-chatbot in action](../screenshots/chat-dark.png)

The pro-grade version of the browser demo: a real retrieval-augmented Q&A service with chunking, TF-IDF retrieval, grounded answers with citations, follow-up resolution and session history - served over a FastAPI REST API.

## Stack

- **Chunking + TF-IDF retrieval** - 110-word overlapping windows, 1-2 gram sublinear TF-IDF, cosine ranking
- **Grounded answer composition** - best-matching sentence per cited passage, numbered citations; ungrounded questions answered honestly
- **Session state** - per-session knowledge base, history, anaphoric follow-up resolution ("more about that")
- **FastAPI** - `/sessions`, `/documents`, `/ask`, `/history`, `/healthz`

Note on the stack choice: the earlier plan mentioned sentence-transformers + FAISS. That pulls in ~2GB of torch and makes free CPU hosting (and local runs) slow and fragile, so this version uses the classic TF-IDF retrieval baseline instead - deterministic, fast and fully free. The honest-limits section below says what that trades away.

## Run locally

```
pip install -r requirements.txt
uvicorn api:app --port 8000
# then: POST /sessions, POST /sessions/{id}/documents, POST /sessions/{id}/ask
```

## Tests

```
python3 tests/test_rag_core.py   # retrieval, grounding, follow-ups
python3 tests/test_api.py        # FastAPI end-to-end
```

## Honest limits

- TF-IDF retrieval is lexical: paraphrases with no shared words miss (an embedding model would fix this at the cost of a large download).
- The answer layer extracts and composes sentences; it does not generate new prose like an LLM.
- Sessions live in process memory; a restart clears them.

## Layout

- `rag_core.py` - chunking, retrieval, session logic (no web imports), unit tested
- `api.py` - FastAPI surface
- `tests/` - core and API tests
