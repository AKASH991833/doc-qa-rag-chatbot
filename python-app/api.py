"""FastAPI layer over the RAG core. Sessions are in-memory per process."""
import uuid
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_core import RagSession

app = FastAPI(title="Doc QA RAG Chatbot", version="1.0.0")
SESSIONS: dict[str, RagSession] = {}


class AddDoc(BaseModel):
    source: str
    text: str


class Ask(BaseModel):
    question: str


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.post("/sessions")
def create_session():
    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = RagSession()
    return {"session_id": sid}


@app.post("/sessions/{sid}/documents")
def add_document(sid: str, doc: AddDoc):
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "Unknown session")
    try:
        s.add_document(doc.source, doc.text)
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"chunks": len(s.kb.chunks)}


@app.post("/sessions/{sid}/ask")
def ask(sid: str, body: Ask):
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "Unknown session")
    try:
        return s.ask(body.question)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.get("/sessions/{sid}/history")
def history(sid: str):
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "Unknown session")
    return {"history": s.history}
