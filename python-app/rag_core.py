"""RAG chatbot core - no web framework imports.

Chunk -> TF-IDF index -> retrieve -> grounded answer composition with
citations. Conversation state tracks which passages were cited so follow-up
questions ("tell me more about the second one") resolve. Fully offline;
the answer layer is template-based extractive composition (no LLM call),
which keeps it free and deterministic.
"""
from __future__ import annotations
import re
import numpy as np


def chunk_text(text: str, target: int = 110, overlap: int = 25) -> list[str]:
    words = re.split(r"\s+", text.strip())
    if not words or words == [""]:
        return []
    if len(words) <= target:
        return [" ".join(words)]
    out, i = [], 0
    step = max(1, target - overlap)
    while i < len(words):
        out.append(" ".join(words[i:i + target]))
        i += step
    return out


class KnowledgeBase:
    def __init__(self):
        self.chunks: list[dict] = []  # {source, text}
        self._vec = None
        self._X = None

    def add(self, source: str, text: str):
        text = (text or "").strip()
        if len(text) < 20:
            raise ValueError(f"Source '{source}' is too short.")
        for c in chunk_text(text):
            self.chunks.append({"source": source, "text": c})
        self._rebuild()

    def _rebuild(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), sublinear_tf=True)
        self._X = self._vec.fit_transform([c["text"].lower() for c in self.chunks])

    def retrieve(self, query: str, top: int = 3) -> list[dict]:
        if not self.chunks:
            return []
        qv = self._vec.transform([query.lower()])
        sims = (qv @ self._X.T).toarray()[0]
        out = []
        for i in np.argsort(-sims)[:top]:
            if sims[i] <= 0:
                break
            out.append({"source": self.chunks[i]["source"],
                        "text": self.chunks[i]["text"],
                        "score": round(float(sims[i]), 4)})
        return out


PRONOUNS = re.compile(r"^\s*(?:it|that|this|they|them|he|she|the (?:first|second|third|last) one|more about)\b", re.I)


class RagSession:
    """Conversation state for one user session."""

    def __init__(self):
        self.kb = KnowledgeBase()
        self.history: list[dict] = []
        self._last_results: list[dict] = []

    def add_document(self, source: str, text: str):
        self.kb.add(source, text)

    def ask(self, question: str) -> dict:
        q = (question or "").strip()
        if not q:
            raise ValueError("Question is empty.")
        if not self.kb.chunks:
            return {"answer": "No documents are loaded yet. Add a document first.",
                    "citations": [], "grounded": False}
        # follow-up resolution: if the question is anaphoric, re-run last query context
        effective_q = q
        if PRONOUNS.match(q) and self.history:
            effective_q = self.history[-1]["question"] + " " + q
        results = self.kb.retrieve(effective_q, top=3)
        self._last_results = results
        if not results:
            ans = "I could not find anything about that in the loaded documents."
            grounded = False
        else:
            parts = []
            for i, r in enumerate(results, 1):
                sent = _best_sentence(r["text"], effective_q)
                parts.append(f"{sent} [{i}]")
            ans = " ".join(parts)
            grounded = True
        self.history.append({"question": q, "answer": ans,
                             "citations": [{"n": i + 1, "source": r["source"], "score": r["score"]}
                                           for i, r in enumerate(results)]})
        return {"answer": ans,
                "citations": [{"n": i + 1, "source": r["source"], "score": r["score"],
                               "excerpt": r["text"][:200]} for i, r in enumerate(results)],
                "grounded": grounded}


def _best_sentence(passage: str, query: str) -> str:
    sents = re.split(r"(?<=[.!?])\s+", passage)
    qterms = {t for t in re.findall(r"[a-z']+", query.lower()) if len(t) > 3}
    best, best_score = sents[0], -1
    for s in sents:
        toks = set(re.findall(r"[a-z']+", s.lower()))
        sc = len(qterms & toks)
        if sc > best_score:
            best, best_score = s, sc
    return best.strip()
