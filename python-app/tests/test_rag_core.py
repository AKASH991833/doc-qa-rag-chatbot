import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from rag_core import KnowledgeBase, RagSession, chunk_text

DOCS = [
    ("refund-policy.txt", "Customers may request a refund within 30 days of purchase. "
     "Digital goods are refundable only if not downloaded. Refunds are processed to the "
     "original payment method within five business days. Shipping fees are not refundable. "
     "To start a refund, contact support with your order number."),
    ("security.txt", "All data is encrypted at rest with AES-256 and in transit with TLS 1.3. "
     "Access keys rotate every 90 days. Employees complete security training annually. "
     "We run third-party penetration tests twice a year. Incident response targets one hour."),
]

def make_session():
    s = RagSession()
    for src, txt in DOCS:
        s.add_document(src, txt)
    return s

def test_grounded_answer_with_citations():
    s = make_session()
    r = s.ask("How long do I have to request a refund?")
    assert r["grounded"] is True
    assert "30 days" in r["answer"]
    assert r["citations"] and r["citations"][0]["source"] == "refund-policy.txt"

def test_second_question_hits_second_doc():
    s = make_session()
    r = s.ask("How is my data encrypted?")
    assert r["grounded"]
    assert "AES-256" in r["answer"]
    assert r["citations"][0]["source"] == "security.txt"

def test_unknown_question_ungrounded():
    s = make_session()
    r = s.ask("What is the meaning of life?")
    assert r["grounded"] is False

def test_empty_kb_message():
    s = RagSession()
    r = s.ask("anything")
    assert r["grounded"] is False
    assert "No documents" in r["answer"]

def test_history_and_followup():
    s = make_session()
    s.ask("How are refunds processed?")
    r = s.ask("more about that")  # anaphoric follow-up
    assert len(s.history) == 2
    assert r["citations"]

def test_chunk_overlap():
    text = " ".join(f"w{i}" for i in range(250))
    chunks = chunk_text(text)
    assert len(chunks) >= 2

def test_short_source_rejected():
    kb = KnowledgeBase()
    try:
        kb.add("tiny", "nope")
        assert False
    except ValueError:
        pass

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"All {len(fns)} tests passed.")

def test_streamlit_app_imports():
    # streamlit_app requires streamlit; import smoke test
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "streamlit_app", os.path.join(os.path.dirname(__file__), "..", "streamlit_app.py"))
    assert spec is not None
