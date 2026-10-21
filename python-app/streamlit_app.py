"""Optional Streamlit front over the RAG core (rag_core.py).

The primary interface is the FastAPI service in api.py; this UI exists so the
app can also deploy to free Streamlit hosting. Same engine, same behavior.
"""
import streamlit as st
from rag_core import RagSession

st.set_page_config(page_title="Doc QA RAG Chatbot (Python)", page_icon="💬", layout="wide")
st.title("Doc QA RAG Chatbot")
st.caption("Retrieval-augmented Q&A over your own documents: chunking, TF-IDF retrieval, grounded answers with citations. Fully offline.")

SAMPLES = [
    ("refund-policy.txt", "Customers may request a refund within 30 days of purchase. Digital goods are refundable only if not downloaded. Refunds are processed to the original payment method within five business days. Shipping fees are not refundable. To start a refund, contact support with your order number."),
    ("security.txt", "All data is encrypted at rest with AES-256 and in transit with TLS 1.3. Access keys rotate every 90 days. Employees complete security training annually. We run third-party penetration tests twice a year. Incident response targets one hour."),
]

if "session" not in st.session_state:
    s = RagSession()
    for src, txt in SAMPLES:
        s.add_document(src, txt)
    st.session_state.session = s
    st.session_state.log = []

s: RagSession = st.session_state.session

with st.sidebar:
    st.header("Knowledge base")
    st.write(f"**{len(s.kb.chunks)}** chunks indexed")
    with st.expander("Add a document"):
        src = st.text_input("Source name", placeholder="notes.txt")
        txt = st.text_area("Text", height=140)
        if st.button("Add") and txt.strip():
            try:
                s.add_document(src.strip() or "document", txt.strip())
                st.rerun()
            except ValueError as e:
                st.error(str(e))

for turn in st.session_state.log:
    with st.chat_message("user"):
        st.write(turn["q"])
    with st.chat_message("assistant"):
        st.write(turn["a"])
        for c in turn["citations"]:
            st.caption(f"[{c['n']}] {c['source']} (score {c['score']:.2f})")

q = st.chat_input("Ask about the loaded documents")
if q:
    r = s.ask(q)
    st.session_state.log.append({"q": q, "a": r["answer"], "citations": r["citations"]})
    st.rerun()
