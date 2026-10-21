import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from fastapi.testclient import TestClient
from api import app

def test_full_flow():
    c = TestClient(app)
    assert c.get("/healthz").json()["status"] == "ok"
    sid = c.post("/sessions").json()["session_id"]
    r = c.post(f"/sessions/{sid}/documents", json={"source": "policy.txt",
        "text": "Refunds are available within 30 days of purchase. Contact support with your order number to start one."})
    assert r.status_code == 200 and r.json()["chunks"] >= 1
    a = c.post(f"/sessions/{sid}/ask", json={"question": "How long do refunds take?"})
    assert a.status_code == 200
    body = a.json()
    assert body["grounded"] and "30 days" in body["answer"]
    h = c.get(f"/sessions/{sid}/history").json()["history"]
    assert len(h) == 1

def test_unknown_session_404():
    c = TestClient(app)
    r = c.post("/sessions/nope/ask", json={"question": "hi"})
    assert r.status_code == 404

def test_short_doc_400():
    c = TestClient(app)
    sid = c.post("/sessions").json()["session_id"]
    r = c.post(f"/sessions/{sid}/documents", json={"source": "x", "text": "tiny"})
    assert r.status_code == 400

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"All {len(fns)} tests passed.")
