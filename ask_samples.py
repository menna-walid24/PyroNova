"""Ask the running RAG server a few real questions and save the JSON replies.
    python ask_samples.py            (server must be running on port 8001)"""
import json, sys, time, urllib.request, pathlib
BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8001"
out = pathlib.Path(__file__).resolve().parent / "docs" / "sample_responses"
out.mkdir(parents=True, exist_ok=True)
QS = [("q1_burn_time_by_id", "What was the burn time in experiment CFI-21153-A5 and how did it end?"),
      ("q2_oxygen_effect", "How does oxygen concentration affect burn time in microgravity?"),
      ("q3_fuels_arabic", "ايه اكتر وقود خطر في التجارب؟"),
      ("q4_hello", "hello")]
for name, q in QS:
    t = time.time()
    req = urllib.request.Request(BASE + "/ask", data=json.dumps({"question": q}).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    b = json.loads(urllib.request.urlopen(req, timeout=600).read())
    (out / f"{name}.json").write_text(json.dumps(b, indent=2, ensure_ascii=False), encoding="utf-8")
    p = b.get("prediction"); p = p[0] if isinstance(p, list) and p else p
    print(f"\n== {name} ({time.time()-t:.0f}s) model={b.get('model_used')} decision={b.get('answerability',{}).get('decision')}")
    print("A:", b["answer"][:600])
    print("sources:", b.get("sources")[:8])
    print("limitations:", [l[:80] for l in b.get("limitations", [])])
