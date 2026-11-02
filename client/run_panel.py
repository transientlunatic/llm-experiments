#!/usr/bin/env python3
"""Run several reviews (e.g. several personas) in ONE Ollama session, so the model is loaded once.

  run_panel.py spec.json
spec.json: {"doc": "doc.txt", "prompt": "review.txt", "evidence": "claim_findings.md", "num_ctx": 16384,
            "max_out": 3000, "jobs": [{"focus_file": "focus_1.txt", "out": "review_1.md"}, ...]}
A failing review is reported and skipped; the exit code is non-zero only if every review failed.
"""
import json, sys, traceback
import review_pdf
from doc import load_document

spec = json.load(open(sys.argv[1]))
text = load_document(spec["doc"])
ok = 0
for j in spec["jobs"]:
    try:
        review_pdf.run(spec["doc"], spec["prompt"], j.get("focus_file"), spec.get("evidence"), j["out"],
                       spec["num_ctx"], spec.get("max_out", 3000), text=text)
        ok += 1
    except BaseException:
        traceback.print_exc()
        print(f"[panel] review {j['out']} FAILED", file=sys.stderr)
sys.exit(0 if ok else 1)
