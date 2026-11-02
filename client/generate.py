#!/usr/bin/env python3
"""Batch prompts through Ollama. Stdlib only, so it runs on bare worker nodes.

  generate.py prompts.jsonl results.jsonl
Each input line: {"prompt": "...", ...extra fields kept...}. Uses $OLLAMA_HOST / $OLLAMA_MODEL.
"""
import json, os, sys, urllib.request

host = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434")
model = os.environ["OLLAMA_MODEL"]

def chat(prompt):
    body = json.dumps({"model": model, "stream": False,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"http://{host}/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.load(r)["message"]["content"]

with open(sys.argv[1]) as fin, open(sys.argv[2], "w") as fout:
    for line in fin:
        row = json.loads(line)
        row["response"] = chat(row["prompt"])
        fout.write(json.dumps(row) + "\n"); fout.flush()
