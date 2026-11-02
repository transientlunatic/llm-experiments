#!/usr/bin/env python3
"""Handling-editor step: combine referee reviews into a report with prioritised actionables.

  editor.py [--reviews reviews] [--plan plan.json] [--prompt editor.txt] [--out report.md]
            [--num-ctx 16384] [--max-out 4000]
Reads reviews/review_<i>.md (missing ones are reported, not fatal). Uses $OLLAMA_HOST/$OLLAMA_MODEL.
"""
import argparse, glob, json, os, re, sys
from doc import est_tokens, TOK_CHARS
from oclient import chat

ap = argparse.ArgumentParser()
ap.add_argument("--reviews", default="reviews"); ap.add_argument("--plan", default="plan.json")
ap.add_argument("--prompt", default="editor.txt"); ap.add_argument("--out", default="report.md")
ap.add_argument("--num-ctx", type=int, default=16384); ap.add_argument("--max-out", type=int, default=4000)
a = ap.parse_args()

plan = json.load(open(a.plan)) if os.path.exists(a.plan) else {"referees": []}
meta = {r["id"]: r for r in plan["referees"]}
reviews = {}
for f in sorted(glob.glob(os.path.join(a.reviews, "review_*.md"))):
    i = int(re.search(r"review_(\d+)\.md", f).group(1))
    reviews[i] = open(f).read().strip()
missing = [i for i in meta if i not in reviews]
if not reviews:
    sys.exit("no referee reviews were produced; nothing to summarise")

instr = open(a.prompt).read().strip()
budget = a.num_ctx - a.max_out - est_tokens(instr) - 600
share = budget // len(reviews)
blocks = []
for i, r in sorted(reviews.items()):
    if est_tokens(r) > share:
        print(f"[editor] WARNING: truncating review {i} to fit the context window", file=sys.stderr)
        r = r[:int(share * TOK_CHARS)]
    m = meta.get(i, {})
    blocks.append(f"=== Referee {i} (model: {m.get('model', '?')}; focus: {m.get('focus_short', 'general')}) ===\n{r}")
note = f"\n\nNOTE: referee(s) {missing} failed to produce a review." if missing else ""
summary = chat(f"{instr}{note}\n\n" + "\n\n".join(blocks), num_ctx=a.num_ctx, num_predict=a.max_out)

with open(a.out, "w") as f:
    f.write(f"# Review report\n\n## Editor's summary\n\n{summary}\n\n## Panel\n\n")
    for i, m in sorted(meta.items()):
        f.write(f"- Referee {i}: `{m['model']}` (ctx {m['ctx']}, {m['mode']} pass) -- {m['focus_short']}"
                f"{'' if i in reviews else '  **[FAILED]**'}\n")
    for i, r in sorted(reviews.items()):
        f.write(f"\n## Referee {i} report\n\n{r}\n")
print(f"[editor] wrote {a.out}", file=sys.stderr)
