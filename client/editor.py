#!/usr/bin/env python3
"""Handling-editor step: combine referee reviews into a report with prioritised actionables.

  editor.py [--reviews reviews] [--plan plan.json] [--prompt editor.txt] [--out report.md]
            [--num-ctx 16384] [--max-out 4000]
Reads reviews/review_<i>.md (missing ones are reported, not fatal). Uses $OLLAMA_HOST/$OLLAMA_MODEL.
"""
import argparse, glob, json, os, re, sys
from doc import est_tokens, TOK_CHARS
from oclient import chat

STARS = ["4*", "3*", "2*", "1*", "U"]
VAL = {"4*": 4, "3*": 3, "2*": 2, "1*": 1, "U": 0}
CRIT = ("originality", "significance", "rigour")


def parse_ref(text):
    """Parse the REF_ASSESSMENT block; return None if absent/unusable."""
    m = re.search(r"REF_ASSESSMENT(.*)", text, re.S)
    if not m:
        return None
    blk, out = m.group(1), {}
    for c in CRIT:
        mm = re.search(c + r"\s*[:=]\s*\**\s*([0-4])\b", blk, re.I)
        out[c] = int(mm.group(1)) if mm else None
    d = {k: float(v) for k, v in re.findall(r"(4\*|3\*|2\*|1\*|U)\s*[=:]\s*([\d.]+)", blk)}
    tot = sum(d.get(k, 0) for k in STARS)
    out["dist"] = {k: d.get(k, 0) / tot for k in STARS} if set(STARS) <= set(d) and tot > 0 else None
    return out


def ref_table(refs, meta):
    """Markdown REF table computed here (not by the model). refs: {id: parsed or None}."""
    lines = ["| Referee | Stance | Originality | Significance | Rigour | P(4*) | P(3*) | P(2*) | P(1*) | P(U) | E[stars] |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    dists = []
    for i, r in sorted(refs.items()):
        who = meta.get(i, {})
        if not r:
            lines.append(f"| {i} (`{who.get('model','?')}`) | {who.get('persona','?')} | " + " | ".join(["n/a"] * 9) + " |")
            continue
        crit = [str(r[c]) if r[c] is not None else "n/a" for c in CRIT]
        if r["dist"]:
            dists.append(r["dist"])
            ps = [f"{100*r['dist'][k]:.0f}%" for k in STARS]
            e = f"{sum(VAL[k]*r['dist'][k] for k in STARS):.2f}"
        else:
            ps, e = ["n/a"] * 5, "n/a"
        lines.append(f"| {i} (`{who.get('model','?')}`) | {who.get('persona','?')} | " + " | ".join(crit + ps + [e]) + " |")
    if dists:
        mix = {k: sum(d[k] for d in dists) / len(dists) for k in STARS}
        mean_c = []
        for c in CRIT:
            v = [r[c] for r in refs.values() if r and r[c] is not None]
            mean_c.append(f"{sum(v)/len(v):.1f}" if v else "n/a")
        lines.append(f"| **Panel (equal-weight mixture)** | | " + " | ".join(mean_c) + " | " +
                     " | ".join(f"{100*mix[k]:.0f}%" for k in STARS) +
                     f" | **{sum(VAL[k]*mix[k] for k in STARS):.2f}** |")
    return "\n".join(lines), bool(dists)


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
budget = a.num_ctx - a.max_out - est_tokens(instr) - 1200
share = budget // len(reviews)
blocks = []
for i, r in sorted(reviews.items()):
    if est_tokens(r) > share:
        print(f"[editor] WARNING: truncating review {i} to fit the context window", file=sys.stderr)
        r = r[:int(share * TOK_CHARS)]
    m = meta.get(i, {})
    blocks.append(f"=== Referee {i} (model: {m.get('model', '?')}; stance: {m.get('persona', 'neutral')}; focus: {m.get('focus_short', 'general')}) ===\n{r}")
refs = {i: parse_ref(r) for i, r in reviews.items()}
table, have_ref = ref_table(refs, meta)
ref_note = ("\n\nREF RATING TABLE (computed from the referees' own ratings; LLM ratings are poorly calibrated):\n" + table) if have_ref else ""
note = f"\n\nNOTE: referee(s) {missing} failed to produce a review." if missing else ""
summary = chat(f"{instr}{note}{ref_note}\n\n" + "\n\n".join(blocks), num_ctx=a.num_ctx, num_predict=a.max_out)

with open(a.out, "w") as f:
    f.write(f"# Review report\n\n## Editor's summary\n\n{summary}\n\n## Panel\n\n")
    for i, m in sorted(meta.items()):
        f.write(f"- Referee {i}: `{m['model']}` (ctx {m['ctx']}, {m['mode']} pass) -- {m.get('persona','')} / {m['focus_short']}"
                f"{'' if i in reviews else '  **[FAILED]**'}\n")
    f.write("\n## Estimated REF ratings\n\n" + (table if have_ref else "No referee produced a parsable REF_ASSESSMENT block.") +
            "\n\n_Ratings are the models' own estimates; small local models are poorly calibrated and personas shift "
            "their rating (hostile rates low, supportive high), so read the spread, not the mean._\n")
    for i, r in sorted(reviews.items()):
        f.write(f"\n## Referee {i} report\n\n{r}\n")
print(f"[editor] wrote {a.out}", file=sys.stderr)
