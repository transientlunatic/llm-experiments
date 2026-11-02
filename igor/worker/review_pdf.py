#!/usr/bin/env python3
"""Review one document with a local Ollama model (one 'referee').

  review_pdf.py DOC [--prompt P] [--focus-file F] [--evidence claim_findings.md] [--out review.md]
                [--num-ctx 32768] [--max-out 3000] [--images N] [--raw]
DOC may be .pdf, .txt, .tex or an arXiv source tarball (see doc.py). If the text doesn't fit
in the context window it is split into chunks, each is reviewed, and the partial reviews are
merged (in several stages if they don't fit in one window either).
"""
import argparse, base64, glob, os, shutil, subprocess, sys, tempfile
from doc import load_document, est_tokens, chunk, TOK_CHARS
from oclient import chat


def render_pages(path, n):
    if not shutil.which("pdftoppm"):
        sys.exit("--images needs pdftoppm (poppler) on the worker")
    d = tempfile.mkdtemp()
    subprocess.run(["pdftoppm", "-r", "100", "-png", "-l", str(n), path, f"{d}/p"], check=True)
    return [base64.b64encode(open(f, "rb").read()).decode() for f in sorted(glob.glob(f"{d}/p*.png"))]


def _int(v):  # tolerate a trailing "# comment" that Condor passes through verbatim
    return int(v.split("#")[0].strip())


def merge_reviews(parts, instr, ctx, max_out):
    """Merge partial reviews; if they don't fit one window, merge in groups first."""
    budget = ctx - max_out - est_tokens(instr) - 800
    while len(parts) > 1:
        if sum(est_tokens(p) for p in parts) <= budget:
            joined = "\n\n".join(f"### Partial review {i}\n{r}" for i, r in enumerate(parts, 1))
            return chat(f"{instr}\n\nBelow are partial reviews of consecutive parts of one document. Merge them "
                        f"into a single coherent review in the requested format, removing duplication.\n\n{joined}",
                        num_ctx=ctx, num_predict=max_out)
        groups, cur, cur_t = [], [], 0
        for p in parts:
            t = est_tokens(p)
            if t > budget:
                print("[review] WARNING: a partial review exceeds the context budget; truncating", file=sys.stderr)
                p, t = p[:int(budget * TOK_CHARS)], budget
            if cur and cur_t + t > budget:
                groups.append(cur); cur, cur_t = [], 0
            cur.append(p); cur_t += t
        groups.append(cur)
        if len(groups) == len(parts):     # nothing fits two-at-a-time: shrink so pairs fit
            print("[review] WARNING: context too small to merge; truncating partials (use a larger --num-ctx)",
                  file=sys.stderr)
            parts = [p[:int(budget * TOK_CHARS / 2)] for p in parts]
            continue
        print(f"[review] merging {len(parts)} partials in {len(groups)} groups", file=sys.stderr)
        parts = [g[0] if len(g) == 1 else merge_reviews(g, instr, ctx, max_out) for g in groups]
    return parts[0]


def run(doc, prompt, focus_file=None, evidence=None, out="review.md", num_ctx=32768, max_out=3000,
        images=0, raw=False, text=None):
    """Review `doc` (or pre-loaded `text`) once; write the review to `out`."""
    instr = open(prompt).read().strip()
    if focus_file:
        instr += "\n\n" + open(focus_file).read().strip()
    if evidence and os.path.exists(evidence) and os.path.getsize(evidence) > 0:
        instr += ("\n\nAUTOMATED CLAIM-CHECK FINDINGS (produced by tools; they can be wrong, but where relevant "
                  "discuss them and say whether you agree):\n" + open(evidence).read().strip())
    if text is None:
        text = load_document(doc, raw=raw)
    print(f"[review] {len(text)} chars (~{est_tokens(text)} tokens), num_ctx={num_ctx}", file=sys.stderr)
    if len(text.strip()) < 200:
        sys.exit("almost no text extracted (scanned PDF?) -- try --images with a vision model")
    imgs = render_pages(doc, images) if images else None
    budget = num_ctx - max_out - est_tokens(instr) - 500
    if est_tokens(text) <= budget:
        review = chat(f"{instr}\n\n--- DOCUMENT ---\n{text}", imgs, num_ctx=num_ctx, num_predict=max_out)
    else:
        parts = chunk(text, budget)
        print(f"[review] too long for one pass: {len(parts)} chunks", file=sys.stderr)
        partial = [chat(f"{instr}\n\nThis is part {i}/{len(parts)} of the document; review only this part.\n\n"
                        f"--- DOCUMENT PART {i} ---\n{p}", imgs if i == 1 else None,
                        num_ctx=num_ctx, num_predict=max_out)
                   for i, p in enumerate(parts, 1)]
        review = merge_reviews(partial, instr, num_ctx, max_out)
    open(out, "w").write(review + "\n")
    print(f"[review] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("doc"); ap.add_argument("--prompt", default="prompts/review.txt")
    ap.add_argument("--focus-file"); ap.add_argument("--evidence"); ap.add_argument("--out", default="review.md")
    ap.add_argument("--num-ctx", type=_int, default=32768); ap.add_argument("--max-out", type=_int, default=3000)
    ap.add_argument("--images", type=_int, default=0, help="attach first N pages as images (vision model only)")
    ap.add_argument("--raw", action="store_true", help="don't clean LaTeX")
    a = ap.parse_args()
    run(a.doc, a.prompt, a.focus_file, a.evidence, a.out, a.num_ctx, a.max_out, a.images, a.raw)
