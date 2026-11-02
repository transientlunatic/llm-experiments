#!/usr/bin/env python3
"""Review one document with a local Ollama model (one 'referee').

  review_pdf.py DOC [--prompt P] [--focus-file F] [--evidence claim_findings.md] [--ref-prompt R]
                [--out review.md] [--num-ctx 32768] [--max-out 3000] [--images N] [--raw]
DOC may be .pdf, .txt, .tex or an arXiv source tarball (see doc.py). If the text doesn't fit
in the context window it is split into chunks, each is reviewed, and the partial reviews are
merged (in several stages if they don't fit in one window either).
"""
import argparse, base64, glob, os, shutil, subprocess, sys, tempfile
import doc as D
from doc import load_document, est_tokens, chunk, split_main_appendix
from oclient import calibrate_tokens, chat, chat_meta


REMINDER = ("Reminder of the task: review the SCIENCE of the main text. Do not review the reference list or the formatting "
            "of citations, and do not review the appendices in their own right (they are supplementary: consult them only "
            "to check claims made in the main text). Prioritise the most important points; at most about 800 words.")
APP_HEAD = ("--- SUPPLEMENTARY APPENDIX (consult only to check claims in the main text; do not review it in its own "
            "right) ---")
OVERFLOW_NOTE = ("[WARNING: the input still exceeded this model's context window after re-chunking, so part of the "
                 "paper (or of the instructions) may have been dropped. Treat this review with caution.]\n\n")
TRUNC = "\n\n[review cut off at the output-length limit]"
PARTIAL_OUT = 1200           # tokens for each chunk's brief partial review


def tail_for(stance):
    """What comes LAST in every prompt: the task, and the assigned stance (so it survives if the front is dropped)."""
    return REMINDER + (f"\nYour assigned stance (keep to it): {stance}" if stance else "")


def overflowed(r, expected, num_ctx):
    """Did the prompt overflow the window? True if the server counts a full window, or far fewer tokens than we sent
    (Ollama drops tokens from near the start of an over-long prompt and reports only what it kept)."""
    used = r.get("prompt_tokens", 0)
    if not used:
        return False
    return used >= num_ctx - 8 or (expected > 0.6 * num_ctx and used < 0.8 * expected)


def rate(review, stance, ref_prompt, ctx):
    """REF rating in its own short call, so a long or truncated review can never lose it."""
    base = open(ref_prompt).read().strip()
    prompt = f"{base}\n\n{stance}\n\n--- REVIEW ---\n{review}"
    for _ in range(2):
        out = chat(prompt, num_ctx=ctx, num_predict=300, temperature=0.0, label="rating")
        i = out.find("REF_ASSESSMENT")
        if i >= 0:
            return out[i:].strip()
        prompt += "\n\nReply with ONLY the REF_ASSESSMENT block in exactly the requested format."
    print("[review] WARNING: no parsable REF_ASSESSMENT from the model", file=sys.stderr)
    return ""


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
            r = chat_meta(f"{instr}\n\nBelow are partial reviews of consecutive parts of one document. Merge them "
                          f"into a single coherent review in the requested format (at most about 800 words), removing "
                          f"duplication.\n\n{joined}", num_ctx=ctx, num_predict=max_out, label="merge")
            return r["text"] + (TRUNC if r["truncated"] else "")
        groups, cur, cur_t = [], [], 0
        for p in parts:
            t = est_tokens(p)
            if t > budget:
                print("[review] WARNING: a partial review exceeds the context budget; truncating", file=sys.stderr)
                p, t = p[:int(budget * D.TOK_CHARS)], budget
            if cur and cur_t + t > budget:
                groups.append(cur); cur, cur_t = [], 0
            cur.append(p); cur_t += t
        groups.append(cur)
        if len(groups) == len(parts):     # nothing fits two-at-a-time: shrink so pairs fit
            print("[review] WARNING: context too small to merge; truncating partials (use a larger --num-ctx)",
                  file=sys.stderr)
            parts = [p[:int(budget * D.TOK_CHARS / 2)] for p in parts]
            continue
        print(f"[review] merging {len(parts)} partials in {len(groups)} groups", file=sys.stderr)
        parts = [g[0] if len(g) == 1 else merge_reviews(g, instr, ctx, max_out) for g in groups]
    return parts[0]


def run(doc, prompt, focus_file=None, evidence=None, out="review.md", num_ctx=32768, max_out=3000,
        images=0, raw=False, text=None, ref_prompt=None, calibrate=True):
    """Review `doc` (or pre-loaded `text`) once; write the review (plus a separate REF rating) to `out`."""
    instr = open(prompt).read().strip()
    stance = ""
    if focus_file:
        stance = open(focus_file).read().strip()
        instr += "\n\n" + stance
    if evidence and os.path.exists(evidence) and os.path.getsize(evidence) > 0:
        instr += ("\n\nAUTOMATED CLAIM-CHECK FINDINGS (produced by tools; they can be wrong, but where relevant "
                  "discuss them and say whether you agree; citation checks are secondary, so do not let them dominate your "
                  "review):\n" + open(evidence).read().strip())
    if text is None:
        text = load_document(doc, raw=raw)
    if len(text.strip()) < 200:
        sys.exit("almost no text extracted (scanned PDF?) -- try --images with a vision model")
    if calibrate:
        calibrate_tokens(text, num_ctx)
    print(f"[review] {len(text)} chars (~{est_tokens(text)} tokens), num_ctx={num_ctx}", file=sys.stderr)
    imgs = render_pages(doc, images) if images else None
    tail = tail_for(stance)
    main, appendix = split_main_appendix(text)
    budget = num_ctx - max_out - est_tokens(instr) - est_tokens(tail) - 500
    review = None
    if est_tokens(main) <= budget:
        doc_part = f"--- MAIN TEXT ---\n{main}"
        room = budget - est_tokens(main) - 50
        if appendix and room > 200:                  # the appendix goes after the main text, only as far as it fits
            app = appendix if est_tokens(appendix) <= room else appendix[:int(room * D.TOK_CHARS)] + "\n[appendix truncated]"
            doc_part += f"\n\n{APP_HEAD}\n{app}"
        elif appendix:
            print("[review] appendix omitted: no room in the context window", file=sys.stderr)
        full = f"{instr}\n\n{doc_part}\n\n--- END OF DOCUMENT ---\n{tail}"
        r = chat_meta(full, imgs, num_ctx=num_ctx, num_predict=max_out, label="review")
        if overflowed(r, est_tokens(full), num_ctx):
            print(f"[review] WARNING: the prompt overflowed the context window (server counted {r['prompt_tokens']} tokens "
                  f"for ~{est_tokens(full)} sent); reading in chunks instead", file=sys.stderr)
        else:
            review = r["text"] + (TRUNC if r["truncated"] else "")
    flagged = False
    if review is None:
        scale = 1.0
        for attempt in range(4):
            parts = chunk(main, int(budget * scale))
            print(f"[review] reading the main text in {len(parts)} chunks (~{est_tokens(main)} tokens, "
                  f"chunk budget {int(budget * scale)})" + ("; appendix not reviewed" if appendix else ""), file=sys.stderr)
            brief = ("\n\nThis is part {i}/{n} of the main text" + (" (appendices are omitted from this reading)" if appendix else "") +
                     ". Review only this part, as BRIEF bullet points (at most 300 words): key strengths, weaknesses and "
                     "unsupported claims, each with a short quote. A merge step follows, so do not write a full review and "
                     "do not give a star rating.")
            partial, over = [], False
            for i, p in enumerate(parts, 1):
                full = (f"{instr}{brief.format(i=i, n=len(parts))}\n\n--- MAIN TEXT, PART {i} ---\n{p}\n\n"
                        f"--- END OF PART ---\n{tail}")
                r = chat_meta(full, imgs if i == 1 else None, num_ctx=num_ctx, num_predict=min(max_out, PARTIAL_OUT),
                              label=f"chunk {i}/{len(parts)}")
                over = over or overflowed(r, est_tokens(full), num_ctx)
                partial.append(r["text"] + (TRUNC if r["truncated"] else ""))
            if not over:
                break
            scale *= 0.7
            print(f"[review] WARNING: a chunk overflowed the context window; retrying with chunks 70% the size",
                  file=sys.stderr)
        else:
            flagged = True
        review = merge_reviews(partial, instr, num_ctx, max_out)
    if flagged:
        review = OVERFLOW_NOTE + review
    if ref_prompt:
        block = rate(review, stance, ref_prompt, num_ctx)
        review += "\n\n" + block if block else ""
    open(out, "w").write(review + "\n")
    print(f"[review] wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("doc"); ap.add_argument("--prompt", default="prompts/review.txt")
    ap.add_argument("--focus-file"); ap.add_argument("--evidence"); ap.add_argument("--out", default="review.md")
    ap.add_argument("--ref-prompt", help="if given, add a REF rating via a separate short call")
    ap.add_argument("--num-ctx", type=_int, default=32768); ap.add_argument("--max-out", type=_int, default=3000)
    ap.add_argument("--images", type=_int, default=0, help="attach first N pages as images (vision model only)")
    ap.add_argument("--raw", action="store_true", help="don't clean LaTeX")
    a = ap.parse_args()
    run(a.doc, a.prompt, a.focus_file, a.evidence, a.out, a.num_ctx, a.max_out, a.images, a.raw, ref_prompt=a.ref_prompt)
