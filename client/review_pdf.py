#!/usr/bin/env python3
"""Review a document with a local Ollama model. Stdlib only (pdftotext/pypdf used if present).

  review_pdf.py paper.pdf [--prompt prompts/review.txt] [--out review.md]
                [--num-ctx 32768] [--images N]
Input may be .pdf, .txt or .tex (LaTeX source gives far better results for equations).
Uses $OLLAMA_HOST / $OLLAMA_MODEL (set by bin/job.sh).
If the text doesn't fit in the context window it is split into chunks, each chunk is
reviewed, then the partial reviews are merged in a final pass.
"""
import argparse, base64, glob, json, os, re, shutil, subprocess, sys, tempfile, urllib.request

def extract_text(path):
    if path.endswith((".txt", ".tex", ".md")):
        return open(path, errors="replace").read()
    if shutil.which("pdftotext"):
        return subprocess.run(["pdftotext", "-layout", path, "-"], check=True,
                              capture_output=True, text=True).stdout
    try:
        import pypdf
    except ImportError:
        print("[review] no pdftotext; installing pypdf --user", file=sys.stderr)
        subprocess.run([sys.executable, "-m", "pip", "install", "--user", "-q", "pypdf"], check=True)
        import pypdf
    return "\n\n".join(p.extract_text() or "" for p in pypdf.PdfReader(path).pages)

def render_pages(path, n):
    if not shutil.which("pdftoppm"):
        sys.exit("--images needs pdftoppm (poppler) on the worker")
    d = tempfile.mkdtemp()
    subprocess.run(["pdftoppm", "-r", "100", "-png", "-l", str(n), path, f"{d}/p"], check=True)
    return [base64.b64encode(open(f, "rb").read()).decode() for f in sorted(glob.glob(f"{d}/p*.png"))]

def chat(content, images=None, *, num_ctx, num_predict=4096):
    host, model = os.environ.get("OLLAMA_HOST", "127.0.0.1:11434"), os.environ["OLLAMA_MODEL"]
    msg = {"role": "user", "content": content}
    if images:
        msg["images"] = images
    body = json.dumps({"model": model, "stream": False, "messages": [msg],
                       "options": {"num_ctx": num_ctx, "num_predict": num_predict, "temperature": 0.2}}).encode()
    req = urllib.request.Request(f"http://{host}/api/chat", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=3600) as r:
        out = json.load(r)
    used = out.get("prompt_eval_count", 0)
    if used >= num_ctx - 8:
        print(f"[review] WARNING: prompt used {used}/{num_ctx} tokens -- likely truncated", file=sys.stderr)
    return out["message"]["content"]

def chunk(text, budget_tokens):
    """Split on blank lines into pieces of <= ~budget_tokens (4 chars ~ 1 token)."""
    limit, parts, cur = budget_tokens * 4, [], ""
    for para in re.split(r"\n\s*\n", text):
        while len(para) > limit:                       # a single huge paragraph
            if cur: parts.append(cur); cur = ""
            parts.append(para[:limit]); para = para[limit:]
        if len(cur) + len(para) + 2 > limit:
            parts.append(cur); cur = ""
        cur += para + "\n\n"
    return parts + ([cur] if cur.strip() else [])

ap = argparse.ArgumentParser()
ap.add_argument("doc"); ap.add_argument("--prompt", default="prompts/review.txt")
ap.add_argument("--out", default="review.md"); ap.add_argument("--num-ctx", type=int, default=32768)
ap.add_argument("--images", type=int, default=0, help="attach first N pages as images (vision model only)")
a = ap.parse_args()

instr = open(a.prompt).read().strip()
text = extract_text(a.doc)
print(f"[review] {len(text)} chars (~{len(text)//4} tokens), num_ctx={a.num_ctx}", file=sys.stderr)
if len(text.strip()) < 200:
    sys.exit("almost no text extracted (scanned PDF?) -- try --images with a vision model")
imgs = render_pages(a.doc, a.images) if a.images else None

reserve = 4096 + len(instr) // 4 + 500                 # output + instructions + slack
budget = a.num_ctx - reserve
if len(text) // 4 <= budget:
    review = chat(f"{instr}\n\n--- DOCUMENT ---\n{text}", imgs, num_ctx=a.num_ctx)
else:
    parts = chunk(text, budget)
    print(f"[review] too long for one pass: {len(parts)} chunks", file=sys.stderr)
    partial = [chat(f"{instr}\n\nThis is part {i}/{len(parts)} of the document; review only this part.\n\n"
                    f"--- DOCUMENT PART {i} ---\n{p}", imgs if i == 1 else None, num_ctx=a.num_ctx)
               for i, p in enumerate(parts, 1)]
    merged = "\n\n".join(f"### Partial review {i}\n{r}" for i, r in enumerate(partial, 1))
    review = chat(f"{instr}\n\nBelow are partial reviews of consecutive parts of one document. "
                  f"Merge them into a single coherent review in the requested format, removing duplication.\n\n{merged}",
                  num_ctx=a.num_ctx)
open(a.out, "w").write(review + "\n")
print(f"[review] wrote {a.out}", file=sys.stderr)
