"""Document loading/cleaning shared by the review tools. Stdlib only
(pdftotext or pypdf is used for PDFs). Handles .pdf, .txt/.md, .tex, single-file
gzip, and arXiv-style .tar.gz source bundles (with or without extension)."""
import gzip, os, re, shutil, subprocess, sys, tarfile, tempfile

TOK_CHARS = 3.5          # conservative chars-per-token for prose + LaTeX
APPENDIX_MARK = "=== APPENDIX (supplementary material) ==="


def split_main_appendix(text):
    """(main text, appendix text); the appendix is whatever follows APPENDIX_MARK."""
    i = text.find(APPENDIX_MARK)
    if i < 0:
        return text, ""
    return text[:i].rstrip(), text[i + len(APPENDIX_MARK):].strip()


def mark_appendix_plain(text):
    """Heuristic for PDF/plain text: the first short heading-like line 'Appendix[ A][: title]' / 'Appendices' /
    'Supplementary material', and not in the first 3000 characters (title, abstract, contents). Long appendices can be most of a paper."""
    pat = (r"(?im)^[ \t]*(?:appendices|appendix(?:[ \t]+[A-Z0-9]{1,3})?|supplementary (?:material|information))"
           r"(?:[ \t]*[.:\-\u2013][ \t]*[^\n]{0,60})?[ \t]*$")
    for m in re.finditer(pat, text):
        if m.start() > 3000:             # skip title/abstract/contents lines near the top
            return text[:m.start()] + APPENDIX_MARK + "\n" + text[m.start():]
    return text


def est_tokens(text):
    return int(len(text) / TOK_CHARS) + 1


# ---------------------------------------------------------------- PDF
def _import_pypdf():
    """Import pypdf, installing it into the node-local cache (pip --target) if needed.
    --target avoids PEP 668 'externally-managed-environment' errors on system Pythons."""
    import importlib
    user = os.environ.get("USER") or subprocess.getoutput("id -un")
    lib = os.path.join(os.environ.get("LLM_CACHE", f"/tmp/{user}-llm-cache"), "pylib")
    sys.path.insert(0, lib)
    try:
        import pypdf
    except ImportError:
        print(f"[doc] no pdftotext; installing pypdf into {lib}", file=sys.stderr)
        py = shutil.which("python3") or sys.executable or "python3"
        for extra in ([], ["--break-system-packages"]):
            if subprocess.run([py, "-m", "pip", "install", "-q", "--target", lib, *extra, "pypdf"]).returncode == 0:
                break
        else:
            sys.exit("could not install pypdf; run pdftotext yourself and pass the .txt instead")
        importlib.invalidate_caches()
        import pypdf
    return pypdf


def pdf_text(path):
    if shutil.which("pdftotext"):   # default reading order copes better with two-column papers
        return subprocess.run(["pdftotext", path, "-"], check=True, capture_output=True, text=True).stdout
    pypdf = _import_pypdf()
    return "\n\n".join(p.extract_text() or "" for p in pypdf.PdfReader(path).pages)


# ---------------------------------------------------------------- LaTeX
_INPUT_RE = re.compile(r"\\(?:input|include)\s*\{([^}]+)\}")


def strip_comments(text):
    text = re.sub(r"\\begin\{comment\}.*?\\end\{comment\}", "", text, flags=re.S)
    return re.sub(r"(?<!\\)%[^\n]*", "", text)


def _braced(text, cmd):
    m = re.search(r"\\" + cmd + r"\s*(?:\[[^\]]*\])?\s*\{", text)
    if not m:
        return ""
    i, depth = m.end(), 1
    while i < len(text) and depth:
        c = text[i]
        if c == "\\":
            i += 2
            continue
        depth += (c == "{") - (c == "}")
        i += 1
    return text[m.end():i - 1]


def inline_inputs(text, roots, depth=0):
    """Replace \\input{x}/\\include{x} by the (comment-stripped) file contents."""
    def repl(m):
        name = m.group(1).strip()
        for root in roots:
            for cand in (name, name + ".tex"):
                p = os.path.join(root, cand)
                if depth < 8 and os.path.isfile(p):
                    return inline_inputs(strip_comments(open(p, errors="replace").read()), roots, depth + 1)
        return ""
    return _INPUT_RE.sub(repl, text)


def clean_latex(text):
    """Keep what a referee needs: title, macros, body. Drop preamble, authors (blind review),
    bibliography, TikZ/pgf pictures, layout commands."""
    pre, sep, body = text.partition("\\begin{document}")
    if not sep:
        pre, body = "", text
    title = _braced(text, "title")
    macros = [l.strip() for l in pre.splitlines()
              if re.match(r"\\(newcommand|renewcommand|def|DeclareMathOperator)\b", l.strip()) and len(l) < 200][:60]
    body = body.split("\\end{document}")[0]
    body = re.sub(r"\\appendix\b|\\begin\{appendices\}", "\n" + APPENDIX_MARK + "\n", body, count=1)
    body = re.sub(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", "", body, flags=re.S)
    body = re.sub(r"\\(bibliography|bibliographystyle)\s*\{[^}]*\}", "", body)
    body = re.sub(r"\\begin\{(tikzpicture|pgfpicture|axis)\}.*?\\end\{\1\}", "", body, flags=re.S)
    body = re.sub(r"\\includegraphics\s*(\[[^\]]*\])?\s*\{[^}]*\}", "", body)
    body = re.sub(r"\\(label|vspace\*?|hspace\*?)\s*\{[^}]*\}", "", body)
    body = re.sub(r"\\(centering|clearpage|newpage|noindent|maketitle)\b", "", body)
    body = re.sub(r"[ \t]+\n", "\n", body)
    body = re.sub(r"\n{3,}", "\n\n", body)
    out = []
    if title:
        out.append("TITLE: " + " ".join(title.split()))
    if macros:
        out.append("AUTHOR MACROS:\n" + "\n".join(macros))
    out.append(body.strip())
    return "\n\n".join(out)


def _safe_extract(tf, dest):
    real = os.path.realpath(dest)
    for m in tf.getmembers():
        if not (m.isfile() or m.isdir()):
            continue
        if not os.path.realpath(os.path.join(dest, m.name)).startswith(real + os.sep):
            continue
        try:
            tf.extract(m, dest, filter="data")
        except TypeError:           # Python without extraction filters
            tf.extract(m, dest)


def _main_tex(root):
    texs = [os.path.join(d, f) for d, _, fs in os.walk(root) for f in fs if f.endswith(".tex")]
    mains = [p for p in texs if "\\documentclass" in open(p, errors="replace").read()]
    if not mains:
        sys.exit("no \\documentclass found in the source bundle")
    full = [p for p in mains if "\\begin{document}" in open(p, errors="replace").read()]
    return max(full or mains, key=os.path.getsize)


def _bib_to_text(bib):
    """Crude .bib -> 'key: Authors. Title. Year.' lines (only used when no .bbl is available)."""
    out = []
    for m in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,(.*?)\n\}", bib, re.S):
        def field(n):
            f = re.search(n + r"\s*=\s*[{\"](.*?)[}\"]\s*(?:,|$)", m.group(2), re.S | re.I)
            return " ".join(f.group(1).split()) if f else ""
        out.append(f"{m.group(1)}: {field('author')}. {field('title')}. {field('journal') or field('booktitle')} {field('year')}.")
    return "\n".join(out)


def _tex_refs(main_text, root):
    """Reference list text from a bundle: \\bibitem entries (.bbl or inline) or .bib files."""
    chunks = []
    for d, _, fs in os.walk(root):
        for f in fs:
            if f.endswith(".bbl"):
                chunks.append(open(os.path.join(d, f), errors="replace").read())
    m = re.search(r"\\begin\{thebibliography\}.*?\\end\{thebibliography\}", main_text, re.S)
    if m:
        chunks.append(m.group(0))
    if not chunks:
        for d, _, fs in os.walk(root):
            for f in fs:
                if f.endswith(".bib"):
                    chunks.append(_bib_to_text(open(os.path.join(d, f), errors="replace").read()))
    t = "\n\n".join(chunks)
    t = re.sub(r"\\bibitem(\[[^\]]*\])?\s*\{([^}]*)\}", r"\n[\2] ", t)
    t = re.sub(r"\\(newblock|begin\{thebibliography\}\{[^}]*\}|end\{thebibliography\})", " ", t)
    return strip_comments(t).strip()


def _pdf_refs(text):
    """Text after the last 'References' heading, if it sits in the final 40% of the document."""
    hits = [m.start() for m in re.finditer(r"(?im)^\s*(references|bibliography)\s*$", text)]
    return text[hits[-1]:] if hits and hits[-1] > 0.6 * len(text) else ""


# ---------------------------------------------------------------- entry point
def load_document(path, raw=False):
    return load_document_ex(path, raw)[0]


def _finish(text, appendix):
    """appendix='drop' removes the appendix from the text altogether; 'separate' keeps it after APPENDIX_MARK."""
    return split_main_appendix(text)[0] if appendix == "drop" else text


def load_document_ex(path, raw=False, appendix="separate"):
    """Return (clean_text, references_text). references_text may be empty. The text has the appendix (if any)
    after APPENDIX_MARK so consumers can treat it as supplementary."""
    with open(path, "rb") as f:
        head = f.read(4)
    roots, text = [os.path.dirname(os.path.abspath(path))], None
    if head == b"%PDF" or path.lower().endswith(".pdf"):
        t = pdf_text(path)
        refs = _pdf_refs(t)
        if refs:                                    # the reference list is for the citation tool, not the referees
            t = t[:len(t) - len(refs)].rstrip()
        return _finish(mark_appendix_plain(t), appendix), refs
    refs = ""
    if tarfile.is_tarfile(path):
        d = tempfile.mkdtemp()
        with tarfile.open(path) as tf:
            _safe_extract(tf, d)
        main = _main_tex(d)
        roots, text = [os.path.dirname(main), d], open(main, errors="replace").read()
        refs = _tex_refs(inline_inputs(strip_comments(text), [os.path.dirname(main), d]), d)
    elif head[:2] == b"\x1f\x8b":
        text = gzip.open(path).read().decode(errors="replace")
    else:
        text = open(path, errors="replace").read()
    if raw or not ("\\documentclass" in text or "\\begin{document}" in text):
        r = refs or _pdf_refs(text)
        if r and text.endswith(r):
            text = text[:len(text) - len(r)].rstrip()
        return _finish(mark_appendix_plain(text), appendix), r
    full = inline_inputs(strip_comments(text), roots)
    return _finish(clean_latex(full), appendix), refs or _tex_refs(full, roots[0])


def chunk(text, budget_tokens):
    """Split on blank lines into pieces of <= ~budget_tokens."""
    limit, parts, cur = int(budget_tokens * TOK_CHARS), [], ""
    for para in re.split(r"\n\s*\n", text):
        while len(para) > limit:
            if cur:
                parts.append(cur); cur = ""
            parts.append(para[:limit]); para = para[limit:]
        if len(cur) + len(para) + 2 > limit:
            parts.append(cur); cur = ""
        cur += para + "\n\n"
    return parts + ([cur] if cur.strip() else [])
