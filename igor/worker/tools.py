#!/usr/bin/env python3
"""Verification tools. Stdlib only. These run INSIDE a sandboxed worker process (see sandbox.py):

    python3 -I -S tools.py   <  {"tool": ..., "arg": ..., "text": ..., "refs": ..., "entry": ...}

Each tool receives only the data it needs and has no side effects:
  python_calc        arithmetic via a restricted AST evaluator (no names, attributes, subscripts, comprehensions)
  search_paper       keyword search over the paper text passed in
  get_reference_entry / resolve_entry   look up entries in the bibliography text passed in
  lookup             INSPIRE-HEP (by arXiv id / DOI / title+author), falling back to Crossref + arXiv, for ONE
                     bibliography entry (only tool that uses the network;
                     https only, fixed host allowlist, redirects refused, response size capped)
"""
import ast, json, math, operator as op, re, sys, urllib.error, urllib.parse, urllib.request
import xml.etree.ElementTree as ET

# ------------------------------------------------------------------ calculator
_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow,
        ast.Mod: op.mod, ast.FloorDiv: op.floordiv, ast.USub: op.neg, ast.UAdd: op.pos}
_FN = {"sqrt": math.sqrt, "log": math.log, "ln": math.log, "log10": math.log10, "log2": math.log2,
       "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan, "abs": abs, "round": round,
       "min": min, "max": max}
_CONST = {"pi": math.pi, "e": math.e}


def safe_eval(expr):
    if len(expr) > 200:
        raise ValueError("expression too long")
    tree = ast.parse(expr.strip().replace("^", "**").replace("×", "*").replace("−", "-"), mode="eval")

    def ev(n, depth=0):
        if depth > 30:
            raise ValueError("expression too deeply nested")
        if isinstance(n, ast.Expression):
            return ev(n.body, depth + 1)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left, depth + 1), ev(n.right, depth + 1)
            if isinstance(n.op, ast.Pow) and (abs(b) > 400 or abs(a) > 1e100):
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand, depth + 1))
        if isinstance(n, ast.Name) and n.id in _CONST:
            return _CONST[n.id]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FN and not n.keywords \
                and len(n.args) <= 3:
            return _FN[n.func.id](*[ev(x, depth + 1) for x in n.args])
        raise ValueError("disallowed expression (arithmetic and math functions only)")
    return ev(tree)


def python_calc(expression):
    try:
        v = safe_eval(expression)
        return f"{expression} = {v:.6g}" if isinstance(v, float) else f"{expression} = {v}"
    except Exception as e:
        return f"error: {e}"


# ------------------------------------------------------------------ text search
def _words(s):
    return set(re.findall(r"[a-z0-9]{3,}", s.lower()))


def _rank(query, items, k):
    sets = [_words(i) for i in items]
    df = {}
    for w in sets:
        for t in w:
            df[t] = df.get(t, 0) + 1
    q = _words(query)
    sc = sorted(((sum(1.0 / df.get(t, 1) for t in q & w), i) for i, w in enumerate(sets)), reverse=True)
    return [items[i] for s, i in sc[:k] if s > 0]


def search_paper(query, text):
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 40]
    return "\n---\n".join(p[:700] for p in _rank(query, paras, 3)) or "no matching passage"


def _entries(refs):
    return [r.strip() for r in re.split(r"\n\s*\n|\n(?=\[\S+\])", refs) if r.strip()]


def resolve_entry(query, refs):
    ents = _entries(refs)
    key = query.strip().strip("[]{}").lower()
    hits = [e for e in ents if key and e.lower().startswith(f"[{key}]")]
    return (hits or _rank(query, ents, 1) or [""])[0][:600]


def get_reference_entry(query, refs):
    if not refs.strip():
        return "the paper's reference list is not available"
    return resolve_entry(query, refs) or "no matching reference entry"


# ------------------------------------------------------------------ network lookup (restricted)
ALLOWED_HOSTS = {"inspirehep.net", "api.crossref.org", "export.arxiv.org"}
MAX_BYTES = 300_000


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def _fetch(url):
    u = urllib.parse.urlparse(url)
    if u.scheme != "https" or u.hostname not in ALLOWED_HOSTS:
        raise ValueError("host not allowed")
    opener = urllib.request.build_opener(_NoRedirect)
    with opener.open(urllib.request.Request(url, headers={"User-Agent": "llm-experiments/0.1"}), timeout=15) as r:
        return r.read(MAX_BYTES)


_ARXIV_RE = re.compile(r"(?:arxiv:?\s*)?\b(\d{4}\.\d{4,5}|[a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?\b", re.I)
_DOI_RE = re.compile(r"\b(10\.\d{4,9}/[^\s,;\"<>]+)", re.I)


def _title_author(q):
    """Heuristic title + first-author surname from a free-text bibliography entry."""
    q = re.sub(r"^\s*\[[^\]]*\]\s*", "", q)
    segs = [x.strip() for x in re.split(r"(?<=[a-z\)])\.\s+|\n", q) if x.strip()]
    good = [x for x in segs if len(x.split()) >= 4 and sum(c.isdigit() for c in x) < 0.2 * len(x)]
    title = max(good, key=len) if good else ""
    first = segs[0] if segs else ""
    names = re.findall(r"\b[A-Z][a-z]{2,}\b", re.split(r"\bet\b|\band\b", first)[0])
    surname = (names[0] if "," in first else names[-1]) if names else ""
    return title, surname


def _inspire(query, titles):
    d = json.loads(_fetch("https://inspirehep.net/api/literature?size=3&sort=bestmatch&q=" + urllib.parse.quote(query)
                          + "&fields=titles,authors.full_name,arxiv_eprints,dois,publication_info,citation_count,earliest_date"))
    out = []
    for h in d["hits"]["hits"]:
        m = h["metadata"]
        au = [a["full_name"] for a in m.get("authors", [])]
        au = ", ".join(au[:4]) + (" et al" if len(au) > 4 else "")
        pub = "; ".join(" ".join(str(x) for x in (p.get("journal_title"), p.get("journal_volume"), p.get("artid") or p.get("page_start")) if x)
                        for p in m.get("publication_info", []) if p.get("journal_title"))
        ax = ",".join("arXiv:" + e["value"] for e in m.get("arxiv_eprints", []))
        doi = ",".join("doi:" + x["value"] for x in m.get("dois", [])[:1])
        out.append(f"[INSPIRE] {m['titles'][0]['title']} | {au} | {m.get('earliest_date','')} | {pub} | {ax} {doi} | cited {m.get('citation_count', '?')}x")
        titles.append(m["titles"][0]["title"])
    return out


def _clean_entry(e):
    e = re.sub(r"\\(newblock|em|emph|textit|textbf|bibinfo\{[^}]*\})\b", " ", e)
    e = e.replace("~", " ")
    e = re.sub(r"\\[A-Za-z]+\*?", " ", e)                 # any other control word
    return re.sub(r"[{}]", "", e)


def _sim(a, b):
    """Word-overlap (Jaccard) between two titles, 0..1."""
    wa, wb = _words(a), _words(b)
    return len(wa & wb) / len(wa | wb) if wa and wb else 0.0


def lookup(entry):
    q = " ".join(_clean_entry(entry).split())[:400]
    if not q:
        return "no reference entry to look up"
    out, how, title = [], "", ""
    try:
        m_ax, m_doi = _ARXIV_RE.search(q), _DOI_RE.search(q)
        queries = ([("arXiv id", "arxiv:" + m_ax.group(1))] if m_ax else []) + \
                  ([("DOI", "doi:" + m_doi.group(1).rstrip(".)"))] if m_doi else [])
        title, surname = _title_author(q)
        titles = []
        if title:
            queries.append(("title+author", f't "{title[:200].replace(chr(34), "")}"' + (f" and a {surname}" if surname else "")))
            queries.append(("title", "t " + title[:200].replace('"', "")))
        for how, qq in queries:
            titles = []
            out = _inspire(qq, titles)
            if out:
                break
        if out:
            out.insert(0, f"(INSPIRE-HEP match by {how}; ranked by INSPIRE relevance; a single hit by id/DOI is definitive; check the title-sim score)")
    except Exception as e:
        out = [f"[INSPIRE] lookup failed: {type(e).__name__}"]
    if len(out) > 1:
        return _annotate(out, title)[:2500]
    # fallback: Crossref + arXiv (INSPIRE has no record or was unavailable)
    out = out[:1] + ["(no INSPIRE record found; falling back to Crossref/arXiv)"]
    try:
        d = json.loads(_fetch("https://api.crossref.org/works?rows=3&select=title,author,issued,DOI,abstract"
                              "&query.bibliographic=" + urllib.parse.quote(q[:300])))
        for it in d["message"]["items"]:
            au = ", ".join(a.get("family", "") for a in it.get("author", [])[:4])
            yr = (it.get("issued", {}).get("date-parts") or [[""]])[0][0]
            ab = re.sub(r"<[^>]+>", "", it.get("abstract", ""))[:250]
            out.append(f"[Crossref] {(it.get('title') or [''])[0]} | {au} | {yr} | doi:{it.get('DOI','')} | {ab}")
    except Exception as e:
        out.append(f"[Crossref] lookup failed: {type(e).__name__}")
    try:
        words = []
        for w in re.findall(r"[A-Za-z]{4,}", q):
            if w.lower() not in words and w.lower() not in ("phys", "rev", "lett", "journal", "et", "al"):
                words.append(w.lower())
        raw = _fetch("https://export.arxiv.org/api/query?max_results=3&search_query="
                     + "+AND+".join("all:" + urllib.parse.quote(w) for w in words[:10]))
        if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
            raise ValueError("unexpected XML")
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for e in ET.fromstring(raw).findall("a:entry", ns):
            g = lambda t: " ".join((e.findtext(f"a:{t}", "", ns) or "").split())
            au = ", ".join(a.findtext("a:name", "", ns).split()[-1] for a in e.findall("a:author", ns)[:4])
            out.append(f"[arXiv] {g('title')} | {au} | {g('published')[:4]} | {g('id')} | {g('summary')[:250]}")
    except Exception as e:
        out.append(f"[arXiv] lookup failed: {type(e).__name__}")
    return _annotate(out, title)[:2500]


def _annotate(lines, want_title):
    """Append a title-similarity score to each hit and a summary verdict line."""
    best, res = 0.0, []
    for l in lines:
        m = re.match(r"\[(INSPIRE|Crossref|arXiv)\] (.*?) \|", l)
        if m and want_title:
            sc = _sim(want_title, m.group(2))
            best = max(best, sc)
            l += f" | title-sim {sc:.2f}"
        res.append(l)
    if want_title:
        verdict = ("strong title match" if best >= 0.7 else "weak/partial match" if best >= 0.4
                   else "NO CLOSE MATCH -- the cited work may not exist as described")
        res.insert(0, f"SUMMARY: best title similarity {best:.2f} ({verdict}) for \"{want_title[:100]}\"")
    return "\n".join(res)


# ------------------------------------------------------------------ worker entry point
def dispatch(req):
    t, arg = req.get("tool"), str(req.get("arg", ""))
    if t == "python_calc":
        return python_calc(arg)
    if t == "search_paper":
        return search_paper(arg, req.get("text", ""))
    if t == "get_reference_entry":
        return get_reference_entry(arg, req.get("refs", ""))
    if t == "resolve_entry":
        return resolve_entry(arg, req.get("refs", ""))
    if t == "lookup":
        return lookup(req.get("entry", ""))
    raise ValueError(f"unknown tool {t!r}")


if __name__ == "__main__":
    try:
        res = {"ok": True, "result": dispatch(json.loads(sys.stdin.read()))}
    except Exception as e:
        res = {"ok": False, "result": f"{type(e).__name__}: {e}"}
    sys.stdout.write(json.dumps(res))
