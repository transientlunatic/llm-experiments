#!/usr/bin/env python3
"""Verification tools. Stdlib only. These run INSIDE a sandboxed worker process (see sandbox.py):

    python3 -I -S tools.py   <  {"tool": ..., "arg": ..., "text": ..., "refs": ..., "entry": ...}

Each tool receives only the data it needs and has no side effects:
  python_calc        arithmetic via a restricted AST evaluator (no names, attributes, subscripts, comprehensions)
  search_paper       keyword search over the paper text passed in
  get_reference_entry / resolve_entry   look up entries in the bibliography text passed in
  lookup             Crossref + arXiv for ONE bibliography entry (only tool that uses the network;
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
ALLOWED_HOSTS = {"api.crossref.org", "export.arxiv.org"}
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


def lookup(entry):
    q = " ".join(entry.split())[:300]
    if not q:
        return "no reference entry to look up"
    out = []
    try:
        d = json.loads(_fetch("https://api.crossref.org/works?rows=3&select=title,author,issued,DOI,abstract"
                              "&query.bibliographic=" + urllib.parse.quote(q)))
        for it in d["message"]["items"]:
            au = ", ".join(a.get("family", "") for a in it.get("author", [])[:4])
            yr = (it.get("issued", {}).get("date-parts") or [[""]])[0][0]
            ab = re.sub(r"<[^>]+>", "", it.get("abstract", ""))[:250]
            out.append(f"[Crossref] {(it.get('title') or [''])[0]} | {au} | {yr} | doi:{it.get('DOI','')} | {ab}")
    except Exception as e:
        out.append(f"[Crossref] lookup failed: {type(e).__name__}")
    try:
        words = []
        for w in re.findall(r"[A-Za-z]{4,}", q):          # significant words only, AND-ed: a well-formed arXiv query
            if w.lower() not in words and w.lower() not in ("phys", "rev", "lett", "journal", "et", "al"):
                words.append(w.lower())
        raw = _fetch("https://export.arxiv.org/api/query?max_results=3&search_query="
                     + "+AND+".join("all:" + urllib.parse.quote(w) for w in words[:10]))
        if b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:                    # refuse XML with DTDs/entities
            raise ValueError("unexpected XML")
        ns = {"a": "http://www.w3.org/2005/Atom"}
        for e in ET.fromstring(raw).findall("a:entry", ns):
            g = lambda t: " ".join((e.findtext(f"a:{t}", "", ns) or "").split())
            au = ", ".join(a.findtext("a:name", "", ns).split()[-1] for a in e.findall("a:author", ns)[:4])
            out.append(f"[arXiv] {g('title')} | {au} | {g('published')[:4]} | {g('id')} | {g('summary')[:250]}")
    except Exception as e:
        out.append(f"[arXiv] lookup failed: {type(e).__name__}")
    return "\n".join(out)[:2500]


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
