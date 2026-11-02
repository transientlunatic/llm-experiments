#!/usr/bin/env python3
"""Claim extraction + tool-assisted verification (runs inside a job.sh Ollama session).

  claims.py DOC [--refs refs.txt] [--num-ctx 16384] [--max-claims 15] [--max-steps 5]
            [--out-json claims.json] [--out-md claim_findings.md]

Fixed pipeline, not a free agent:  1. extract checkable claims (JSON, quotes must occur verbatim in the
paper)  2. verify each claim with a small, type-specific tool set via Ollama tool calling.
Tools (no arbitrary code execution):
  search_paper(query)         keyword search over the paper's paragraphs
  python_calc(expression)     arithmetic only: + - * / ** %, sqrt/log/log10/exp/sin/cos/tan/abs/round, pi, e
  get_reference_entry(query)  look up an entry in the paper's own reference list
  lookup_reference(query)     Crossref + arXiv search (needs outbound internet)
Needs a model with tool support (e.g. qwen2.5, llama3.1). Always exits 0 and writes outputs, so a
failure here never blocks the referees; the findings file then says verification was unavailable.
"""
import argparse, ast, json, math, operator as op, re, sys, traceback
import urllib.parse, urllib.request
import xml.etree.ElementTree as ET
from doc import load_document_ex, chunk, est_tokens
from oclient import chat_messages

# ------------------------------------------------------------------ tools
_OPS = {ast.Add: op.add, ast.Sub: op.sub, ast.Mult: op.mul, ast.Div: op.truediv, ast.Pow: op.pow,
        ast.Mod: op.mod, ast.FloorDiv: op.floordiv, ast.USub: op.neg, ast.UAdd: op.pos}
_FN = {"sqrt": math.sqrt, "log": math.log, "ln": math.log, "log10": math.log10, "log2": math.log2,
       "exp": math.exp, "sin": math.sin, "cos": math.cos, "tan": math.tan, "abs": abs, "round": round,
       "min": min, "max": max}
_CONST = {"pi": math.pi, "e": math.e}


def safe_eval(expr):
    tree = ast.parse(expr.strip().replace("^", "**").replace("×", "*").replace("−", "-"), mode="eval")

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)) and not isinstance(n.value, bool):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow) and (abs(b) > 400 or abs(a) > 1e100):
                raise ValueError("exponent too large")
            return _OPS[type(n.op)](a, b)
        if isinstance(n, ast.UnaryOp) and type(n.op) in _OPS:
            return _OPS[type(n.op)](ev(n.operand))
        if isinstance(n, ast.Name) and n.id in _CONST:
            return _CONST[n.id]
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FN and not n.keywords:
            return _FN[n.func.id](*[ev(x) for x in n.args])
        raise ValueError("disallowed expression (arithmetic and math functions only)")
    return ev(tree)


class Tools:
    def __init__(self, text, refs):
        self.paras = [p.strip() for p in re.split(r"\n\s*\n", text) if len(p.strip()) > 40]
        self.words = [set(re.findall(r"[a-z0-9]{3,}", p.lower())) for p in self.paras]
        df = {}
        for w in self.words:
            for t in w:
                df[t] = df.get(t, 0) + 1
        self.df = df
        self.refs = [r.strip() for r in re.split(r"\n\s*\n|\n(?=\[\S+\])", refs) if r.strip()] if refs else []

    def _rank(self, query, texts, wordsets, k):
        q = set(re.findall(r"[a-z0-9]{3,}", query.lower()))
        sc = [(sum(1.0 / self.df.get(t, 1) if self.df else 1.0 for t in q & w), i) for i, w in enumerate(wordsets)]
        return [i for s, i in sorted(sc, reverse=True)[:k] if s > 0]

    def search_paper(self, query):
        idx = self._rank(query, self.paras, self.words, 3)
        return "\n---\n".join(self.paras[i][:700] for i in idx) or "no matching passage"

    def python_calc(self, expression):
        try:
            v = safe_eval(expression)
            return f"{expression} = {v:.6g}" if isinstance(v, float) else f"{expression} = {v}"
        except Exception as e:
            return f"error: {e}"

    def get_reference_entry(self, query):
        if not self.refs:
            return "the paper's reference list is not available"
        ws = [set(re.findall(r"[a-z0-9]{3,}", r.lower())) for r in self.refs]
        key = query.strip().strip("[]{}").lower()
        hits = [r for r in self.refs if r.lower().startswith(f"[{key}]")] if key else []
        if not hits:
            hits = [self.refs[i] for i in self._rank(query, self.refs, ws, 2)]
        return "\n---\n".join(h[:500] for h in hits) or "no matching reference entry"

    def lookup_reference(self, query):
        out = []
        try:
            u = "https://api.crossref.org/works?rows=3&select=title,author,issued,DOI,abstract&query.bibliographic=" \
                + urllib.parse.quote(query[:300])
            with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "llm-experiments/0.1"}), timeout=15) as r:
                for it in json.load(r)["message"]["items"]:
                    au = ", ".join(a.get("family", "") for a in it.get("author", [])[:4])
                    yr = (it.get("issued", {}).get("date-parts") or [[""]])[0][0]
                    ab = re.sub(r"<[^>]+>", "", it.get("abstract", ""))[:250]
                    out.append(f"[Crossref] {(it.get('title') or [''])[0]} | {au} | {yr} | doi:{it.get('DOI','')} | {ab}")
        except Exception as e:
            out.append(f"[Crossref] lookup failed: {e}")
        try:
            u = "https://export.arxiv.org/api/query?max_results=3&search_query=all:" + urllib.parse.quote(query[:200])
            with urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "llm-experiments/0.1"}), timeout=15) as r:
                ns = {"a": "http://www.w3.org/2005/Atom"}
                for e in ET.fromstring(r.read()).findall("a:entry", ns):
                    g = lambda t: " ".join((e.findtext(f"a:{t}", "", ns) or "").split())
                    au = ", ".join(a.findtext("a:name", "", ns).split()[-1] for a in e.findall("a:author", ns)[:4])
                    out.append(f"[arXiv] {g('title')} | {au} | {g('published')[:4]} | {g('id')} | {g('summary')[:250]}")
        except Exception as e:
            out.append(f"[arXiv] lookup failed: {e}")
        return "\n".join(out)[:2500]


def _spec(name, desc, arg, adesc):
    return {"type": "function", "function": {"name": name, "description": desc, "parameters": {
        "type": "object", "properties": {arg: {"type": "string", "description": adesc}}, "required": [arg]}}}


SPECS = {
    "search_paper": (_spec("search_paper", "Search the paper's text; returns the most relevant passages.", "query", "keywords"), "query"),
    "python_calc": (_spec("python_calc", "Evaluate an arithmetic expression exactly (no variables; use numbers from the paper).", "expression", "e.g. sqrt(2*9.8)/3"), "expression"),
    "get_reference_entry": (_spec("get_reference_entry", "Find an entry in the paper's own reference list by cite key, number or words.", "query", "cite key or words"), "query"),
    "lookup_reference": (_spec("lookup_reference", "Search Crossref and arXiv for a paper by title/authors/year.", "query", "title and authors"), "query"),
}
TYPE_TOOLS = {"numerical": ["search_paper", "python_calc"],
              "citation": ["get_reference_entry", "lookup_reference", "search_paper"]}

# ------------------------------------------------------------------ LLM steps
def parse_json(s):
    s = s.strip()
    for cand in (s, *re.findall(r"\{.*\}", s, re.S)[:1]):
        try:
            return json.loads(cand)
        except Exception:
            pass
    return None


def norm(t):
    return " ".join(t.split()).lower()


def extract_claims(text, a):
    budget = a.num_ctx - 2500
    raw = []
    for n, part in enumerate(chunk(text, budget), 1):
        prompt = (
            "Extract the most important CHECKABLE claims from this part of a scientific paper: numerical results "
            "(values, errors, ratios, significances), statements that rely on a citation, and methodological assertions "
            "(e.g. 'the sampler converged').\nReturn JSON: {\"claims\": [{\"type\": \"numerical|citation|methodological\", "
            "\"quote\": \"<exact sentence copied verbatim from the text>\", \"importance\": <1-5>}]} with at most 8 claims. "
            "Quotes MUST be copied exactly.\n\n--- PAPER PART ---\n" + part)
        m = chat_messages([{"role": "user", "content": prompt}], fmt="json", num_ctx=a.num_ctx, num_predict=2000)
        j = parse_json(m["content"]) or {}
        raw += j.get("claims", []) if isinstance(j, dict) else []
    haystack, seen, claims = norm(text), set(), []
    for c in raw:
        if not isinstance(c, dict) or not isinstance(c.get("quote"), str):
            continue
        q = norm(c["quote"])
        if len(q) < 20 or q in seen or q not in haystack:        # drop paraphrased/hallucinated quotes
            continue
        seen.add(q)
        t = c.get("type") if c.get("type") in ("numerical", "citation", "methodological") else "other"
        imp = c.get("importance") if isinstance(c.get("importance"), (int, float)) else 3
        claims.append({"type": t, "quote": " ".join(c["quote"].split()), "importance": imp})
    claims.sort(key=lambda c: -c["importance"])
    print(f"[claims] {len(raw)} raw -> {len(claims)} verbatim claims", file=sys.stderr)
    return claims[:a.max_claims]


def verify(claim, tools, a):
    names = TYPE_TOOLS.get(claim["type"], ["search_paper"])
    specs = [SPECS[n][0] for n in names]
    msgs = [{"role": "user", "content": (
        f"Verify this {claim['type']} claim from a paper:\n\"{claim['quote']}\"\n\n"
        "Use the tools to check it against the paper itself (and, for citations, the literature). Recompute numbers "
        "with python_calc rather than in your head. When finished, reply with ONLY JSON: "
        "{\"verdict\": \"supported|partly_supported|contradicted|unverifiable\", \"evidence\": \"1-3 sentences quoting numbers/text\", "
        "\"confidence\": <0-1>}. Say 'unverifiable' if the tools give no basis; do not guess.")}]
    used = []
    for _ in range(a.max_steps):
        m = chat_messages(msgs, tools=specs, num_ctx=a.num_ctx, num_predict=1200)
        calls = m.get("tool_calls") or []
        msgs.append({"role": "assistant", "content": m.get("content", ""), "tool_calls": calls})
        if not calls:
            break
        for tc in calls:
            fn = tc["function"]["name"]
            args = tc["function"].get("arguments") or {}
            if isinstance(args, str):
                args = parse_json(args) or {}
            if fn in names:
                try:
                    res = getattr(tools, fn)(str(args.get(SPECS[fn][1], "")))
                except Exception as e:
                    res = f"tool error: {e}"
                used.append(fn)
            else:
                res = f"unknown or unavailable tool {fn}"
            msgs.append({"role": "tool", "tool_name": fn, "content": res[:3000]})
    j = parse_json(msgs[-1].get("content", "")) if msgs[-1]["role"] == "assistant" else None
    if not (isinstance(j, dict) and "verdict" in j):               # ask once more, JSON-constrained
        msgs.append({"role": "user", "content": "Give your final answer now as JSON with keys verdict, evidence, confidence."})
        j = parse_json(chat_messages(msgs, fmt="json", num_ctx=a.num_ctx, num_predict=500)["content"])
    if not (isinstance(j, dict) and j.get("verdict") in ("supported", "partly_supported", "contradicted", "unverifiable")):
        j = {"verdict": "unverifiable", "evidence": "no parsable verdict from the model", "confidence": 0}
    return {**claim, "verdict": j["verdict"], "evidence": str(j.get("evidence", ""))[:400],
            "confidence": j.get("confidence", 0), "tools_used": sorted(set(used))}


ORDER = {"contradicted": 0, "partly_supported": 1, "unverifiable": 2, "supported": 3}


def findings_md(results, model, note=""):
    lines = ["# Claim verification (automated; may contain errors)", "",
             f"{len(results)} claims checked by `{model}` with tools: search_paper, python_calc, "
             "get_reference_entry, lookup_reference. Verdicts are evidence, not certainty. " + note, ""]
    for i, r in enumerate(sorted(results, key=lambda r: ORDER.get(r["verdict"], 9)), 1):
        lines += [f"## C{i} [{r['type']}] {r['verdict'].upper()} (confidence {r['confidence']})",
                  f"Claim: \"{r['quote']}\"", f"Evidence: {r['evidence']}",
                  f"Tools used: {', '.join(r['tools_used']) or 'none'}", ""]
    return "\n".join(lines)


if __name__ == "__main__":
    import os
    ap = argparse.ArgumentParser()
    ap.add_argument("doc"); ap.add_argument("--refs"); ap.add_argument("--num-ctx", type=int, default=16384)
    ap.add_argument("--max-claims", type=int, default=15); ap.add_argument("--max-steps", type=int, default=5)
    ap.add_argument("--out-json", default="claims.json"); ap.add_argument("--out-md", default="claim_findings.md")
    a = ap.parse_args()
    model = os.environ.get("OLLAMA_MODEL", "?")
    try:
        text = open(a.doc, errors="replace").read()
        refs = open(a.refs, errors="replace").read() if a.refs and os.path.exists(a.refs) else ""
        tools = Tools(text, refs)
        results = []
        for n, c in enumerate(extract_claims(text, a), 1):
            print(f"[claims] verifying {n}: [{c['type']}] {c['quote'][:80]}", file=sys.stderr)
            try:
                results.append(verify(c, tools, a))
            except Exception as e:
                print(f"[claims] claim {n} failed: {e}", file=sys.stderr)
                results.append({**c, "verdict": "unverifiable", "confidence": 0, "tools_used": [],
                                "evidence": f"verification step errored: {str(e)[:150]}"})
        json.dump(results, open(a.out_json, "w"), indent=2)
        open(a.out_md, "w").write(findings_md(results, model) if results else
                                  "# Claim verification\n\nNo claims could be extracted/verified automatically.\n")
    except Exception:
        traceback.print_exc()
        open(a.out_md, "w").write("# Claim verification\n\nUnavailable: the verification step failed.\n")
        json.dump([], open(a.out_json, "w"))
    print(f"[claims] wrote {a.out_md}", file=sys.stderr)
