#!/usr/bin/env python3
"""Plan a multi-referee review as an HTCondor DAG.

  plan_review.py DOC [--name N] [--n 3] [--vram 23000] [--ctx-cap 32768] [--submit]

Reads the document (pdf / tex / arXiv tarball / txt), estimates its size, and picks a panel of
referee models from config/models.json that (a) fit in the GPU memory budget at the context
size needed, (b) come from different model families where possible (independent opinions), and
(c) prefer single-pass reading over chunking. Each referee gets a different review focus.
A final 'editor' node (DAGMan FINAL node, so it runs even if a referee fails) merges the reports.
Output: runs/<name>/{doc.txt,plan.json,referee_*.sub,editor.sub,review.dag}
"""
import argparse, json, math, os, shutil, subprocess, sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "client"))
from doc import load_document_ex, est_tokens  # noqa: E402

OUT_RESERVE = 3000          # tokens reserved for each referee's output
OVERHEAD_MB = 1500          # CUDA context + compute buffers (rough)
FINDINGS_RESERVE = 3500   # tokens of claim findings added to each referee prompt
EDITOR_REVIEW_EST = 2000  # typical tokens per referee review fed to the editor


PERSONAS = {
    "neutral": ("neutral", "Stance: neutral and even-handed. Weigh strengths and weaknesses equally."),
    "hostile": ("hostile", "Stance: a blunt, sceptical, antagonistic referee (the proverbial 'Reviewer 2'). Look hard for "
                "flaws, overclaiming and gaps, and do not soften your criticism. But you are honest: every criticism "
                "must quote or point to the specific text, and you must not invent flaws. Rate conservatively."),
    "supportive": ("supportive", "Stance: a constructive, sympathetic referee who looks for what is valuable and how "
                   "weaknesses could be fixed. You are still honest: do not overlook real problems."),
    "pedantic": ("pedantic", "Stance: a meticulous pedant. Check definitions, notation, units, numbers, references and "
                 "internal consistency line by line, and list every discrepancy you find with a quote."),
}
DEFAULT_PERSONAS = ["neutral", "hostile", "supportive", "pedantic"]


def vram_need(m, ctx):
    return (m["weights_mb"] + m["kv_mb_per_1k"] * ctx / 1000 + OVERHEAD_MB) * 1.1


def fit(m, tokens, budget, ctx_cap):
    """Return (mode, ctx) for model m on a document of `tokens`, or None if it cannot run."""
    need = tokens + OUT_RESERVE + 800
    single = math.ceil(need / 4096) * 4096
    limit = min(m["max_ctx"], ctx_cap)
    if single <= limit and vram_need(m, single) <= budget:
        return "single", single
    ctx = min(limit, max(single - 4096, 8192)) // 4096 * 4096
    while ctx >= 8192:
        if vram_need(m, ctx) <= budget:
            return "chunked", ctx
        ctx -= 4096
    return None


def score(m, mode):
    return m["quality"] - (1.5 if mode == "chunked" else 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("doc"); ap.add_argument("--name")
    ap.add_argument("--n", type=int, default=3, help="number of distinct models in the panel")
    ap.add_argument("--personas", default="neutral,hostile,supportive",
                    help="comma list from: " + ", ".join(PERSONAS) + " -- each is run on every model")
    ap.add_argument("--vram", type=int, default=23000, help="max GPU memory per job in MB (24 GB cards: 23000; 5090: 31000)")
    ap.add_argument("--ctx-cap", type=int, default=32768)
    ap.add_argument("--split", action="store_true", help="one DAG node per (model, persona)")
    ap.add_argument("--no-claims", action="store_true", help="skip claim extraction/verification")
    ap.add_argument("--max-claims", type=int, default=15)
    ap.add_argument("--catalog", default=os.path.join(ROOT, "config", "models.json"))
    ap.add_argument("--prompt", default=os.path.join(ROOT, "prompts", "review.txt"))
    ap.add_argument("--raw", action="store_true", help="don't clean LaTeX")
    ap.add_argument("--submit", action="store_true", help="run condor_submit_dag afterwards")
    a = ap.parse_args()

    name = a.name or os.path.splitext(os.path.basename(a.doc.rstrip("/")))[0]
    run = os.path.join(ROOT, "runs", name)
    for d in ("reviews", "logs"):
        os.makedirs(os.path.join(run, d), exist_ok=True)
    pkeys = [k.strip() for k in a.personas.split(",") if k.strip()]
    bad = [k for k in pkeys if k not in PERSONAS]
    if bad:
        sys.exit(f"unknown persona(s) {bad}; choose from {list(PERSONAS)}")

    text, refs = load_document_ex(a.doc, raw=a.raw)
    if len(text.strip()) < 200:
        sys.exit("almost no text extracted from the document")
    open(os.path.join(run, "doc.txt"), "w").write(text)
    open(os.path.join(run, "refs.txt"), "w").write(refs)
    tokens = est_tokens(text)
    prompt_tok = est_tokens(open(a.prompt).read()) + 250       # + REF rubric/persona text

    models = json.load(open(a.catalog))["models"]
    claims = None
    if not a.no_claims:
        tc = [(m, *fit(m, 12000, a.vram, 16384)) for m in models if m.get("tools") and fit(m, 12000, a.vram, 16384)]
        if tc:
            m, mode, ctx = max(tc, key=lambda c: c[0]["quality"])
            claims = {"model": m["name"], "ctx": 16384, "weights_mb": m["weights_mb"],
                      "vram_req_mb": min(a.vram, math.ceil(vram_need(m, 16384) / 1000) * 1000)}
        else:
            print("WARNING: no tool-capable model fits the budget; skipping claim checking")

    ref_tokens = tokens + prompt_tok + (FINDINGS_RESERVE if claims else 0)
    cands = []
    for m in models:
        f = fit(m, ref_tokens, a.vram, a.ctx_cap)
        if f:
            cands.append((m, f[0], f[1]))
    if not cands:
        sys.exit(f"no catalogue model fits in {a.vram} MB; raise --vram or edit {a.catalog}")
    cands.sort(key=lambda c: -score(c[0], c[1]))
    panel, fams = [], set()
    for c in cands:                                   # best model of each family first
        if c[0]["family"] not in fams and len(panel) < a.n:
            panel.append(c); fams.add(c[0]["family"])
    for c in cands:
        if c not in panel and len(panel) < a.n:
            panel.append(c)

    P = len(pkeys)
    referees, nodes = [], {}
    for mi, (m, mode, ctx) in enumerate(panel, 1):
        for pi, pk in enumerate(pkeys, 1):
            rid = (mi - 1) * P + pi
            pshort, ptext = PERSONAS[pk]
            node = f"R{mi}_{pi}" if a.split else f"R{mi}"
            referees.append({"id": rid, "node": node, "model": m["name"], "family": m["family"], "mode": mode,
                             "ctx": ctx, "weights_mb": m["weights_mb"],
                             "vram_req_mb": min(a.vram, math.ceil(vram_need(m, ctx) / 1000) * 1000),
                             "persona": pshort, "persona_text": ptext, "focus_short": "general"})
            nodes.setdefault(node, []).append(referees[-1])

    ed_tokens = len(referees) * EDITOR_REVIEW_EST + (3000 if claims else 0) + 2500
    ed_c = [(m, *fit(m, ed_tokens, a.vram, a.ctx_cap)) for m in models if fit(m, ed_tokens, a.vram, a.ctx_cap)]
    ed_c = [c for c in ed_c if c[1] == "single"] or ed_c
    m, mode, ctx = max(ed_c, key=lambda c: score(c[0], c[1]))
    editor = {"model": m["name"], "ctx": ctx, "mode": mode, "weights_mb": m["weights_mb"],
              "vram_req_mb": min(a.vram, math.ceil(vram_need(m, ctx) / 1000) * 1000)}

    plan = {"name": name, "doc_tokens": tokens, "vram_budget_mb": a.vram, "personas": pkeys,
            "claims": claims, "referees": referees, "editor": editor}
    json.dump(plan, open(os.path.join(run, "plan.json"), "w"), indent=2)

    def mem(w):
        return "32GB" if w > 12000 else "16GB"

    def gpu(w, v):
        return f"request_memory          = {mem(w)}\nrequire_gpus            = (GlobalMemoryMb >= {v})\n"

    common = f"""universe                = vanilla
executable              = {ROOT}/bin/job.sh
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
request_gpus            = 1
request_cpus            = 4
request_disk            = 30GB
log                     = logs/nodes.log
"""
    client = ", ".join(f"{ROOT}/client/{f}" for f in ("doc.py", "oclient.py"))
    prompt_name = os.path.basename(a.prompt)
    dag = []
    if claims:
        open(os.path.join(run, "claims.sub"), "w").write(common + f"""arguments               = {claims['model']} python3 claims.py doc.txt --refs refs.txt --num-ctx {claims['ctx']} --max-claims {a.max_claims}
transfer_input_files    = {ROOT}/client/claims.py, {client}, doc.txt, refs.txt
transfer_output_remaps  = "ollama-server.log = logs/ollama-server_claims.log"
""" + gpu(claims["weights_mb"], claims["vram_req_mb"]) + """output                  = logs/claims.out
error                   = logs/claims.err
queue
""")
        dag += ["JOB C claims.sub", f"SCRIPT POST C {ROOT}/bin/dag_post_claims.sh"]
    for node, rs in nodes.items():
        spec = {"doc": "doc.txt", "prompt": prompt_name, "num_ctx": rs[0]["ctx"], "max_out": 2500,
                "evidence": "claim_findings.md" if claims else None,
                "jobs": [{"focus_file": f"focus_{r['id']}.txt", "out": f"review_{r['id']}.md"} for r in rs]}
        json.dump(spec, open(os.path.join(run, f"spec_{node}.json"), "w"), indent=2)
        for r in rs:
            open(os.path.join(run, f"focus_{r['id']}.txt"), "w").write(r["persona_text"] + "\n")
        remaps = "; ".join([f"review_{r['id']}.md = reviews/review_{r['id']}.md" for r in rs] +
                           [f"ollama-server.log = logs/ollama-server_{node}.log"])
        inputs = [f"{ROOT}/client/run_panel.py", f"{ROOT}/client/review_pdf.py", client, a.prompt, "doc.txt",
                  f"spec_{node}.json"] + [f"focus_{r['id']}.txt" for r in rs] + (["claim_findings.md"] if claims else [])
        open(os.path.join(run, f"referee_{node}.sub"), "w").write(common + f"""arguments               = {rs[0]['model']} python3 run_panel.py spec_{node}.json
transfer_input_files    = {', '.join(inputs)}
transfer_output_remaps  = "{remaps}"
""" + gpu(rs[0]["weights_mb"], rs[0]["vram_req_mb"]) + f"""output                  = logs/referee_{node}.out
error                   = logs/referee_{node}.err
queue
""")
        dag += [f"JOB {node} referee_{node}.sub", f"RETRY {node} 1"]
    if claims:
        dag.append(f"PARENT C CHILD {' '.join(nodes)}")
    open(os.path.join(run, "editor.sub"), "w").write(common + f"""arguments               = {editor['model']} python3 editor.py --prompt editor.txt --num-ctx {editor['ctx']}
transfer_input_files    = {ROOT}/client/editor.py, {client}, {ROOT}/prompts/editor.txt, plan.json, reviews{', claim_findings.md' if claims else ''}
transfer_output_remaps  = "ollama-server.log = logs/ollama-server_editor.log"
""" + gpu(editor["weights_mb"], editor["vram_req_mb"]) + """output                  = logs/editor.out
error                   = logs/editor.err
queue
""")
    dag.append("FINAL E editor.sub")
    open(os.path.join(run, "review.dag"), "w").write("\n".join(dag) + "\n")

    print(f"\nDocument: ~{tokens} tokens   GPU budget: {a.vram} MB   run dir: {run}\n")
    if claims:
        print(f"claims    {claims['model']}  (tool use; ctx {claims['ctx']}, <= {a.max_claims} claims, then referees wait for it)")
    print(f"{'node':<7}{'model':<18}{'family':<9}{'ctx':>6}  {'mode':<8}{'VRAM req':>9}  personas")
    for node, rs in nodes.items():
        r = rs[0]
        print(f"{node:<7}{r['model']:<18}{r['family']:<9}{r['ctx']:>6}  {r['mode']:<8}{r['vram_req_mb']:>8}M  {', '.join(x['persona'] for x in rs)}")
    print(f"{'editor':<7}{editor['model']:<18}{'':<9}{editor['ctx']:>6}  {editor['mode']:<8}{editor['vram_req_mb']:>8}M")
    print(f"\n{len(referees)} reviews = {len(panel)} models x {P} personas in {len(nodes)} job(s).")
    if any(r["mode"] == "chunked" for r in referees):
        print("Note: 'chunked' referees read the paper in pieces and merge -- expect weaker reviews.")
    if len({r["family"] for r in referees}) < len(panel):
        print("Note: fewer distinct model families fit than models requested.")
    print(f"\nSubmit with:\n  cd {run} && condor_submit_dag -f review.dag\n")
    if a.submit:
        if not shutil.which("condor_submit_dag"):
            sys.exit("condor_submit_dag not found")
        subprocess.run(["condor_submit_dag", "-f", "review.dag"], cwd=run, check=True)


if __name__ == "__main__":
    main()
