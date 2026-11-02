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
from doc import load_document, est_tokens  # noqa: E402

OUT_RESERVE = 3000          # tokens reserved for each referee's output
OVERHEAD_MB = 1500          # CUDA context + compute buffers (rough)
FOCI = [
    ("statistics & methodology",
     "Focus especially on statistical and methodological soundness: modelling assumptions, priors, likelihoods, "
     "uncertainty quantification, model comparison, and reproducibility."),
    ("novelty & conclusions",
     "Focus especially on novelty, significance, and whether the conclusions actually follow from the results shown."),
    ("technical correctness",
     "Focus especially on technical correctness: derivations, equations, units, numerical results, internal "
     "consistency, and clarity of presentation."),
]


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
    ap.add_argument("doc"); ap.add_argument("--name"); ap.add_argument("--n", type=int, default=3)
    ap.add_argument("--vram", type=int, default=23000, help="max GPU memory per job in MB (24 GB cards: 23000; 5090: 31000)")
    ap.add_argument("--ctx-cap", type=int, default=32768)
    ap.add_argument("--personas", default=",".join(DEFAULT_PERSONAS),
                    help="comma list from: " + ", ".join(PERSONAS) + " (cycled over referees)")
    ap.add_argument("--catalog", default=os.path.join(ROOT, "config", "models.json"))
    ap.add_argument("--prompt", default=os.path.join(ROOT, "prompts", "review.txt"))
    ap.add_argument("--raw", action="store_true", help="don't clean LaTeX")
    ap.add_argument("--submit", action="store_true", help="run condor_submit_dag afterwards")
    a = ap.parse_args()

    name = a.name or os.path.splitext(os.path.basename(a.doc.rstrip("/")))[0]
    run = os.path.join(ROOT, "runs", name)
    os.makedirs(os.path.join(run, "reviews"), exist_ok=True)
    os.makedirs(os.path.join(run, "logs"), exist_ok=True)

    text = load_document(a.doc, raw=a.raw)
    if len(text.strip()) < 200:
        sys.exit("almost no text extracted from the document")
    open(os.path.join(run, "doc.txt"), "w").write(text)
    tokens = est_tokens(text)
    prompt_tok = est_tokens(open(a.prompt).read()) + 150

    models = json.load(open(a.catalog))["models"]
    cands = []
    for m in models:
        f = fit(m, tokens + prompt_tok, a.vram, a.ctx_cap)
        if f:
            cands.append((m, f[0], f[1]))
    if not cands:
        sys.exit(f"no catalogue model fits in {a.vram} MB; raise --vram or edit {a.catalog}")
    cands.sort(key=lambda c: -score(c[0], c[1]))

    panel, fams = [], set()
    for c in cands:                                   # first: best model of each family
        if c[0]["family"] not in fams and len(panel) < a.n:
            panel.append(c); fams.add(c[0]["family"])
    for c in cands:                                   # then fill with the best remaining
        if c not in panel and len(panel) < a.n:
            panel.append(c)

    referees = []
    for i, (m, mode, ctx) in enumerate(panel, 1):
        short, focus = FOCI[(i - 1) % len(FOCI)]
        pkeys = a.personas.split(",")
        pshort, ptext = PERSONAS[pkeys[(i - 1) % len(pkeys)].strip()]
        referees.append({"id": i, "model": m["name"], "family": m["family"], "mode": mode, "ctx": ctx,
                         "vram_req_mb": min(a.vram, math.ceil(vram_need(m, ctx) / 1000) * 1000),
                         "weights_mb": m["weights_mb"], "focus_short": short, "focus": focus,
                         "persona": pshort, "persona_text": ptext})

    # Editor: needs all reviews in one window; best model that can do that in a single pass.
    ed_tokens = len(referees) * OUT_RESERVE + 2000
    ed_c = [(m, *fit(m, ed_tokens, a.vram, a.ctx_cap)) for m in models if fit(m, ed_tokens, a.vram, a.ctx_cap)]
    ed_c = [c for c in ed_c if c[1] == "single"] or ed_c
    m, mode, ctx = max(ed_c, key=lambda c: score(c[0], c[1]))
    editor = {"model": m["name"], "ctx": ctx, "mode": mode, "weights_mb": m["weights_mb"],
              "vram_req_mb": min(a.vram, math.ceil(vram_need(m, ctx) / 1000) * 1000)}

    plan = {"name": name, "doc_tokens": tokens, "vram_budget_mb": a.vram, "referees": referees, "editor": editor}
    json.dump(plan, open(os.path.join(run, "plan.json"), "w"), indent=2)

    def mem(w):  # host RAM request
        return "32GB" if w > 12000 else "16GB"

    common = f"""universe                = vanilla
executable              = {ROOT}/bin/job.sh
should_transfer_files   = YES
when_to_transfer_output = ON_EXIT
request_gpus            = 1
request_cpus            = 4
request_disk            = 30GB
log                     = logs/nodes.log
"""
    py = "review_pdf.py doc.txt --prompt " + os.path.basename(a.prompt) + " --focus-file focus_{i}.txt --out review_{i}.md --num-ctx {ctx}"
    dag = []
    for r in referees:
        i = r["id"]
        open(os.path.join(run, f"focus_{i}.txt"), "w").write(r["persona_text"] + "\n\n" + r["focus"] + "\n")
        open(os.path.join(run, f"referee_{i}.sub"), "w").write(common + f"""arguments               = {r['model']} python3 {py.format(i=i, ctx=r['ctx'])}
transfer_input_files    = {ROOT}/client/review_pdf.py, {ROOT}/client/doc.py, {ROOT}/client/oclient.py, {a.prompt}, doc.txt, focus_{i}.txt
transfer_output_remaps  = "review_{i}.md = reviews/review_{i}.md; ollama-server.log = logs/ollama-server_{i}.log"
request_memory          = {mem(r['weights_mb'])}
require_gpus            = (GlobalMemoryMb >= {r['vram_req_mb']})
output                  = logs/referee_{i}.out
error                   = logs/referee_{i}.err
queue
""")
        dag += [f"JOB R{i} referee_{i}.sub", f"RETRY R{i} 1"]
    open(os.path.join(run, "editor.sub"), "w").write(common + f"""arguments               = {editor['model']} python3 editor.py --prompt editor.txt --num-ctx {editor['ctx']}
transfer_input_files    = {ROOT}/client/editor.py, {ROOT}/client/doc.py, {ROOT}/client/oclient.py, {ROOT}/prompts/editor.txt, plan.json, reviews
transfer_output_remaps  = "ollama-server.log = logs/ollama-server_editor.log"
request_memory          = {mem(editor['weights_mb'])}
require_gpus            = (GlobalMemoryMb >= {editor['vram_req_mb']})
output                  = logs/editor.out
error                   = logs/editor.err
queue
""")
    dag.append("FINAL E editor.sub")
    open(os.path.join(run, "review.dag"), "w").write("\n".join(dag) + "\n")

    print(f"\nDocument: ~{tokens} tokens   GPU budget: {a.vram} MB   run dir: {run}\n")
    print(f"{'role':<10}{'model':<18}{'family':<9}{'ctx':>6}  {'mode':<8}{'VRAM req':>9}  persona / focus")
    for r in referees:
        print(f"{'referee ' + str(r['id']):<10}{r['model']:<18}{r['family']:<9}{r['ctx']:>6}  {r['mode']:<8}{r['vram_req_mb']:>8}M  {r['persona']} / {r['focus_short']}")
    print(f"{'editor':<10}{editor['model']:<18}{'':<9}{editor['ctx']:>6}  {editor['mode']:<8}{editor['vram_req_mb']:>8}M")
    if any(r["mode"] == "chunked" for r in referees):
        print("\nNote: 'chunked' referees read the paper in pieces and merge -- expect weaker reviews.")
    if len({r["family"] for r in referees}) < len(referees):
        print("Note: fewer distinct model families fit than referees requested.")
    print(f"\nSubmit with:\n  cd {run} && condor_submit_dag -f review.dag\n")
    if a.submit:
        if not shutil.which("condor_submit_dag"):
            sys.exit("condor_submit_dag not found")
        subprocess.run(["condor_submit_dag", "-f", "review.dag"], cwd=run, check=True)


if __name__ == "__main__":
    main()
