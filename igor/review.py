"""Build a multi-referee review as an HTCondor DAG.

Reads the document (pdf / tex / arXiv tarball / txt), estimates its size, and picks a panel of N models from the
catalogue that fit the GPU budget at the context needed (different families where possible, single pass preferred).
EVERY persona runs on EVERY model (a models x personas matrix) so model effects can be told apart from stance effects:

    CLAIMS (extract + tool-verify claims)  ->  one referee node per model (all personas, one model load)
                                           ->  FINAL editor (report.md)

Output (in <out_dir>/<name>/): doc.txt, refs.txt, plan.json, *.sub, spec_*.json, focus_*.txt, review.dag
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from . import catalog as cat
from . import condor
from .catalog import DATA

WORKER = Path(__file__).parent / "worker"
sys.path.insert(0, str(WORKER))            # worker scripts are flat, stdlib-only modules; reuse their doc loader
from doc import est_tokens, load_document_ex  # noqa: E402

FINDINGS_RESERVE = 3500   # tokens of claim findings added to each referee prompt
EDITOR_REVIEW_EST = 2000  # typical tokens per referee review fed to the editor

PERSONAS = {
    "neutral": "Stance: neutral and even-handed. Weigh strengths and weaknesses equally.",
    "hostile": ("Stance: a blunt, sceptical, antagonistic referee (the proverbial 'Reviewer 2'). Look hard for flaws, "
                "overclaiming and gaps, and do not soften your criticism. But you are honest: every criticism must "
                "quote or point to the specific text, and you must not invent flaws. Rate conservatively."),
    "supportive": ("Stance: a constructive, sympathetic referee who looks for what is valuable and how weaknesses could "
                   "be fixed. You are still honest: do not overlook real problems."),
    "pedantic": ("Stance: a meticulous pedant. Check definitions, notation, units, numbers, references and internal "
                 "consistency line by line, and list every discrepancy you find with a quote."),
}
DEFAULT_PERSONAS = ("neutral", "hostile", "supportive")


class PlanError(Exception):
    pass


def _client_files():
    return [str(WORKER / f) for f in ("doc.py", "oclient.py")]


def build(doc, *, name=None, out_dir="runs", n=3, personas=DEFAULT_PERSONAS, vram=23000, ctx_cap=32768,
          split=False, claims=True, max_claims=15, sandbox="strict", network=True, catalog=None, prompt=None,
          raw=False) -> tuple[Path, dict]:
    """Create the run directory; returns (run_dir, plan)."""
    bad = [p for p in personas if p not in PERSONAS]
    if bad:
        raise PlanError(f"unknown persona(s) {bad}; choose from {sorted(PERSONAS)}")
    if not personas:
        raise PlanError("need at least one persona")
    prompt = Path(prompt) if prompt else DATA / "prompts" / "review.txt"
    name = name or Path(str(doc).rstrip("/")).name.split(".")[0] or "review"
    run = Path(out_dir) / name
    (run / "reviews").mkdir(parents=True, exist_ok=True)
    (run / "logs").mkdir(exist_ok=True)

    text, refs = load_document_ex(str(doc), raw=raw)
    if len(text.strip()) < 200:
        raise PlanError("almost no text extracted from the document (scanned PDF?)")
    (run / "doc.txt").write_text(text)
    (run / "refs.txt").write_text(refs)
    tokens = est_tokens(text)
    prompt_tok = est_tokens(prompt.read_text()) + 250        # + REF rubric / persona text

    models = cat.load_catalog(catalog)
    claims_cfg, warnings = None, []
    if claims:
        tc = [(m, *cat.fit(m, 12000, vram, 16384)) for m in models if m.get("tools") and cat.fit(m, 12000, vram, 16384)]
        if tc:
            m = max(tc, key=lambda c: c[0]["quality"])[0]
            claims_cfg = {"model": m["name"], "ctx": 16384, "weights_mb": m["weights_mb"],
                          "vram_req_mb": cat.vram_request(m, 16384, vram)}
        else:
            warnings.append("no tool-capable model fits the budget; claim checking skipped")

    ref_tokens = tokens + prompt_tok + (FINDINGS_RESERVE if claims_cfg else 0)
    cands = [(m, *f) for m in models if (f := cat.fit(m, ref_tokens, vram, ctx_cap))]
    if not cands:
        raise PlanError(f"no catalogue model fits in {vram} MB; raise --vram or edit the catalogue")
    cands.sort(key=lambda c: -cat.score(c[0], c[1]))
    panel, fams = [], set()
    for c in cands:                                           # best model of each family first
        if c[0]["family"] not in fams and len(panel) < n:
            panel.append(c); fams.add(c[0]["family"])
    for c in cands:
        if c not in panel and len(panel) < n:
            panel.append(c)

    referees, nodes = [], {}
    for mi, (m, mode, ctx) in enumerate(panel, 1):
        for pi, pk in enumerate(personas, 1):
            rid = (mi - 1) * len(personas) + pi
            node = f"R{mi}_{pi}" if split else f"R{mi}"
            referees.append({"id": rid, "node": node, "model": m["name"], "family": m["family"], "mode": mode,
                             "ctx": ctx, "weights_mb": m["weights_mb"], "vram_req_mb": cat.vram_request(m, ctx, vram),
                             "persona": pk, "persona_text": PERSONAS[pk], "focus_short": "general"})
            nodes.setdefault(node, []).append(referees[-1])

    ed_tokens = len(referees) * EDITOR_REVIEW_EST + (3000 if claims_cfg else 0) + 2500
    ed_c = [(m, *f) for m in models if (f := cat.fit(m, ed_tokens, vram, ctx_cap))]
    ed_c = [c for c in ed_c if c[1] == "single"] or ed_c
    m, mode, ctx = max(ed_c, key=lambda c: cat.score(c[0], c[1]))
    editor = {"model": m["name"], "ctx": ctx, "mode": mode, "weights_mb": m["weights_mb"],
              "vram_req_mb": cat.vram_request(m, ctx, vram)}

    plan = {"name": name, "doc_tokens": tokens, "vram_budget_mb": vram, "personas": list(personas),
            "claims": claims_cfg, "referees": referees, "editor": editor, "warnings": warnings,
            "split": split, "max_claims": max_claims}
    (run / "plan.json").write_text(json.dumps(plan, indent=2))

    dag = []
    if claims_cfg:
        cmd = ["python3", "claims.py", "doc.txt", "--refs", "refs.txt", "--num-ctx", claims_cfg["ctx"],
               "--max-claims", max_claims, "--sandbox", sandbox] + ([] if network else ["--no-network"])
        (run / "claims.sub").write_text(condor.sub(
            model=claims_cfg["model"], command=cmd, name="claims",
            inputs=[WORKER / f for f in ("claims.py", "sandbox.py", "tools.py")] + _client_files() + ["doc.txt", "refs.txt"],
            remaps={"ollama-server.log": "logs/ollama-server_claims.log"},
            gpu_mb=claims_cfg["vram_req_mb"], mem=condor.host_mem(claims_cfg["weights_mb"])))
        dag += ["JOB C claims.sub", f"SCRIPT POST C /bin/sh {condor.DAG_POST}"]

    for node, rs in nodes.items():
        spec = {"doc": "doc.txt", "prompt": prompt.name, "ref_prompt": "ref_rating.txt", "num_ctx": rs[0]["ctx"], "max_out": 2500,
                "evidence": "claim_findings.md" if claims_cfg else None,
                "jobs": [{"focus_file": f"focus_{r['id']}.txt", "out": f"review_{r['id']}.md"} for r in rs]}
        (run / f"spec_{node}.json").write_text(json.dumps(spec, indent=2))
        for r in rs:
            (run / f"focus_{r['id']}.txt").write_text(r["persona_text"] + "\n")
        remaps = {f"review_{r['id']}.md": f"reviews/review_{r['id']}.md" for r in rs}
        remaps["ollama-server.log"] = f"logs/ollama-server_{node}.log"
        inputs = ([WORKER / f for f in ("run_panel.py", "review_pdf.py")] + _client_files() + [prompt, DATA / "prompts" / "ref_rating.txt", "doc.txt", f"spec_{node}.json"]
                  + [f"focus_{r['id']}.txt" for r in rs] + (["claim_findings.md"] if claims_cfg else []))
        (run / f"referee_{node}.sub").write_text(condor.sub(
            model=rs[0]["model"], command=["python3", "run_panel.py", f"spec_{node}.json"], name=f"referee_{node}",
            inputs=inputs, remaps=remaps, gpu_mb=rs[0]["vram_req_mb"], mem=condor.host_mem(rs[0]["weights_mb"])))
        dag += [f"JOB {node} referee_{node}.sub", f"RETRY {node} 1"]
    if claims_cfg:
        dag.append(f"PARENT C CHILD {' '.join(nodes)}")

    (run / "editor.sub").write_text(condor.sub(
        model=editor["model"], command=["python3", "editor.py", "--prompt", "editor.txt", "--num-ctx", editor["ctx"]],
        name="editor", inputs=[WORKER / "editor.py"] + _client_files() + [DATA / "prompts" / "editor.txt", "plan.json", "reviews"]
        + (["claim_findings.md"] if claims_cfg else []),
        remaps={"ollama-server.log": "logs/ollama-server_editor.log"},
        gpu_mb=editor["vram_req_mb"], mem=condor.host_mem(editor["weights_mb"])))
    dag.append("FINAL E editor.sub")
    (run / "review.dag").write_text("\n".join(dag) + "\n")
    return run, plan


def render(plan: dict) -> str:
    """Human-readable summary of a plan."""
    o = [f"Document: ~{plan['doc_tokens']} tokens   GPU budget: {plan['vram_budget_mb']} MB", ""]
    if plan["claims"]:
        c = plan["claims"]
        o.append(f"claims  {c['model']}  (tool use; ctx {c['ctx']}, <= {plan['max_claims']} claims; referees wait for it)")
    o.append(f"{'node':<7}{'model':<18}{'family':<9}{'ctx':>6}  {'mode':<8}{'VRAM req':>9}  personas")
    nodes: dict = {}
    for r in plan["referees"]:
        nodes.setdefault(r["node"], []).append(r)
    for node, rs in nodes.items():
        r = rs[0]
        o.append(f"{node:<7}{r['model']:<18}{r['family']:<9}{r['ctx']:>6}  {r['mode']:<8}{r['vram_req_mb']:>8}M  "
                 f"{', '.join(x['persona'] for x in rs)}")
    e = plan["editor"]
    o.append(f"{'editor':<7}{e['model']:<18}{'':<9}{e['ctx']:>6}  {e['mode']:<8}{e['vram_req_mb']:>8}M")
    models = {r["model"] for r in plan["referees"]}
    o.append(f"\n{len(plan['referees'])} reviews = {len(models)} models x {len(plan['personas'])} personas "
             f"in {len(nodes)} job(s).")
    if any(r["mode"] == "chunked" for r in plan["referees"]):
        o.append("Note: 'chunked' referees read the paper in pieces and merge -- expect weaker reviews.")
    if len({r["family"] for r in plan["referees"]}) < len(models):
        o.append("Note: fewer distinct model families fit than models requested.")
    o += [f"WARNING: {w}" for w in plan.get("warnings", [])]
    return "\n".join(o)
