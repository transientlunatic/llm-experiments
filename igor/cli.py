"""Igor command line: build and submit local-LLM jobs on HTCondor."""
from __future__ import annotations

import functools
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

import click

from . import __version__
from . import catalog as cat
from . import condor
from . import review as rv


def _catch(fn):
    """Turn Igor's own errors into clean click errors (no tracebacks)."""
    @functools.wraps(fn)
    def wrapper(*a, **k):
        try:
            return fn(*a, **k)
        except (rv.PlanError, condor.CondorError) as e:
            raise click.ClickException(str(e))
    return wrapper


def _split_csv(ctx, param, value):
    return tuple(x.strip() for x in value.split(",") if x.strip())


@click.group(context_settings={"help_option_names": ["-h", "--help"]})
@click.version_option(__version__, prog_name="igor")
def main():
    """Igor: run local LLMs (via Ollama) as HTCondor jobs.

    \b
    igor review run PAPER        multi-model, multi-persona paper review with claim checking
    igor batch MODEL PROMPTS     run a JSONL file of prompts through one model
    igor serve MODEL             start an interactive model server job (then: igor connect JOBID)
    igor models                  list the model catalogue and what fits where
    """


# --------------------------------------------------------------------------- review
def review_options(f):
    opts = [
        click.argument("doc", type=click.Path(exists=True)),
        click.option("--name", help="run name (default: from the file name)"),
        click.option("--out-dir", default="runs", show_default=True, help="parent directory for run directories"),
        click.option("-n", "--models", "n", default=3, show_default=True, help="number of distinct models in the panel"),
        click.option("--personas", default=",".join(rv.DEFAULT_PERSONAS), show_default=True, callback=_split_csv,
                     help="comma list from: " + ", ".join(rv.PERSONAS) + " -- each runs on every model"),
        click.option("--vram", default=23000, show_default=True,
                     help="max GPU memory per job in MB (24 GB cards: 23000; 5090: 31000)"),
        click.option("--ctx-cap", default=32768, show_default=True, help="largest context window to use"),
        click.option("--split", is_flag=True, help="one DAG node per (model, persona) instead of per model"),
        click.option("--claims/--no-claims", default=True, show_default=True, help="extract and verify claims first"),
        click.option("--max-claims", default=15, show_default=True),
        click.option("--sandbox", type=click.Choice(["strict", "best-effort"]), default="strict", show_default=True,
                     help="strict: claim tools refuse to run unless user+net+mount namespaces work on the node"),
        click.option("--network/--no-network", default=True, show_default=True,
                     help="allow INSPIRE/Crossref/arXiv reference lookups (the only tool that uses the network)"),
        click.option("--catalog", type=click.Path(exists=True), help="model catalogue JSON (default: built in)"),
        click.option("--prompt", type=click.Path(exists=True), help="review prompt file (default: built in)"),
        click.option("--raw", is_flag=True, help="do not clean LaTeX source"),
        click.option("--appendix", type=click.Choice(["separate", "drop"]), default="separate", show_default=True,
                     help="separate: appendices are supplementary, shown to referees only if they fit; drop: remove them"),
    ]
    for o in reversed(opts):
        f = o(f)
    return f


def _build_review(doc, **o):
    run, plan = rv.build(doc, name=o["name"], out_dir=o["out_dir"], n=o["n"], personas=o["personas"], vram=o["vram"],
                         ctx_cap=o["ctx_cap"], split=o["split"], claims=o["claims"], max_claims=o["max_claims"],
                         sandbox=o["sandbox"], network=o["network"], catalog=o["catalog"], prompt=o["prompt"],
                         raw=o["raw"], appendix=o["appendix"])
    click.echo("\n" + rv.render(plan) + f"\n\nRun directory: {run}")
    return run


@main.group()
def review():
    """Multi-referee paper review: claims -> models x personas -> editor report."""


@review.command("build")
@review_options
@_catch
def review_build(doc, **o):
    """Build the run directory and DAG for DOC (a .pdf, .tex, .txt or arXiv source tarball), without submitting."""
    run = _build_review(doc, **o)
    click.echo(f"Submit with:  igor review submit {run}")


@review.command("submit")
@click.argument("run", type=click.Path(exists=True, file_okay=False))
@click.option("--dry-run", is_flag=True, help="print the command instead of running it")
@_catch
def review_submit(run, dry_run):
    """Submit a built run directory to HTCondor (condor_submit_dag)."""
    run = Path(run)
    if not (run / "review.dag").exists():
        raise click.ClickException(f"{run} has no review.dag -- build it first with 'igor review build'")
    click.echo(condor.submit_dag("review.dag", run, dry_run))
    if not dry_run:
        click.echo(f"\nFollow progress with:  igor review status {run}")


@review.command("run")
@review_options
@click.option("--dry-run", is_flag=True, help="build, but only print the submit command")
@_catch
def review_run(doc, dry_run, **o):
    """Build and submit a review in one step."""
    run = _build_review(doc, **o)
    click.echo("\n" + condor.submit_dag("review.dag", run, dry_run))
    if not dry_run:
        click.echo(f"\nFollow progress with:  igor review status {run}")


@review.command("status")
@click.argument("run", type=click.Path(exists=True, file_okay=False))
def review_status(run):
    """Show what a review run has produced so far."""
    run = Path(run)
    if not (run / "plan.json").exists():
        raise click.ClickException(f"{run} is not an Igor run directory (no plan.json)")
    plan = json.loads((run / "plan.json").read_text())
    done = {int(m.group(1)) for p in (run / "reviews").glob("review_*.md") if (m := re.search(r"(\d+)", p.name))}
    ids = [r["id"] for r in plan["referees"]]
    click.echo(f"Run {plan['name']}: ~{plan['doc_tokens']} tokens")
    if plan["claims"]:
        cf = run / "claim_findings.md"
        n = cf.read_text().count("\n## C") if cf.exists() else 0
        click.echo(f"  claims:   {'done, %d claims checked' % n if cf.exists() else 'pending'}")
    click.echo(f"  reviews:  {len(done & set(ids))}/{len(ids)}"
               + (f"  missing: {sorted(set(ids) - done)}" if done and set(ids) - done else ""))
    click.echo(f"  report:   {'READY -> ' + str(run / 'report.md') if (run / 'report.md').exists() else 'pending'}")
    out = run / "review.dag.dagman.out"
    if out.exists():
        click.echo("\nDAGMan log (tail):\n  " + "\n  ".join(out.read_text().splitlines()[-6:]))
    else:
        click.echo("\n(not submitted yet, or DAGMan has not started)")


# --------------------------------------------------------------------------- shared helpers for simple jobs
def _gpu_req(model, ctx, catalog_path=None, vram=None):
    """GlobalMemoryMb requirement for MODEL at ctx: --vram if given, else estimated from the catalogue."""
    if vram:
        return vram, "from --vram"
    m = cat.find_model(model, cat.load_catalog(catalog_path))
    if not m:
        click.echo(f"note: {model} is not in the catalogue, so no GPU memory requirement is set (use --vram to set one)",
                   err=True)
        return None, None
    return cat.vram_request(m, ctx, 10**6), f"estimated for {m['name']} at ctx {ctx}"


def _finish(run, sub_name, build_only, dry_run):
    if build_only:
        click.echo(f"Built {run}/{sub_name}. Submit with: (cd {run} && condor_submit {sub_name})")
        return None
    return condor.submit(sub_name, run, dry_run)


def _simple_options(f):
    for o in reversed([
        click.argument("model"),
        click.option("--name", help="run name"),
        click.option("--out-dir", default="runs", show_default=True),
        click.option("--ctx", default=8192, show_default=True, help="context window (sizes the GPU memory request)"),
        click.option("--vram", type=int, help="GlobalMemoryMb requirement in MB (default: estimated from the catalogue)"),
        click.option("--catalog", type=click.Path(exists=True)),
        click.option("--build-only", is_flag=True, help="write the submit file but do not submit"),
        click.option("--dry-run", is_flag=True, help="print the submit command instead of running it"),
    ]):
        f = o(f)
    return f


@main.command()
@_simple_options
@click.argument("prompts", type=click.Path(exists=True, dir_okay=False))
@_catch
def batch(model, prompts, name, out_dir, ctx, vram, catalog, build_only, dry_run):
    """Run a JSONL file of prompts ({"prompt": ...} per line) through MODEL; results.jsonl comes back in the run dir."""
    run = Path(out_dir) / (name or f"batch-{Path(prompts).stem}")
    (run / "logs").mkdir(parents=True, exist_ok=True)
    shutil.copy(prompts, run / "prompts.jsonl")
    gpu, why = _gpu_req(model, ctx, catalog, vram)
    (run / "batch.sub").write_text(condor.sub(
        model=model, command=["python3", "generate.py", "prompts.jsonl", "results.jsonl"], name="batch",
        inputs=[cat.DATA.parent / "worker" / "generate.py", "prompts.jsonl"],
        remaps={"ollama-server.log": "logs/ollama-server.log"}, gpu_mb=gpu, log="logs/batch.log"))
    click.echo(f"Run directory: {run}" + (f"   GPU memory >= {gpu} MB ({why})" if gpu else ""))
    out = _finish(run, "batch.sub", build_only, dry_run)
    if out:
        click.echo(out)
        click.echo(f"Results will appear in {run}/results.jsonl")


@main.command()
@_simple_options
@click.option("--hours", default=4.0, show_default=True, help="remove the job after this many hours of running")
@_catch
def serve(model, name, out_dir, ctx, vram, catalog, build_only, dry_run, hours):
    """Start an interactive Ollama server job for MODEL. Connect with 'igor connect JOBID'."""
    run = Path(out_dir) / (name or f"serve-{re.sub(r'[^A-Za-z0-9]+', '_', model)}")
    (run / "logs").mkdir(parents=True, exist_ok=True)
    gpu, why = _gpu_req(model, ctx, catalog, vram)
    (run / "serve.sub").write_text(condor.sub(
        model=model, command=["serve"], name="serve", gpu_mb=gpu, log="logs/serve.log",
        remaps={"ollama-server.log": "logs/ollama-server.log"},
        extra=f"periodic_remove = (time() - JobCurrentStartDate) > {int(hours * 3600)}"))
    click.echo(f"Run directory: {run}" + (f"   GPU memory >= {gpu} MB ({why})" if gpu else ""))
    out = _finish(run, "serve.sub", build_only, dry_run)
    if out:
        click.echo(out)
        click.echo("\nOnce it is running:  igor connect <JOBID>      (job is removed after %.4g h; condor_rm ends it sooner)" % hours)


@main.command()
@click.argument("jobid")
@click.option("--local-port", default=11434, show_default=True)
@click.option("--dry-run", is_flag=True)
@_catch
def connect(jobid, local_port, dry_run):
    """Open an SSH tunnel to a running 'igor serve' job; the model is then at http://localhost:LOCAL_PORT."""
    if shutil.which("condor_tail") is None and not dry_run:
        raise click.ClickException("condor_tail not found -- run this on the submit host")
    port = None
    if not dry_run:
        r = subprocess.run(["condor_tail", "-stderr", jobid], capture_output=True, text=True)
        found = re.findall(r"serving on worker localhost:(\d+)", r.stdout + r.stderr)
        if not found:
            raise click.ClickException(f"job {jobid} is not serving yet (model may still be loading); try again shortly")
        port = found[-1]
    cmd = ["condor_ssh_to_job", "-ssh", f"ssh -L {local_port}:127.0.0.1:{port or '<port>'}", jobid]
    click.echo(f"Tunnelling localhost:{local_port} -> worker:{port or '<port>'}   (leave this session open; Ctrl-D to close)")
    click.echo("In another terminal:  OLLAMA_HOST=127.0.0.1:%d ollama run MODEL   (or curl localhost:%d/api/tags)" % (local_port, local_port))
    if dry_run:
        click.echo("$ " + " ".join(cmd))
        return
    os.execvp(cmd[0], cmd)


# --------------------------------------------------------------------------- models
@main.command()
@click.option("--vram", default=23000, show_default=True, help="GPU memory budget in MB")
@click.option("--tokens", type=int, help="size of the document in tokens, to show how each model would run")
@click.option("--doc", type=click.Path(exists=True), help="a document to measure instead of --tokens")
@click.option("--ctx-cap", default=32768, show_default=True)
@click.option("--catalog", type=click.Path(exists=True))
@_catch
def models(vram, tokens, doc, ctx_cap, catalog):
    """List the model catalogue and how each model fits the GPU budget (for a document, if given)."""
    if doc:
        tokens = rv.est_tokens(rv.split_main_appendix(rv.load_document_ex(doc)[0])[0])
    ms = cat.load_catalog(catalog)
    click.echo(f"GPU budget {vram} MB" + (f", document ~{tokens} tokens" if tokens else "") + "\n")
    click.echo(f"{'model':<18}{'family':<9}{'tools':<6}{'weights':>8}{'quality':>8}   fit")
    for m in ms:
        f = cat.fit(m, tokens or 4000, vram, ctx_cap)
        fit_s = (f"{f[0]:<8} ctx {f[1]:<6} needs ~{cat.vram_request(m, f[1], 10**6)} MB" if f else "does not fit")
        click.echo(f"{m['name']:<18}{m['family']:<9}{'yes' if m.get('tools') else '-':<6}"
                   f"{m['weights_mb'] / 1000:>6.1f}GB{m['quality']:>8}   {fit_s}")
