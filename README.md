# llm-experiments

Run local LLMs (Ollama) as HTCondor GPU jobs, with no root, containers or shared filesystem.

Each worker downloads the standalone Ollama release and the model weights into a **node-local cache**
(`/tmp/$USER-llm-cache`, override with `LLM_CACHE`), so only the first job per node is slow.
Workers need outbound internet, `curl`, `python3` and `flock`; `zstd` is optional (falls back to `pip install --target` of `zstandard` into the cache).
Pin a release with `OLLAMA_VERSION=v0.x.y`, or use a local install with `OLLAMA_BIN=$(which ollama)` (e.g. laptop minicondor).

## Batch
```bash
cp prompts.example.jsonl prompts.jsonl
condor_submit ollama.sub -append 'MODEL=llama3.2:1b'    # or edit the defaults at the top
```
`bin/job.sh MODEL CMD...` starts `ollama serve` on a random free port, pulls the model, runs
`CMD` with `OLLAMA_HOST` and `OLLAMA_MODEL` set, then shuts the server down.

## Review a PDF
```bash
condor_submit review.sub -append 'PDF=paper.pdf' -append 'MODEL=llama3.1:8b'    # result: review.md
```
`client/review_pdf.py` extracts text (`pdftotext`, else `pypdf`), sets a large `num_ctx`
(Ollama's default silently truncates long input), and map-reduces over chunks if the paper doesn't fit.
Edit `prompts/review.txt` to change what is asked. Tips:
- Text/`.tex` input also works; arXiv LaTeX source handles equations much better than PDF text.
- Scanned PDFs or figures: add `--images 5` and use a vision model (e.g. `qwen2.5vl`); needs `pdftoppm` on the worker.
- Reviews from 8B models are shallow; try a 30B+ model on the 24/32 GB cards.

## Interactive
```bash
condor_submit serve.sub
condor_tail -stderr <jobid>       # prints the port once the model is loaded
# on the submit host (leave running):
condor_ssh_to_job -ssh "ssh -L 11434:127.0.0.1:<port>" <jobid>
# then, on the submit host (or via a second -L hop from your laptop):
OLLAMA_HOST=127.0.0.1:11434 ollama run llama3.1:8b
```
The server only listens on the worker's localhost (it has no authentication), hence the tunnel.
The job removes itself after 4 h (`periodic_remove` in `serve.sub`); `condor_rm <jobid>` ends it sooner.
Batch jobs exit on their own when the command finishes.

## Notes
- Pool GPUs: 24 GB (3090/4090), 32 GB (5090), 10.6 GB (2080 Ti). Use `MIN_VRAM_MB` to pick;
  ~8B models at Q4 fit anywhere, 30B+ need a 24 GB card or more.
- `request_disk` must cover model weights in the cache if the node has little `/tmp` space.
- Untested on the cluster. The bundled CUDA libraries must be compatible with the worker's NVIDIA driver.

## Multi-referee review (claims -> models x personas -> editor)
```bash
bin/plan_review.py 2401.08709 --n 3 --personas neutral,hostile,supportive --vram 23000
cd runs/2401.08709 && condor_submit_dag -f review.dag
```
Paper: `.pdf`, `.tex`, `.txt` or an arXiv source tarball. `plan_review.py` cleans the document once (LaTeX: inlines
`\input`s, strips comments, bibliography, TikZ and the author block; keeps the reference list in `refs.txt`),
estimates its size and picks a **panel of N models** from `config/models.json` that fit the `--vram` budget at the
context needed (different families where possible, single pass preferred). **Every persona runs on every model**
(`neutral, hostile, supportive, pedantic`; hostile must quote its criticisms and may not invent flaws), so model
effects can be separated from stance effects. DAG:

    CLAIMS  ->  one node per model (all personas, one model load)  ->  FINAL editor -> report.md

`--split` makes one node per (model, persona); `--no-claims` skips claim checking; `--submit` submits directly.

**Claim checking** (`client/claims.py`, a tool-capable model e.g. qwen2.5): extracts checkable claims as JSON
(quotes must occur verbatim in the paper, otherwise dropped), then verifies each with a small type-specific tool set
via Ollama tool calling: `search_paper`, `python_calc` (arithmetic only -- a restricted AST evaluator, *no* model-written
code is executed), `get_reference_entry` (the paper's own bibliography), `lookup_reference` (**INSPIRE-HEP** by arXiv id / DOI /
title+author, falling back to Crossref + arXiv; workers need outbound internet; every hit gets a title-similarity score
and a "NO CLOSE MATCH" flag, so non-existent citations stand out). Verdicts: supported / partly_supported / contradicted / unverifiable, with evidence. The findings
go into every referee prompt and the editor prompt. The claims node's POST script never fails the workflow: if it
produces nothing, referees run without findings. Empirical claims (convergence, figure contents) are mostly
`unverifiable` without data/code; lookups are fuzzy, so citation checks often need judgement.

**REF ratings**: each review ends with a `REF_ASSESSMENT` block (originality / significance / rigour 0-4 and a probability
distribution over 4*/3*/2*/1*/U). `editor.py` parses these and computes, in code, the per-referee table, the panel mixture
and an expected-stars **model x stance matrix** (see `report.md`). LLM ratings are poorly calibrated -- read the spread.
Rubric: `prompts/review.txt` (REF 2021 criteria).

Notes: logs in `runs/<name>/logs/`, reviews in `runs/<name>/reviews/`, final `report.md` in the run dir. The DAG mechanics
(`FINAL`, POST script, `transfer_output_remaps`, directory transfer) are untested on a real pool -- check those first.

## Tool sandboxing
Tool calls never run model-written code, and each one runs in its own short-lived subprocess (`client/sandbox.py`,
workers in `client/tools.py`):
- **Least privilege by construction:** the calculator is a restricted AST evaluator; `lookup_reference` takes a *cite key
  only*, resolved against the paper's own bibliography, so the model cannot choose text to send to Crossref/arXiv; each
  worker is handed only the data its tool needs (the lookup worker sees one bibliography entry, never the paper).
- **OS isolation per call:** `python -I -S -B`, scrubbed environment (no `HOME`, tokens or credentials), empty working
  directory, rlimits (CPU, memory, no file writes, no forking), hard timeout, output cap.
- **Namespaces (self-tested at start-up):** user + mount namespaces with a `chroot` into a tiny tmpfs root (interpreter and
  libraries only -- no home, no job directory, no `/etc`, no paper), plus a network namespace for offline tools (no network
  at all). The lookup worker keeps the host network but sees no files; it may only reach `inspirehep.net`, `api.crossref.org` and
  `export.arxiv.org` over https, refuses redirects, caps response size and refuses XML with DTDs/entities.
- **`--sandbox strict` (default)** refuses to run any tool unless the namespace stack works on that node (then the claims
  stage reports itself unavailable and referees run without findings). `--sandbox best-effort` accepts rlimits-only.
  `--no-network` disables lookups entirely, so no tool touches the network. The achieved level is printed in the job log
  and in the findings header.
- **Prompt injection:** the paper, tool output and referee reports are labelled untrusted data in every prompt. Tools have no
  side effects, so the worst case is a skewed verdict.

Known limits: binaries under the bound `/usr` can still be executed inside the sandbox (they have no files, network or
credentials to work with); the no-fork limit relies on `RLIMIT_NPROC`, which does not apply to a root user (couldn't be
demonstrated in the development sandbox); the sandbox protects the *tools*, not the Ollama server itself or the
Condor job as a whole; and the lookups reveal your paper's reference list (not its text) to Crossref/arXiv -- use `--no-network`
for confidential manuscripts if even that matters.
