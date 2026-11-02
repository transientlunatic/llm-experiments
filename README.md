# Igor

Run local LLMs (via [Ollama](https://ollama.com)) as HTCondor jobs, for the IGR. Igor builds and submits the jobs;
nothing needs root, containers or a shared filesystem.

```bash
pip install .            # or: pip install -e .   (needs Python >= 3.9 and click, on the submit host)
igor --help
```

| Command | What it does |
|---|---|
| `igor review run PAPER` | multi-model, multi-persona paper review with claim checking and an editor's report |
| `igor review build / submit / status` | the same in steps: build the DAG, submit it, check progress |
| `igor batch MODEL prompts.jsonl` | run a JSONL file of prompts through one model |
| `igor serve MODEL` + `igor connect JOBID` | an interactive model server on a GPU node, reached over an SSH tunnel |
| `igor models` | the model catalogue and what fits in your GPU budget (`--doc paper.pdf` to size it for a paper) |

Every job command takes `--dry-run` (print the `condor_submit` command) and the build commands take `--build-only`,
so you can inspect the generated files in `runs/<name>/` before anything is queued.

## How jobs run on the cluster
Each job runs `job.sh MODEL COMMAND...` on a GPU worker: it downloads the standalone Ollama release and the model into a
**node-local cache** (`/tmp/$USER-llm-cache`; first job per node is slow, later ones reuse it), starts `ollama serve` on a
random localhost port, runs the command, and shuts the server down. Workers need outbound internet, `curl`, `python3` and
`flock`. `zstd` is optional (falls back to `pip install --target` of `zstandard` into the cache). Overrides:
`OLLAMA_VERSION=v0.x.y`, `LLM_CACHE=...`, `OLLAMA_BIN=$(which ollama)` (use a local install, e.g. laptop minicondor).

## Paper review
```bash
igor review run 2401.08709 -n 3 --personas neutral,hostile,supportive --vram 23000
igor review status runs/2401.08709        # then read runs/2401.08709/report.md
```
Input may be a `.pdf`, `.tex`, `.txt`, or an arXiv source tarball (LaTeX is cleaned: `\input`s inlined, comments,
bibliography, TikZ and the author block removed; the reference list is kept for citation checks).

The planner estimates the paper's size and picks **N models** from the catalogue that fit `--vram` at the context needed
(different families where possible, single pass preferred). **Every persona runs on every model** (`neutral`, `hostile`,
`supportive`, `pedantic`; the hostile one must quote its criticisms and may not invent flaws), so model effects can be
told apart from stance effects. The DAG:

    CLAIMS  ->  one node per model (all personas, one model load)  ->  FINAL editor  ->  report.md

`--split` makes one node per (model, persona); `--no-claims` skips claim checking; `--network/--no-network` controls
reference lookups; `--sandbox strict|best-effort` (below).

**Claim checking** extracts checkable claims as JSON (quotes must occur verbatim in the paper, otherwise they are dropped)
and verifies each with a small type-specific tool set via Ollama tool calling: `search_paper`, `python_calc` (arithmetic
only), `get_reference_entry` (the paper's own bibliography) and `lookup_reference` (**INSPIRE-HEP** by arXiv id / DOI /
title+author, falling back to Crossref + arXiv; every hit gets a title-similarity score and a "NO CLOSE MATCH" flag so
non-existent citations stand out). Findings go into every referee prompt and the editor prompt. The claims node's POST
script never fails the workflow: if it produces nothing, the referees run without findings. Empirical claims
(convergence, figure contents) are mostly "unverifiable" without data or code.

**REF ratings:** each review ends with a `REF_ASSESSMENT` block (originality / significance / rigour on 0-4, and a
probability distribution over 4*/3*/2*/1*/U). The editor step parses these and computes, in code, a per-referee table, the
panel mixture and an expected-stars model x stance matrix (`report.md`). LLM ratings are poorly calibrated -- read the
spread, not the mean. The rubric (REF 2021 criteria) is in `igor/data/prompts/review.txt`; pass `--prompt` to use your own.

## Tool sandboxing
Tool calls never run model-written code, and each runs in its own short-lived subprocess (`igor/worker/sandbox.py`):
- **Least privilege by construction:** the calculator is a restricted AST evaluator; `lookup_reference` takes a *cite key
  only*, resolved against the paper's own bibliography, so the model cannot choose text to send to a third party; each
  worker is handed only the data its tool needs (the lookup worker sees one bibliography entry, never the paper).
- **OS isolation per call:** `python -I -S -B`, scrubbed environment, empty working directory, rlimits (CPU, memory, no
  file writes, no forking), hard timeout, output cap.
- **Namespaces (self-tested at start-up):** user + mount namespaces with a `chroot` into a tiny tmpfs root (interpreter and
  libraries only: no home, job directory, `/etc` or paper), and a network namespace for offline tools. The lookup worker
  keeps the host network but sees no files and may only reach `inspirehep.net`, `api.crossref.org` and `export.arxiv.org`
  over https (redirects refused, response size capped, XML with DTDs refused).
- **`--sandbox strict` (default)** refuses to run any tool unless the namespace stack works on that node (the claims stage
  then reports itself unavailable and the referees run without findings); `best-effort` accepts rlimits-only. `--no-network`
  removes lookups so no tool touches the network. The achieved level is printed in the job log and the findings header.
- **Prompt injection:** the paper, tool output and referee reports are labelled untrusted data in every prompt.

Known limits: binaries under the bound `/usr` can still be executed inside the sandbox (with no files, network or
credentials to use); the no-fork limit relies on `RLIMIT_NPROC`, which does not apply to root (not demonstrable in the
development sandbox); the sandbox protects the *tools*, not the Ollama server or the Condor job as a whole; lookups reveal
the paper's reference list (not its text) to INSPIRE/Crossref/arXiv.

## Status and caveats
- Developed without access to a cluster: the pieces were tested against mock Ollama servers, and the generated submit files
  and DAGs are unit-tested for content (`python -m unittest discover -s tests`), but **not yet run on a real pool**. First things
  to check: the DAG mechanics (`FINAL` node, POST script, `transfer_output_remaps`, directory transfer), that your chosen
  models support tool calling, and the sandbox level reported by the claims job.
- `igor/data/models.json` holds rough size/quality estimates; edit it or pass `--catalog` (and check the tags exist in the
  Ollama library).

## Layout
`igor/cli.py` (click commands), `igor/review.py` (review builder), `igor/catalog.py` (model fitting), `igor/condor.py`
(submit-file helpers), `igor/worker/` (stdlib-only scripts copied to the nodes), `igor/data/` (`job.sh`, prompts, catalogue).
A new job type is a click command that builds a submit file with `condor.sub(...)` and submits it.
