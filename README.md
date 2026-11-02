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

## Multi-referee review (planner + DAG)
```bash
bin/plan_review.py 2401.08709 --n 3 --vram 23000     # paper: .pdf, .tex, .txt or an arXiv source tarball
cd runs/2401.08709 && condor_submit_dag -f review.dag
```
`plan_review.py` cleans the document once (LaTeX: inlines `\input`s, strips comments, bibliography, TikZ
and the author block), estimates its size, and picks a **panel from `config/models.json`**:
models that fit the `--vram` budget at the context needed, from *different families* where possible,
preferring single-pass reading over chunking, each given a different focus (statistics / novelty /
technical correctness). It prints the plan and writes a DAG: referees run in parallel on separate GPUs,
then an **editor** node (a DAG FINAL node, so it still runs if a referee fails) writes `report.md`:
verdict, consensus, disagreements, a prioritised action list tagged by referee, and all referee reports.
- `--vram 31000` also allows the 5090 and bigger models; `--submit` submits straight away.
- Edit `config/models.json` (sizes/quality are rough estimates; check tags exist with the Ollama library).
- Logs and server logs land in `runs/<name>/logs/`; reports in `runs/<name>/reviews/` and `report.md`.
- Untested on the cluster: DAG semantics (`FINAL`, `transfer_output_remaps`, directory transfer) are the thing to check first.

### Personas and REF ratings
Each referee gets a stance (default order: `neutral, hostile, supportive, pedantic`, cycled) on top of its
focus; choose with `--personas hostile,hostile,neutral`. The hostile stance must back every criticism with
a quote and may not invent flaws. Every review ends with a `REF_ASSESSMENT` block (originality / significance /
rigour on 0-4, plus a probability distribution over 4*/3*/2*/1*/U). `editor.py` parses these and computes the
rating table and an equal-weight mixture in code (the editor model does not write the numbers); see
"Estimated REF ratings" in `report.md`. LLM ratings are poorly calibrated -- read the spread, not the mean.
The rubric lives in `prompts/review.txt` (REF 2021 criteria; edit it if you want different ones).
