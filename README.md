# llm-experiments

Run local LLMs (Ollama) as HTCondor GPU jobs, with no root, containers or shared filesystem.

Each worker downloads the standalone Ollama release and the model weights into a **node-local cache**
(`/tmp/$USER-llm-cache`, override with `LLM_CACHE`), so only the first job per node is slow.
Workers need outbound internet, `curl`, `python3` and `flock`; `zstd` is optional (falls back to `pip install --user zstandard`).
Pin a release with `OLLAMA_VERSION=v0.x.y`, or use a local install with `OLLAMA_BIN=$(which ollama)` (e.g. laptop minicondor).

## Batch
```bash
cp prompts.example.jsonl prompts.jsonl
condor_submit ollama.sub          # edit MODEL / MIN_VRAM_MB at the top
```
`bin/job.sh MODEL CMD...` starts `ollama serve` on a random free port, pulls the model, runs
`CMD` with `OLLAMA_HOST` and `OLLAMA_MODEL` set, then shuts the server down.

## Review a PDF
```bash
condor_submit review.sub PDF=paper.pdf MODEL=llama3.1:8b     # result: review.md
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
