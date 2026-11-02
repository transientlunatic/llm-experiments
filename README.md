# llm-experiments

Run local LLMs (Ollama) as HTCondor GPU jobs, with no shared filesystem needed.

Each worker pulls the Ollama Apptainer image and model weights into a **node-local cache**
(`/tmp/$USER-llm-cache`, override with `LLM_CACHE`), so only the first job per node is slow.

## Batch
```bash
cp prompts.example.jsonl prompts.jsonl
condor_submit ollama.sub          # edit MODEL / MIN_VRAM_MB at the top
```
`bin/job.sh MODEL CMD...` starts `ollama serve` on a random free port, pulls the model, runs
`CMD` with `OLLAMA_HOST` and `OLLAMA_MODEL` set, then shuts the server down.

## Interactive
```bash
condor_submit serve.sub
condor_tail -stderr <jobid>       # prints host:port and an ssh tunnel hint
ssh -L 11434:<worker>:<port> <submit-host>   # then OLLAMA_HOST=localhost:11434 ollama run ...
```

## Notes
- Pool GPUs: 24 GB (3090/4090), 32 GB (5090), 10.6 GB (2080 Ti). Use `MIN_VRAM_MB` to pick;
  ~8B models at Q4 fit anywhere, 30B+ need a 24 GB card or more.
- `request_disk` must cover model weights in the cache if the node has little `/tmp` space.
- Untested on the cluster; first thing to check is that `apptainer` is on workers' PATH.
