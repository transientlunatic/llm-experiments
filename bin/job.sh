#!/usr/bin/env bash
# Run a command against a temporary Ollama server on a Condor GPU worker.
#
# Usage: job.sh MODEL [CMD ARGS...]
#   MODEL  e.g. llama3.1:8b  (pulled into a node-local cache if absent)
#   CMD    run with OLLAMA_HOST / OLLAMA_MODEL set. If omitted (or "serve"),
#          the server is kept alive until the job ends (interactive use).
#
# Env overrides: OLLAMA_IMAGE (default docker://ollama/ollama:latest),
#                LLM_CACHE (default /tmp/$USER-llm-cache; node-local, reused by later jobs)
set -euo pipefail

MODEL="${1:?usage: job.sh MODEL [CMD...]}"; shift || true
IMAGE="${OLLAMA_IMAGE:-docker://ollama/ollama:latest}"
CACHE="${LLM_CACHE:-/tmp/${USER:-$(id -un)}-llm-cache}"
SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
mkdir -p "$CACHE"/{models,apptainer,tmp}

export APPTAINER_CACHEDIR="$CACHE/apptainer" APPTAINER_TMPDIR="$CACHE/tmp"
SIF="$CACHE/ollama-$(echo "$IMAGE" | tr -c 'A-Za-z0-9\n' _).sif"

# Fetch the image once per node; flock stops concurrent jobs racing.
(
  flock 9
  if [ ! -f "$SIF" ]; then
    rm -f "$SIF.part"
    apptainer pull "$SIF.part" "$IMAGE" >&2
    mv -f "$SIF.part" "$SIF"
  fi
) 9>"$CACHE/image.lock"

PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1])')
export OLLAMA_HOST="127.0.0.1:$PORT" OLLAMA_MODEL="$MODEL"
# --writable-tmpfs: ollama wants a writable $HOME; --nv exposes the GPU.
AP=(apptainer exec --nv --writable-tmpfs --bind "$CACHE/models:/models"
    --env OLLAMA_MODELS=/models --env OLLAMA_HOST="$OLLAMA_HOST" "$SIF")

echo "[job] host=$(hostname -f) port=$PORT model=$MODEL gpu=${CUDA_VISIBLE_DEVICES:-?}" >&2
"${AP[@]}" ollama serve >"$SCRATCH/ollama-server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true' EXIT

for _ in $(seq 120); do
  curl -sf "http://$OLLAMA_HOST/api/tags" >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "[job] server died:" >&2; cat "$SCRATCH/ollama-server.log" >&2; exit 1; }
  sleep 1
done

# Per-model lock so concurrent jobs on one node don't pull the same weights twice.
( flock 8; "${AP[@]}" ollama pull "$MODEL" >&2 ) 8>"$CACHE/pull.lock"

if [ $# -eq 0 ] || [ "$1" = serve ]; then
  echo "[job] serving at $(hostname -f):$PORT -- tunnel: ssh -L 11434:$(hostname -f):$PORT <submit-host>" >&2
  wait $SERVER
else
  "$@"
fi
