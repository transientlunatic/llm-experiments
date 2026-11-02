#!/usr/bin/env bash
# Run a command against a temporary Ollama server on a Condor GPU worker.
# No root, container or shared filesystem needed: the standalone Ollama release is unpacked
# into a node-local cache that later jobs on the same node reuse.
#
# Usage: job.sh MODEL [CMD ARGS...]
#   MODEL  e.g. llama3.1:8b  (pulled into the node-local cache if absent)
#   CMD    run with OLLAMA_HOST / OLLAMA_MODEL set. If omitted (or "serve"),
#          the server is kept alive until the job ends (interactive use).
#
# Env overrides: OLLAMA_VERSION (e.g. v0.12.3; default: latest)
#                LLM_CACHE      (default /tmp/$USER-llm-cache)
#                OLLAMA_BIN     (use this ollama binary instead, e.g. on a laptop with ollama installed)
set -euo pipefail

MODEL="${1:?usage: job.sh MODEL [CMD...]}"; shift || true
CACHE="${LLM_CACHE:-/tmp/$(id -un)-llm-cache}"
export LLM_CACHE="$CACHE"   # so client scripts share the same node-local cache
SCRATCH="${_CONDOR_SCRATCH_DIR:-$PWD}"
VERSION="${OLLAMA_VERSION:-latest}"
mkdir -p "$CACHE/models" "$SCRATCH/home"
export HOME="$SCRATCH/home"          # ollama writes its key under $HOME/.ollama

if [ -n "${OLLAMA_BIN:-}" ]; then
  BIN="$OLLAMA_BIN"
else
  DIR="$CACHE/ollama-$VERSION"
  BIN="$DIR/bin/ollama"
  if [ "$VERSION" = latest ]; then
    URL="https://ollama.com/download/ollama-linux-amd64.tar.zst"
  else
    URL="https://github.com/ollama/ollama/releases/download/$VERSION/ollama-linux-amd64.tar.zst"
  fi
  (
    flock 9   # concurrent jobs on one node wait here rather than racing the download
    if [ ! -x "$BIN" ]; then
      echo "[job] installing ollama ($VERSION) into $DIR" >&2
      rm -rf "$DIR.part"; mkdir -p "$DIR.part"
      if command -v zstd >/dev/null; then
        curl -fsSL "$URL" | zstd -dc | tar -x -C "$DIR.part"
      else
        # --target sidesteps PEP 668 "externally-managed-environment" errors; fall back if pip still objects
        PYLIB="$CACHE/pylib"
        python3 -c "import sys; sys.path.insert(0,'$PYLIB'); import zstandard" 2>/dev/null || \
          python3 -m pip install -q --target "$PYLIB" zstandard >&2 || \
          python3 -m pip install -q --target "$PYLIB" --break-system-packages zstandard >&2
        curl -fsSL "$URL" | PYTHONPATH="$PYLIB" python3 -c '
import sys, zstandard
zstandard.ZstdDecompressor().copy_stream(sys.stdin.buffer, sys.stdout.buffer)' | tar -x -C "$DIR.part"
      fi
      mv "$DIR.part" "$DIR"
    fi
  ) 9>"$CACHE/install.lock"
fi

PORT=$(python3 -c 'import socket;s=socket.socket();s.bind(("",0));print(s.getsockname()[1])')
export OLLAMA_HOST="127.0.0.1:$PORT" OLLAMA_MODEL="$MODEL" OLLAMA_MODELS="$CACHE/models"

echo "[job] host=$(hostname -f) port=$PORT model=$MODEL gpu=${CUDA_VISIBLE_DEVICES:-?}" >&2
"$BIN" serve >"$SCRATCH/ollama-server.log" 2>&1 &
SERVER=$!
cleanup() {
  rc=$?
  [ $rc -ne 0 ] && { echo "[job] FAILED (exit $rc); tail of ollama-server.log:" >&2; tail -n 40 "$SCRATCH/ollama-server.log" >&2 || true; }
  kill $SERVER 2>/dev/null || true; wait $SERVER 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 120); do
  curl -sf "http://$OLLAMA_HOST/api/tags" >/dev/null && break
  kill -0 $SERVER 2>/dev/null || { echo "[job] server died:" >&2; cat "$SCRATCH/ollama-server.log" >&2; exit 1; }
  sleep 1
done

# Lock so concurrent jobs on one node don't pull the same weights twice.
( flock 8; "$BIN" pull "$MODEL" >&2 ) 8>"$CACHE/pull.lock"

if [ $# -eq 0 ] || [ "$1" = serve ]; then
  echo "[job] serving on worker localhost:$PORT -- from submit host: condor_ssh_to_job -ssh \"ssh -L 11434:127.0.0.1:$PORT\" ${CONDOR_JOB_ID:-<jobid>}" >&2
  wait $SERVER
else
  "$@"
fi
