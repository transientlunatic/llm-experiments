"""HTCondor helpers shared by every Igor job type: submit-file text and condor_submit wrappers."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .catalog import DATA

JOB_SH = DATA / "job.sh"                    # starts a temporary Ollama server, runs a command, cleans up
DAG_POST = DATA / "dag_post_claims.sh"


class CondorError(Exception):
    pass


def quote_args(*items) -> str:
    """HTCondor 'new syntax' arguments: whole string in double quotes, items with spaces in single quotes."""
    out = []
    for a in map(str, items):
        a = a.replace('"', '""').replace("'", "''")
        out.append(f"'{a}'" if (not a or " " in a or "\t" in a) else a)
    return '"' + " ".join(out) + '"'


def sub(*, model: str, command: list, inputs=(), remaps: dict | None = None, gpu_mb: int | None = None,
        mem: str = "16GB", disk: str = "30GB", cpus: int = 4, name: str = "job", log: str = "logs/nodes.log",
        extra: str = "") -> str:
    """Submit-file text for one GPU job: `job.sh MODEL COMMAND...` (an Ollama session around COMMAND).

    command: argv run on the worker after Ollama is up, e.g. ["python3", "claims.py", "doc.txt"].
    inputs:  files to transfer (absolute paths on the submit host, or paths relative to the submit dir).
    remaps:  {file written by the job: where to put it on return}, so parallel jobs don't overwrite each other.
    """
    lines = [
        "universe                = vanilla",
        "executable              = /bin/bash",
        "transfer_executable     = False",                      # use the node's bash; job.sh is shipped as an input
        f"arguments               = {quote_args('job.sh', model, *command)}",
        f"transfer_input_files    = {', '.join([str(JOB_SH), *map(str, inputs)])}",
        "should_transfer_files   = YES",
        "when_to_transfer_output = ON_EXIT",
    ]
    if remaps:
        rm = "; ".join(f"{k} = {v}" for k, v in remaps.items())
        lines.append(f'transfer_output_remaps  = "{rm}"')
    lines += ["request_gpus            = 1", f"request_cpus            = {cpus}",
              f"request_memory          = {mem}", f"request_disk            = {disk}"]
    if gpu_mb:
        lines.append(f"require_gpus            = (GlobalMemoryMb >= {gpu_mb})")
    lines += [f"output                  = logs/{name}.out", f"error                   = logs/{name}.err",
              f"log                     = {log}"]
    if extra:
        lines.append(extra.rstrip())
    lines.append("queue")
    return "\n".join(lines) + "\n"


def host_mem(weights_mb: int) -> str:
    return "32GB" if weights_mb > 12000 else "16GB"


def _run(cmd: list, cwd: Path, dry_run: bool) -> str:
    text = "$ (cd %s && %s)" % (cwd, " ".join(cmd))
    if dry_run:
        return text
    if not shutil.which(cmd[0]):
        raise CondorError(f"{cmd[0]} not found -- run this on an HTCondor submit host (or use --dry-run / --build-only)")
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if r.returncode:
        raise CondorError(f"{cmd[0]} failed:\n{r.stdout}{r.stderr}")
    return text + "\n" + r.stdout.strip()


def submit(sub_file: str, cwd, dry_run=False) -> str:
    Path(cwd, "logs").mkdir(exist_ok=True)
    return _run(["condor_submit", sub_file], Path(cwd), dry_run)


def submit_dag(dag_file: str, cwd, dry_run=False) -> str:
    Path(cwd, "logs").mkdir(exist_ok=True)
    return _run(["condor_submit_dag", "-f", dag_file], Path(cwd), dry_run)
