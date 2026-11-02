"""Run each tool call in a short-lived, least-privilege subprocess.

Layers (outermost first):
  1. one process per call, started `python3 -I -S -B` (no site/user dirs, PYTHON* env ignored), new session
  2. scrubbed environment (no HOME, no tokens/credentials) and an empty temporary working directory
  3. resource limits: CPU seconds, address space, no file writes (RLIMIT_FSIZE=0), few fds, no core dumps,
     and no forking
  4. hard wall-clock timeout (whole process group killed) and an output cap
  5. where the kernel allows UNPRIVILEGED namespaces (detected and self-tested at start-up):
       - user namespace (the worker is "root" of nothing)
       - mount namespace + chroot into a tiny tmpfs root holding only the interpreter, its libraries and a
         copy of tools.py: the worker cannot see your home, the job directory, the paper, /etc/passwd...
       - network namespace for offline tools: no network at all
  6. least data: each worker is given only what its tool needs (the lookup worker sees one bibliography
     entry, never the paper; offline workers have neither network nor files)
Only `lookup` uses the network (host allowlist, https only, redirects refused: see tools.py). The workers run
fixed code (tools.py) and no model-written code, so these layers are defence in depth.

strict=True refuses to run any tool unless layer 5 passes its self-test on this node.
"""
import json, os, resource, shutil, signal, subprocess, sys, tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
OFFLINE = {"python_calc", "search_paper", "get_reference_entry", "resolve_entry"}
LIMITS = {"python_calc": 5, "search_paper": 10, "get_reference_entry": 10, "resolve_entry": 10, "lookup": 45}
PY = sys.executable or "python3"
SYSTEM_BINDS = ["/usr", "/lib", "/lib32", "/lib64", "/bin", "/sbin"]
NET_BINDS = ["/etc/resolv.conf", "/etc/hosts", "/etc/nsswitch.conf", "/etc/host.conf", "/etc/gai.conf",
             "/etc/ssl", "/etc/ca-certificates", "/etc/pki"]
NET_ENV = ["HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy", "NO_PROXY", "no_proxy",
           "SSL_CERT_FILE", "SSL_CERT_DIR"]

# Runs inside the new user+mount namespace: build a minimal root and chroot into it.
_BOOT = r'''
R=$1; BINDS=$2; WORK=$3; shift 3
mount --make-rprivate / 2>/dev/null || true
mount -t tmpfs -o size=2m,mode=755 tmpfs "$R" || exit 97
for d in $BINDS; do
  [ -e "$d" ] || continue
  if [ -L "$d" ]; then mkdir -p "$R$(dirname "$d")"; ln -s "$(readlink "$d")" "$R$d"; continue; fi
  if [ -d "$d" ]; then mkdir -p "$R$d"; else mkdir -p "$R$(dirname "$d")"; : > "$R$d"; fi
  mount --rbind "$d" "$R$d" || exit 98
  mount -o remount,bind,ro "$R$d" 2>/dev/null || true
done
mkdir -p "$R/work" && mount --rbind "$WORK" "$R/work" || exit 99
ulimit -u 0 2>/dev/null || true
cd "$R" && exec chroot . "$@"
'''


class SandboxError(Exception):
    pass


def _unshare_ok(flags):
    if not shutil.which("unshare"):
        return False
    try:
        return subprocess.run(["unshare", "--user", "--map-root-user", *flags, "true"],
                              capture_output=True, timeout=10).returncode == 0
    except Exception:
        return False


def _py_prefixes():
    ps = {os.path.realpath(p) for p in (sys.prefix, sys.base_prefix, sys.exec_prefix, sys.base_exec_prefix)}
    ps.add(os.path.dirname(os.path.dirname(os.path.realpath(PY))))
    return sorted(p for p in ps if p not in ("/", "") and not any(p == b or p.startswith(b + "/") for b in SYSTEM_BINDS))


class Sandbox:
    def __init__(self, strict=True, allow_network=True):
        self.strict, self.allow_network = strict, allow_network
        self.userns_net = _unshare_ok(["--net"])
        self.userns_mnt = _unshare_ok(["--mount"])
        self.fs_hidden = self.fs_hidden_net = False
        if self.userns_net and self.userns_mnt:
            self.fs_hidden = self._selftest(offline=True)
            self.fs_hidden_net = self._selftest(offline=False)
        parts = []
        if self.userns_net:
            parts.append("user+net namespaces")
        if self.fs_hidden:
            parts.append("hidden filesystem (mount ns + chroot)")
        parts.append("rlimits")
        self.level = ", ".join(parts)
        if not allow_network:
            self.level += "; lookups disabled"
        elif not self.fs_hidden_net:
            self.level += "; lookup worker: filesystem NOT hidden"
        if strict and not self.fs_hidden:
            raise SandboxError("strict sandbox needs unprivileged user+net+mount namespaces with a working chroot "
                               f"(self-test failed on this node: userns_net={self.userns_net}, "
                               f"userns_mount={self.userns_mnt}); use --sandbox best-effort to accept weaker isolation")

    # ---- process plumbing
    @staticmethod
    def _limits(cpu, nproc):
        def f():
            resource.setrlimit(resource.RLIMIT_CPU, (cpu, cpu))
            resource.setrlimit(resource.RLIMIT_AS, (1 << 30, 1 << 30))
            resource.setrlimit(resource.RLIMIT_FSIZE, (0, 0))
            resource.setrlimit(resource.RLIMIT_NOFILE, (32, 32))
            resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
            if nproc:
                try:
                    resource.setrlimit(resource.RLIMIT_NPROC, (0, 0))
                except (ValueError, OSError):
                    pass
        return f

    def _command(self, wd, inner, offline, hide_fs):
        """Wrap `inner` (argv run inside the sandbox) according to the available layers.
        Returns (argv, apply_nproc_limit_in_preexec)."""
        if not hide_fs:
            if offline and self.userns_net:
                return ["unshare", "--user", "--map-root-user", "--net", *inner], False
            return inner, True
        root = os.path.join(wd, "root")
        os.mkdir(root)
        binds = SYSTEM_BINDS + _py_prefixes()
        if not offline:
            binds += NET_BINDS + [os.environ["SSL_CERT_FILE"]] if os.environ.get("SSL_CERT_FILE") else NET_BINDS
        flags = ["--user", "--map-root-user", "--mount"] + (["--net"] if offline else [])
        return ["unshare", *flags, "sh", "-c", _BOOT, "sbx", root,
                " ".join(b for b in binds if " " not in b), os.path.join(wd, "work"), *inner], False

    def _run(self, inner, payload, timeout, cpu, offline, hide_fs, stage_tools=True):
        wd = tempfile.mkdtemp(prefix="sbx-")
        try:
            os.mkdir(os.path.join(wd, "work"))
            if stage_tools:
                shutil.copy(os.path.join(HERE, "tools.py"), os.path.join(wd, "work", "tools.py"))
            cmd, nproc = self._command(wd, inner, offline, hide_fs)
            env = {"PATH": "/usr/sbin:/usr/bin:/sbin:/bin", "LANG": "C.UTF-8"}   # sbin: mount/chroot live there
            if not offline:
                env.update({k: os.environ[k] for k in NET_ENV if k in os.environ})
            p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                 cwd=os.path.join(wd, "work"), env=env, preexec_fn=self._limits(cpu, nproc),
                                 start_new_session=True)
            try:
                out, _ = p.communicate(payload.encode(), timeout=timeout)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                p.communicate()
                raise SandboxError(f"timed out after {timeout}s")
            finally:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except OSError:
                    pass
            return out[:50_000].decode(errors="replace")
        finally:
            shutil.rmtree(wd, ignore_errors=True)

    @staticmethod
    def _tool_argv(hide_fs):
        # inside the chroot the staged copy lives at /work/tools.py; otherwise use the real path
        return [PY, "-I", "-S", "-B", "/work/tools.py" if hide_fs else os.path.join(HERE, "tools.py")]

    def _selftest(self, offline):
        """Does the full isolation stack work here, and does it really hide the filesystem?"""
        try:
            probe = ("import os,json;print(json.dumps({'passwd':os.path.exists('/etc/passwd'),"
                     "'home':os.path.exists('/home')}))")
            seen = json.loads(self._run([PY, "-I", "-S", "-B", "-c", probe], "", 20, 10, offline, True,
                                        stage_tools=False))
            if seen["passwd"] and offline:          # offline workers must not see /etc
                return False
            res = json.loads(self._run(self._tool_argv(True), json.dumps({"tool": "python_calc", "arg": "6*7"}),
                                       20, 10, offline, True))
            return res.get("result", "").endswith("= 42")
        except Exception:
            return False

    # ---- public API
    def call(self, tool, arg="", **data):
        """Run one tool call; returns a result string. Failures come back as text, never as exceptions."""
        if tool not in LIMITS:
            return f"error: unknown tool {tool}"
        if tool == "lookup" and not self.allow_network:
            return "error: network lookups are disabled"
        offline = tool in OFFLINE
        hide = self.fs_hidden if offline else self.fs_hidden_net
        if self.strict and not hide:
            return "error: tool refused (strict sandbox not satisfied for this tool on this node)"
        payload = json.dumps({"tool": tool, "arg": arg, **data})
        try:
            raw = self._run(self._tool_argv(hide), payload, LIMITS[tool] + 5, LIMITS[tool], offline, hide)
            return str(json.loads(raw)["result"])[:3000]
        except SandboxError as e:
            return f"error: tool failed ({e})"
        except Exception as e:
            return f"error: tool failed ({type(e).__name__})"

    def probe(self, code, offline=True):
        """TEST HELPER: run arbitrary python under the same restrictions; returns its stdout."""
        hide = self.fs_hidden if offline else self.fs_hidden_net
        return self._run([PY, "-I", "-S", "-B", "-c", code], "", 15, 5, offline, hide, stage_tools=False)
