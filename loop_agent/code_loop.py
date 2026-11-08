"""
Autonomous code verification and correction loop.

Flow (agent.py orchestrates this module):

    the model writes or edits files
            │
            ▼
    ┌─► GATE: the model is ready to finish → always verify the code
    │       1. local static checks (syntax, JSON)
    │       2. upload files to the sandbox
    │       3. install dependencies and run tests / the program
    │       │
    │       ├─ OK ────────────► stop the loop and return the response
    │       └─ FAIL ─► send the error back to the model as a message
    │                    │  the model fixes it (removes / adds / changes)
    └────────────────────┘

Safeguards that ensure the loop ALWAYS terminates:
  - maximum attempts (MAX_FIX_ATTEMPTS)
  - same error 3 times in a row → stop and report it honestly
  - environment errors (missing API key, no database, etc.) are not bugs → do not retry them
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Protocol

REMOTE_ROOT = "/home/user/project"

VERIFIABLE_EXT = {".py", ".js", ".mjs", ".json"}
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".venv", "venv", "generated", ".pytest_cache"}
MAX_SYNC_FILES = 150
MAX_SYNC_BYTES = 512 * 1024
TAIL_CHARS = 12000
SAME_ERROR_LIMIT = 3

ENTRY_NAMES = {"main", "app", "run", "bot", "server", "index", "cli", "__main__"}

# Conservative allowlist: never install arbitrary module names with pip/npm
# extracted from a traceback (it could be a typo, a local module, or a malicious package).
PIP_NAMES = {
    "cv2": "opencv-python-headless",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "bs4": "beautifulsoup4",
    "sklearn": "scikit-learn",
    "dotenv": "python-dotenv",
    "discord": "discord.py",
    "dateutil": "python-dateutil",
    "serial": "pyserial",
    "jwt": "pyjwt",
    "requests": "requests",
    "aiohttp": "aiohttp",
    "numpy": "numpy",
    "pandas": "pandas",
    "matplotlib": "matplotlib",
    "flask": "flask",
    "fastapi": "fastapi",
    "uvicorn": "uvicorn",
    "pydantic": "pydantic",
    "sqlalchemy": "sqlalchemy",
    "httpx": "httpx",
    "openai": "openai",
    "anthropic": "anthropic",
    "redis": "redis",
    "pytest": "pytest",
    "lxml": "lxml",
    "openpyxl": "openpyxl",
    "docx": "python-docx",
    "aiofiles": "aiofiles",
}

# Errors caused by the sandbox environment rather than the code.
_ENV_ERROR = re.compile(
    r"(Missing environment variable|KeyError: ['\"][A-Z][A-Z0-9_]{2,}['\"]|"
    r"ConnectionRefusedError|Connection refused|could not connect to server|"
    r"Name or service not known|getaddrinfo failed|"
    r"(api[_ ]?key|token|secret|credentials?)\b.{0,40}\b(missing|not set|required|invalid))",
    re.IGNORECASE,
)
# Executor (allows tests to run without E2B)
class Executor(Protocol):
    async def put_files(self, files: dict[str, bytes]) -> None: ...
    async def run(self, command: str, timeout: float) -> dict[str, Any]: ...


class SandboxExecutor:
    """Ejecutor real: el sandbox E2B de sandbox.py."""

    def __init__(self, sb: Any, remote_root: str = REMOTE_ROOT) -> None:
        self.sb = sb
        self.remote_root = remote_root

    async def put_files(self, files: dict[str, bytes]) -> None:
        # Start clean for each verification to avoid leftovers from previous attempts.
        await self.sb.run_shell(
            f"rm -rf {shlex.quote(self.remote_root)} && mkdir -p {shlex.quote(self.remote_root)}",
            allow_dangerous=True,
        )
        res = await self.sb.put_files(
            {f"{self.remote_root}/{rel}": data for rel, data in files.items()}
        )
        if not res.get("ok"):
            raise RuntimeError(f"Could not upload files to the sandbox: {res.get('error')}")

    async def run(self, command: str, timeout: float) -> dict[str, Any]:
        return await self.sb.run_shell(command, timeout=timeout)
# Results
@dataclass
class Check:
    name: str
    ok: bool
    command: str = ""
    returncode: int = 0
    stdout: str = ""
    stderr: str = ""
    inconclusive: bool = False
    note: str = ""


@dataclass
class VerifyReport:
    checks: list[Check] = field(default_factory=list)
    sandbox_used: bool = False
    notes: list[str] = field(default_factory=list)
    reused: bool = False

    @property
    def failed(self) -> Check | None:
        for c in self.checks:
            if not c.ok and not c.inconclusive:
                return c
        return None

    @property
    def status(self) -> str:
        if self.failed:
            return "fail"
        if any(c.inconclusive for c in self.checks) or not self.sandbox_used:
            return "inconclusive"
        return "ok"

    @property
    def signature(self) -> str:
        c = self.failed
        if not c:
            return ""
        combined = "\n".join(part for part in (c.stdout, c.stderr) if part)
        lines = [l.strip() for l in combined.splitlines() if l.strip()]
        last = lines[-1] if lines else ""
        last = re.sub(r"0x[0-9a-fA-F]+|\d+(\.\d+)?s\b", "", last)
        return hashlib.blake2b(f"{c.name}|{last}".encode("utf-8", "ignore"), digest_size=6).hexdigest()

    def feedback(self, attempt: int, max_attempts: int, repeated: int) -> str:
        c = self.failed
        assert c is not None
        # Keep both streams; some tools write tracebacks to stdout.
        combined_output = "\n".join(part for part in (c.stdout, c.stderr) if part)
        body = _tail(combined_output or "(no output)")
        parts = [
            f"🔁 AUTOMATIC VERIFICATION — attempt {attempt}/{max_attempts}: FAILED",
            f"Check: {c.name}",
        ]
        if c.command:
            parts.append(f"Command: {c.command}")
        parts.append(f"Exit code: {c.returncode}")
        if self.notes:
            parts.append("Notes: " + " | ".join(self.notes))
        parts += ["--- error ---", body, "--- end of error ---"]
        if self.reused:
            parts.append(
                "⚠️ You did not modify any files since the previous attempt, "
                "so the result is identical."
            )
        elif repeated >= 2:
            parts.append(
                f"⚠️ The SAME error has occurred {repeated}ª times: the previous fix did not work. "
                "Try a different approach and read the file with file_read before editing it."
            )
        parts.append(
            "INSTRUCTIONS: find the root cause in the code and fix it with "
            "file_replace_many (targeted edits) or file_write (full rewrite). "
            "Do not just explain or ask for permission: fix it and the system will run the verification again. "
            "Do not remove functionality to work around the error. "
            "If the failure is caused by an environment variable, API key, or external service "
            "that is unavailable in the sandbox, it is not a code bug: do not invent values."
        )
        return "\n".join(parts)

    def success_note(self, attempts_used: int) -> str:
        n = sum(1 for c in self.checks if c.ok)
        txt = f"✅ Verified in sandbox ({n} checks passed"
        txt += f", {attempts_used} automatic fixes)" if attempts_used else ")"
        return txt

    def inconclusive_note(self) -> str:
        why = [c.note or c.name for c in self.checks if c.inconclusive]
        if not self.sandbox_used:
            why = self.notes or ["sandbox unavailable; only syntax was checked"]
        return "⚠️ I could not fully verify the result: " + "; ".join(why)


def _tail(text: str, limit: int = TAIL_CHARS) -> str:
    text = text.strip()
    return text if len(text) <= limit else "[...]\n" + text[-limit:]
# Estado por turno
@dataclass
class LoopState:
    workspace: Path
    max_attempts: int = 8
    written: dict[str, None] = field(default_factory=dict)   # conjunto ordenado
    dirty: bool = False
    attempts: int = 0                                         # verificaciones fallidas
    sig_counts: dict[str, int] = field(default_factory=dict)
    last_report: VerifyReport | None = None

    def pending(self) -> bool:
        return bool(self.written)

    def mark_written(self, path: str) -> None:
        p = Path(path)
        if p.is_absolute():
            try:
                p = p.relative_to(self.workspace)
            except ValueError:
                return
        if p.suffix.lower() in VERIFIABLE_EXT:
            self.written[str(p)] = None
            self.dirty = True

    def register_failure(self, report: VerifyReport) -> int:
        self.attempts += 1
        sig = report.signature
        self.sig_counts[sig] = self.sig_counts.get(sig, 0) + 1
        self.last_report = report
        return self.sig_counts[sig]

    def exhausted(self) -> bool:
        return (
            self.attempts >= self.max_attempts
            or any(n >= SAME_ERROR_LIMIT for n in self.sig_counts.values())
        )

    def giveup_text(self, report: VerifyReport) -> str:
        c = report.failed
        why = (
            "the same error occurred three times"
            if any(n >= SAME_ERROR_LIMIT for n in self.sig_counts.values())
            else f"I used all {self.max_attempts} attempts"
        )
        combined_output = "\n".join(part for part in (c.stdout, c.stderr) if part) if c else ""
        err = _tail(combined_output or "(no output)", 6000)
        return (
            f"\n\n❌ I could not get it working: {why}.\n"
            f"Last error ({c.name if c else '?'}):\n{err}"
        )

    def track(self, tool_name: str, args: dict[str, Any], result: Any) -> None:
        """Record which code files a tool modified."""
        if not isinstance(result, dict) or result.get("error"):
            return
        if tool_name in ("file_write", "file_append"):
            if result.get("path"):
                self.mark_written(result["path"])
        elif tool_name == "file_replace_many":
            if result.get("changed") and result.get("path"):
                self.mark_written(result["path"])


@dataclass
class Gate:
    """Gate decision made when the model is ready to return its final response."""
    action: str      # "skip" | "pass" | "retry" | "giveup"
    message: str = ""  # retry: message for the model · pass/giveup: text to append to the response
# Verificador
class CodeLoop:
    def __init__(
        self,
        executor_factory: Callable[[], Executor | None],
        run_timeout: int = 20,
        remote_root: str = REMOTE_ROOT,
    ) -> None:
        self.executor_factory = executor_factory
        self.run_timeout = run_timeout
        self.remote_root = remote_root

    # ---------------- API pública ----------------

    async def gate(self, st: LoopState) -> Gate:
        """Called when the model no longer requests tools."""
        if not st.pending():
            return Gate("skip")
        report = await self.verify(st)
        if report.status == "fail":
            repeated = st.register_failure(report)
            if st.exhausted():
                return Gate("giveup", st.giveup_text(report))
            return Gate("retry", report.feedback(st.attempts, st.max_attempts, repeated))
        st.written.clear()   # Verification complete.
        if report.status == "ok":
            return Gate("pass", "\n\n" + report.success_note(st.attempts))
        return Gate("pass", "\n\n" + report.inconclusive_note())

    async def verify(self, st: LoopState) -> VerifyReport:
        if not st.dirty and st.last_report is not None and st.last_report.failed:
            st.last_report.reused = True
            return st.last_report

        report = VerifyReport()
        files = {
            rel: st.workspace / rel
            for rel in st.written
            if (st.workspace / rel).is_file()
        }
        st.dirty = False
        if not files:
            report.notes.append("there are no code files to verify")
            return report

        # 1) Static, local, free
        for rel, path in files.items():
            err = self._static_check(rel, path)
            if err:
                report.checks.append(Check(name=f"sintaxis {rel}", ok=False, returncode=1, stderr=err))
        if report.failed:
            report.notes.append("static checks failed; sandbox resources were not used")
            return report
        for rel in files:
            report.checks.append(Check(name=f"sintaxis {rel}", ok=True))

        runnable = [r for r in files if Path(r).suffix.lower() in {".py", ".js", ".mjs"}]
        if not runnable:
            report.notes.append("only data files were present; their formats were validated")
            return report

        # 2) Sandbox
        try:
            ex = self.executor_factory()
        except Exception as exc:  # for example, E2B_API_KEY is missing
            ex = None
            report.notes.append(f"sandbox unavailable ({exc})")
        if ex is None:
            if not report.notes:
                report.notes.append("sandbox is not configured")
            return report

        try:
            project = self._collect_project(files, st.workspace)
            await ex.put_files({rel: p.read_bytes() for rel, p in project.items()})
            report.sandbox_used = True
            await self._run_checks(ex, report, files, project)
        except Exception as exc:
            report.sandbox_used = False
            report.notes.append(f"error communicating with the sandbox: {type(exc).__name__}: {exc}")
        return report

    # ---------------- internos ----------------

    @staticmethod
    def _static_check(rel: str, path: Path) -> str:
        ext = path.suffix.lower()
        try:
            text = path.read_text(encoding="utf-8")
        except Exception as exc:
            return f"{rel}: could not be read ({exc})"
        if ext == ".py":
            try:
                ast.parse(text, filename=rel)
            except SyntaxError as exc:
                line = (exc.text or "").rstrip("\n")
                caret = " " * max((exc.offset or 1) - 1, 0) + "^"
                return f"SyntaxError in {rel}, line {exc.lineno}: {exc.msg}\n    {line}\n    {caret}"
        elif ext == ".json":
            try:
                json.loads(text)
            except json.JSONDecodeError as exc:
                return f"Invalid JSON in {rel}, line {exc.lineno}, col {exc.colno}: {exc.msg}"
        return ""

    def _collect_project(self, written: dict[str, Path], ws: Path) -> dict[str, Path]:
        """Written files plus files from the same project (for imports/tests)."""
        out: dict[str, Path] = dict(written)
        roots: set[str | None] = {
            (Path(rel).parts[0] if len(Path(rel).parts) > 1 else None) for rel in written
        }
        for root in roots:
            if root is None:
                candidates = [p for p in ws.iterdir() if p.is_file()]
            else:
                candidates = [p for p in (ws / root).rglob("*") if p.is_file()]
            for p in candidates:
                if len(out) >= MAX_SYNC_FILES:
                    break
                rel_parts = p.relative_to(ws).parts
                if any(part in SKIP_DIRS for part in rel_parts):
                    continue
                try:
                    if p.stat().st_size > MAX_SYNC_BYTES:
                        continue
                except OSError:
                    continue
                out.setdefault(str(p.relative_to(ws)), p)
        return out

    async def _exec(self, ex: Executor, cmd: str, cwd: str, timeout: float) -> dict[str, Any]:
        full = (
            f"cd {shlex.quote(cwd)} && export PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 && {cmd}"
        )
        r = await ex.run(full, timeout)
        return {
            "rc": r.get("returncode", -1 if not r.get("ok") else 0),
            "stdout": r.get("stdout", "") or "",
            "stderr": r.get("stderr", "") or "",
            "error": r.get("error"),
        }

    def _cwd(self, rel_dir: str) -> str:
        return f"{self.remote_root}/{rel_dir}".rstrip("/") if rel_dir not in ("", ".") else self.remote_root

    async def _run_checks(
        self, ex: Executor, report: VerifyReport,
        written: dict[str, Path], project: dict[str, Path],
    ) -> None:
        t = self.run_timeout
        py_written = [r for r in written if r.endswith(".py")]
        js_written = [r for r in written if r.endswith((".js", ".mjs"))]
        local_modules = {Path(r).stem for r in project if r.endswith(".py")}

        # --- dependencias ---
        if py_written and "requirements.txt" in project:
            r = await self._exec(ex, "python -m pip install -q -r requirements.txt", self.remote_root, 300)
            report.checks.append(Check(
                "instalar requirements.txt", r["rc"] == 0,
                "pip install -r requirements.txt", r["rc"], r["stdout"], r["stderr"],
            ))
            if r["rc"] != 0:
                return
        if js_written and "package.json" in project:
            r = await self._exec(ex, "npm install --no-audit --no-fund --silent", self.remote_root, 300)
            report.checks.append(Check(
                "npm install", r["rc"] == 0, "npm install", r["rc"], r["stdout"], r["stderr"],
            ))
            if r["rc"] != 0:
                return

        # --- Python ---
        if py_written:
            test_files = [
                r for r in project
                if r.endswith(".py") and (Path(r).name.startswith("test_") or Path(r).name.endswith("_test.py"))
            ]
            if test_files:
                await self._pytest(ex, report)
            else:
                for rel in self._pick_entries(py_written, written):
                    await self._run_script(ex, report, rel, local_modules)
                    if report.failed:
                        return
            if report.failed:
                return

        # --- JavaScript ---
        for rel in js_written:
            cwd = self._cwd(str(Path(rel).parent))
            name = Path(rel).name
            r = await self._exec(ex, f"node --check {shlex.quote(name)}", cwd, 30)
            report.checks.append(Check(
                f"node --check {rel}", r["rc"] == 0, f"node --check {name}", r["rc"], r["stdout"], r["stderr"],
            ))
            if r["rc"] != 0:
                return
        for rel in self._pick_entries(js_written, written):
            await self._run_script(ex, report, rel, set(), kind="node")
            if report.failed:
                return

    @staticmethod
    def _pick_entries(candidates: list[str], written: dict[str, Path]) -> list[str]:
        """When many files changed, test only the entry points."""
        if len(candidates) <= 3:
            return candidates
        entries = []
        for rel in candidates:
            stem = Path(rel).stem
            try:
                has_main = "__main__" in written[rel].read_text(encoding="utf-8", errors="ignore")
            except Exception:
                has_main = False
            if stem in ENTRY_NAMES or has_main:
                entries.append(rel)
        return entries[:5] or candidates[:3]

    async def _pytest(self, ex: Executor, report: VerifyReport) -> None:
        await self._exec(
            ex, "python -m pytest --version >/dev/null 2>&1 || python -m pip install -q pytest",
            self.remote_root, 180,
        )
        cmd = "python -m pytest -x -q --no-header -p no:cacheprovider"
        r = await self._exec(ex, cmd, self.remote_root, max(self.run_timeout * 3, 90))
        ok = r["rc"] in (0, 5)   # 5 = no tests were collected
        c = Check("pytest", ok, cmd, r["rc"], r["stdout"], r["stderr"] or (r["stdout"] if not ok else ""))
        out = r["stdout"] + r["stderr"]
        if not ok and "No module named pytest" in out:
            c.ok, c.inconclusive = False, True
            c.note = "pytest could not be installed in the sandbox; tests were not run"
        elif not ok and _ENV_ERROR.search(out):
            c.ok, c.inconclusive = False, True
            c.note = "the tests require an environment variable or external service unavailable in the sandbox"
        report.checks.append(c)

    async def _run_script(
        self, ex: Executor, report: VerifyReport, rel: str,
        local_modules: set[str], kind: str = "python",
    ) -> None:
        name = Path(rel).name
        cwd = self._cwd(str(Path(rel).parent))
        t = self.run_timeout
        interp = "python" if kind == "python" else "node"
        cmd = f"timeout {t} {interp} {shlex.quote(name)} < /dev/null"
        installed: list[str] = []
        r: dict[str, Any] = {"rc": 0, "stdout": "", "stderr": ""}

        for _ in range(4):   # retries only for installing missing dependencies
            r = await self._exec(ex, cmd, cwd, t + 60)
            if r["rc"] in (0, 124):
                break
            missing = self._missing_module(r["stderr"], kind)
            if not missing or missing in local_modules:
                break
            # Node packages must be declared in package.json and are installed
            # during dependency setup; packages are not guessed or downloaded here.
            if kind != "python":
                break
            pkg = PIP_NAMES.get(missing)
            if not pkg:
                # Return the exact ModuleNotFoundError to the model so it can
                # add an explicit dependency or fix a local import.
                break
            install = f"python -m pip install -q {shlex.quote(pkg)}"
            ins = await self._exec(ex, install, self.remote_root if kind == "node" else cwd, 180)
            if ins["rc"] != 0:
                r["stderr"] += f"\n[automatic installation of '{pkg}' failed: {ins['stderr'][-300:]}]"
                break
            installed.append(pkg)

        out = r["stdout"] + r["stderr"]
        ok = r["rc"] in (0, 124)
        note = ""
        if r["rc"] == 124:
            note = f"kept running {t}s without failing (normal for servers and bots)"
        if not ok and "EOFError" in out and "input(" in (r.get("stderr", "")):
            ok, note = True, "requests keyboard input (expected EOF)"
        c = Check(f"run {rel}", ok, f"{interp} {name}", r["rc"], r["stdout"], r["stderr"], note=note)
        if not ok and _ENV_ERROR.search(out):
            c.inconclusive = True
            c.note = f"{rel} requires an environment variable or external service unavailable in the sandbox"
        if installed:
            report.notes.append("Dependencies installed automatically: " + ", ".join(installed))
        report.checks.append(c)

    @staticmethod
    def _missing_module(stderr: str, kind: str) -> str | None:
        if kind == "python":
            m = re.search(r"No module named '([\w\.\-]+)'", stderr)
            return m.group(1).split(".")[0] if m else None
        m = re.search(r"Cannot find module '([^'./][^']*)'", stderr)   # ignore relative paths
        return m.group(1) if m else None
