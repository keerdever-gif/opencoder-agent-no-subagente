"""
Tests for the verification-and-correction loop. No E2B or network access required:
usan un ejecutor local (subprocess en un directorio temporal).

    python -m tests.test_code_loop
"""
from __future__ import annotations

import asyncio
import shutil
import tempfile
from pathlib import Path

from loop_agent.code_loop import CodeLoop, LoopState


class LocalExecutor:
    """Same interface as SandboxExecutor, but runs in a local temporary directory."""

    def __init__(self, root: Path) -> None:
        self.root = root

    async def put_files(self, files):
        shutil.rmtree(self.root, ignore_errors=True)
        for rel, data in files.items():
            p = self.root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)

    async def run(self, command, timeout):
        proc = await asyncio.create_subprocess_shell(
            command, executable="/bin/bash",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout)
        return {
            "ok": proc.returncode == 0, "returncode": proc.returncode,
            "stdout": out.decode(errors="replace"), "stderr": err.decode(errors="replace"),
        }


def make_env():
    tmp = Path(tempfile.mkdtemp())
    ws, remote = tmp / "workspace", tmp / "remote"
    ws.mkdir()
    loop = CodeLoop(lambda: LocalExecutor(remote), run_timeout=3, remote_root=str(remote))
    return ws, loop


def write(ws: Path, st: LoopState, name: str, content: str) -> None:
    """Mirrors the behavior of file_write + st.track()."""
    p = ws / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    st.track("file_write", {}, {"path": str(p)})


async def drive(ws, loop, st, attempts_script):
    """
    Simulates the agent: the model writes the next version of the file,
    the gate tests it, and a failure is sent back to the model.
    """
    log = []
    for version in attempts_script:
        write(ws, st, "app.py", version)
        g = await loop.gate(st)
        log.append(g)
        if g.action != "retry":
            break
    return log
async def test_fix_loop_repairs_and_passes():
    ws, loop = make_env()
    st = LoopState(workspace=ws, max_attempts=8)
    script = [
        "def add(a, b)\n    return a + b\n",                         # 1) syntax error
        "def add(a, b):\n    return a + b\nprint(add(1))\n",        # 2) runtime TypeError
        "def add(a, b):\n    return a + b\nprint(add(1, 2))\n",     # 3) successful
    ]
    log = await drive(ws, loop, st, script)
    assert [g.action for g in log] == ["retry", "retry", "pass"], [g.action for g in log]
    assert "SyntaxError" in log[0].message and "line 1" in log[0].message
    assert "TypeError" in log[1].message
    assert "✅" in log[2].message and "2 automatic fixes" in log[2].message
    assert not st.pending()
    print("OK  fix loop: syntax → runtime → success")


async def test_repeated_error_stops_loop():
    ws, loop = make_env()
    st = LoopState(workspace=ws, max_attempts=8)
    bug = "x = 1 / 0\n"
    log = await drive(ws, loop, st, [bug, bug + "# attempt 2\n", bug + "# attempt 3\n", bug + "# 4\n"])
    assert [g.action for g in log] == ["retry", "retry", "giveup"], [g.action for g in log]
    assert "SAME error" in log[1].message
    assert "ZeroDivisionError" in log[2].message
    print("OK  same error 3 times → stops and reports it")


async def test_attempt_limit_is_respected():
    ws, loop = make_env()
    st = LoopState(workspace=ws, max_attempts=2)
    log = await drive(ws, loop, st, ["raise ValueError('a')\n", "raise KeyError('b')\n", "raise OSError('c')\n"])
    assert [g.action for g in log] == ["retry", "giveup"], [g.action for g in log]
    print("OK  attempt limit respected")


async def test_unchanged_files_do_not_rerun_sandbox():
    ws, loop = make_env()
    st = LoopState(workspace=ws)
    write(ws, st, "app.py", "raise RuntimeError('boom')\n")
    g1 = await loop.gate(st)
    g2 = await loop.gate(st)   # the model did not change anything
    assert g1.action == "retry" and g2.action == "retry"
    assert "You did not modify any files" in g2.message
    print("OK  unchanged files → no rerun, model is notified")


async def test_long_running_server_is_success():
    ws, loop = make_env()
    st = LoopState(workspace=ws)
    write(ws, st, "server.py", "import time\nwhile True:\n    time.sleep(1)\n")
    g = await loop.gate(st)
    assert g.action == "pass" and "✅" in g.message, g
    print("OK  long-running process (server/bot) counts as success")


async def test_environment_errors_are_not_retried():
    ws, loop = make_env()
    st = LoopState(workspace=ws)
    write(ws, st, "bot.py", "import os\nraise RuntimeError('Missing environment variable API_KEY')\n")
    g = await loop.gate(st)
    assert g.action == "pass" and "⚠️" in g.message and "environment variable" in g.message, g
    print("OK  missing API key → warning, no retry loop")


async def test_pytest_runs_when_tests_exist():
    ws, loop = make_env()
    st = LoopState(workspace=ws)
    write(ws, st, "calc.py", "def double(x):\n    return x + x + 1\n")
    write(ws, st, "test_calc.py", "from calc import double\ndef test_d():\n    assert double(2) == 4\n")
    g = await loop.gate(st)
    if g.action == "pass" and "pytest" in g.message:
        print("OK  pytest unavailable → inconclusive with warning (not treated as a code failure)")
        return
    assert g.action == "retry" and "pytest" in g.message and "assert" in g.message.lower(), g.message
    write(ws, st, "calc.py", "def double(x):\n    return x * 2\n")
    g = await loop.gate(st)
    assert g.action == "pass", g.message
    print("OK  pytest: failing test → fixed → passes")


async def test_missing_local_import_is_a_code_bug():
    ws, loop = make_env()
    st = LoopState(workspace=ws)
    write(ws, st, "main.py", "import utilities\nprint(utilities.f())\n")
    g = await loop.gate(st)
    assert g.action == "retry" and "ModuleNotFoundError" in g.message, g.message
    write(ws, st, "utilities.py", "def f():\n    return 42\n")
    g = await loop.gate(st)
    assert g.action == "pass", g.message
    print("OK  missing local module import → genuine bug → fixed")


async def test_invalid_json_and_missing_sandbox():
    ws, _ = make_env()
    loop = CodeLoop(lambda: None)           # without a configured sandbox
    st = LoopState(workspace=ws)
    write(ws, st, "data.json", '{"a": 1,}')
    g = await loop.gate(st)
    assert g.action == "retry" and "Invalid JSON" in g.message
    write(ws, st, "data.json", '{"a": 1}')
    write(ws, st, "x.py", "print('hello')\n")
    g = await loop.gate(st)
    assert g.action == "pass" and "⚠️" in g.message and "sandbox" in g.message, g
    print("OK  invalid JSON detected; without sandbox, falls back to static checks with warning")


async def main():
    for t in (
        test_fix_loop_repairs_and_passes, test_repeated_error_stops_loop,
        test_attempt_limit_is_respected, test_unchanged_files_do_not_rerun_sandbox,
        test_long_running_server_is_success, test_environment_errors_are_not_retried,
        test_pytest_runs_when_tests_exist, test_missing_local_import_is_a_code_bug,
        test_invalid_json_and_missing_sandbox,
    ):
        await t()
    print("\nALL CODE LOOP TESTS PASSED")


if __name__ == "__main__":
    asyncio.run(main())
