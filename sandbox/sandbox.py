from __future__ import annotations

import asyncio
import ast
import base64
import hashlib
import json
import os
import re
import shlex
import tempfile
import time
from pathlib import Path
from typing import Any

import aiohttp

# e2b-code-interpreter 1.x re-exports AsyncSandbox. If only the base SDK
# is installed, use its AsyncSandbox and run Python through commands.run.
try:
    from e2b_code_interpreter import AsyncSandbox  # type: ignore
    _HAS_CODE_INTERPRETER = True
except ModuleNotFoundError as import_error:
    if import_error.name != "e2b_code_interpreter":
        raise
    try:
        from e2b import AsyncSandbox  # type: ignore
    except ModuleNotFoundError:
        AsyncSandbox = None  # type: ignore[assignment,misc]
    _HAS_CODE_INTERPRETER = False


class Sandbox:

    def __init__(
        self,
        timeout: float = 30.0,
        max_output: int = 100000,
        max_download: int = 50 * 1024 * 1024,
        max_retries: int = 2,
    ):
        self.timeout = timeout
        self.max_output = max_output
        self.max_download = max_download
        self.max_retries = max_retries
        self._sandbox: Any | None = None
        self._lock = asyncio.Lock()
        self._deploy_lock = asyncio.Lock()
        self.sandbox_timeout_seconds = 3600

    def _success(self, **data: Any) -> dict[str, Any]:
        return {"ok": True, **data}

    def _error(self, operation: str, exc: Exception | str, *, retryable: bool = False, **data: Any) -> dict[str, Any]:
        if isinstance(exc, Exception):
            error_type = type(exc).__name__
            message = str(exc)
        else:
            error_type = "Error"
            message = str(exc)
        return {
            "ok": False,
            "error": {"type": error_type, "message": message, "operation": operation, "retryable": retryable},
            **data,
        }

    def _limit(self, value: Any) -> str:
        return str(value or "")[: self.max_output]

    async def _get_sandbox(self) -> Any:
        async with self._lock:
            if self._sandbox is not None:
                return self._sandbox
            api_key = os.getenv("E2B_API_KEY")
            if not api_key:
                raise RuntimeError("E2B_API_KEY is not configured in the environment.")
            if AsyncSandbox is None:
                raise RuntimeError(
                    "E2B SDK is not installed. Run: pip install 'e2b-code-interpreter>=1.0,<2'"
                )
            self._sandbox = await AsyncSandbox.create(
                timeout=self.sandbox_timeout_seconds,
                api_key=api_key,
                cpu_count=4,
                memory_mb=8192,
            )
            return self._sandbox

    async def close(self) -> None:
        async with self._lock:
            if self._sandbox is None:
                return
            try:
                await self._sandbox.kill()
            except Exception:
                pass
            finally:
                self._sandbox = None

    async def _reset_sandbox(self) -> None:
        await self.close()
        await self._get_sandbox()

    async def health_check(self) -> dict[str, Any]:
        start = time.monotonic()
        try:
            sandbox = await self._get_sandbox()
            result = await sandbox.commands.run("printf 'SANDBOX_OK\\n'", timeout=10)
            duration = time.monotonic() - start
            if result.exit_code != 0:
                return self._error("health_check", "The sandbox returned a non-zero exit code.",
                                   retryable=True, stdout=self._limit(result.stdout),
                                   stderr=self._limit(result.stderr), returncode=result.exit_code, duration=round(duration, 3))
            return self._success(operation="health_check", status="healthy",
                                 stdout=self._limit(result.stdout), stderr=self._limit(result.stderr),
                                 returncode=result.exit_code, duration=round(duration, 3))
        except Exception as exc:
            try:
                await self._reset_sandbox()
            except Exception:
                pass
            return self._error("health_check", exc, retryable=True, duration=round(time.monotonic() - start, 3))

    async def run_python(self, code: str) -> dict[str, Any]:
        if not code.strip():
            return self._error("run_python", "No Python code was provided.")
        start = time.monotonic()
        for attempt in range(self.max_retries + 1):
            try:
                sandbox = await self._get_sandbox()
                if callable(getattr(sandbox, "run_code", None)):
                    execution = await sandbox.run_code(code, timeout=self.timeout)
                    stdout = self._limit(execution.text)
                    stderr = ""
                    returncode = 0
                    if execution.error:
                        stderr = f"{execution.error.name}: {execution.error.value}"
                        if execution.error.traceback:
                            stderr += "\n" + execution.error.traceback
                        stderr = self._limit(stderr)
                        returncode = 1
                else:
                    encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
                    python_expr = (
                        "import base64; exec(compile(base64.b64decode(" + repr(encoded)
                        + ").decode('utf-8'), '<sandbox>', 'exec'))"
                    )
                    command = "python3 -c " + shlex.quote(python_expr)
                    result = await self.run_shell(command, allow_dangerous=True, timeout=self.timeout)
                    stdout = self._limit(result.get("stdout", ""))
                    stderr = self._limit(result.get("stderr", ""))
                    returncode = int(result.get("returncode", 1))
                return {
                    "ok": returncode == 0,
                    "operation": "run_python",
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": returncode,
                    "duration": round(time.monotonic() - start, 3),
                    "attempt": attempt + 1,
                }
            except Exception as exc:
                if attempt < self.max_retries:
                    await asyncio.sleep(0.5 * (attempt + 1))
                    try:
                        await self._reset_sandbox()
                    except Exception:
                        pass
                    continue
                try:
                    await self._reset_sandbox()
                except Exception:
                    pass
                return self._error("run_python", exc, retryable=True,
                                   stdout="", stderr="", returncode=-1,
                                   duration=round(time.monotonic() - start, 3), attempt=attempt + 1)
        return self._error("run_python", "Execution timed out.", retryable=True)

    async def run_shell(self, command: str, *, allow_dangerous: bool = False,
                        timeout: float | None = None) -> dict[str, Any]:
        if not command.strip():
            return self._error("run_shell", "No command was provided.")
        if not allow_dangerous:
            blocked = self._check_dangerous_command(command)
            if blocked:
                return self._error("run_shell", blocked, retryable=False)
        start = time.monotonic()
        try:
            sandbox = await self._get_sandbox()
            result = await sandbox.commands.run(command, timeout=timeout or self.timeout)
            return {
                "ok": result.exit_code == 0,
                "operation": "run_shell",
                "command": command,
                "stdout": self._limit(result.stdout),
                "stderr": self._limit(result.stderr),
                "returncode": result.exit_code,
                "duration": round(time.monotonic() - start, 3),
            }
        except Exception as exc:
            # E2B lanza CommandExitException when the command exits with
            # a non-zero exit code. This is NOT a sandbox failure; it is a normal
            # result (the program failed), and the sandbox must NOT be reset.
            if hasattr(exc, "exit_code") and hasattr(exc, "stderr"):
                return {
                    "ok": False,
                    "operation": "run_shell",
                    "command": command,
                    "stdout": self._limit(getattr(exc, "stdout", "")),
                    "stderr": self._limit(getattr(exc, "stderr", "")),
                    "returncode": getattr(exc, "exit_code", 1),
                    "duration": round(time.monotonic() - start, 3),
                }
            await self._safe_reset()
            return self._error("run_shell", exc, retryable=True,
                               stdout="", stderr="", returncode=-1,
                               duration=round(time.monotonic() - start, 3))

    async def deploy_web(self, html_content: str) -> dict[str, Any]:
        """Serves a public static website on port 8080 of the current sandbox."""
        if not isinstance(html_content, str) or not html_content.strip():
            return self._error("deploy_web", "html_content is empty.")
        if len(html_content.encode("utf-8")) > 1_000_000:
            return self._error("deploy_web", "HTML exceeds the 1 MB limit.")

        async with self._deploy_lock:
            try:
                sandbox = await self._get_sandbox()
                web_root = "/tmp/opencoder-agent-temporary-website"
                await sandbox.commands.run(f"mkdir -p {shlex.quote(web_root)}", timeout=10)
                await sandbox.files.write(f"{web_root}/index.html", html_content.encode("utf-8"))

                socket_code = (
                    "import socket,sys; s=socket.socket(); s.settimeout(0.5); "
                    "sys.exit(0 if s.connect_ex(('127.0.0.1',8080)) == 0 else 1)"
                )
                probe = await sandbox.commands.run("python3 -c " + shlex.quote(socket_code), timeout=5)
                if int(getattr(probe, "exit_code", 1)) != 0:
                    start_command = (
                        "nohup python3 -m http.server 8080 --bind 0.0.0.0 "
                        f"--directory {shlex.quote(web_root)} "
                        ">/tmp/opencoder-agent-http-8080.log 2>&1 < /dev/null & echo $!"
                    )
                    started = await sandbox.commands.run(start_command, timeout=10)
                    if int(getattr(started, "exit_code", 1)) != 0:
                        raise RuntimeError(
                            "Could not start the HTTP server: "
                            + self._limit(getattr(started, "stderr", ""))
                        )

                readiness_code = (
                    "import time,urllib.request\n"
                    "for attempt in range(20):\n"
                    "    try:\n"
                    "        response = urllib.request.urlopen('http://127.0.0.1:8080/', timeout=1)\n"
                    "        assert response.status == 200\n"
                    "        break\n"
                    "    except Exception:\n"
                    "        time.sleep(0.25)\n"
                    "else:\n"
                    "    raise SystemExit(1)\n"
                )
                ready = await sandbox.commands.run(
                    "python3 -c " + shlex.quote(readiness_code), timeout=12
                )
                if int(getattr(ready, "exit_code", 1)) != 0:
                    log = await sandbox.commands.run(
                        "tail -n 40 /tmp/opencoder-agent-http-8080.log 2>/dev/null || true", timeout=5
                    )
                    raise RuntimeError(
                        "The server did not respond on port 8080. "
                        + self._limit(getattr(log, "stdout", ""))
                    )

                host = sandbox.get_host(8080)
                if hasattr(host, "__await__"):
                    host = await host
                url = str(host)
                if not url.startswith(("http://", "https://")):
                    url = "https://" + url
                return self._success(
                    operation="deploy_web",
                    url=url,
                    port=8080,
                    path=f"{web_root}/index.html",
                    max_lifetime_seconds=self.sandbox_timeout_seconds,
                    warning=(
                        "URL temporal: desaparece al apagar o expirar el sandbox. "
                        f"The configured maximum is {self.sandbox_timeout_seconds} seconds "
                        "from sandbox creation; remaining availability may be shorter. "
                        "Availability for seven days is not guaranteed."
                    ),
                )
            except Exception as exc:
                return self._error("deploy_web", exc, retryable=True)

    def _check_dangerous_command(self, command: str) -> str | None:
        normalized = command.strip().lower()
        dangerous_patterns = [
            r"\brm\s+-rf\s+/", r"\brm\s+-rf\s+\*", r"\bmkfs\b",
            r"\bshutdown\b", r"\breboot\b", r"\bpoweroff\b",
            r"\binit\s+0\b", r"\bdd\s+if=",
        ]
        for pattern in dangerous_patterns:
            if re.search(pattern, normalized):
                return "Command blocked for safety. Set allow_dangerous=True only when execution is necessary."
        return None

    async def run_bash(self, script: str, *, allow_dangerous: bool = False) -> dict[str, Any]:
        return await self.run_shell(script, allow_dangerous=allow_dangerous)

    async def run_node(self, code: str) -> dict[str, Any]:
        if not code.strip():
            return self._error("run_node", "No JavaScript code was provided.")
        start = time.monotonic()
        try:
            sandbox = await self._get_sandbox()
            # Base64 prevents accidental injection of the heredoc delimiter
            # when the code itself contains a line named SANDBOX_EOF.
            encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
            command = (
                "printf %s " + shlex.quote(encoded)
                + " | base64 -d > /tmp/sandbox_script.js && node /tmp/sandbox_script.js"
            )
            result = await sandbox.commands.run(command, timeout=self.timeout)
            return {
                "ok": result.exit_code == 0,
                "operation": "run_node",
                "stdout": self._limit(result.stdout),
                "stderr": self._limit(result.stderr),
                "returncode": result.exit_code,
                "duration": round(time.monotonic() - start, 3),
            }
        except Exception as exc:
            await self._safe_reset()
            return self._error("run_node", exc, retryable=True, stdout="", stderr="", returncode=-1,
                               duration=round(time.monotonic() - start, 3))

    async def pip_install(self, packages: list[str]) -> dict[str, Any]:
        if not packages:
            return self._error("pip_install", "Sin paquetes que instalar.")
        for package in packages:
            if not re.fullmatch(r"[A-Za-z0-9_.-]+(?:\[[A-Za-z0-9_,.-]+\])?(?:[<>=!~]=?.*)?", package):
                return self._error("pip_install", f"Invalid package name: {package}")
        command = "python -m pip install --disable-pip-version-check --no-input " + " ".join(shlex.quote(p) for p in packages)
        return await self.run_shell(command)

    async def check_python(self, code: str) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ok": True, "operation": "check_python",
            "syntax": {"ok": False}, "compile": {"ok": False},
            "imports": [], "functions": [], "classes": [],
            "errors": [], "warnings": [],
        }
        try:
            tree = ast.parse(code)
            result["syntax"] = {"ok": True}
        except SyntaxError as exc:
            result["ok"] = False
            result["errors"].append(f"SINTAXIS: {exc}")
            return result
        try:
            compile(code, "<sandbox_check>", "exec")
            result["compile"] = {"ok": True}
        except Exception as exc:
            result["ok"] = False
            result["errors"].append(f"COMPILE: {exc}")
        imports: set[str] = set()
        functions: set[str] = set()
        classes: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.add(node.module)
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                functions.add(node.name)
            elif isinstance(node, ast.ClassDef):
                classes.add(node.name)
        result["imports"] = sorted(imports)
        result["functions"] = sorted(functions)
        result["classes"] = sorted(classes)
        return result

    async def check_python_file(self, path: str) -> dict[str, Any]:
        safe_path = shlex.quote(path)
        result = await self.run_shell(f"cat {safe_path}")
        if not result.get("ok"):
            return result
        return await self.check_python(result.get("stdout", ""))

    async def run_tests(self, path: str = ".", *, test_command: str | None = None) -> dict[str, Any]:
        if test_command:
            command = test_command
        else:
            command = f"cd {shlex.quote(path)} && python -m pytest -q"
        return await self.run_shell(command)

    async def run_lint(self, path: str = ".") -> dict[str, Any]:
        command = f"cd {shlex.quote(path)} && ruff check ."
        return await self.run_shell(command)

    async def compile_python_project(self, path: str = ".") -> dict[str, Any]:
        command = f"cd {shlex.quote(path)} && python -m compileall -q ."
        return await self.run_shell(command)

    async def full_python_check(self, path: str = ".", *, run_tests: bool = False, run_lint: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {"ok": True, "operation": "full_python_check", "path": path, "checks": {}}
        compile_result = await self.compile_python_project(path)
        result["checks"]["compile"] = compile_result
        if not compile_result.get("ok"):
            result["ok"] = False
        if run_tests:
            tests = await self.run_tests(path)
            result["checks"]["tests"] = tests
            if not tests.get("ok"):
                result["ok"] = False
        if run_lint:
            lint = await self.run_lint(path)
            result["checks"]["lint"] = lint
            if not lint.get("ok"):
                result["ok"] = False
        return result

    async def analyze_json(self, json_str: str) -> dict[str, Any]:
        try:
            data = json.loads(json_str)
            return self._success(operation="analyze_json", type=type(data).__name__,
                                 keys=(list(data.keys()) if isinstance(data, dict) else None),
                                 length=(len(data) if isinstance(data, (list, dict, str)) else None))
        except Exception as exc:
            return self._error("analyze_json", f"Invalid JSON: {exc}")

    async def regex_test(self, pattern: str, text: str, flags: str = "") -> dict[str, Any]:
        try:
            fl = 0
            if "i" in flags: fl |= re.IGNORECASE
            if "m" in flags: fl |= re.MULTILINE
            if "s" in flags: fl |= re.DOTALL
            rx = re.compile(pattern, fl)
            matches = [m.group(0) for m in rx.finditer(text)]
            return self._success(operation="regex_test", matches=matches, count=len(matches))
        except Exception as exc:
            return self._error("regex_test", f"Invalid regular expression: {exc}")

    async def http_request(
        self,
        url: str,
        method: str = "GET",
        body: str | None = None,
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        timeout_seconds: float = 20,
        *,
        query: dict[str, Any] | None = None,
        json_data: Any = None,
        timeout: float | None = None,
        max_bytes: int = 30_000,
    ) -> dict[str, Any]:
        """Performs the request from inside E2B, not from the bot process."""
        from urllib.parse import urlsplit, urlunsplit, urlencode, parse_qsl

        url = str(url).strip()
        if not url or len(url) > 4096:
            return self._error("http_request", "URL is empty or too long (maximum 4096 characters).")
        parsed = urlsplit(url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            return self._error("http_request", "Only valid HTTP(S) URLs are allowed.")
        if parsed.username or parsed.password:
            return self._error("http_request", "Credentials embedded in URLs are not allowed.")
        method = str(method or "GET").upper().strip()
        if method not in {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS"}:
            return self._error("http_request", "HTTP method is not allowed.")
        if not (1 <= int(max_bytes) <= 60_000):
            return self._error("http_request", "max_bytes must be between 1 and 60,000.")
        request_timeout = float(timeout if timeout is not None else timeout_seconds)
        if not (1 <= request_timeout <= 60):
            return self._error("http_request", "timeout must be between 1 and 60 seconds.")

        query_values = query if query is not None else params
        query_pairs = list(parse_qsl(parsed.query, keep_blank_values=True))
        for key, value in (query_values or {}).items():
            if isinstance(value, list):
                query_pairs.extend((str(key), str(item)) for item in value)
            else:
                query_pairs.append((str(key), str(value)))
        final_url = urlunsplit((parsed.scheme, parsed.netloc, parsed.path or "/", urlencode(query_pairs), ""))

        safe_headers: dict[str, str] = {"User-Agent": "OpenCoder-Agent/1.0", "Accept": "*/*"}
        for key, value in (headers or {}).items():
            key, value = str(key).strip(), str(value)
            if not key or any(ch in key + value for ch in "\r\n"):
                return self._error("http_request", "Invalid HTTP header.")
            if key.lower() in {"host", "content-length", "connection"}:
                continue
            safe_headers[key] = value[:8000]

        payload: bytes | None = None
        if json_data is not None:
            payload = json.dumps(json_data, ensure_ascii=False).encode("utf-8")
            safe_headers.setdefault("Content-Type", "application/json; charset=utf-8")
        elif body is not None:
            payload = str(body).encode("utf-8")
        if payload is not None and len(payload) > 1_000_000:
            return self._error("http_request", "HTTP request body exceeds 1 MB.")

        cfg = {
            "url": final_url,
            "method": method,
            "headers": safe_headers,
            "body_b64": base64.b64encode(payload).decode("ascii") if payload is not None else None,
            "timeout": request_timeout,
            "max_bytes": int(max_bytes),
        }
        encoded = base64.b64encode(json.dumps(cfg, ensure_ascii=False).encode("utf-8")).decode("ascii")
        code = (
            "import base64,ipaddress,json,socket,urllib.request,urllib.error,urllib.parse\n"
            f"cfg=json.loads(base64.b64decode({encoded!r}).decode('utf-8'))\n"
            "def check_public_target(target):\n"
            " p=urllib.parse.urlsplit(target)\n"
            " if p.scheme not in ('http','https') or not p.hostname or p.username or p.password: raise RuntimeError('HTTP redirect or destination is not allowed')\n"
            " port=p.port or (443 if p.scheme=='https' else 80)\n"
            " ips={x[4][0] for x in socket.getaddrinfo(p.hostname,port,type=socket.SOCK_STREAM)}\n"
            " if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips): raise RuntimeError('Destination blocked: private, local, or reserved IP')\n"
            "check_public_target(cfg['url'])\n"
            "class PublicRedirect(urllib.request.HTTPRedirectHandler):\n"
            " def redirect_request(self,req,fp,code,msg,headers,newurl):\n"
            "  check_public_target(newurl)\n"
            "  return super().redirect_request(req,fp,code,msg,headers,newurl)\n"
            "opener=urllib.request.build_opener(PublicRedirect())\n"
            "data=base64.b64decode(cfg['body_b64']) if cfg.get('body_b64') else None\n"
            "req=urllib.request.Request(cfg['url'],data=data,headers=cfg['headers'],method=cfg['method'])\n"
            "try: response=opener.open(req,timeout=cfg['timeout'])\n"
            "except urllib.error.HTTPError as exc: response=exc\n"
            "raw=response.read(cfg['max_bytes']+1)\n"
            "truncated=len(raw)>cfg['max_bytes']; raw=raw[:cfg['max_bytes']]\n"
            "text=raw.decode(response.headers.get_content_charset() or 'utf-8',errors='replace')\n"
            "try: parsed=json.loads(text)\n"
            "except Exception: parsed=None\n"
            "print(json.dumps({'ok':200<=response.status<400,'status':response.status,'url':response.geturl(),'headers':dict(response.headers.items()),'text':text,'json':parsed,'truncated':truncated,'bytes':len(raw)},ensure_ascii=False))\n"
        )
        result = await self.run_python(code)
        if not result.get("ok"):
            return {"ok": False, "operation": "http_request", "error": result.get("stderr") or result.get("error"), "returncode": result.get("returncode"), "executed_in": "E2B"}
        stdout = str(result.get("stdout", "")).strip()
        try:
            parsed_result = json.loads(stdout.splitlines()[-1])
        except (json.JSONDecodeError, IndexError):
            return self._error("http_request", "E2B did not return a parseable HTTP response.", stdout=self._limit(stdout))
        parsed_result["operation"] = "http_request"
        parsed_result["executed_in"] = "E2B"
        return parsed_result

    async def hash_text(self, text: str, algorithm: str = "sha256") -> dict[str, Any]:
        try:
            h = hashlib.new(algorithm)
            h.update(text.encode("utf-8"))
            return self._success(operation="hash_text", algorithm=algorithm, hash=h.hexdigest())
        except Exception as exc:
            return self._error("hash_text", exc)

    async def ocr_image(self, path: str, lang: str = "spa+eng") -> dict[str, Any]:
        command = (
            "which tesseract >/dev/null 2>&1 || "
            "(apt-get update -qq && apt-get install -y -qq tesseract-ocr tesseract-ocr-spa); "
            f"tesseract {shlex.quote(path)} stdout -l {shlex.quote(lang)}"
        )
        return await self.run_shell(command)

    async def extract_text_pdf(self, path: str) -> dict[str, Any]:
        code = f"""
from pypdf import PdfReader
reader = PdfReader({path!r})
for page in reader.pages:
    print(page.extract_text() or "")
"""
        return await self.run_python(code)

    async def git_clone(self, repo_url: str, dest: str | None = None) -> dict[str, Any]:
        command = "git clone " + shlex.quote(repo_url)
        if dest: command += " " + shlex.quote(dest)
        return await self.run_shell(command)

    async def git_status(self, path: str = ".") -> dict[str, Any]:
        command = f"cd {shlex.quote(path)} && git status --short --branch"
        return await self.run_shell(command)

    async def git_diff(self, path: str = ".") -> dict[str, Any]:
        command = f"cd {shlex.quote(path)} && git diff"
        return await self.run_shell(command)

    async def file_exists(self, path: str) -> dict[str, Any]:
        command = f"test -e {shlex.quote(path)}"
        result = await self.run_shell(command)
        return {**result, "path": path, "exists": result.get("returncode") == 0}

    async def read_file(self, path: str) -> dict[str, Any]:
        return await self.run_shell(f"cat {shlex.quote(path)}")

    async def write_file(self, path: str, content: str) -> dict[str, Any]:
        try:
            sandbox = await self._get_sandbox()
            command = "cat > " + shlex.quote(path) + " << 'SANDBOX_EOF'\n" + content + "\nSANDBOX_EOF"
            result = await sandbox.commands.run(command, timeout=self.timeout)
            return {"ok": result.exit_code == 0, "operation": "write_file", "path": path,
                    "stdout": self._limit(result.stdout), "stderr": self._limit(result.stderr), "returncode": result.exit_code}
        except Exception as exc:
            await self._safe_reset()
            return self._error("write_file", exc, retryable=True)

    async def put_files(self, files: dict[str, str | bytes]) -> dict[str, Any]:
        """Uploads multiple files to the sandbox (creating parent directories)."""
        try:
            sandbox = await self._get_sandbox()
            await asyncio.gather(*(sandbox.files.write(path, data) for path, data in files.items()))
            return self._success(operation="put_files", count=len(files))
        except Exception as exc:
            return self._error("put_files", exc, retryable=True)

    async def list_files(self, path: str = ".") -> dict[str, Any]:
        command = f"find {shlex.quote(path)} -maxdepth 2 -type f -print"
        return await self.run_shell(command)

    async def search_files(self, pattern: str, path: str = ".") -> dict[str, Any]:
        command = f"grep -RIn --exclude-dir=.git {shlex.quote(pattern)} {shlex.quote(path)}"
        return await self.run_shell(command)

    async def delete_file(self, path: str) -> dict[str, Any]:
        if path in {"/", ".", "..", ""}:
            return self._error("delete_file", "Path is not allowed.")
        return await self.run_shell(f"rm -f -- {shlex.quote(path)}")

    async def download(self, url: str, output: str) -> dict[str, Any]:
        try:
            timeout = aiohttp.ClientTimeout(total=120)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(url) as resp:
                    if resp.status >= 400:
                        return self._error("download", f"HTTP {resp.status}")
                    total = 0
                    with open(output, "wb") as file:
                        async for chunk in resp.content.iter_chunked(1024 * 1024):
                            total += len(chunk)
                            if total > self.max_download:
                                return self._error("download", "File is too large.")
                            file.write(chunk)
                    return self._success(operation="download", path=output, size=total, status=resp.status)
        except Exception as exc:
            return self._error("download", exc, retryable=True)

    async def verify_python_change(self, file_path: str, old_content: str, new_content: str) -> dict[str, Any]:
        errors: list[str] = []
        warnings: list[str] = []
        old_lines_count = len(old_content.splitlines())
        new_lines_count = len(new_content.splitlines())
        try:
            ast.parse(new_content)
        except SyntaxError as exc:
            return {"ok": False, "operation": "verify_python_change", "file": file_path,
                    "errors": [f"SINTAXIS: {exc}"], "warnings": [],
                    "old_lines": old_lines_count, "new_lines": new_lines_count}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "check.py"
            path.write_text(new_content, encoding="utf-8")
            try:
                import py_compile
                py_compile.compile(str(path), doraise=True)
            except py_compile.PyCompileError as exc:
                return {"ok": False, "operation": "verify_python_change", "file": file_path,
                        "errors": [f"COMPILE: {exc}"], "warnings": [],
                        "old_lines": old_lines_count, "new_lines": new_lines_count}
        old_set = set(old_content.splitlines())
        new_set = set(new_content.splitlines())
        similarity = 1.0
        if old_set:
            similarity = len(old_set & new_set) / max(len(old_set), len(new_set))
        if similarity < 0.80:
            warnings.append(f"SIMILITUD BAJA: {similarity:.2%} (< 80%).")
        old_tree = ast.parse(old_content)
        new_tree = ast.parse(new_content)
        old_classes = {node.name for node in ast.walk(old_tree) if isinstance(node, ast.ClassDef)}
        new_classes = {node.name for node in ast.walk(new_tree) if isinstance(node, ast.ClassDef)}
        missing_classes = old_classes - new_classes
        if missing_classes:
            errors.append(f"CLASES ELIMINADAS: {sorted(missing_classes)}")
        old_funcs = {node.name for node in ast.walk(old_tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        new_funcs = {node.name for node in ast.walk(new_tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
        missing_funcs = old_funcs - new_funcs
        if missing_funcs:
            errors.append(f"FUNCIONES ELIMINADAS: {sorted(missing_funcs)}")
        def extract_imports(tree: ast.AST) -> set[str]:
            result: set[str] = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        result.add(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        result.add(node.module)
            return result
        old_imports = extract_imports(old_tree)
        new_imports = extract_imports(new_tree)
        missing_imports = old_imports - new_imports
        if missing_imports:
            warnings.append(f"IMPORTS ELIMINADOS: {sorted(missing_imports)}")
        if old_lines_count > 20 and new_lines_count < old_lines_count * 0.90:
            errors.append(f"FILE TRUNCATED: {old_lines_count} -> {new_lines_count} lines")
        return {"ok": len(errors) == 0, "operation": "verify_python_change", "file": file_path,
                "errors": errors, "warnings": warnings, "similarity": round(similarity, 4),
                "old_lines": old_lines_count, "new_lines": new_lines_count}

    async def verify_and_test_python_change(self, file_path: str, old_content: str, new_content: str,
                                             *, project_path: str = ".", run_tests: bool = True) -> dict[str, Any]:
        verification = await self.verify_python_change(file_path, old_content, new_content)
        result = {"ok": verification.get("ok", False), "operation": "verify_and_test_python_change",
                  "verification": verification, "tests": None}
        if not verification.get("ok"):
            return result
        if run_tests:
            tests = await self.run_tests(project_path)
            result["tests"] = tests
            if not tests.get("ok"):
                result["ok"] = False
        return result

    async def create_zip(self, source: str, output: str) -> dict[str, Any]:
        command = f"cd {shlex.quote(source)} && zip -r {shlex.quote(output)} ."
        return await self.run_shell(command)

    async def extract_zip(self, archive: str, destination: str) -> dict[str, Any]:
        command = (f"mkdir -p {shlex.quote(destination)} && "
                   f"unzip -o {shlex.quote(archive)} -d {shlex.quote(destination)}")
        return await self.run_shell(command)

    async def _safe_reset(self) -> None:
        try:
            await self._reset_sandbox()
        except Exception:
            pass


sandbox = Sandbox()
