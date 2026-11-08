#!/usr/bin/env python3
"""file_tools.py
File, web, and utility tools for OpenCoder-Agent.

Integrates the enhanced search engine (search.py):
TTL cache, retries, heuristic reranking, deduplication, and URL validation.
"""
from __future__ import annotations

import asyncio
import base64
import csv
import hashlib
import inspect
import io
import json
import mimetypes
import os
import re
import shutil
import urllib.request
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, quote, unquote, urlparse

import aiohttp
from bs4 import BeautifulSoup


try:
    from PIL import Image  # type: ignore
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False

try:
    from pypdf import PdfReader  # type: ignore
    _HAS_PYPDF = True
except Exception:
    try:
        from PyPDF2 import PdfReader  # type: ignore
        _HAS_PYPDF = True
    except Exception:
        _HAS_PYPDF = False

try:
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:
    _HAS_YAML = False
# ENHANCED SEARCH ENGINE
try:
    from search.search import (
        search as _ddg_search,
        SearchError as _SearchError,
    )
    _HAS_SEARCH = True
except Exception:
    _ddg_search = None
    _SearchError = Exception  # type: ignore
    _HAS_SEARCH = False


TEXT_EXTENSIONS = {
    ".txt", ".md", ".py", ".js", ".ts", ".tsx", ".jsx",
    ".html", ".htm", ".css", ".scss", ".json", ".yaml",
    ".yml", ".toml", ".ini", ".cfg", ".conf", ".env",
    ".sh", ".bash", ".zsh", ".ps1", ".bat", ".cmd",
    ".sql", ".xml", ".csv", ".tsv", ".log", ".rst",
    ".c", ".h", ".cpp", ".hpp", ".cc", ".java", ".kt",
    ".go", ".rs", ".rb", ".php", ".pl", ".lua", ".r",
    ".swift", ".dart", ".vue", ".svelte", ".gitignore",
    ".dockerfile", ".makefile",
}

IMAGE_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".webp",
    ".bmp", ".tiff", ".tif", ".ico",
}

PDF_EXTENSIONS = {".pdf"}

MAX_TEXT_RESULT = 30000
MAX_LINES_RESULT = 2000


class FileToolsError(Exception):
    """File tools error."""

class FileTools:
    """API for an AI agent to work with files, the web, and utilities."""
    _USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:125.0) Gecko/20100101 Firefox/125.0",
    ]

    def __init__(self, workspace: str | Path = "."):
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace.mkdir(parents=True, exist_ok=True)
        self._ua_index = 0

    def _next_ua(self) -> str:
        ua = self._USER_AGENTS[self._ua_index % len(self._USER_AGENTS)]
        self._ua_index += 1
        return ua
    # HELPERS
    def _path(self, name: str | Path) -> Path:
        p = Path(name)
        if not p.is_absolute():
            p = self.workspace / p
        p = p.resolve()
        try:
            p.relative_to(self.workspace)
        except ValueError as exc:
            raise PermissionError(
                f"Path outside the workspace is not allowed: {p}"
            ) from exc
        return p

    @staticmethod
    def _truncate(text: str, limit: int = MAX_TEXT_RESULT) -> str:
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n\n[...truncado a {limit} caracteres de {len(text)}...]"

    def ensure_inside(self, name: str | Path) -> str:
        p = self._path(name)
        return str(p.relative_to(self.workspace))
    # BASIC CRUD
    def create_file(
        self, name: str, content: str = "",
        encoding: str = "utf-8", overwrite: bool = False,
    ) -> str:
        path = self._path(name)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Ya existe: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)
        return str(path)

    def write_file(self, name: str, content: str, encoding: str = "utf-8") -> str:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding=encoding)
        return str(path)

    def append_file(self, name: str, content: str, encoding: str = "utf-8") -> str:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding=encoding) as f:
            f.write(content)
        return str(path)

    def read_file(self, name: str, encoding: str = "utf-8") -> str:
        content = self._path(name).read_text(encoding=encoding)
        return self._truncate(content)

    def create_binary_file(self, name: str, data_base64: str) -> str:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(base64.b64decode(data_base64))
        return str(path)

    def read_binary_file_base64(self, name: str) -> str:
        return base64.b64encode(self._path(name).read_bytes()).decode("ascii")

    def mkdir(self, name: str) -> str:
        path = self._path(name)
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def touch(self, name: str) -> str:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch(exist_ok=True)
        return str(path)

    def list_files(self, directory: str = ".") -> list[dict[str, Any]]:
        path = self._path(directory)
        if not path.is_dir():
            raise NotADirectoryError(str(path))
        result = []
        for item in sorted(path.iterdir(), key=lambda x: x.name.lower()):
            result.append({
                "name": item.name,
                "path": str(item.relative_to(self.workspace)),
                "type": "directory" if item.is_dir() else "file",
                "size": item.stat().st_size if item.is_file() else None,
            })
        return result

    def info(self, name: str) -> dict[str, Any]:
        path = self._path(name)
        stat = path.stat()
        return {
            "name": path.name,
            "path": str(path.relative_to(self.workspace)),
            "absolute_path": str(path),
            "type": "directory" if path.is_dir() else "file",
            "size": stat.st_size,
        }

    def copy(self, source: str, destination: str) -> str:
        src = self._path(source)
        dst = self._path(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        if src.is_dir():
            shutil.copytree(src, dst, dirs_exist_ok=True)
        else:
            shutil.copy2(src, dst)
        return str(dst)

    def move(self, source: str, destination: str) -> str:
        src = self._path(source)
        dst = self._path(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        return str(shutil.move(str(src), str(dst)))

    def delete(self, name: str, recursive: bool = False) -> bool:
        path = self._path(name)
        if path.is_dir():
            if not recursive:
                path.rmdir()
            else:
                shutil.rmtree(path)
        else:
            path.unlink()
        return True
    # ZIP
    def zip_create(
        self, output_zip: str, sources: list[str],
        compression: int = zipfile.ZIP_DEFLATED,
    ) -> str:
        zip_path = self._path(output_zip)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        resolved_sources: list[Path] = []
        for source in sources:
            src = self._path(source)
            if not src.exists():
                raise FileNotFoundError(str(src))
            if src.is_file() and src.resolve() == zip_path:
                raise ValueError("The output ZIP cannot also be one of the source files.")
            resolved_sources.append(src)

        seen: set[str] = set()
        with zipfile.ZipFile(zip_path, "w", compression=compression) as zf:
            for src in resolved_sources:
                candidates = src.rglob("*") if src.is_dir() else (src,)
                for item in candidates:
                    try:
                        if not item.is_file():
                            continue
                        resolved = item.resolve(strict=True)
                        # Do not follow symlinks that escape the workspace.
                        relative = resolved.relative_to(self.workspace)
                    except (OSError, ValueError, RuntimeError):
                        continue
                    if resolved == zip_path or str(relative) in seen:
                        continue
                    seen.add(str(relative))
                    zf.write(resolved, relative)
        return str(zip_path)

    def zip_list(self, zip_name: str) -> list[dict[str, Any]]:
        zip_path = self._path(zip_name)
        with zipfile.ZipFile(zip_path, "r") as zf:
            return [
                {
                    "name": info.filename,
                    "size": info.file_size,
                    "compressed_size": info.compress_size,
                }
                for info in zf.infolist()
            ]

    def zip_extract(
        self,
        zip_name: str,
        destination: str = "extracted",
        *,
        overwrite: bool = False,
        max_entries: int = 5000,
        max_total_bytes: int = 500 * 1024 * 1024,
        max_file_bytes: int = 100 * 1024 * 1024,
        max_compression_ratio: int = 200,
    ) -> list[str]:
        """Extract a ZIP safely, rejecting traversal, symlinks, and oversized archives."""
        zip_path = self._path(zip_name)
        if not zip_path.is_file():
            raise FileNotFoundError(f"ZIP archive not found: {zip_name}")
        if not zipfile.is_zipfile(zip_path):
            raise FileToolsError(f"The file is not a valid ZIP archive: {zip_name}")

        dest = self._path(destination)
        dest.mkdir(parents=True, exist_ok=True)
        dest_root = dest.resolve()
        extracted: list[str] = []
        total_uncompressed = 0

        with zipfile.ZipFile(zip_path, "r") as zf:
            members = zf.infolist()
            if len(members) > max_entries:
                raise FileToolsError(
                    f"ZIP rejected: contains {len(members)} entries; maximum {max_entries}."
                )

            for member in members:
                raw_name = member.filename
                # ZIP archives use POSIX separators, even when created on Windows.
                safe_name = raw_name.replace("\\", "/")
                pure = Path(safe_name)
                if (
                    not safe_name
                    or safe_name.startswith("/")
                    or pure.is_absolute()
                    or any(part in {"..", ""} for part in safe_name.split("/") if part != "")
                    or re.match(r"^[A-Za-z]:", safe_name)
                ):
                    raise PermissionError(f"ZIP rejected due to unsafe path: {raw_name}")

                # Reject symlinks that could redirect writes outside the destination.
                unix_mode = (member.external_attr >> 16) & 0xFFFF
                if (unix_mode & 0o170000) == 0o120000:
                    raise PermissionError(f"ZIP rejected: symbolic link: {raw_name}")

                target = (dest_root / safe_name).resolve()
                try:
                    target.relative_to(dest_root)
                except ValueError as exc:
                    raise PermissionError(f"ZIP rejected due to unsafe path: {raw_name}") from exc

                if member.is_dir() or safe_name.endswith("/"):
                    target.mkdir(parents=True, exist_ok=True)
                    continue

                if member.flag_bits & 0x1:
                    raise FileToolsError(f"Encrypted ZIP archives are not supported: {raw_name}")
                if member.file_size > max_file_bytes:
                    raise FileToolsError(
                        f"ZIP member is too large ({member.file_size} bytes): {raw_name}"
                    )
                total_uncompressed += member.file_size
                if total_uncompressed > max_total_bytes:
                    raise FileToolsError(
                        f"ZIP exceeds the maximum total extraction size ({max_total_bytes} bytes)."
                    )
                if member.file_size and member.compress_size == 0:
                    raise FileToolsError(f"Abnormal compression ratio: {raw_name}")
                if member.compress_size and member.file_size / member.compress_size > max_compression_ratio:
                    raise FileToolsError(f"Possible ZIP bomb: excessive compression ratio in {raw_name}")

                target.parent.mkdir(parents=True, exist_ok=True)
                if target.exists() and not overwrite:
                    raise FileExistsError(
                        f"Destination already exists: {target.relative_to(dest_root)}. "
                        "Usa overwrite=true para reemplazarlo."
                    )
                mode = "wb" if overwrite else "xb"
                written = 0
                with zf.open(member, "r") as source, target.open(mode) as output:
                    while True:
                        chunk = source.read(1024 * 1024)
                        if not chunk:
                            break
                        written += len(chunk)
                        if written > max_file_bytes or written > member.file_size:
                            raise FileToolsError(f"Unexpected actual size while extracting: {raw_name}")
                        output.write(chunk)
                if written != member.file_size:
                    raise FileToolsError(f"Incomplete extraction: {raw_name}")
                extracted.append(str(target))

        return extracted

    @staticmethod
    def tool_capability_catalog() -> list[dict[str, Any]]:
        """Lightweight catalog of 300 executable recipes/capabilities built on existing engines.

        Does not create 300 independent external integrations: it indexes 300 tasks and execution
        paths by reusing FileTools and the E2B sandbox.
        """
        cached = getattr(FileTools, "_capability_catalog_cache", None)
        if cached is not None:
            return cached

        groups = [
            ("filesystem", "Files and folders", [
                ("list", "file_list"), ("read", "file_read"), ("write", "file_write"),
                ("append", "file_append"), ("copy", "sandbox_run_python"), ("move", "sandbox_run_python"),
                ("delete", "sandbox_run_python"), ("create folder", "file_mkdir"),
                ("inspect metadata", "file_info"), ("calculate hash", "sandbox_run_python"),
                ("create backup", "file_backup"), ("show tree", "file_tree"),
                ("find files", "file_search"), ("search text", "file_grep"),
                ("compare files", "sandbox_run_python"),
            ]),
            ("text", "Text and documents", [
                ("read first lines", "sandbox_run_python"), ("read last lines", "sandbox_run_python"),
                ("count lines", "sandbox_run_python"), ("count words", "file_count_words"),
                ("find and replace", "file_search_and_replace_dir"), ("normalize text", "sandbox_run_python"),
                ("remove duplicates", "sandbox_run_python"), ("sort lines", "sandbox_run_python"),
                ("extract regular expressions", "sandbox_run_python"), ("merge documents", "sandbox_run_python"),
                ("split document", "sandbox_run_python"), ("generate diff", "sandbox_run_python"),
                ("detect encoding", "sandbox_run_python"), ("convert format", "sandbox_run_python"),
                ("summarize document", "sandbox_run_python"),
            ]),
            ("structured_data", "Structured data", [
                ("read JSON", "sandbox_run_python"), ("write JSON", "sandbox_run_python"),
                ("validate JSON", "sandbox_run_python"), ("query JSONPath", "sandbox_run_python"),
                ("read CSV", "sandbox_run_python"), ("filter CSV", "sandbox_run_python"),
                ("group CSV", "sandbox_run_python"), ("convert CSV to JSON", "sandbox_run_python"),
                ("convert JSON to CSV", "sandbox_run_python"), ("read YAML", "sandbox_run_python"),
                ("validate YAML", "sandbox_run_python"), ("read XML", "sandbox_run_python"),
                ("validate XML", "sandbox_run_python"), ("query SQLite", "sandbox_run_python"),
                ("export data", "sandbox_run_python"),
            ]),
            ("archives", "Compressed archives", [
                ("create ZIP", "file_create_zip"), ("list ZIP", "sandbox_run_python"),
                ("extract ZIP", "file_extract_zip"), ("validate ZIP", "sandbox_run_python"),
                ("compare ZIP", "sandbox_run_python"), ("create TAR", "sandbox_run_python"),
                ("extract TAR", "sandbox_run_python"), ("create GZIP", "sandbox_run_python"),
                ("extract GZIP", "sandbox_run_python"), ("create BZIP2", "sandbox_run_python"),
                ("extract BZIP2", "sandbox_run_python"), ("calculate compressed size", "sandbox_run_python"),
                ("detect archive format", "sandbox_run_python"), ("verify integrity", "sandbox_run_python"),
                ("reorganize contents", "sandbox_run_python"),
            ]),
            ("development", "Development and code", [
                ("run Python", "sandbox_run_python"), ("run shell", "sandbox_run_shell"),
                ("run JavaScript", "sandbox_run_node"), ("run tests", "sandbox_run_shell"),
                ("compile Python", "sandbox_run_python"), ("analyze AST", "sandbox_run_python"),
                ("format code", "sandbox_run_shell"), ("run lint", "sandbox_run_shell"),
                ("check dependencies", "sandbox_run_shell"), ("install package", "sandbox_run_shell"),
                ("create virtual environment", "sandbox_run_shell"), ("inspect project", "sandbox_run_shell"),
                ("generate documentation", "sandbox_run_python"), ("generate changelog", "sandbox_run_python"),
                ("prepare build", "sandbox_run_shell"),
            ]),
            ("images_media", "Images and media", [
                ("inspect image metadata", "sandbox_run_python"), ("resize image", "sandbox_run_python"),
                ("create thumbnail", "sandbox_run_python"), ("convert image", "sandbox_run_python"),
                ("extract EXIF", "sandbox_run_python"), ("crop image", "sandbox_run_python"),
                ("compress image", "sandbox_run_python"), ("detect dimensions", "sandbox_run_python"),
                ("compare images", "sandbox_run_python"), ("create image collage", "sandbox_run_python"),
                ("convert audio", "sandbox_run_shell"), ("inspect audio", "sandbox_run_shell"),
                ("extract video frames", "sandbox_run_shell"), ("get media duration", "sandbox_run_shell"),
                ("generate subtitles", "sandbox_run_shell"),
            ]),
            ("pdf_documents", "PDF and documents", [
                ("extract PDF text", "file_read_pdf"), ("count PDF pages", "sandbox_run_python"),
                ("extract PDF metadata", "sandbox_run_python"), ("split PDF", "sandbox_run_python"),
                ("merge PDF", "sandbox_run_python"), ("rotate PDF pages", "sandbox_run_python"),
                ("extract PDF images", "sandbox_run_python"), ("search PDF text", "sandbox_run_python"),
                ("create PDF", "sandbox_run_python"), ("convert Markdown", "sandbox_run_shell"),
                ("convert HTML", "sandbox_run_shell"), ("validate document", "sandbox_run_python"),
                ("clean metadata", "sandbox_run_python"), ("extract tables", "sandbox_run_python"),
                ("generate index", "sandbox_run_python"),
            ]),
            ("web_api", "Web and APIs", [
                ("search the web", "web_search"), ("read web page", "web_fetch"),
                ("HTTP GET request", "sandbox_http_request"), ("HTTP POST request", "sandbox_http_request"),
                ("HTTP PUT request", "sandbox_http_request"), ("HTTP PATCH request", "sandbox_http_request"),
                ("HTTP DELETE request", "sandbox_http_request"), ("query JSON API", "sandbox_http_request"),
                ("download file", "sandbox_download"), ("clone repository", "sandbox_git_clone"),
                ("check endpoint", "sandbox_http_request"), ("inspect headers", "sandbox_http_request"),
                ("validate URL", "sandbox_run_python"), ("query DNS", "sandbox_run_python"),
                ("measure latency", "sandbox_run_python"),
            ]),
            ("quality", "Quality and diagnostics", [
                ("run test suite", "sandbox_run_shell"), ("detect syntax errors", "sandbox_run_python"),
                ("check imports", "sandbox_run_python"), ("audit dependencies", "sandbox_run_shell"),
                ("review structure", "sandbox_run_python"), ("measure complexity", "sandbox_run_python"),
                ("find TODO and FIXME", "sandbox_run_shell"), ("detect large files", "sandbox_run_shell"),
                ("detect duplicates", "sandbox_run_python"), ("review formats", "sandbox_run_python"),
                ("validate configuration", "sandbox_run_python"), ("analyze logs", "sandbox_run_python"),
                ("generate quality report", "sandbox_run_python"), ("verify build", "sandbox_run_shell"),
                ("test endpoint", "sandbox_http_request"),
            ]),
            ("automation", "Automation", [
                ("automate file task", "sandbox_run_python"), ("create Python script", "file_write"),
                ("create Bash script", "file_write"), ("create JavaScript script", "file_write"),
                ("schedule cron task", "sandbox_run_shell"), ("run pipeline", "sandbox_run_shell"),
                ("batch-process files", "sandbox_run_python"), ("batch rename files", "sandbox_run_python"),
                ("batch copy files", "sandbox_run_python"), ("generate report", "sandbox_run_python"),
                ("back up project", "file_create_zip"), ("deploy temporary website", "deploy_temp_website"),
                ("download resources", "sandbox_download"), ("check service status", "sandbox_http_request"),
                ("run project workflow", "sandbox_run_shell"),
            ]),
        ]
        variants = {
            "filesystem": ["local workspace", "subfolder"],
            "text": ["UTF-8", "long document"],
            "structured_data": ["JSON", "CSV"],
            "archives": ["ZIP", "TAR/GZIP"],
            "development": ["Python", "JavaScript"],
            "images_media": ["image", "audio/video"],
            "pdf_documents": ["PDF", "HTML/Markdown"],
            "web_api": ["public website", "API REST"],
            "quality": ["complete project", "single file"],
            "automation": ["single task", "batch process"],
        }
        catalog: list[dict[str, Any]] = []
        for category, category_label, operations in groups:
            for operation, backend in operations:
                for variant in variants[category]:
                    idx = len(catalog) + 1
                    slug = re.sub(r"[^a-z0-9]+", "_", f"{category}_{operation}_{variant}".lower()).strip("_")
                    catalog.append({
                        "id": f"cap_{idx:03d}_{slug[:48]}",
                        "name": f"{operation.capitalize()} · {variant}",
                        "category": category,
                        "category_label": category_label,
                        "description": f"{operation.capitalize()} for {variant}. Execute through {backend} with explicit parameters, then inspect and validate the result.",
                        "executor": backend,
                        "tags": [category, operation, variant, backend],
                    })
        # 10 categories x 15 operations x 2 variants = 300 capabilities.
        if len(catalog) != 300:
            raise RuntimeError(f"The catalog must contain 300 capabilities; found {len(catalog)}")
        FileTools._capability_catalog_cache = catalog
        return catalog

    def discover_tools(self, query: str, limit: int = 10) -> dict[str, Any]:
        """Searches the capability catalog without sending all 300 entries in the prompt."""
        query = " ".join(str(query or "").casefold().split())[:500]
        limit = max(1, min(int(limit), 20))
        words = {w for w in re.findall(r"[a-z0-9_-]+", query) if len(w) > 2}
        ranked: list[tuple[int, dict[str, Any]]] = []
        for item in self.tool_capability_catalog():
            corpus = " ".join([item["name"], item["description"], *item["tags"]]).casefold()
            score = sum(3 if word in item["name"].casefold() else 1 for word in words if word in corpus)
            if score:
                ranked.append((score, item))
        ranked.sort(key=lambda pair: (-pair[0], pair[1]["id"]))
        return {
            "total_capabilities": 300,
            "returned": min(limit, len(ranked)),
            "results": [item for _, item in ranked[:limit]],
            "usage": "Use tool_execute with capability_id and arguments. For sandbox_run_python or sandbox_run_shell, include code or command. The E2B sandbox is isolated.",
        }

    def search(self, pattern: str, directory: str = ".") -> list[str]:
        root = self._path(directory)
        return [str(p.relative_to(self.workspace)) for p in root.rglob(pattern)]
    # TIPO
    def detect_kind(self, name: str) -> str:
        path = self._path(name)
        if path.is_dir():
            return "directory"
        ext = path.suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return "image"
        if ext in PDF_EXTENSIONS:
            return "pdf"
        if ext in TEXT_EXTENSIONS:
            return "text"
        mime, _ = mimetypes.guess_type(str(path))
        if mime:
            if mime.startswith("image/"):
                return "image"
            if mime == "application/pdf":
                return "pdf"
            if mime.startswith("text/"):
                return "text"
            if mime in ("application/json", "application/xml"):
                return "text"
        return "binary"
    # IMAGES
    def read_image_base64(self, name: str) -> dict[str, Any]:
        path = self._path(name)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        raw = path.read_bytes()
        mime, _ = mimetypes.guess_type(str(path))
        if not mime or not mime.startswith("image/"):
            mime = "image/png"
        b64 = base64.b64encode(raw).decode("ascii")
        return {
            "path": str(path.relative_to(self.workspace)),
            "filename": path.name,
            "mime": mime,
            "base64": b64,
            "data_url": f"data:{mime};base64,{b64}",
            "size": len(raw),
        }

    def image_info(self, name: str) -> dict[str, Any]:
        path = self._path(name)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        info: dict[str, Any] = {
            "path": str(path.relative_to(self.workspace)),
            "filename": path.name,
            "size": path.stat().st_size,
        }
        if _HAS_PIL:
            try:
                with Image.open(path) as img:
                    info.update({
                        "width": img.width,
                        "height": img.height,
                        "format": img.format,
                        "mode": img.mode,
                    })
            except Exception as exc:
                info["pillow_error"] = str(exc)
        return info

    def image_resize(
        self, name: str, output: str, width: int,
        height: int | None = None, keep_aspect: bool = True,
    ) -> str:
        if not _HAS_PIL:
            raise RuntimeError("Pillow is not installed.")
        src = self._path(name)
        dst = self._path(output)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as img:
            if keep_aspect:
                if height is None:
                    ratio = width / img.width
                    height = int(img.height * ratio)
                else:
                    img.thumbnail((width, height))
                    img.save(dst)
                    return str(dst)
            resized = img.resize((width, height or img.height))
            resized.save(dst)
        return str(dst)

    def image_thumbnail(self, name: str, output: str, size: int = 256) -> str:
        if not _HAS_PIL:
            raise RuntimeError("Pillow is not installed.")
        src = self._path(name)
        dst = self._path(output)
        dst.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(src) as img:
            img.thumbnail((size, size))
            img.save(dst)
        return str(dst)

    def image_convert(self, name: str, output: str, format: str | None = None) -> str:
        if not _HAS_PIL:
            raise RuntimeError("Pillow is not installed.")
        src = self._path(name)
        dst = self._path(output)
        dst.parent.mkdir(parents=True, exist_ok=True)
        fmt = (format or dst.suffix.lstrip(".")).upper() or "PNG"
        with Image.open(src) as img:
            if fmt == "JPG":
                fmt = "JPEG"
            img.save(dst, format=fmt)
        return str(dst)

    def image_exif(self, name: str) -> dict[str, Any]:
        if not _HAS_PIL:
            raise RuntimeError("Pillow is not installed.")
        path = self._path(name)
        result: dict[str, Any] = {}
        with Image.open(path) as img:
            exif = img.getexif()
            for key, value in exif.items():
                try:
                    result[str(key)] = str(value)
                except Exception:
                    pass
        return result
    # TEXTO
    def read_text_auto(self, name: str) -> str:
        path = self._path(name)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        text = None
        for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
            try:
                text = path.read_text(encoding=enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            text = path.read_text(encoding="utf-8", errors="replace")
        return self._truncate(text)

    def head_file(self, name: str, lines: int = 20) -> str:
        path = self._path(name)
        out: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f):
                if i >= lines:
                    break
                out.append(line)
        return "".join(out)

    def tail_file(self, name: str, lines: int = 20) -> str:
        path = self._path(name)
        from collections import deque
        with path.open("r", encoding="utf-8", errors="replace") as f:
            return "".join(deque(f, maxlen=lines))

    def read_lines(self, name: str, start: int = 1, end: int | None = None) -> str:
        path = self._path(name)
        out: list[str] = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for i, line in enumerate(f, start=1):
                if i < start:
                    continue
                if end is not None and i > end:
                    break
                out.append(line)
                if len(out) >= MAX_LINES_RESULT:
                    out.append(f"\n[...truncated to {MAX_LINES_RESULT} lines...]")
                    break
        return "".join(out)

    def count_lines(self, name: str) -> int:
        path = self._path(name)
        count = 0
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for _ in f:
                count += 1
        return count

    def replace_in_file(
        self, name: str, old: str, new: str, count: int = -1,
    ) -> dict[str, Any]:
        path = self._path(name)
        content = self.read_text_auto(name)
        replaced = content.replace(old, new, count) if count >= 0 else content.replace(old, new)
        path.write_text(replaced, encoding="utf-8")
        return {
            "path": str(path.relative_to(self.workspace)),
            "replaced": content != replaced,
            "occurrences": content.count(old) if count < 0 else min(count, content.count(old)),
        }

    def grep(
        self, pattern: str, directory: str = ".",
        extensions: list[str] | None = None,
        max_results: int = 100,
    ) -> list[dict[str, Any]]:
        root = self._path(directory)
        results: list[dict[str, Any]] = []
        if not root.is_dir():
            raise NotADirectoryError(str(root))
        ext_filter = {e.lower() for e in extensions} if extensions else None
        for file in root.rglob("*"):
            if not file.is_file():
                continue
            if ext_filter and file.suffix.lower() not in ext_filter:
                continue
            try:
                with file.open("r", encoding="utf-8", errors="replace") as f:
                    for i, line in enumerate(f, start=1):
                        if pattern in line:
                            results.append({
                                "file": str(file.relative_to(self.workspace)),
                                "line": i,
                                "text": line.rstrip("\n")[:200],
                            })
                            if len(results) >= max_results:
                                return results
            except Exception:
                continue
        return results
    # PDF
    def read_pdf_text(self, name: str) -> str:
        if not _HAS_PYPDF:
            raise RuntimeError("pypdf is not installed.")
        path = self._path(name)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        reader = PdfReader(str(path))
        parts: list[str] = []
        for i, page in enumerate(reader.pages):
            try:
                text = page.extract_text() or ""
            except Exception as exc:
                text = f"[error extracting page {i}: {exc}]"
            parts.append(f"--- Page {i + 1} ---\n{text}")
            if sum(len(p) for p in parts) > MAX_TEXT_RESULT:
                parts.append("\n[...PDF truncated due to size...]")
                break
        return self._truncate("\n\n".join(parts))
    # GENERIC READING
    def read_any(self, name: str) -> dict[str, Any]:
        path = self._path(name)
        kind = self.detect_kind(name)
        if kind == "directory":
            return {
                "kind": "directory",
                "path": str(path.relative_to(self.workspace)),
                "listing": self.list_files(name)[:200],
            }
        if kind == "image":
            data = self.read_image_base64(name)
            data["kind"] = "image"
            data["info"] = self.image_info(name)
            return data
        if kind == "pdf":
            try:
                return {
                    "kind": "pdf",
                    "path": str(path.relative_to(self.workspace)),
                    "filename": path.name,
                    "text": self.read_pdf_text(name),
                }
            except Exception as exc:
                return {
                    "kind": "pdf",
                    "path": str(path.relative_to(self.workspace)),
                    "filename": path.name,
                    "error": str(exc),
                }
        if kind == "text":
            return {
                "kind": "text",
                "path": str(path.relative_to(self.workspace)),
                "filename": path.name,
                "text": self.read_text_auto(name),
            }
        return {
            "kind": "binary",
            "path": str(path.relative_to(self.workspace)),
            "filename": path.name,
            "size": path.stat().st_size,
            "note": "Binary file; it cannot be displayed as text.",
        }
    # UTILIDADES
    def exists(self, name: str) -> bool:
        try:
            return self._path(name).exists()
        except Exception:
            return False

    def is_file(self, name: str) -> bool:
        try:
            return self._path(name).is_file()
        except Exception:
            return False

    def is_dir(self, name: str) -> bool:
        try:
            return self._path(name).is_dir()
        except Exception:
            return False

    def file_hash(self, name: str, algorithm: str = "sha256") -> dict[str, Any]:
        path = self._path(name)
        if not path.is_file():
            raise FileNotFoundError(str(path))
        h = hashlib.new(algorithm)
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return {
            "path": str(path.relative_to(self.workspace)),
            "algorithm": algorithm,
            "hash": h.hexdigest(),
        }

    @staticmethod
    def human_size(size: int) -> str:
        units = ["B", "KB", "MB", "GB", "TB"]
        s = float(size)
        for u in units:
            if s < 1024 or u == units[-1]:
                return f"{s:.2f} {u}"
            s /= 1024
        return f"{size} B"

    def tree(self, directory: str = ".", max_depth: int = 3) -> dict[str, Any]:
        root = self._path(directory)
        if not root.is_dir():
            raise NotADirectoryError(str(root))
        def walk(p: Path, depth: int) -> dict[str, Any]:
            node: dict[str, Any] = {
                "name": p.name,
                "type": "directory",
                "children": [],
            }
            if depth >= max_depth:
                return node
            try:
                for item in sorted(p.iterdir(), key=lambda x: x.name.lower()):
                    if item.is_dir():
                        node["children"].append(walk(item, depth + 1))
                    else:
                        node["children"].append({
                            "name": item.name,
                            "type": "file",
                            "size": item.stat().st_size,
                            "human_size": self.human_size(item.stat().st_size),
                        })
            except Exception as exc:
                node["error"] = str(exc)
            return node
        return walk(root, 0)

    def concat_files(
        self, output: str, sources: list[str], separator: str = "\n",
    ) -> str:
        parts: list[str] = []
        for s in sources:
            parts.append(self.read_text_auto(s))
        dst = self._path(output)
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text(separator.join(parts), encoding="utf-8")
        return str(dst)

    def split_file(
        self, name: str, lines_per_chunk: int = 1000, output_dir: str = ".",
    ) -> list[str]:
        src = self._path(name)
        out_dir = self._path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        if lines_per_chunk <= 0:
            raise ValueError("lines_per_chunk must be > 0")
        parts: list[str] = []
        buffer: list[str] = []
        index = 1
        with src.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                buffer.append(line)
                if len(buffer) >= lines_per_chunk:
                    dst = out_dir / f"{src.stem}_part{index:03d}{src.suffix}"
                    dst.write_text("".join(buffer), encoding="utf-8")
                    parts.append(str(dst.relative_to(self.workspace)))
                    buffer = []
                    index += 1
            if buffer:
                dst = out_dir / f"{src.stem}_part{index:03d}{src.suffix}"
                dst.write_text("".join(buffer), encoding="utf-8")
                parts.append(str(dst.relative_to(self.workspace)))
        return parts

    def diff_files(self, a: str, b: str) -> str:
        import difflib
        text_a = self.read_text_auto(a).splitlines(keepends=True)
        text_b = self.read_text_auto(b).splitlines(keepends=True)
        return "".join(difflib.unified_diff(text_a, text_b, fromfile=a, tofile=b))

    @staticmethod
    def sanitize_filename(name: str) -> str:
        keep = "-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
        return "".join(c for c in name if c in keep).strip() or "file"

    def safe_name(self, name: str) -> str:
        return self.sanitize_filename(name)

    def read_json(self, name: str) -> Any:
        return json.loads(self.read_text_auto(name))

    def write_json(self, name: str, data: Any, indent: int = 2) -> str:
        path = self._path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(data, indent=indent, ensure_ascii=False),
            encoding="utf-8",
        )
        return str(path)

    def read_csv(self, name: str, max_rows: int | None = None) -> list[dict[str, Any]]:
        path = self._path(name)
        rows: list[dict[str, Any]] = []
        with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            for i, row in enumerate(reader):
                rows.append(dict(row))
                if max_rows is not None and i + 1 >= max_rows:
                    break
        return rows

    def read_yaml(self, name: str) -> Any:
        if not _HAS_YAML:
            raise RuntimeError("PyYAML is not installed.")
        return yaml.safe_load(self.read_text_auto(name))

    def download_url(
        self, url: str, destination: str, timeout: int = 60,
    ) -> dict[str, Any]:
        dst = self._path(destination)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                data = resp.read()
            dst.write_bytes(data)
            return {
                "path": str(dst.relative_to(self.workspace)),
                "size": len(data),
                "human_size": self.human_size(len(data)),
                "url": url,
            }
        except Exception as exc:
            raise FileToolsError(f"Error downloading {url}: {exc}") from exc

    def find_duplicates(self, directory: str = ".") -> list[list[str]]:
        root = self._path(directory)
        by_hash: dict[str, list[str]] = {}
        for file in root.rglob("*"):
            if not file.is_file():
                continue
            try:
                h = hashlib.sha256(file.read_bytes()).hexdigest()
                by_hash.setdefault(h, []).append(
                    str(file.relative_to(self.workspace))
                )
            except Exception:
                continue
        return [paths for paths in by_hash.values() if len(paths) > 1]

    def backup(self, name: str) -> str:
        src = self._path(name)
        if not src.exists():
            raise FileNotFoundError(str(src))
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if src.is_dir():
            dst = src.with_name(f"{src.name}.bak_{stamp}")
            shutil.copytree(src, dst)
        else:
            dst = src.with_name(f"{src.stem}.bak_{stamp}{src.suffix}")
            shutil.copy2(src, dst)
        return str(dst.relative_to(self.workspace))

    def empty_dir(self, name: str) -> str:
        path = self._path(name)
        if not path.is_dir():
            raise NotADirectoryError(str(path))
        for item in path.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
        return str(path)

    def clear_dir(self, name: str) -> str:
        return self.empty_dir(name)

    def count_words(self, name: str) -> int:
        text = self.read_text_auto(name)
        return len(text.split())

    def file_stats(self, name: str) -> dict[str, Any]:
        path = self._path(name)
        stat = path.stat()
        return {
            "name": path.name,
            "path": str(path.relative_to(self.workspace)),
            "size": stat.st_size,
            "human_size": self.human_size(stat.st_size),
            "modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "created": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        }

    def replace_many(self, name: str, replacements: dict) -> dict[str, Any]:
        content = self.read_text_auto(name)
        original = content
        for old, new in replacements.items():
            content = content.replace(old, new)
        if content != original:
            self.write_file(name, content)
        return {
            "path": str(self._path(name).relative_to(self.workspace)),
            "changed": content != original,
            "count": len(replacements),
        }

    def search_and_replace_dir(
        self, directory: str, old: str, new: str,
    ) -> dict[str, Any]:
        root = self._path(directory)
        files = [p for p in root.rglob("*") if p.is_file()]
        changed = 0
        errors = 0
        for f in files:
            try:
                text = f.read_text(encoding="utf-8", errors="ignore")
                if old in text:
                    f.write_text(text.replace(old, new), encoding="utf-8")
                    changed += 1
            except Exception:
                errors += 1
                continue
        return {
            "files_scanned": len(files),
            "files_changed": changed,
            "errors": errors,
        }
    # WEB
    async def web_search(
        self,
        query: str,
        max_results: int = 5,
        timeout: float = 15.0,
        *,
        region: str | None = None,
        safesearch: str | None = None,
        use_cache: bool = True,
        rerank: bool = True,
    ) -> dict[str, Any]:
        """
        Searches the web using the enhanced search engine (search.py).

        Returns the standard format:
            {"query": ..., "count": N, "results": [
                {"title", "url", "snippet", "score", "rank", "source"},
                ...
            ]}
        On error, returns {"error": "...", "results": []}.
        """
        if not query or not query.strip():
            return {"error": "query is empty", "results": []}

        try:
            n = int(max_results)
        except (TypeError, ValueError):
            n = 5
        n = max(1, min(n, 10))

        if not _HAS_SEARCH or _ddg_search is None:
            return {
                "error": (
                    "Search engine is unavailable. "
                    "Make sure you have 'search.py'  "
                    "and the 'ddgs' package installed."
                ),
                "results": [],
            }

        try:
            results = await _ddg_search(
                query,
                max_results=n,
                region=region,
                safesearch=safesearch,
                timeout_s=timeout,
                use_cache=use_cache,
                rerank=rerank,
            )
        except _SearchError as exc:  # type: ignore[misc]
            code = getattr(exc, "code", "search_error")
            message = getattr(exc, "message", str(exc))
            return {"error": f"{code}: {message}", "results": []}
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "results": []}

        return {
            "query": query,
            "count": len(results),
            "results": [
                {
                    "title": r.title,
                    "url": r.url,
                    "snippet": r.snippet,
                    "score": getattr(r, "score", 0.0),
                    "rank": getattr(r, "rank", idx + 1),
                    "source": getattr(r, "source", "duckduckgo"),
                }
                for idx, r in enumerate(results)
            ],
        }

    async def web_fetch(
        self, url: str, max_chars: int = 8000, timeout: float = 20.0,
    ) -> dict[str, Any]:
        if not url or not url.strip():
            return {"error": "URL is empty"}
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        headers = {
            "User-Agent": self._next_ua(),
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "es-ES,es;q=0.9,en;q=0.8",
        }
        client_timeout = aiohttp.ClientTimeout(total=timeout)
        try:
            async with aiohttp.ClientSession(
                headers=headers, timeout=client_timeout,
            ) as session:
                async with session.get(url, allow_redirects=True) as resp:
                    if resp.status >= 400:
                        return {"error": f"HTTP {resp.status}", "url": url}
                    content_type = (resp.headers.get("Content-Type", "")).lower()
                    if ("html" not in content_type and "text" not in content_type
                            and "xml" not in content_type):
                        return {"error": f"unsupported type: {content_type}", "url": url}
                    html = await resp.text()
                    final_url = str(resp.url)
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}", "url": url}

        soup = BeautifulSoup(html, "lxml")
        for tag in soup([
            "script", "style", "noscript", "iframe",
            "nav", "header", "footer", "aside", "form",
            "button", "svg", "img", "video", "audio",
            "canvas", "figure", "menu",
        ]):
            tag.decompose()

        title = ""
        if soup.title and soup.title.string:
            title = soup.title.string.strip()

        main = (
            soup.find("article") or soup.find("main")
            or soup.find(attrs={"role": "main"})
            or soup.find(attrs={"id": "content"})
            or soup.find(attrs={"class": "content"})
            or soup.body or soup
        )
        text = main.get_text("\n", strip=True)

        import re as _re
        text = _re.sub(r"[ \t]+", " ", text)
        text = _re.sub(r"\n{3,}", "\n\n", text)

        truncated = len(text) > max_chars
        if truncated:
            text = text[:max_chars] + "\n\n[...contenido truncado...]"

        return {
            "url": final_url,
            "title": title,
            "chars": len(text),
            "truncated": truncated,
            "text": text,
        }
# MODULE API
TOOLS = FileTools(os.getenv("SUBTOM_WORKSPACE", "./workspace"))


def create_file(name: str, content: str = "", overwrite: bool = False) -> str:
    return TOOLS.create_file(name, content, overwrite=overwrite)

def read_file(name: str) -> str:
    return TOOLS.read_file(name)

def write_file(name: str, content: str) -> str:
    return TOOLS.write_file(name, content)

def append_file(name: str, content: str) -> str:
    return TOOLS.append_file(name, content)

def mkdir(name: str) -> str:
    return TOOLS.mkdir(name)

def touch(name: str) -> str:
    return TOOLS.touch(name)

def list_files(directory: str = ".") -> list[dict[str, Any]]:
    return TOOLS.list_files(directory)

def copy_file(source: str, destination: str) -> str:
    return TOOLS.copy(source, destination)

def move_file(source: str, destination: str) -> str:
    return TOOLS.move(source, destination)

def delete_file(name: str, recursive: bool = False) -> bool:
    return TOOLS.delete(name, recursive)

def create_zip(output_zip: str, sources: list[str]) -> str:
    return TOOLS.zip_create(output_zip, sources)

def list_zip(zip_name: str) -> list[dict[str, Any]]:
    return TOOLS.zip_list(zip_name)

def unzip_file(zip_name: str, destination: str = ".") -> list[str]:
    return TOOLS.zip_extract(zip_name, destination)

def file_info(name: str) -> dict[str, Any]:
    return TOOLS.info(name)

def search_files(pattern: str, directory: str = ".") -> list[str]:
    return TOOLS.search(pattern, directory)

def detect_kind(name: str) -> str:
    return TOOLS.detect_kind(name)

def read_image_base64(name: str) -> dict[str, Any]:
    return TOOLS.read_image_base64(name)

def image_info(name: str) -> dict[str, Any]:
    return TOOLS.image_info(name)

def read_text_auto(name: str) -> str:
    return TOOLS.read_text_auto(name)

def read_pdf_text(name: str) -> str:
    return TOOLS.read_pdf_text(name)

def read_any(name: str) -> dict[str, Any]:
    return TOOLS.read_any(name)

def exists(name: str) -> bool:
    return TOOLS.exists(name)

def is_file(name: str) -> bool:
    return TOOLS.is_file(name)

def is_dir(name: str) -> bool:
    return TOOLS.is_dir(name)

def file_hash(name: str, algorithm: str = "sha256") -> dict[str, Any]:
    return TOOLS.file_hash(name, algorithm)

def human_size(size: int) -> str:
    return TOOLS.human_size(size)

def tree(directory: str = ".", max_depth: int = 3) -> dict[str, Any]:
    return TOOLS.tree(directory, max_depth)

def head_file(name: str, lines: int = 20) -> str:
    return TOOLS.head_file(name, lines)

def tail_file(name: str, lines: int = 20) -> str:
    return TOOLS.tail_file(name, lines)

def read_lines(name: str, start: int = 1, end: int | None = None) -> str:
    return TOOLS.read_lines(name, start, end)

def count_lines(name: str) -> int:
    return TOOLS.count_lines(name)

def replace_in_file(name: str, old: str, new: str, count: int = -1) -> dict[str, Any]:
    return TOOLS.replace_in_file(name, old, new, count)

def grep(pattern: str, directory: str = ".", extensions: list[str] | None = None) -> list[dict[str, Any]]:
    return TOOLS.grep(pattern, directory, extensions)

def concat_files(output: str, sources: list[str], separator: str = "\n") -> str:
    return TOOLS.concat_files(output, sources, separator)

def split_file(name: str, lines_per_chunk: int = 1000, output_dir: str = ".") -> list[str]:
    return TOOLS.split_file(name, lines_per_chunk, output_dir)

def diff_files(a: str, b: str) -> str:
    return TOOLS.diff_files(a, b)

def sanitize_filename(name: str) -> str:
    return TOOLS.sanitize_filename(name)

def read_json(name: str) -> Any:
    return TOOLS.read_json(name)

def write_json(name: str, data: Any, indent: int = 2) -> str:
    return TOOLS.write_json(name, data, indent)

def read_csv(name: str, max_rows: int | None = None) -> list[dict[str, Any]]:
    return TOOLS.read_csv(name, max_rows)

def read_yaml(name: str) -> Any:
    return TOOLS.read_yaml(name)

def download_url(url: str, destination: str, timeout: int = 60) -> dict[str, Any]:
    return TOOLS.download_url(url, destination, timeout)

def find_duplicates(directory: str = ".") -> list[list[str]]:
    return TOOLS.find_duplicates(directory)

def backup(name: str) -> str:
    return TOOLS.backup(name)

def empty_dir(name: str) -> str:
    return TOOLS.empty_dir(name)

def image_resize(name: str, output: str, width: int, height: int | None = None, keep_aspect: bool = True) -> str:
    return TOOLS.image_resize(name, output, width, height, keep_aspect)

def image_thumbnail(name: str, output: str, size: int = 256) -> str:
    return TOOLS.image_thumbnail(name, output, size)

def image_convert(name: str, output: str, format: str | None = None) -> str:
    return TOOLS.image_convert(name, output, format)

def image_exif(name: str) -> dict[str, Any]:
    return TOOLS.image_exif(name)

def ensure_inside(name: str) -> str:
    return TOOLS.ensure_inside(name)

def count_words(name: str) -> int:
    return TOOLS.count_words(name)

def file_stats(name: str) -> dict[str, Any]:
    return TOOLS.file_stats(name)

def replace_many(name: str, replacements: dict) -> dict[str, Any]:
    return TOOLS.replace_many(name, replacements)

def search_and_replace_dir(directory: str, old: str, new: str) -> dict[str, Any]:
    return TOOLS.search_and_replace_dir(directory, old, new)

async def web_search(
    query: str,
    max_results: int = 5,
    timeout: float = 15.0,
    **kwargs: Any,
) -> dict[str, Any]:
    return await TOOLS.web_search(query, max_results, timeout, **kwargs)

async def web_fetch(url: str, max_chars: int = 8000) -> dict[str, Any]:
    return await TOOLS.web_fetch(url, max_chars)
