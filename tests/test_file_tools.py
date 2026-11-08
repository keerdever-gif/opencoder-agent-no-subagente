"""Local tests for ZIP extraction and the capability catalog; no E2B required."""
from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

from tools.file_tools import FileTools


def run() -> None:
    with tempfile.TemporaryDirectory(prefix="opencoder-filetools-") as temp:
        root = Path(temp)
        tools = FileTools(root)

        # Valid extraction and directory structure preservation.
        source = root / "source"
        source.mkdir()
        (source / "hello.txt").write_text("hello OpenCoder", encoding="utf-8")
        with zipfile.ZipFile(root / "valid.zip", "w", zipfile.ZIP_DEFLATED) as archive:
            archive.write(source / "hello.txt", "project/docs/hello.txt")
        extracted = tools.zip_extract("valid.zip", "output")
        assert len(extracted) == 1
        assert Path(extracted[0]).read_text(encoding="utf-8") == "hello OpenCoder"

        # Zip Slip must be blocked.
        with zipfile.ZipFile(root / "zip-slip.zip", "w") as archive:
            archive.writestr("../../outside.txt", "no")
        try:
            tools.zip_extract("zip-slip.zip", "blocked")
        except PermissionError:
            pass
        else:
            raise AssertionError("Zip Slip was not blocked")

        # Existing files are not overwritten by default.
        try:
            tools.zip_extract("valid.zip", "output")
        except FileExistsError:
            pass
        else:
            raise AssertionError("Extraction overwrote a file without permission")

        # The maximum entry count is enforced.
        try:
            tools.zip_extract("valid.zip", "limited", max_entries=0)
        except Exception:
            pass
        else:
            raise AssertionError("max_entries was not enforced")

        catalog = tools.tool_capability_catalog()
        assert len(catalog) == 300
        assert len({entry["id"] for entry in catalog}) == 300
        matches = tools.discover_tools("extract ZIP archive", limit=5)
        assert matches["total_capabilities"] == 300
        assert matches["results"]
        print("OK: ZIP extraction, Zip Slip protection, no-overwrite behavior, and 300-capability catalog")


if __name__ == "__main__":
    run()
