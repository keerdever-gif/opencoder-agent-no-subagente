"""Small GitHub REST API client used by the agent."""
from __future__ import annotations

import json
from typing import Any

import aiohttp


async def github_request(
    token: str | None, method: str, path: str, **kwargs: Any
) -> dict[str, Any]:
    """Send an authenticated request to the GitHub REST API."""
    if not token:
        return {"error": "GITHUB_TOKEN is not configured."}

    if not path.startswith("/") or path.startswith("//"):
        return {"error": "GitHub API paths must be relative paths beginning with '/'."}

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    headers.update(kwargs.pop("headers", {}))
    timeout = aiohttp.ClientTimeout(total=60)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.request(
            method.upper(), f"https://api.github.com{path}",
            headers=headers, **kwargs
        ) as response:
            body = await response.text()
            if response.status >= 400:
                return {"error": f"GitHub HTTP {response.status}", "detail": body[:3000]}
            try:
                value = json.loads(body)
            except json.JSONDecodeError:
                value = {"content": body}
            return value if isinstance(value, dict) else {"data": value}
