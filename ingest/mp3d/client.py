"""Matterport Model API transport.

The API is a single GraphQL endpoint using HTTP Basic auth, where the username
is the token *id* (``MATTERPORT_API_KEY``) and the password is the token
*secret* (``MATTERPORT_API_TOKEN``).

Note: the API omits null fields from responses entirely rather than returning
them as ``null``, so every consumer must use ``.get()`` and never index.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Mapping, Optional

ENDPOINT = "https://api.matterport.com/api/models/graph"
# Assumed: generous ceiling for an offline ingest call; a GEOMETRY fetch measures ~225 ms (see fetch.py), so 60 s only
# trips on a hung connection.
DEFAULT_TIMEOUT_S = 60.0
# Tuned heuristic: 3 tries with linear backoff (0.75 s, then 1.5 s) rides out a brief 429/5xx blip while keeping a
# dead endpoint's worst case to roughly 3 timeouts plus 2.25 s of sleep.
MAX_ATTEMPTS = 3
BACKOFF_S = 0.75


class MatterportError(RuntimeError):
    """A transport, auth or GraphQL-level failure."""


def _load_dotenv_once() -> None:
    try:
        from dotenv import load_dotenv
    except Exception:  # noqa: BLE001 - python-dotenv is optional at runtime
        return
    load_dotenv()


class MatterportClient:
    """Minimal GraphQL client. Stateless apart from credentials."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        api_token: Optional[str] = None,
        endpoint: str = ENDPOINT,
        timeout_s: float = DEFAULT_TIMEOUT_S,
    ) -> None:
        if api_key is None or api_token is None:
            _load_dotenv_once()
        self.api_key = api_key or os.environ.get("MATTERPORT_API_KEY") or ""
        self.api_token = api_token or os.environ.get("MATTERPORT_API_TOKEN") or ""
        if not self.api_key or not self.api_token:
            raise MatterportError(
                "Missing credentials: set MATTERPORT_API_KEY (token id) and "
                "MATTERPORT_API_TOKEN (token secret) in .env - see .env.example."
            )
        self.endpoint = endpoint
        self.timeout_s = timeout_s
        self._auth = base64.b64encode(
            f"{self.api_key}:{self.api_token}".encode()
        ).decode()

    def query(self, document: str, variables: Optional[Mapping[str, Any]] = None) -> dict:
        """Execute a GraphQL document and return ``data``.

        Raises :class:`MatterportError` on transport failure or GraphQL errors.
        Retries 5xx and 429 responses with linear backoff.
        """
        body = json.dumps(
            {"query": document, "variables": dict(variables or {})}
        ).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Basic {self._auth}",
            "Accept": "application/json",
        }
        last: Optional[str] = None
        for attempt in range(1, MAX_ATTEMPTS + 1):
            request = urllib.request.Request(self.endpoint, data=body, headers=headers)
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
                    payload = json.loads(response.read().decode())
                break
            except urllib.error.HTTPError as exc:
                detail = exc.read()[:400].decode("utf8", "ignore")  # cap the body echoed into the error message
                if exc.code in (429, 500, 502, 503, 504) and attempt < MAX_ATTEMPTS:
                    last = f"HTTP {exc.code}: {detail}"
                    time.sleep(BACKOFF_S * attempt)
                    continue
                if exc.code in (401, 403):
                    raise MatterportError(
                        f"Authentication rejected (HTTP {exc.code}). Check that "
                        "MATTERPORT_API_KEY is the token id and MATTERPORT_API_TOKEN "
                        f"is the secret. Server said: {detail}"
                    ) from exc
                raise MatterportError(f"HTTP {exc.code}: {detail}") from exc
            except Exception as exc:  # noqa: BLE001 - urllib raises a wide range
                if attempt < MAX_ATTEMPTS:
                    last = f"{type(exc).__name__}: {exc}"
                    time.sleep(BACKOFF_S * attempt)
                    continue
                raise MatterportError(f"{type(exc).__name__}: {exc}") from exc
        else:  # pragma: no cover - loop always breaks or raises
            raise MatterportError(last or "request failed")

        if payload.get("errors"):
            # First 4 GraphQL errors only: enough to diagnose, short enough to read in a traceback.
            messages = "; ".join(
                str(e.get("message", e)) for e in payload["errors"][:4]
            )
            raise MatterportError(f"GraphQL error: {messages}")
        data = payload.get("data")
        if data is None:
            raise MatterportError("GraphQL response carried no data")
        return data
