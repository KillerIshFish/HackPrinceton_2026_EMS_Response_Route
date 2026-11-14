"""Shared test setup.

* Call events go to a temporary call log, never data/processed/call_log.jsonl.
* The suite stays offline: a new console session starts with the rule parser whatever key is
  in ``.env`` (``PARSER_MODE=rules``), and no prompt-cache warm-up request is sent
  (``LLM_PREWARM=0``). Tests of the model path inject a stub client.
"""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _temp_call_log(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    path = tmp_path / "call_log.jsonl"
    monkeypatch.setenv("CALL_LOG_PATH", str(path))
    return path


@pytest.fixture(autouse=True)
def _offline_parser(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PARSER_MODE", "rules")
    monkeypatch.setenv("LLM_PREWARM", "0")
    for name in ("LLM_THINKING", "LLM_EFFORT", "LLM_STRICT_TOOLS"):
        monkeypatch.delenv(name, raising=False)
