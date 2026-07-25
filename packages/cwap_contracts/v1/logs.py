"""Epic 4 contract — the streaming log feed.

Output contract from the brief:
    {"timestamp": ..., "node": "...", "level": "INFO", "data": ...}
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import Field

from cwap_contracts.v1.base import ContractModel, utcnow


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


class LogEvent(ContractModel):
    """One observable moment in a run. Every node entry, external call payload,
    state change and error becomes one of these — that is what makes the run
    report a trace the user can actually trust."""

    run_id: str = Field(..., min_length=1)
    seq: int = Field(default=0, ge=0, description="Monotonic within a run; lets the UI order.")
    timestamp: datetime = Field(default_factory=utcnow)
    node: str | None = Field(default=None, description="Node id, or None for run-level events.")
    level: LogLevel = LogLevel.INFO
    event: str = Field(..., min_length=1, description="Short machine-readable event name.")
    message: str = Field(default="", max_length=2000)
    data: dict[str, Any] = Field(default_factory=dict)
