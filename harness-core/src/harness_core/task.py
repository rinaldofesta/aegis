"""Scheduled-task contracts (status-only in v1; cancel deferred)."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class JobSpec:
    cron_expr: str
    prompt: str
    enabled_toolsets: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class JobState:
    job_id: str
    status: JobStatus
    last_run: str | None = None
    next_run: str | None = None
    output: str | None = None
    error: str | None = None


@runtime_checkable
class TaskScheduler(Protocol):
    def register_job(self, spec: JobSpec) -> str: ...
    def get_status(self, job_id: str) -> JobState: ...
