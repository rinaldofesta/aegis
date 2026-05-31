"""Agent config + the load-bearing `Turn` result.

NEUTRALITY RULE for `Turn`: every field must be populatable by ANY engine adapter
(Hermes, Claude Agent SDK, ...) with the SAME semantics. Engine-specific extras go in
`raw` and core consumers MUST NOT depend on them — otherwise the 'vendor-free' core
becomes Hermes-shaped.

Verified field-by-field against the Claude Agent SDK (2026-05): `text`, `stop_reason`,
`tool_calls`, `usage`, `cost`, `session_id` all map cleanly; `raw` is the explicit
escape hatch. Per-turn observability is emitted through the adapter's span side-channel,
NOT carried on `Turn` (the old `events` field was vestigial — never populated, read by no
consumer — and was dropped to keep a single home for the decision-log).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from .cost import CanonicalUsage, CostResult
from .tools import ToolCall


class StopReason(str, Enum):
    """Why a turn ended — a portable concept BOTH engines expose (Hermes `finish_reason`
    + `turn_exit_reason`; Claude `ResultMessage.stop_reason` + `subtype`). Adapters MUST
    map their engine's native signal onto one of these; the verbose engine-native string
    (and any structured error payload) goes in `Turn.raw`."""

    COMPLETED = "completed"      # finished normally (end_turn / stop)
    MAX_TURNS = "max_turns"      # hit the agentic turn/step ceiling
    LENGTH = "length"            # truncated on the output-token limit
    ERROR = "error"              # the turn errored out
    INTERRUPTED = "interrupted"  # cancelled / cut short before finishing
    BLOCKED = "blocked"          # halted by the gate (action refused)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str
    provider_name: str
    system_prompt: str | None = None
    session_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Agent(Protocol):
    """Opaque handle to a spawned agent. Vendor-free consumers use only `session_id`."""

    @property
    def session_id(self) -> str: ...


@dataclass(slots=True)
class Turn:
    text: str
    stop_reason: StopReason
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: CanonicalUsage = field(default_factory=CanonicalUsage)
    cost: CostResult | None = None
    session_id: str | None = None
    raw: dict[str, Any] | None = None  # adapter escape hatch — NOT part of the contract
