"""Agent config + the load-bearing `Turn` result.

NEUTRALITY RULE for `Turn`: every field must be populatable by ANY engine adapter
(Hermes, Claude Agent SDK, ...). Engine-specific extras go in `raw` and core consumers
MUST NOT depend on them — otherwise the 'vendor-free' core becomes Hermes-shaped.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from .cost import CanonicalUsage, CostResult
from .observability import ObservabilityEvent
from .tools import ToolCall


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
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: CanonicalUsage = field(default_factory=CanonicalUsage)
    cost: CostResult | None = None
    session_id: str | None = None
    events: list[ObservabilityEvent] = field(default_factory=list)
    raw: dict[str, Any] | None = None  # adapter escape hatch — NOT part of the contract
