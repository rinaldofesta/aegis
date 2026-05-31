"""Tool contracts — MCP as the canonical tool/data protocol.

Tool schemas are MCP-compatible JSON Schema (`inputSchema`), modeled here as plain
dicts so the core needs no engine/SDK import. The OpenAI-shape <-> MCP normalization
lives in the ADAPTER, never here (that boundary is what keeps the standard vendor-free).
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

JSONSchema = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolDef:
    name: str
    description: str
    input_schema: JSONSchema  # MCP `inputSchema` (JSON Schema)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """A tool call that EXECUTED during a turn.

    `Turn.tool_calls` is EXECUTED-only — "what actually ran". This is the one semantic
    every engine produces cleanly (Claude: the tool_use the SDK ran; Hermes: the calls
    that passed the gate). Attempts the gate BLOCKED are NOT here — they live on the
    action span in the decision-log (`harness.gate.decision="block"`), the single
    canonical home for that judgement. (The older "attempted-incl-blocked" semantic was a
    Hermes artifact: it forced a Claude adapter to fake-merge `permission_denials` to
    reproduce it. Dropped.) `id` is the engine's stable call id — never empty.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ToolResult:
    call_id: str
    content: str
    is_error: bool = False


@dataclass(frozen=True, slots=True)
class ToolScope:
    """An operator's tool allowlist — the tools it may SEE and DISPATCH.

    Allowlist, FAIL-CLOSED: anything not explicitly listed is denied; an empty scope
    grants NO tools. This is the vendor-free expression of per-operator tool isolation —
    the product is many vertical operators sharing one runtime, and a marketing operator
    must not even see (let alone call) finance's `send_payment`. Every engine restricts
    tools per agent (Hermes derives `agent.tools`/`valid_tool_names` from enabled toolsets;
    the Claude SDK takes `allowed_tools`), so the CONCEPT is neutral and lives here; the
    adapter ENFORCES it in two layers — visibility (what the model is offered) as the
    primary guard, and a fail-closed dispatch backstop. Mirrors `GatePolicy`: the core owns
    the policy, the adapter owns enforcement, and the unknown never passes.
    """

    allowed: frozenset[str] = frozenset()

    @classmethod
    def from_names(cls, names: Iterable[str]) -> "ToolScope":
        return cls(allowed=frozenset(names))

    def allows(self, name: str) -> bool:
        return name in self.allowed

    def filter(self, tools: Iterable[ToolDef]) -> list[ToolDef]:
        """Visibility filter: keep only in-scope tools (drop everything else)."""
        return [t for t in tools if t.name in self.allowed]


@runtime_checkable
class ToolRegistry(Protocol):
    def list_tools(self) -> list[ToolDef]: ...
    def get(self, name: str) -> ToolDef | None: ...
