"""Tool contracts — MCP as the canonical tool/data protocol.

Tool schemas are MCP-compatible JSON Schema (`inputSchema`), modeled here as plain
dicts so the core needs no engine/SDK import. The OpenAI-shape <-> MCP normalization
lives in the ADAPTER, never here (that boundary is what keeps the standard vendor-free).
"""
from __future__ import annotations

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


@runtime_checkable
class ToolRegistry(Protocol):
    def list_tools(self) -> list[ToolDef]: ...
    def get(self, name: str) -> ToolDef | None: ...
