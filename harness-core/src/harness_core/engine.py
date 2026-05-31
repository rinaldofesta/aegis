"""The Engine Protocol — THE adapter surface.

This is the ONLY thing that varies per runtime. Everything else in `harness_core` is
vendor-free. An adapter (hermes_adapter, claude_adapter, ...) implements these methods by
wrapping its engine; core/consumer code depends only on this Protocol — never on an engine.

The v1 tracer-bullet exercises the first block end-to-end (spawn -> run_turn calling a
typed `echo` tool -> cost -> observability). The rest are declared now and implemented
post-bullet; they are the contracts an adapter must eventually satisfy.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable

from .agent import Agent, AgentConfig, Turn
from .cost import CanonicalUsage, CostResult
from .hooks import HookEvent, HookHandler, HookResult
from .memory import MemoryProvider
from .observability import ObservabilityEvent
from .provider import ProviderInfo
from .subagent import SubagentResult, SubagentTask
from .task import JobSpec, JobState
from .tenant import TenantContext
from .tools import ToolDef, ToolResult


@runtime_checkable
class Engine(Protocol):
    # --- v1 tracer-bullet subset ---
    def spawn_agent(self, config: AgentConfig, tenant: TenantContext) -> Agent: ...
    def run_turn(
        self, agent: Agent, user_input: str, system_prompt: str | None = None
    ) -> Turn: ...
    def resolve_provider(self, name: str) -> ProviderInfo: ...
    def list_tools(self) -> list[ToolDef]: ...
    def dispatch_tool(
        self, name: str, arguments: dict[str, Any], context: dict[str, Any]
    ) -> ToolResult: ...
    def estimate_cost(
        self, provider: str, model: str, usage: CanonicalUsage
    ) -> CostResult: ...
    def emit_observability_event(self, event: ObservabilityEvent) -> None: ...

    # --- post-bullet (declared; implemented once the bullet passes) ---
    def delegate(
        self, tasks: Sequence[SubagentTask], tenant: TenantContext
    ) -> list[SubagentResult]: ...
    def get_memory_provider(self) -> MemoryProvider: ...
    def register_hook_handler(self, event: HookEvent, handler: HookHandler) -> None: ...
    def fire_hook(self, event: HookEvent, context: dict[str, Any]) -> HookResult | None: ...
    def register_job(self, spec: JobSpec) -> str: ...
    def get_job_status(self, job_id: str) -> JobState: ...
