"""harness-core — vendor-free contracts for the Aegis agent harness.

Zero engine/vendor imports. Adopts MCP (tool/data) and OpenTelemetry GenAI (observability)
at the wire level. Adapters (hermes_adapter, claude_adapter) implement `Engine`.
"""
from __future__ import annotations

from .agent import Agent, AgentConfig, StopReason, Turn
from .cost import CanonicalUsage, CostResult, CostStatus
from .engine import Engine
from .gate import GateEvaluator, GatePolicy, GateResult, GateRule, GatedAction
from .hooks import (
    ApprovalPolicy,
    HookDecision,
    HookEvent,
    HookHandler,
    HookRegistry,
    HookResult,
)
from .memory import MemoryProvider
from .observability import (
    HARNESS_ATTR_KEYS,
    GenAIAttr,
    HarnessAttr,
    ObservabilityEmitter,
    ObservabilityEvent,
    SpanKind,
)
from .provider import ProviderInfo, ProviderRegistry
from .task import JobSpec, JobState, JobStatus, TaskScheduler
from .tenant import TenantContext
from .tools import JSONSchema, ToolCall, ToolDef, ToolRegistry, ToolResult

__version__ = "0.1.0"

__all__ = [
    "Agent",
    "AgentConfig",
    "StopReason",
    "Turn",
    "Engine",
    "GatedAction",
    "GateRule",
    "GatePolicy",
    "GateResult",
    "GateEvaluator",
    "CanonicalUsage",
    "CostResult",
    "CostStatus",
    "ApprovalPolicy",
    "HookDecision",
    "HookEvent",
    "HookHandler",
    "HookRegistry",
    "HookResult",
    "MemoryProvider",
    "HARNESS_ATTR_KEYS",
    "GenAIAttr",
    "HarnessAttr",
    "ObservabilityEmitter",
    "ObservabilityEvent",
    "SpanKind",
    "ProviderInfo",
    "ProviderRegistry",
    "JobSpec",
    "JobState",
    "JobStatus",
    "TaskScheduler",
    "TenantContext",
    "JSONSchema",
    "ToolCall",
    "ToolDef",
    "ToolRegistry",
    "ToolResult",
]
