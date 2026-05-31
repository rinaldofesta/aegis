"""Observability — the moat. THIN on transport, PRECISE on vocabulary.

Transport: we ADOPT the OpenTelemetry GenAI semantic conventions (`gen_ai.*`). HOW spans
are emitted is adapter-specific (Hermes builds them; the Claude SDK maps its hooks->spans).

Vocabulary: the fields that have NO `gen_ai.*` equivalent — gate decision, provenance,
approval outcome, kill switch — are pinned here as `harness.*` keys. This is the precise
exception to "adopt, don't invent": without pinned key NAMES the derived decision-log would
not be portable/queryable across adapters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class SpanKind(str, Enum):
    TURN = "turn"
    TOOL = "tool"
    SUBAGENT = "subagent"
    LLM = "llm"


class GenAIAttr:
    """OTel GenAI semantic-convention keys we ADOPT (do not invent)."""

    SYSTEM = "gen_ai.system"
    OPERATION_NAME = "gen_ai.operation.name"
    REQUEST_MODEL = "gen_ai.request.model"
    USAGE_INPUT_TOKENS = "gen_ai.usage.input_tokens"
    USAGE_OUTPUT_TOKENS = "gen_ai.usage.output_tokens"
    TOOL_NAME = "gen_ai.tool.name"


class HarnessAttr:
    """The pinned `harness.*` vocabulary — keys with NO `gen_ai.*` equivalent.

    The decision-log is DERIVED from these attributes; pinning the names in the core is
    what makes it portable across the Hermes and Claude adapters.
    """

    GATE_DECISION = "harness.gate.decision"          # HookDecision value
    APPROVAL_POLICY = "harness.approval.policy"       # ApprovalPolicy value
    APPROVAL_REQUIRED = "harness.approval.required"   # bool
    PROVENANCE_REFS = "harness.provenance.refs"       # list[str] of source refs
    KILL_REQUESTED = "harness.kill.requested"         # bool
    TENANT_ID = "harness.tenant.id"
    AGENT_ID = "harness.agent.id"
    COST_USD = "harness.cost.usd"


# Frozen set of allowed harness.* keys — guards against typo'd/ad-hoc keys leaking in.
HARNESS_ATTR_KEYS: frozenset[str] = frozenset(
    v for k, v in vars(HarnessAttr).items() if not k.startswith("_") and isinstance(v, str)
)


@dataclass(frozen=True, slots=True)
class ObservabilityEvent:
    kind: SpanKind
    name: str
    attributes: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class ObservabilityEmitter(Protocol):
    def emit(self, event: ObservabilityEvent) -> None: ...
