"""Lifecycle hooks + approval policy — the control-plane primitives.

`HookEvent` names double as the vocabulary for observability span events. The gate
EVALUATOR (logic: given action + policy -> allow/block/needs_approval) is the moat and
will be CORE-resident (first post-bullet contract); here we define only the enums/shapes.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class HookEvent(str, Enum):
    PRE_LLM_CALL = "pre_llm_call"
    POST_LLM_CALL = "post_llm_call"
    PRE_TOOL_CALL = "pre_tool_call"
    POST_TOOL_CALL = "post_tool_call"
    POST_TURN = "post_turn"


class HookDecision(str, Enum):
    ALLOW = "allow"
    BLOCK = "block"
    NEEDS_APPROVAL = "needs_approval"


class ApprovalPolicy(str, Enum):
    AUTO_APPROVE = "auto_approve"
    AUTO_DENY = "auto_deny"
    PROMPT = "prompt"


@dataclass(frozen=True, slots=True)
class HookResult:
    decision: HookDecision = HookDecision.ALLOW
    reason: str | None = None
    context_injection: str | None = None


HookHandler = Callable[[HookEvent, dict[str, Any]], "HookResult | None"]


@runtime_checkable
class HookRegistry(Protocol):
    def register(self, event: HookEvent, handler: HookHandler) -> None: ...
    def fire(self, event: HookEvent, context: dict[str, Any]) -> HookResult | None: ...
