"""Observability emission (adapter-side). OTel-GenAI envelope + pinned `harness.*` keys.

The core defines the KEY NAMES; the adapter emits (the HOW is vendor-specific). The tool
(ACTION) span is a CHILD of the turn span, and gate/approval/provenance keys live on the
ACTION span — that placement is what makes the derived decision-log correct. Gate VALUES
are now real (computed by the core GateEvaluator), no longer placeholders.
"""
from __future__ import annotations

import json
from contextlib import contextmanager

from harness_core import GenAIAttr, HarnessAttr, ObservabilityEvent


class SpanEmitter:
    def __init__(self, tracer):
        self._tracer = tracer

    @contextmanager
    def turn_span(self, *, agent_id: str, model: str):
        with self._tracer.start_as_current_span("turn") as span:
            span.set_attribute(GenAIAttr.OPERATION_NAME, "turn")
            if model:
                span.set_attribute(GenAIAttr.REQUEST_MODEL, model)
            if agent_id:
                span.set_attribute(HarnessAttr.AGENT_ID, agent_id)
            yield span

    def emit_tool_span(
        self,
        *,
        tool_name: str,
        parent_ctx=None,
        gate_decision: str = "block",
        approval_policy: str = "auto_deny",
        approval_required: bool = False,
        reason: str = "",
    ) -> None:
        # ACTION span, explicit child of the turn span (thread-safe). Defaults fail-closed.
        with self._tracer.start_as_current_span("tool", context=parent_ctx) as span:
            span.set_attribute(GenAIAttr.OPERATION_NAME, "execute_tool")
            span.set_attribute(GenAIAttr.TOOL_NAME, tool_name)
            span.set_attribute(HarnessAttr.GATE_DECISION, gate_decision)
            span.set_attribute(HarnessAttr.GATE_REASON, reason)
            span.set_attribute(HarnessAttr.APPROVAL_POLICY, approval_policy)
            span.set_attribute(HarnessAttr.APPROVAL_REQUIRED, approval_required)
            span.set_attribute(HarnessAttr.PROVENANCE_REFS, json.dumps([]))  # wired with memory/retrieval later

    def annotate_turn(self, span, *, cost) -> None:
        if cost is not None and cost.amount_usd is not None:
            span.set_attribute(HarnessAttr.COST_USD, float(cost.amount_usd))

    def emit_event(self, event: ObservabilityEvent) -> None:
        with self._tracer.start_as_current_span(event.name) as span:
            for key, value in event.attributes.items():
                span.set_attribute(key, value)
