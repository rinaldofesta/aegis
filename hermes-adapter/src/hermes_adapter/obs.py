"""Observability emission (adapter-side). OTel-GenAI envelope + pinned `harness.*` keys.

The core defines the KEY NAMES; the adapter does the actual emission (the HOW is
vendor-specific). Span structure matters: the tool (ACTION) span is a CHILD of the turn
span, and gate/approval/provenance keys live on the ACTION span — that is what makes the
derived decision-log correct rather than a bag of scattered attributes.
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

    def emit_tool_span(self, *, tool_name: str, parent_ctx=None) -> None:
        # ACTION span, explicit child of the turn span (thread-safe even if Hermes runs
        # tools on a worker thread).
        with self._tracer.start_as_current_span("tool", context=parent_ctx) as span:
            span.set_attribute(GenAIAttr.OPERATION_NAME, "execute_tool")
            span.set_attribute(GenAIAttr.TOOL_NAME, tool_name)
            # PALETTO #4: pin KEYS + PLACEMENT here; VALUES are placeholders until the
            # core-resident gate evaluator exists (post-bullet).
            span.set_attribute(HarnessAttr.GATE_DECISION, "allow")
            span.set_attribute(HarnessAttr.APPROVAL_POLICY, "none")
            span.set_attribute(HarnessAttr.APPROVAL_REQUIRED, False)
            span.set_attribute(HarnessAttr.PROVENANCE_REFS, json.dumps([]))  # JSON-encoded list

    def annotate_turn(self, span, *, cost) -> None:
        if cost is not None and cost.amount_usd is not None:
            span.set_attribute(HarnessAttr.COST_USD, float(cost.amount_usd))

    def emit_event(self, event: ObservabilityEvent) -> None:
        with self._tracer.start_as_current_span(event.name) as span:
            for key, value in event.attributes.items():
                span.set_attribute(key, value)
