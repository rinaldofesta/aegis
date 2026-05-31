"""Tracer-bullet contract test — the four teeth that separate a green bullet from a
truly green one:

  1. dir.1 ≠ dir.2: registration (MCP inputSchema -> OpenAI 'parameters') preserves the
     schema teeth (required/types/additionalProperties) — a different normalization from
     the dispatch round-trip.
  2. multi-step usage SUM: a tool turn is 2 model calls; cost must sum both.
  3. span STRUCTURE: tool span is a child of the turn span; gate/approval/provenance on the
     ACTION span; cost on the turn span.
  4. harness.* VALUES are placeholders (allow/none) until the gate evaluator exists; the
     bullet pins keys + placement, not values.
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from openai.types.chat import ChatCompletion
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from harness_core import AgentConfig, CostStatus, GenAIAttr, HarnessAttr, TenantContext
from hermes_adapter import HermesAdapter, ScriptedTransport, SpanEmitter
from hermes_adapter.echo import ECHO_TOOL, echo_handler

# Hermes' real pricing — used to compute the EXPECTED cost (paletto #3: not hardcoded).
from agent.usage_pricing import CanonicalUsage as HermesUsage, estimate_usage_cost
from tools.registry import registry as hermes_registry


def _resp(payload: dict) -> ChatCompletion:
    return ChatCompletion.model_validate(payload)


def _scripted() -> list[ChatCompletion]:
    return [
        _resp({
            "id": "c1", "object": "chat.completion", "created": 0, "model": "gpt-4o-mini",
            "choices": [{"index": 0, "finish_reason": "tool_calls", "message": {
                "role": "assistant", "content": None,
                "tool_calls": [{"id": "call_1", "type": "function", "function": {
                    "name": "echo", "arguments": json.dumps({"message": "pong", "times": 2})}}],
            }}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }),
        _resp({
            "id": "c2", "object": "chat.completion", "created": 0, "model": "gpt-4o-mini",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "Echoed for you."}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
        }),
    ]


def _run():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("aegis.test")
    transport = ScriptedTransport(_scripted())
    adapter = HermesAdapter(SpanEmitter(tracer), model_transport=transport)
    adapter.register_tool(ECHO_TOOL, echo_handler)
    cfg = AgentConfig(model="gpt-4o-mini", provider_name="openai",
                      system_prompt="spike", extra={"toolsets": ["aegis"]})
    handle = adapter.spawn_agent(cfg, TenantContext(tenant_id="t1", root=Path("/tmp")))
    turn = adapter.run_turn(handle, "Please echo 'pong' twice.")
    return turn, exporter.get_finished_spans()


def test_dir1_registration_preserves_schema_teeth():
    adapter = HermesAdapter(SpanEmitter(TracerProvider().get_tracer("x")))
    adapter.register_tool(ECHO_TOOL, echo_handler)
    params = hermes_registry.get_entry("echo").schema["parameters"]
    assert params == ECHO_TOOL.input_schema
    assert params["required"] == ["message"]
    assert params["properties"]["times"]["type"] == "integer"
    assert params["additionalProperties"] is False


def test_dispatch_usage_and_span_structure():
    turn, spans = _run()

    # dir.2: typed args round-tripped through the OpenAI tool_call
    assert turn.text == "Echoed for you."
    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.name == "echo"
    assert call.arguments == {"message": "pong", "times": 2}

    # paletto #2: usage summed across BOTH model calls (11+20 in, 7+4 out)
    assert turn.usage.input_tokens == 31
    assert turn.usage.output_tokens == 11
    assert turn.usage.request_count == 2

    # paletto #3: cost from Hermes' real pricing table, not a hardcoded guess
    expected = estimate_usage_cost(
        "gpt-4o-mini", HermesUsage(input_tokens=31, output_tokens=11, request_count=2),
        provider="openai",
    )
    exp_amt = getattr(expected, "amount_usd", None)
    if exp_amt is None:
        assert turn.cost.status == CostStatus.UNKNOWN
    else:
        assert turn.cost.amount_usd == Decimal(str(exp_amt))

    # paletto #3 (structure): tool span is a CHILD of the turn span
    turn_spans = [s for s in spans if s.name == "turn"]
    tool_spans = [s for s in spans if s.name == "tool"]
    assert len(turn_spans) == 1
    assert len(tool_spans) == 1
    tspan, toolspan = turn_spans[0], tool_spans[0]
    assert toolspan.parent is not None
    assert toolspan.parent.span_id == tspan.context.span_id

    # gate/approval/provenance keys live on the ACTION span (placeholder values ok)
    assert HarnessAttr.GATE_DECISION in toolspan.attributes
    assert HarnessAttr.APPROVAL_POLICY in toolspan.attributes
    assert HarnessAttr.PROVENANCE_REFS in toolspan.attributes
    assert toolspan.attributes[GenAIAttr.TOOL_NAME] == "echo"

    # cost on the TURN span
    assert HarnessAttr.COST_USD in tspan.attributes
