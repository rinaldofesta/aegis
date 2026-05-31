"""Tracer-bullet + gate contract tests.

Bullet teeth (unchanged): dir.1 registration preserves schema teeth; dir.2 typed args
round-trip; multi-step usage SUM; cost from Hermes' real pricing; span structure
(tool child-of turn, harness.* placement).

Gate teeth (the moat — enforcement, not labels):
  - ALLOW policy -> echo executes, span gate.decision="allow".
  - BLOCK policy -> echo handler is NOT called (execution prevented), span gate.decision="block".
  - UNMATCHED action + safe PROMPT default + no approver -> fail-closed BLOCK with
    approval.required=true (the HITL queue marker), reason "awaiting approval".
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

from openai.types.chat import ChatCompletion
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from harness_core import (
    AgentConfig,
    ApprovalPolicy,
    CostStatus,
    GatePolicy,
    GateRule,
    GenAIAttr,
    HarnessAttr,
    TenantContext,
)
from hermes_adapter import HermesAdapter, ScriptedTransport, SpanEmitter
from hermes_adapter.echo import ECHO_TOOL, echo_handler

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
                    "name": "echo", "arguments": json.dumps({"message": "pong", "times": 2})}}]}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }),
        _resp({
            "id": "c2", "object": "chat.completion", "created": 0, "model": "gpt-4o-mini",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "Echoed for you."}}],
            "usage": {"prompt_tokens": 20, "completion_tokens": 4, "total_tokens": 24},
        }),
    ]


_ALLOW_ECHO = GatePolicy([GateRule("echo", ApprovalPolicy.AUTO_APPROVE)], default=ApprovalPolicy.AUTO_DENY)


def _run(policy: GatePolicy, approval_callback=None):
    """Returns (turn, spans, exec_calls). `exec_calls` counts real echo executions."""
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("aegis.test")

    exec_calls: list[dict] = []

    def spy(args):
        exec_calls.append(args)
        return echo_handler(args)

    adapter = HermesAdapter(
        SpanEmitter(tracer),
        model_transport=ScriptedTransport(_scripted()),
        gate_policy=policy,
        approval_callback=approval_callback,
    )
    adapter.register_tool(ECHO_TOOL, spy)
    cfg = AgentConfig(model="gpt-4o-mini", provider_name="openai",
                      system_prompt="spike", extra={"toolsets": ["aegis"]})
    handle = adapter.spawn_agent(cfg, TenantContext(tenant_id="t1", root=Path("/tmp")))
    turn = adapter.run_turn(handle, "Please echo 'pong' twice.")
    return turn, exporter.get_finished_spans(), exec_calls


def _tool_span(spans):
    return next(s for s in spans if s.name == "tool")


def _turn_span(spans):
    return next(s for s in spans if s.name == "turn")


def test_dir1_registration_preserves_schema_teeth():
    adapter = HermesAdapter(SpanEmitter(TracerProvider().get_tracer("x")), gate_policy=_ALLOW_ECHO)
    adapter.register_tool(ECHO_TOOL, echo_handler)
    params = hermes_registry.get_entry("echo").schema["parameters"]
    assert params == ECHO_TOOL.input_schema
    assert params["required"] == ["message"]
    assert params["properties"]["times"]["type"] == "integer"
    assert params["additionalProperties"] is False


def test_allow_dispatch_usage_and_span_structure():
    turn, spans, exec_calls = _run(_ALLOW_ECHO)

    # dir.2: typed args round-tripped + actually executed
    assert turn.text == "Echoed for you."
    assert len(exec_calls) == 1
    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.name == "echo"
    assert call.arguments == {"message": "pong", "times": 2}

    # usage summed across BOTH model calls
    assert turn.usage.input_tokens == 31
    assert turn.usage.output_tokens == 11
    assert turn.usage.request_count == 2

    # cost from Hermes' real pricing table, not hardcoded
    expected = estimate_usage_cost(
        "gpt-4o-mini", HermesUsage(input_tokens=31, output_tokens=11, request_count=2), provider="openai")
    exp_amt = getattr(expected, "amount_usd", None)
    if exp_amt is None:
        assert turn.cost.status == CostStatus.UNKNOWN
    else:
        assert turn.cost.amount_usd == Decimal(str(exp_amt))

    # span structure: tool is a CHILD of turn; gate keys on the ACTION span, cost on turn
    tspan, toolspan = _turn_span(spans), _tool_span(spans)
    assert toolspan.parent is not None and toolspan.parent.span_id == tspan.context.span_id
    assert toolspan.attributes[GenAIAttr.TOOL_NAME] == "echo"
    assert toolspan.attributes[HarnessAttr.GATE_DECISION] == "allow"
    assert HarnessAttr.GATE_REASON in toolspan.attributes
    assert toolspan.attributes[HarnessAttr.APPROVAL_REQUIRED] is False
    assert HarnessAttr.COST_USD in tspan.attributes


def test_block_prevents_execution():
    deny = GatePolicy([GateRule("echo", ApprovalPolicy.AUTO_DENY)], default=ApprovalPolicy.AUTO_DENY)
    turn, spans, exec_calls = _run(deny)
    # THE moat tooth: blocked => the real handler never ran
    assert exec_calls == []
    # the attempt is still recorded (decision-log shows requested-but-blocked)
    assert len(turn.tool_calls) == 1
    toolspan = _tool_span(spans)
    assert toolspan.attributes[HarnessAttr.GATE_DECISION] == "block"
    assert toolspan.attributes[HarnessAttr.APPROVAL_REQUIRED] is False  # a real deny, not awaiting approval


def test_unmatched_action_fails_closed():
    # No rule for echo, safe PROMPT default, NO approver -> fail-closed block + HITL marker.
    prompt_default = GatePolicy([], default=ApprovalPolicy.PROMPT)
    turn, spans, exec_calls = _run(prompt_default)
    assert exec_calls == []  # fail closed on the unmatched path
    toolspan = _tool_span(spans)
    assert toolspan.attributes[HarnessAttr.GATE_DECISION] == "block"
    assert toolspan.attributes[HarnessAttr.APPROVAL_REQUIRED] is True  # the HITL queue marker
    assert "awaiting approval" in toolspan.attributes[HarnessAttr.GATE_REASON]


def test_needs_approval_with_approver_executes():
    prompt_default = GatePolicy([GateRule("echo", ApprovalPolicy.PROMPT)], default=ApprovalPolicy.AUTO_DENY)
    turn, spans, exec_calls = _run(prompt_default, approval_callback=lambda action: True)
    assert len(exec_calls) == 1  # approver said yes -> executed
    toolspan = _tool_span(spans)
    assert toolspan.attributes[HarnessAttr.GATE_DECISION] == "allow"
    assert toolspan.attributes[HarnessAttr.APPROVAL_REQUIRED] is True
