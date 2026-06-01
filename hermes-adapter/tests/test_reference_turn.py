"""Tracer-bullet + gate contract tests.

Bullet teeth (unchanged): dir.1 registration preserves schema teeth; dir.2 typed args
round-trip; multi-step usage SUM; cost from Hermes' real pricing; span structure
(tool child-of turn, harness.* placement).

Gate teeth (the moat — enforcement, not labels):
  - ALLOW policy -> echo executes, span gate.decision="allow", recorded with a real id.
  - BLOCK policy -> echo handler is NOT called; tool_calls is EXECUTED-only so a blocked
    call is ABSENT from Turn.tool_calls but its block verdict IS on the decision-log span.
  - UNMATCHED action + safe PROMPT default + no approver -> fail-closed BLOCK with
    approval.required=true (the HITL queue marker), reason "awaiting approval".
"""
from __future__ import annotations

import json
from decimal import Decimal
from pathlib import Path

import pytest

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
    StopReason,
    SubagentTask,
    TenantContext,
    ToolDef,
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
                      system_prompt="spike", tools=("echo",), extra={"toolsets": ["aegis"]})
    handle = adapter.spawn_agent(cfg, TenantContext(tenant_id="t1", root=Path("/tmp")))
    turn = adapter.run_turn(handle, "Please echo 'pong' twice.")
    return turn, exporter.get_finished_spans(), exec_calls


class _ZeroSuccessTransport:
    """Simulates Hermes degrading a TOTAL API failure: text comes back, but the transport
    OBSERVED zero successful model responses (calls stays 0) — as when every attempt 404'd."""

    def __init__(self):
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0  # deliberately never incremented: zero successes observed
        self.last_finish_reason = None

    def __call__(self, api_kwargs, *a, **k):
        return _resp({
            "id": "err", "object": "chat.completion", "created": 0, "model": "x",
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": "API call failed after retries: HTTP 404"}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
        })


def test_errored_turn_maps_to_error_stop_reason():
    """Observed-zero successful model responses -> StopReason.ERROR (not COMPLETED). Guards the
    orchestrator's partial-failure invariant: a failed turn must never read as completed."""
    adapter = HermesAdapter(SpanEmitter(TracerProvider().get_tracer("aegis.err")),
                            model_transport=_ZeroSuccessTransport(), gate_policy=_ALLOW_ECHO)
    adapter.register_tool(ECHO_TOOL, echo_handler)
    cfg = AgentConfig(model="gpt-4o-mini", provider_name="openai", system_prompt="x",
                      tools=("echo",), extra={"toolsets": ["aegis"]})
    handle = adapter.spawn_agent(cfg, TenantContext(tenant_id="t", root=Path("/tmp")))
    turn = adapter.run_turn(handle, "anything")
    assert turn.stop_reason == StopReason.ERROR     # observed-zero successes -> errored, not completed
    assert turn.usage.request_count is None         # no successful request observed


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
    # stop_reason is REAL — derived from the terminal finish_reason ("stop"), not hardcoded
    assert turn.stop_reason == StopReason.COMPLETED
    assert len(exec_calls) == 1
    assert len(turn.tool_calls) == 1
    call = turn.tool_calls[0]
    assert call.name == "echo"
    assert call.arguments == {"message": "pong", "times": 2}
    assert call.id and call.id.startswith("echo")  # real, stable id — never the old ""

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
    # executed-only: a blocked call is ABSENT from Turn.tool_calls...
    assert turn.tool_calls == []
    # ...but the attempt + its block verdict still live on the decision-log span (the audit).
    toolspan = _tool_span(spans)
    assert toolspan.attributes[HarnessAttr.GATE_DECISION] == "block"
    assert toolspan.attributes[HarnessAttr.APPROVAL_REQUIRED] is False  # a real deny, not awaiting approval


def test_unmatched_action_fails_closed():
    # No rule for echo, safe PROMPT default, NO approver -> fail-closed block + HITL marker.
    prompt_default = GatePolicy([], default=ApprovalPolicy.PROMPT)
    turn, spans, exec_calls = _run(prompt_default)
    assert exec_calls == []  # fail closed on the unmatched path
    assert turn.tool_calls == []  # nothing executed -> nothing in the executed-only list
    toolspan = _tool_span(spans)
    assert toolspan.attributes[HarnessAttr.GATE_DECISION] == "block"
    assert toolspan.attributes[HarnessAttr.APPROVAL_REQUIRED] is True  # the HITL queue marker
    assert "awaiting approval" in toolspan.attributes[HarnessAttr.GATE_REASON]


def test_needs_approval_with_approver_executes():
    prompt_default = GatePolicy([GateRule("echo", ApprovalPolicy.PROMPT)], default=ApprovalPolicy.AUTO_DENY)
    turn, spans, exec_calls = _run(prompt_default, approval_callback=lambda action: True)
    assert len(exec_calls) == 1  # approver said yes -> executed
    assert len(turn.tool_calls) == 1  # approved -> executed -> in the executed-only list
    toolspan = _tool_span(spans)
    assert toolspan.attributes[HarnessAttr.GATE_DECISION] == "allow"
    assert toolspan.attributes[HarnessAttr.APPROVAL_REQUIRED] is True


# --- per-operator tool isolation (the multi-operator product invariant) ---

_PAY_TOOL = ToolDef(
    name="pay",
    description="Send a payment (finance-only).",
    input_schema={
        "type": "object",
        "properties": {"amount": {"type": "integer", "minimum": 1}},
        "required": ["amount"],
        "additionalProperties": False,
    },
)


def test_operator_tool_isolation():
    """Two operators share the process-global registry; the marketing operator must NOT
    see — nor be able to dispatch — finance's `pay`. Visibility (primary, native) plus our
    own dispatch backstop (independent of Hermes' valid_tool_names check)."""
    tracer = TracerProvider().get_tracer("aegis.iso")
    adapter = HermesAdapter(SpanEmitter(tracer), gate_policy=_ALLOW_ECHO)
    adapter.register_tool(ECHO_TOOL, echo_handler)            # marketing's tool

    pay_calls: list = []

    def pay_handler(args):
        pay_calls.append(args)
        return "{}"

    adapter.register_tool(_PAY_TOOL, pay_handler)             # finance's tool, same registry

    # marketing operator — scope is echo only
    cfg = AgentConfig(model="gpt-4o-mini", provider_name="openai", system_prompt="marketing",
                      tools=("echo",), extra={"toolsets": ["aegis"]})
    handle = adapter.spawn_agent(cfg, TenantContext(tenant_id="mkt", root=Path("/tmp")))
    hagent = handle._agent

    # VISIBILITY (primary, native): the model is offered ONLY echo; pay is invisible AND
    # absent from Hermes' own dispatch allowlist (valid_tool_names derived from agent.tools).
    assert hagent.valid_tool_names == {"echo"}
    assert all(t["function"]["name"] != "pay" for t in (hagent.tools or []))

    # OUR DISPATCH BACKSTOP (independent of Hermes): pay is refused, its handler never runs.
    blocked = adapter.dispatch_tool("pay", {"amount": 1000}, {"scope": handle.scope})
    assert blocked.is_error
    assert pay_calls == []  # the security invariant: marketing cannot reach send-payment

    # in-scope dispatch still works
    ok = adapter.dispatch_tool("echo", {"message": "hi"}, {"scope": handle.scope})
    assert not ok.is_error


def test_empty_scope_fails_closed():
    """Allowlist, not denylist: an operator with no declared tools sees and dispatches none."""
    tracer = TracerProvider().get_tracer("aegis.iso2")
    adapter = HermesAdapter(SpanEmitter(tracer), gate_policy=_ALLOW_ECHO)
    adapter.register_tool(ECHO_TOOL, echo_handler)
    cfg = AgentConfig(model="gpt-4o-mini", provider_name="openai", tools=(),
                      extra={"toolsets": ["aegis"]})
    handle = adapter.spawn_agent(cfg, TenantContext(tenant_id="x", root=Path("/tmp")))
    assert handle._agent.valid_tool_names == set()  # sees nothing — fail-closed
    blocked = adapter.dispatch_tool("echo", {"message": "hi"}, {"scope": handle.scope})
    assert blocked.is_error  # nothing in scope -> blocked


def test_concurrent_turn_refused_fail_closed():
    """The gated_handler reads scope from self._active, so a second overlapping turn on one
    adapter would run under the wrong operator's scope. The re-entrancy guard refuses it
    (fail-closed) instead of risking a cross-operator execution."""
    tracer = TracerProvider().get_tracer("aegis.reentry")
    adapter = HermesAdapter(SpanEmitter(tracer), model_transport=ScriptedTransport(_scripted()),
                            gate_policy=_ALLOW_ECHO)
    adapter.register_tool(ECHO_TOOL, echo_handler)
    cfg = AgentConfig(model="gpt-4o-mini", provider_name="openai", system_prompt="op",
                      tools=("echo",), extra={"toolsets": ["aegis"]})
    handle = adapter.spawn_agent(cfg, TenantContext(tenant_id="t", root=Path("/tmp")))
    # simulate a turn already in flight on this adapter
    adapter._active = {"turn_ctx": None, "recorded": [], "tool_seq": 0, "scope": handle.scope}
    with pytest.raises(RuntimeError, match="concurrent/nested"):
        adapter.run_turn(handle, "echo please")


# --- subagent delegation (the 'lead' fanning work to scoped specialists) ---

def _scoped_cfg(prompt: str, tools: tuple) -> AgentConfig:
    return AgentConfig(model="gpt-4o-mini", provider_name="openai", system_prompt=prompt,
                       tools=tools, extra={"toolsets": ["aegis"]})


def test_delegate_runs_scoped_subagents_in_order():
    """The orchestrator contract: delegate scoped subagents -> await their Turns, in order.
    Reference realization is sequential (the re-entrancy guard holds across delegations)."""
    transport = ScriptedTransport(_scripted() + _scripted())  # 2 subagents x 2 model calls
    adapter = HermesAdapter(SpanEmitter(TracerProvider().get_tracer("aegis.deleg")),
                            model_transport=transport, gate_policy=_ALLOW_ECHO)
    adapter.register_tool(ECHO_TOOL, echo_handler)
    tasks = [
        SubagentTask(config=_scoped_cfg("researcher", ("echo",)), input="echo A", label="researcher"),
        SubagentTask(config=_scoped_cfg("writer", ("echo",)), input="echo B", label="writer"),
    ]
    results = adapter.delegate(tasks, TenantContext(tenant_id="t", root=Path("/tmp")))
    assert len(results) == 2  # one result per task, in order
    assert results[0].task.label == "researcher"  # explicit coupling, not positional trust
    assert results[0].turn.text == "Echoed for you."
    assert all(r.turn.stop_reason == StopReason.COMPLETED for r in results)


def test_delegate_isolates_scope_per_subagent():
    """Each delegated subagent runs under ITS OWN scope, never the parent's: an echo-scoped
    subagent executes echo; an empty-scope sibling is isolated and cannot."""
    transport = ScriptedTransport(_scripted() + _scripted())
    adapter = HermesAdapter(SpanEmitter(TracerProvider().get_tracer("aegis.deleg2")),
                            model_transport=transport, gate_policy=_ALLOW_ECHO)
    exec_calls: list = []

    def spy(args):
        exec_calls.append(args)
        return echo_handler(args)

    adapter.register_tool(ECHO_TOOL, spy)
    tasks = [
        SubagentTask(config=_scoped_cfg("has-echo", ("echo",)), input="echo", label="has-echo"),
        SubagentTask(config=_scoped_cfg("no-tools", ()), input="echo", label="no-tools"),
    ]
    results = adapter.delegate(tasks, TenantContext(tenant_id="t", root=Path("/tmp")))
    assert len(results) == 2
    # only the echo-scoped subagent could execute echo; the empty-scope one is isolated
    assert len(exec_calls) == 1


def test_delegate_between_turns_only():
    """Orchestration is a sequence of turns: delegate() during an active parent turn is
    refused (nesting would trip the re-entrancy guard) — a DELEGATION error, fail-loud."""
    adapter = HermesAdapter(SpanEmitter(TracerProvider().get_tracer("aegis.deleg3")),
                            model_transport=ScriptedTransport(_scripted()), gate_policy=_ALLOW_ECHO)
    adapter.register_tool(ECHO_TOOL, echo_handler)
    # simulate a parent turn already in flight
    adapter._active = {"turn_ctx": None, "recorded": [], "tool_seq": 0, "scope": None}
    with pytest.raises(RuntimeError, match="active turn"):
        adapter.delegate([SubagentTask(config=_scoped_cfg("a", ("echo",)), input="x", label="a")],
                         TenantContext(tenant_id="t", root=Path("/tmp")))


def test_delegate_partial_failure_does_not_kill_fanout(monkeypatch):
    """A subagent's OWN failure comes back AS an ERROR Turn, explicitly coupled to its task;
    siblings still complete; delegate() does NOT raise on a subagent failure."""
    adapter = HermesAdapter(SpanEmitter(TracerProvider().get_tracer("aegis.deleg4")),
                            model_transport=ScriptedTransport(_scripted()), gate_policy=_ALLOW_ECHO)
    adapter.register_tool(ECHO_TOOL, echo_handler)
    real_spawn = adapter.spawn_agent
    n = {"i": 0}

    def flaky_spawn(config, tenant):
        n["i"] += 1
        if n["i"] == 1:
            raise RuntimeError("spawn boom")  # first subagent fails to start
        return real_spawn(config, tenant)

    monkeypatch.setattr(adapter, "spawn_agent", flaky_spawn)
    tasks = [
        SubagentTask(config=_scoped_cfg("alpha", ("echo",)), input="x", label="alpha"),
        SubagentTask(config=_scoped_cfg("beta", ("echo",)), input="y", label="beta"),
    ]
    results = adapter.delegate(tasks, TenantContext(tenant_id="t", root=Path("/tmp")))
    assert len(results) == 2  # fan-out NOT killed by one failure
    assert results[0].task.label == "alpha"  # explicit coupling (B)
    assert results[0].turn.stop_reason == StopReason.ERROR  # failed subagent -> ERROR Turn (A)
    assert results[1].task.label == "beta"
    assert results[1].turn.stop_reason == StopReason.COMPLETED  # sibling still ran
