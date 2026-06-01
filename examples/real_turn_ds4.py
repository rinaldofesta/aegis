"""Aegis Stage C — a REAL turn through the full chain against a live local model (ds4 /
DwarfStar serving DeepSeek V4 Flash on Metal, OpenAI-compatible on :8000).

No mock: HermesAdapter is built WITHOUT a model_transport, so spawn_agent wraps Hermes'
real model call in a PassthroughTransport — the same boundary as the mock, now hitting ds4.
No creds, no external egress (localhost), tool-capable. Proves the plumbing end-to-end:
scoped spawn -> real turn -> gate on the echo action -> real usage/cost -> OTel spans.

    PYTHONPATH=harness-core/src:hermes-adapter/src python examples/real_turn_ds4.py
"""
from __future__ import annotations

from pathlib import Path

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from harness_core import AgentConfig, ApprovalPolicy, GatePolicy, GateRule, HarnessAttr, TenantContext
from hermes_adapter import HermesAdapter, SpanEmitter
from hermes_adapter.echo import ECHO_TOOL, echo_handler

POLICY = GatePolicy([GateRule("echo", ApprovalPolicy.AUTO_APPROVE)], default=ApprovalPolicy.AUTO_DENY)


def main() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("aegis.real.ds4")

    # NO model_transport => real mode: the adapter wraps Hermes' real call (-> ds4).
    adapter = HermesAdapter(SpanEmitter(tracer), gate_policy=POLICY)
    adapter.register_tool(ECHO_TOOL, echo_handler)

    cfg = AgentConfig(
        model="deepseek-chat",          # non-thinking mode on ds4 (fast, few tokens)
        provider_name="custom",         # local OpenAI-compatible endpoint
        system_prompt="You are an Aegis operator. When asked to echo, call the echo tool.",
        tools=("echo",),                # operator scope: echo only
        extra={
            "toolsets": ["aegis"],
            "base_url": "http://localhost:8000/v1",
            "api_key": "ds4-local",     # dummy: ds4 ignores auth
        },
    )
    handle = adapter.spawn_agent(cfg, TenantContext(tenant_id="t1", root=Path("/tmp")))
    turn = adapter.run_turn(handle, "Use the echo tool to echo the message 'pong' exactly two times.")

    print("=== REAL turn via ds4 (DeepSeek V4 Flash) ===")
    print("text:        ", repr(turn.text))
    print("stop_reason: ", turn.stop_reason)
    print("tool_calls:  ", [(c.name, c.arguments) for c in turn.tool_calls])
    print("usage:       ", turn.usage)
    print("cost:        ", turn.cost)
    print("session_id:  ", turn.session_id)
    print("spans:")
    spans = exporter.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    for s in spans:
        parent = by_id.get(s.parent.span_id).name if s.parent else "—"
        gate = s.attributes.get(HarnessAttr.GATE_DECISION)
        extra = f" gate={gate}" if gate else ""
        print(f"  - {s.name:5s} parent={parent:5s}{extra}")


if __name__ == "__main__":
    main()
