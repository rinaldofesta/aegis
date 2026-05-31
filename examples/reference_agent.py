"""Aegis tracer-bullet (runnable).

One real turn through the boundary: harness_core contracts -> hermes-adapter -> Hermes'
real loop dispatching the typed `echo` tool, with a scripted (mock) model at the raw
model-output boundary. No creds, no spend. Spans go to an in-memory exporter and are
printed so the structure is visible.

    PYTHONPATH=harness-core/src:hermes-adapter/src python examples/reference_agent.py
"""
from __future__ import annotations

import json
from pathlib import Path

from openai.types.chat import ChatCompletion
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from harness_core import AgentConfig, TenantContext
from hermes_adapter import HermesAdapter, ScriptedTransport, SpanEmitter
from hermes_adapter.echo import ECHO_TOOL, echo_handler


def _resp(payload: dict) -> ChatCompletion:
    return ChatCompletion.model_validate(payload)


SCRIPTED = [
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


def main() -> None:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("aegis.reference")

    adapter = HermesAdapter(SpanEmitter(tracer), model_transport=ScriptedTransport(SCRIPTED))
    adapter.register_tool(ECHO_TOOL, echo_handler)

    cfg = AgentConfig(model="gpt-4o-mini", provider_name="openai", system_prompt="bullet",
                      extra={"toolsets": ["aegis"]})
    handle = adapter.spawn_agent(cfg, TenantContext(tenant_id="t1", root=Path("/tmp")))
    turn = adapter.run_turn(handle, "Please echo 'pong' twice.")

    print("text:        ", turn.text)
    print("tool_calls:  ", [(c.name, c.arguments) for c in turn.tool_calls])
    print("usage:       ", turn.usage)
    print("cost:        ", turn.cost)
    print("spans:")
    spans = exporter.get_finished_spans()
    by_id = {s.context.span_id: s for s in spans}
    for s in spans:
        parent = by_id.get(s.parent.span_id).name if s.parent else "—"
        harness_keys = sorted(k for k in s.attributes if k.startswith("harness."))
        print(f"  - {s.name:5s} parent={parent:5s} harness.*={harness_keys}")


if __name__ == "__main__":
    main()
