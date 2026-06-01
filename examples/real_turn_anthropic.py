"""Aegis Stage C — a REAL turn against Anthropic (Claude), the cloud-provider variant.

Unlike ds4 (local, OpenAI chat_completions, $0), this exercises:
  * api_mode="anthropic_messages" (the native Messages API, not chat_completions),
  * the PassthroughTransport reading Anthropic's raw shape (usage.input/output_tokens,
    stop_reason — NOT prompt_tokens/choices),
  * a REAL cost from Hermes' pricing table (CostStatus.ESTIMATED, a few $0.000…).

The key is NEVER passed on the command line / in chat: it is read from the environment
(ANTHROPIC_API_KEY) or from ~/.hermes/.env. Drop it there first:
    printf 'ANTHROPIC_API_KEY=sk-ant-...\n' >> ~/.hermes/.env && chmod 600 ~/.hermes/.env

    PYTHONPATH=harness-core/src:hermes-adapter/src python examples/real_turn_anthropic.py
"""
from __future__ import annotations

import os
from pathlib import Path

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from harness_core import AgentConfig, ApprovalPolicy, GatePolicy, GateRule, HarnessAttr, TenantContext
from hermes_adapter import HermesAdapter, SpanEmitter
from hermes_adapter.echo import ECHO_TOOL, echo_handler

POLICY = GatePolicy([GateRule("echo", ApprovalPolicy.AUTO_APPROVE)], default=ApprovalPolicy.AUTO_DENY)
MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")


def _load_key() -> str:
    """Read ANTHROPIC_API_KEY from the environment, else ~/.hermes/.env. Never logged."""
    key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if key:
        return key
    env_file = Path.home() / ".hermes" / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line.startswith("ANTHROPIC_API_KEY="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit(
        "ANTHROPIC_API_KEY not found. Drop it in ~/.hermes/.env:\n"
        "  printf 'ANTHROPIC_API_KEY=sk-ant-...\\n' >> ~/.hermes/.env && chmod 600 ~/.hermes/.env"
    )


def main() -> None:
    key = _load_key()
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("aegis.real.anthropic")

    adapter = HermesAdapter(SpanEmitter(tracer), gate_policy=POLICY)  # NO transport -> real mode
    adapter.register_tool(ECHO_TOOL, echo_handler)

    cfg = AgentConfig(
        model=MODEL,
        provider_name="anthropic",
        system_prompt="You are an Aegis operator. When asked to echo, call the echo tool.",
        tools=("echo",),
        extra={
            "toolsets": ["aegis"],
            "api_mode": "anthropic_messages",            # native Messages API, not chat_completions
            "base_url": "https://api.anthropic.com",
            "api_key": key,
        },
    )
    handle = adapter.spawn_agent(cfg, TenantContext(tenant_id="t1", root=Path("/tmp")))
    turn = adapter.run_turn(handle, "Use the echo tool to echo the message 'pong' exactly two times.")

    print(f"=== REAL turn via Anthropic ({MODEL}) ===")
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
