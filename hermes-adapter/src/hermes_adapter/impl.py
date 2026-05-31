"""HermesAdapter — implements harness_core.Engine over Hermes.

This is the ONLY module where Hermes types appear. harness_core stays vendor-free; the
import-boundary test guarantees nothing here leaks back into it.
"""
from __future__ import annotations

from decimal import Decimal
from typing import Any, Callable

from opentelemetry import trace as _otel_trace

from harness_core import (
    Agent,
    AgentConfig,
    CanonicalUsage,
    CostResult,
    CostStatus,
    ProviderInfo,
    TenantContext,
    ToolCall,
    ToolDef,
    ToolResult,
    Turn,
)

from .normalize import mcp_tooldef_to_openai_function
from .obs import SpanEmitter

# --- Hermes (vendor) imports: confined to the adapter ---
from run_agent import AIAgent
from tools.registry import registry as hermes_registry
from agent.usage_pricing import CanonicalUsage as HermesUsage, estimate_usage_cost

_STATUS = {
    "actual": CostStatus.ACTUAL,
    "estimated": CostStatus.ESTIMATED,
    "included": CostStatus.INCLUDED,
    "unknown": CostStatus.UNKNOWN,
}


def _map_cost(hres: Any) -> CostResult:
    amount = getattr(hres, "amount_usd", None)
    status = _STATUS.get(str(getattr(hres, "status", "unknown")), CostStatus.UNKNOWN)
    source = str(getattr(hres, "source", "") or "")
    if amount is None:
        return CostResult(amount_usd=Decimal(0), status=CostStatus.UNKNOWN, source=source)
    return CostResult(amount_usd=Decimal(str(amount)), status=status, source=source)


class ScriptedTransport:
    """Mock model at the RAW model-output boundary (OpenAI shape). Accumulates usage
    across every model call so a multi-step turn sums correctly (not under-reported)."""

    def __init__(self, responses: list[Any]):
        self._responses = list(responses)
        self._i = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0

    def __call__(self, api_kwargs: Any, *args: Any, **kwargs: Any) -> Any:
        resp = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        self.calls += 1
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        return resp


class _Handle:
    """Opaque agent handle satisfying harness_core.Agent (vendor-free consumers use only
    `session_id`). Holds Hermes-specific state internally."""

    def __init__(self, agent: Any, state: dict[str, Any]):
        self._agent = agent
        self._state = state

    @property
    def session_id(self) -> str:
        return str(getattr(self._agent, "session_id", "") or "")


class HermesAdapter:
    def __init__(self, span_emitter: SpanEmitter, model_transport: Callable[..., Any] | None = None):
        self._obs = span_emitter
        self._transport = model_transport

    # --- tool registration: dir.1 normalization (MCP -> OpenAI) ---
    def register_tool(self, td: ToolDef, handler: Callable[[dict], str], toolset: str = "aegis") -> None:
        schema = mcp_tooldef_to_openai_function(td)
        hermes_registry.register(
            name=td.name,
            toolset=toolset,
            schema=schema,
            handler=lambda args, **kw: handler(args),
            check_fn=lambda: True,
            description=td.description,
            override=True,
        )
        # PALETTO #2: positively assert the tool registered — the fail-soft registry could
        # otherwise swallow a broken registration and leave a green bullet on a missing tool.
        if hermes_registry.get_entry(td.name) is None:
            raise RuntimeError(f"tool {td.name!r} failed to register in Hermes")

    def resolve_provider(self, name: str) -> ProviderInfo:
        return ProviderInfo(canonical_name=name, api_mode="chat_completions", supports_tools=True)

    def list_tools(self) -> list[ToolDef]:
        return []  # not exercised by the bullet

    def dispatch_tool(self, name: str, arguments: dict, context: dict) -> ToolResult:
        out = hermes_registry.dispatch(name, arguments)
        return ToolResult(call_id=str(context.get("call_id", "")), content=str(out), is_error=False)

    def estimate_cost(self, provider: str, model: str, usage: CanonicalUsage) -> CostResult:
        hu = HermesUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            request_count=usage.request_count,
        )
        return _map_cost(estimate_usage_cost(model, hu, provider=provider))

    def spawn_agent(self, config: AgentConfig, tenant: TenantContext) -> Agent:
        state: dict[str, Any] = {"recorded": [], "turn_ctx": None}

        def on_tool_complete(tc_id: Any, name: Any, args: Any, result: Any) -> None:
            # Translate Hermes' callback shape into the neutral Turn.tool_calls field —
            # do NOT let the callback shape leak into Turn.
            state["recorded"].append(
                ToolCall(
                    id=str(tc_id),
                    name=str(name),
                    arguments=dict(args) if isinstance(args, dict) else {},
                )
            )
            self._obs.emit_tool_span(tool_name=str(name), parent_ctx=state["turn_ctx"])

        agent = AIAgent(
            provider=config.provider_name,
            model=config.model,
            api_key=str(config.extra.get("api_key", "sk-aegis-mock")),
            base_url=str(config.extra.get("base_url", "https://api.openai.com/v1")),
            api_mode="chat_completions",
            enabled_toolsets=list(config.extra.get("toolsets", ["aegis"])),
            skip_context_files=True,
            skip_memory=True,
            quiet_mode=True,
            ephemeral_system_prompt=config.system_prompt or "You are an Aegis operator.",
            session_id=config.session_id,
            tool_complete_callback=on_tool_complete,
        )
        if self._transport is not None:
            agent._disable_streaming = True
            agent._interruptible_api_call = self._transport
        return _Handle(agent, state)

    def run_turn(self, agent: Agent, user_input: str, system_prompt: str | None = None) -> Turn:
        handle: _Handle = agent  # type: ignore[assignment]
        hagent = handle._agent
        model = str(getattr(hagent, "model", "") or "")
        provider = str(getattr(hagent, "provider", "") or "")
        with self._obs.turn_span(agent_id=handle.session_id, model=model) as turn_span:
            handle._state["turn_ctx"] = _otel_trace.set_span_in_context(turn_span)
            text = hagent.chat(user_input)
            transport = self._transport
            usage = CanonicalUsage(
                input_tokens=getattr(transport, "input_tokens", 0),
                output_tokens=getattr(transport, "output_tokens", 0),
                request_count=getattr(transport, "calls", 1) or 1,
            )
            cost = self.estimate_cost(provider, model, usage)
            self._obs.annotate_turn(turn_span, cost=cost)
        return Turn(
            text=str(text),
            tool_calls=list(handle._state["recorded"]),
            usage=usage,
            cost=cost,
            session_id=handle.session_id,
        )

    def emit_observability_event(self, event: Any) -> None:
        self._obs.emit_event(event)
