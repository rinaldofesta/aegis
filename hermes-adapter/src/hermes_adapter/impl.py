"""HermesAdapter — implements harness_core.Engine over Hermes.

Only module where Hermes types appear. The gate enforcement lives in the tool wrapper (the
path we fully control); the DECISION comes from the core GateEvaluator. Everything fails
CLOSED: no policy, an evaluator error, or any unexpected path -> do not execute.
"""
from __future__ import annotations

import json
from decimal import Decimal
from typing import Any, Callable

from opentelemetry import trace as _otel_trace

from harness_core import (
    Agent,
    AgentConfig,
    ApprovalPolicy,
    CanonicalUsage,
    CostResult,
    CostStatus,
    GateEvaluator,
    GatePolicy,
    GateResult,
    GatedAction,
    HookDecision,
    ProviderInfo,
    StopReason,
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


def _stop_from_finish(finish_reason: str | None) -> StopReason:
    """Map an OpenAI-shape finish_reason onto the portable StopReason. chat() only returns
    once the loop sees a terminal stop, so a clean return is COMPLETED unless truncated.
    (Stage C / real Hermes additionally maps turn_exit_reason; errors -> ERROR upstream.)"""
    if finish_reason == "length":
        return StopReason.LENGTH
    if finish_reason == "content_filter":
        return StopReason.ERROR
    return StopReason.COMPLETED


def _map_cost(hres: Any) -> CostResult:
    amount = getattr(hres, "amount_usd", None)
    status = _STATUS.get(str(getattr(hres, "status", "unknown")), CostStatus.UNKNOWN)
    source = str(getattr(hres, "source", "") or "")
    if amount is None:
        return CostResult(amount_usd=Decimal(0), status=CostStatus.UNKNOWN, source=source)
    return CostResult(amount_usd=Decimal(str(amount)), status=status, source=source)


class ScriptedTransport:
    """Mock model at the RAW model-output boundary (OpenAI shape). Accumulates usage across
    every model call so a multi-step turn sums correctly."""

    def __init__(self, responses: list[Any]):
        self._responses = list(responses)
        self._i = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.calls = 0
        self.last_finish_reason: str | None = None

    def __call__(self, api_kwargs: Any, *args: Any, **kwargs: Any) -> Any:
        resp = self._responses[min(self._i, len(self._responses) - 1)]
        self._i += 1
        self.calls += 1
        usage = getattr(resp, "usage", None)
        if usage is not None:
            self.input_tokens += int(getattr(usage, "prompt_tokens", 0) or 0)
            self.output_tokens += int(getattr(usage, "completion_tokens", 0) or 0)
        choices = getattr(resp, "choices", None) or []
        if choices:
            self.last_finish_reason = getattr(choices[0], "finish_reason", None)
        return resp


class _Handle:
    """Opaque agent handle satisfying harness_core.Agent (consumers use only session_id)."""

    def __init__(self, agent: Any):
        self._agent = agent

    @property
    def session_id(self) -> str:
        return str(getattr(self._agent, "session_id", "") or "")


class HermesAdapter:
    def __init__(
        self,
        span_emitter: SpanEmitter,
        model_transport: Callable[..., Any] | None = None,
        gate_policy: GatePolicy | None = None,
        approval_callback: Callable[[GatedAction], bool] | None = None,
    ):
        self._obs = span_emitter
        self._transport = model_transport
        self._gate = GateEvaluator()
        self._gate_policy = gate_policy
        self._approval_callback = approval_callback
        self._active: dict[str, Any] | None = None  # per-turn state (turn_ctx + recorded)

    # --- enforcement (adapter): map a pure GateResult + approval mechanism to an outcome ---
    def _enforce(self, res: GateResult, action: GatedAction):
        """Returns (execute, decision, approval_required, reason). Fails closed."""
        if res.decision == HookDecision.ALLOW:
            return True, HookDecision.ALLOW, False, res.reason
        if res.decision == HookDecision.BLOCK:
            return False, HookDecision.BLOCK, False, res.reason
        # NEEDS_APPROVAL — asking is enforcement, not evaluation.
        if self._approval_callback is not None:
            try:
                approved = bool(self._approval_callback(action))
            except Exception as exc:
                return False, HookDecision.BLOCK, True, (
                    f"approval callback error: {type(exc).__name__}; fail-closed block"
                )
            if approved:
                return True, HookDecision.ALLOW, True, "approved by approver"
            return False, HookDecision.BLOCK, True, "approval denied by approver"
        # No approver -> fail-safe block, BUT mark approval.required: this is the HITL queue
        # marker (decision=block AND approval.required=true), distinct from a real deny.
        return False, HookDecision.BLOCK, True, "awaiting approval, no approver"

    def register_tool(self, td: ToolDef, handler: Callable[[dict], str], toolset: str = "aegis") -> None:
        def gated_handler(args: Any, **kwargs: Any) -> str:
            action = GatedAction(
                kind="tool",
                name=td.name,
                arguments=dict(args) if isinstance(args, dict) else {},
            )
            # DECISION (core) — fail closed on no-policy or evaluator error.
            try:
                if self._gate_policy is None:
                    res = GateResult(HookDecision.BLOCK, ApprovalPolicy.AUTO_DENY, None,
                                     "no gate policy configured; fail-closed block")
                else:
                    res = self._gate.evaluate(action, self._gate_policy)
            except Exception as exc:
                res = GateResult(HookDecision.BLOCK, ApprovalPolicy.AUTO_DENY, None,
                                 f"gate evaluation error: {type(exc).__name__}; fail-closed block")
            execute, decision, approval_required, reason = self._enforce(res, action)

            active = self._active
            if active is not None:
                # The action span is the decision-log: EVERY attempt + its gate verdict
                # (blocks included). This is the single home for the blocked-vs-executed
                # judgement, so Turn.tool_calls need not carry it.
                seq = active["tool_seq"]
                active["tool_seq"] = seq + 1
                self._obs.emit_tool_span(
                    tool_name=td.name,
                    parent_ctx=active.get("turn_ctx"),
                    gate_decision=decision.value,
                    approval_policy=res.policy.value,
                    approval_required=approval_required,
                    reason=reason,
                )
                # Turn.tool_calls is EXECUTED-only ("what ran"): record ONLY when the gate
                # allows, with a stable non-empty id. Blocked attempts stay on the span
                # above — not faked into the list (which is what kept Claude portable).
                if execute:
                    active["recorded"].append(
                        ToolCall(id=f"{td.name}#{seq}", name=td.name, arguments=action.arguments)
                    )
            if execute:
                return handler(args)
            return json.dumps({"blocked": True, "decision": decision.value, "reason": reason})

        hermes_registry.register(
            name=td.name,
            toolset=toolset,
            schema=mcp_tooldef_to_openai_function(td),  # dir.1: MCP inputSchema -> OpenAI function
            handler=lambda args, **kw: gated_handler(args, **kw),
            check_fn=lambda: True,
            description=td.description,
            override=True,
        )
        # PALETTO #2: positively assert registration (fail-soft registry could hide a miss).
        if hermes_registry.get_entry(td.name) is None:
            raise RuntimeError(f"tool {td.name!r} failed to register in Hermes")

    def resolve_provider(self, name: str) -> ProviderInfo:
        return ProviderInfo(canonical_name=name, api_mode="chat_completions", supports_tools=True)

    def list_tools(self) -> list[ToolDef]:
        return []

    def dispatch_tool(self, name: str, arguments: dict, context: dict) -> ToolResult:
        out = hermes_registry.dispatch(name, arguments)
        return ToolResult(call_id=str(context.get("call_id", "")), content=str(out), is_error=False)

    def estimate_cost(self, provider: str, model: str, usage: CanonicalUsage) -> CostResult:
        hu = HermesUsage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_read_tokens=usage.cache_read_tokens,
            cache_write_tokens=usage.cache_write_tokens,
            # Hermes-internal default: its estimator wants a positive count. Our contract
            # keeps request_count=None for "unknown"; coalesce only here, not on the Turn.
            request_count=usage.request_count if usage.request_count is not None else 1,
        )
        return _map_cost(estimate_usage_cost(model, hu, provider=provider))

    def spawn_agent(self, config: AgentConfig, tenant: TenantContext) -> Agent:
        # enabled_toolsets=["aegis"] excludes Hermes built-ins from the operator, so the
        # gated set (adapter-registered tools) == the set offered to the model. Documented
        # scope: dispatch-level isolation of built-ins is post-bullet (global registry).
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
        )
        if self._transport is not None:
            agent._disable_streaming = True
            agent._interruptible_api_call = self._transport
        return _Handle(agent)

    def run_turn(self, agent: Agent, user_input: str, system_prompt: str | None = None) -> Turn:
        handle: _Handle = agent  # type: ignore[assignment]
        hagent = handle._agent
        model = str(getattr(hagent, "model", "") or "")
        provider = str(getattr(hagent, "provider", "") or "")
        self._active = {"turn_ctx": None, "recorded": [], "tool_seq": 0}
        try:
            with self._obs.turn_span(agent_id=handle.session_id, model=model) as turn_span:
                self._active["turn_ctx"] = _otel_trace.set_span_in_context(turn_span)
                text = hagent.chat(user_input)
                transport = self._transport
                usage = CanonicalUsage(
                    input_tokens=getattr(transport, "input_tokens", 0),
                    output_tokens=getattr(transport, "output_tokens", 0),
                    # real API-call count when known; None ("unknown") when no transport.
                    request_count=getattr(transport, "calls", None) or None,
                )
                # stop_reason is REAL: derived from the model's terminal finish_reason
                # (not hardcoded). Real Hermes (Stage C) additionally maps turn_exit_reason.
                stop_reason = _stop_from_finish(getattr(transport, "last_finish_reason", None))
                cost = self.estimate_cost(provider, model, usage)
                self._obs.annotate_turn(turn_span, cost=cost)
            recorded = list(self._active["recorded"])
        finally:
            self._active = None
        return Turn(
            text=str(text),
            stop_reason=stop_reason,
            tool_calls=recorded,
            usage=usage,
            cost=cost,
            session_id=handle.session_id,
        )

    def emit_observability_event(self, event: Any) -> None:
        self._obs.emit_event(event)
