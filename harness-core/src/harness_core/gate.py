"""Gate evaluator — the moat. PURE, DETERMINISTIC, FAILS CLOSED.

`(action, policy) -> decision` where decision ∈ {allow, block, needs_approval}. The
DECISION lives here (core); ENFORCEMENT (execute/skip, and the act of ASKING for approval)
lives in the adapter. `needs_approval` is a terminal decision — `evaluate()` never asks; the
adapter's approval_callback does. Anything the evaluator cannot resolve to an explicit ALLOW
resolves to a non-executing decision (fail closed). A fail-OPEN gate is worse than no gate.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .hooks import ApprovalPolicy, HookDecision

__all__ = ["GatedAction", "GateRule", "GatePolicy", "GateResult", "GateEvaluator"]


@dataclass(frozen=True, slots=True)
class GatedAction:
    kind: str  # "tool" (extensible to non-tool actions later)
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)
    attributes: dict[str, Any] = field(default_factory=dict)  # e.g. budget_usd, target — for future rules


@dataclass(frozen=True, slots=True)
class GateRule:
    match: str  # exact action name, or "*" wildcard
    policy: ApprovalPolicy


@dataclass(frozen=True, slots=True)
class GateResult:
    decision: HookDecision
    policy: ApprovalPolicy
    matched: str | None
    reason: str


_POLICY_TO_DECISION: dict[ApprovalPolicy, HookDecision] = {
    ApprovalPolicy.AUTO_APPROVE: HookDecision.ALLOW,
    ApprovalPolicy.AUTO_DENY: HookDecision.BLOCK,
    ApprovalPolicy.PROMPT: HookDecision.NEEDS_APPROVAL,
}


class GatePolicy:
    """Ordered rules + a SAFE default. Precedence is most-specific-wins:
    exact > '*' > default. Duplicate exact rules and multiple '*' rules are rejected at
    construction (config error). `default` leans safe (PROMPT) — never a silent ALLOW.

    The future "safe starter" taxonomy (spend/deploy/email -> deny/prompt) ships as one of
    these policies (data handed to an operator), NOT as evaluator logic — keeping the
    evaluator a pure, reusable mechanism.
    """

    def __init__(
        self,
        rules: tuple[GateRule, ...] | list[GateRule] = (),
        *,
        default: ApprovalPolicy = ApprovalPolicy.PROMPT,
    ):
        self.rules = tuple(rules)
        self.default = default
        self._exact: dict[str, ApprovalPolicy] = {}
        wild: list[ApprovalPolicy] = []
        for rule in self.rules:
            if rule.match == "*":
                wild.append(rule.policy)
            elif rule.match in self._exact:
                raise ValueError(f"duplicate exact gate rule for {rule.match!r}")
            else:
                self._exact[rule.match] = rule.policy
        if len(wild) > 1:
            raise ValueError("multiple '*' wildcard gate rules")
        self._wild: ApprovalPolicy | None = wild[0] if wild else None


class GateEvaluator:
    """Pure and deterministic. Unknown/garbage policy -> BLOCK (fail closed)."""

    def evaluate(self, action: GatedAction, policy: GatePolicy) -> GateResult:
        if action.name in policy._exact:  # exact wins
            return self._resolve(policy._exact[action.name], action.name,
                                 f"matched exact rule {action.name!r}")
        if policy._wild is not None:  # then wildcard
            return self._resolve(policy._wild, "*", "matched wildcard '*'")
        return self._resolve(policy.default, None, "fell through to default")  # then safe default

    @staticmethod
    def _resolve(policy: ApprovalPolicy, matched: str | None, why: str) -> GateResult:
        decision = _POLICY_TO_DECISION.get(policy)
        if decision is None:  # unknown policy value -> fail closed
            return GateResult(HookDecision.BLOCK, ApprovalPolicy.AUTO_DENY, matched,
                              f"unknown policy {policy!r}; fail-closed block")
        return GateResult(decision, policy, matched, why)
