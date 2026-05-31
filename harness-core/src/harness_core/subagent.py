"""Subagent delegation — the multi-operator core (a 'lead' fanning work to specialists).

NEUTRAL on execution: the contract says only "delegate scoped subagents -> await their
Turns". A batch of tasks MAY run concurrently (fan-out) — putting fan-out in the contract
is a CONTRACT-level choice — but HOW (out-of-process workers vs threads) is an adapter
choice, and isolation is non-negotiable: every subagent runs under ITS OWN scope
(`AgentConfig.tools`), never the parent's.

INVOCATION MODEL (explicit): orchestration is a SEQUENCE OF TURNS, not nested delegation.
The parent turn DECIDES what to delegate and returns; the orchestration loop then calls
`delegate` BETWEEN turns. `delegate` must NOT be called while a parent turn is active —
nesting a subagent's turn inside the parent's would trip the adapter's re-entrancy guard
(and that guard is exactly what keeps every turn single-active, the regime where
per-operator scope stays correct). A future mid-turn realization would break that guard, so
the posture is pinned here, not left implicit.

PARTIAL FAILURE (explicit): a subagent's OWN failure comes back AS a Turn
(`stop_reason=StopReason.ERROR`) — so one failing specialist never kills the whole fan-out.
`delegate` raises ONLY for DELEGATION failures (called mid-turn, invalid tenant, …), never
for a subagent's own error.

Decided production realization: **out-of-process workers** (one adapter/agent per process)
— a real kill-switch (kill a process, not a half-finished thread) and OS-enforced isolation,
the strongest isolation story for a trust-first product, and it dissolves shared mutable
state entirely. The in-repo REFERENCE realization is sequential and in-process (one active
turn at a time, between turns); it proves the contract and per-subagent scope isolation
without infra.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .agent import AgentConfig, Turn
from .tenant import TenantContext


@dataclass(frozen=True, slots=True)
class SubagentTask:
    """One scoped delegation: run `config` (whose `.tools` is the subagent's scope) on
    `input`, and return its Turn. `label` names the subagent in the decision-log."""

    config: AgentConfig
    input: str
    label: str = ""


@dataclass(frozen=True, slots=True)
class SubagentResult:
    """A subagent's Turn EXPLICITLY coupled to the task that produced it.

    Never correlate a fan-out's Turns by list position: a concurrent realization could drop
    or reorder a result and silently misattribute it (A's Turn read as B's). The pairing is
    carried in the data. A failed subagent has `turn.stop_reason == StopReason.ERROR`.
    """

    task: SubagentTask
    turn: Turn


@runtime_checkable
class SubagentOrchestrator(Protocol):
    """Delegate scoped subagents and await their results.

    Isolation is the invariant: each subagent runs under its own `task.config.tools` scope.
    Parallelism is realization-dependent (sequential reference; concurrent out-of-process
    workers in production) — the contract permits fan-out, it does not mandate threads.
    Call BETWEEN turns (see module docstring): delegating mid-turn is a delegation error.
    A subagent's own failure is returned as a `SubagentResult` whose `turn.stop_reason` is
    ERROR; `delegate` raises only for delegation failures.
    """

    def delegate(
        self, tasks: Sequence[SubagentTask], tenant: TenantContext
    ) -> list[SubagentResult]: ...
