"""Memory provider contract — an archive of ATTRIBUTED BELIEFS, not a key-value store.

The anti-Polsia pillar. Observability records WHAT happened; provenance records WHAT a
belief was grounded ON. Polsia's sin was not acting — it was asserting the unfounded as
fact. So every memory entry carries explicit provenance, READING memory leaves a grounding
trace (`harness.provenance.refs` on the span), and "asserted-by-the-model-without-a-source"
is a VISIBLE provenance category — never silently equal to a grounded fact.

NEUTRAL: the concept (attributed beliefs + provenance) lives here; STORAGE is the adapter's
job (Hermes already carries write metadata — write_origin/task_id/tool_call_id — onto which
this maps; `agent/memory_manager.py`). The old `prefetch/sync/build_prompt` shape was
Hermes-shaped (prompt-injection); `remember/recall` is engine-neutral (how recalled beliefs
re-enter the context is an adapter detail). FAIL-CLOSED: a belief that cites no concrete
source cannot prove grounding, so it is recorded as MODEL_ASSERTED — never grounded in
silence.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class ProvenanceKind(str, Enum):
    """How a belief came to be known — the audit category separating grounded from made-up.

    DEFINITIONAL GUARD: the grounded kinds mean the value was taken DIRECTLY from that source
    — the verbatim retrieval result, the tool's actual output, what the user said — NOT the
    model's SYNTHESIS over them. A conclusion the model draws *from* a tool output is
    MODEL_ASSERTED (honestly ungrounded), never TOOL_OUTPUT: labelling a synthesis as
    directly-sourced is exactly Polsia's lie. The fail-closed downgrade in `Provenance.make`
    covers "no source"; THIS guard covers "has a source but is a derivation" — it can only be
    a contract obligation on the writer, not machine-checkable.

    Today MODEL_ASSERTED safely ABSORBS honest derivations (a conclusion reasoned from
    grounded inputs) — the SAFE error (grounding not proven → treated as unfounded). A future
    `DERIVED` kind would separate "reasoned from refs X,Y" from "invented from nothing", but
    it is deliberately NOT added yet: it is ahead of the runtime (Hermes' write metadata
    attributes the SOURCE, not "derived-from-these-refs") and couples to the deferred
    ref→action correlation. The current state errs safe — leave it.
    """

    RETRIEVED = "retrieved"            # the retrieval result itself (verbatim), not a summary of it
    TOOL_OUTPUT = "tool_output"        # the tool's actual output, not the model's reading of it
    USER_PROVIDED = "user_provided"    # what the user/principal actually said (source = user/session)
    MODEL_ASSERTED = "model_asserted"  # the model said it: no source, OR a synthesis/derivation — UNGROUNDED

    @property
    def grounded(self) -> bool:
        return self is not ProvenanceKind.MODEL_ASSERTED


# kinds that CLAIM grounding and therefore REQUIRE a concrete source
_GROUNDED_KINDS = frozenset(
    {ProvenanceKind.RETRIEVED, ProvenanceKind.TOOL_OUTPUT, ProvenanceKind.USER_PROVIDED}
)


@dataclass(frozen=True, slots=True)
class Provenance:
    """Where a belief came from. `kind` is the audit category; `source` is the concrete
    citation (tool name, doc id, retrieval ref, …). Build via `make`, which enforces the
    fail-closed rule.

    `grounded` means "attributed to a non-model source", NOT "verified true": a USER_PROVIDED
    belief is grounded because it is traceable to the user, not because the user is right. The
    downgrade is a PRESENCE check (is there a source?); source VALIDITY is out of scope — not
    generally verifiable. For USER_PROVIDED always pass `source` (the user/session id, which
    always exists) so it never accidentally downgrades to MODEL_ASSERTED."""

    kind: ProvenanceKind
    source: str = ""
    operator_id: str = ""
    session_id: str = ""
    at: str = ""  # ISO-8601 timestamp, stamped by the adapter (kept out of the core for determinism)

    @classmethod
    def make(
        cls,
        kind: ProvenanceKind,
        *,
        source: str = "",
        operator_id: str = "",
        session_id: str = "",
        at: str = "",
    ) -> "Provenance":
        # FAIL-CLOSED: a kind that CLAIMS grounding but cites no source cannot prove it, so it
        # is downgraded to MODEL_ASSERTED — never accepted as grounded in silence.
        if kind in _GROUNDED_KINDS and not source.strip():
            kind = ProvenanceKind.MODEL_ASSERTED
        return cls(kind=kind, source=source, operator_id=operator_id, session_id=session_id, at=at)

    @property
    def grounded(self) -> bool:
        return self.kind.grounded


@dataclass(frozen=True, slots=True)
class MemoryEntry:
    """An attributed belief — never a bare (key, value). `provenance` is mandatory; an entry
    written without a real source is MODEL_ASSERTED (see `Provenance.make`), so an audit can
    always tell a grounded belief from one the operator invented. `ref` is the stable id
    cited in `harness.provenance.refs` when this entry grounds a later action."""

    key: str
    value: str
    provenance: Provenance
    ref: str = ""


@runtime_checkable
class MemoryProvider(Protocol):
    """An archive of attributed beliefs.

    `remember` stores one belief and returns its `ref`. `recall` reads scoped beliefs back
    (each still carrying its provenance) — and the realization emits `harness.provenance.refs`
    so that USING memory leaves a grounding trace on the decision-log, with any
    model-asserted (ungrounded) belief visible in that trace.
    """

    def remember(self, entry: MemoryEntry) -> str: ...
    def recall(self, query: str, *, operator_id: str = "", limit: int = 10) -> list[MemoryEntry]: ...
