"""Reference MemoryProvider — an in-process archive of ATTRIBUTED beliefs.

Proves the harness_core MemoryProvider contract without infra: every entry keeps its
provenance, `recall` emits `harness.provenance.refs` (the grounding trace), and a
MODEL_ASSERTED belief used as grounding is VISIBLE on that span — never silently grounded.
PRODUCTION backing is Hermes' `MemoryManager` (the memory 'Stage C', deferred to infra);
the contract is identical (Hermes already carries write metadata — write_origin/task_id/
tool_call_id — onto which Provenance maps).
"""
from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable

from harness_core import MemoryEntry

from .obs import SpanEmitter


class ReferenceMemory:
    def __init__(self, span_emitter: SpanEmitter, active_getter: Callable[[], Any] | None = None):
        self._entries: list[MemoryEntry] = []
        self._obs = span_emitter
        self._active = active_getter or (lambda: None)
        self._seq = 0

    def remember(self, entry: MemoryEntry) -> str:
        ref = entry.ref or f"mem#{self._seq}"
        self._seq += 1
        self._entries.append(entry if entry.ref else replace(entry, ref=ref))
        return ref

    def recall(self, query: str, *, operator_id: str = "", limit: int = 10) -> list[MemoryEntry]:
        q = query.lower()
        hits = [
            e
            for e in self._entries
            if (not operator_id or e.provenance.operator_id == operator_id)
            and (q in e.key.lower() or q in e.value.lower())
        ][:limit]
        # grounding trace: using memory leaves provenance.refs on the decision-log, parented
        # under the active turn when there is one.
        active = self._active()
        self._obs.emit_memory_recall(
            query=query,
            refs=[
                {"ref": e.ref, "kind": e.provenance.kind.value, "grounded": e.provenance.grounded}
                for e in hits
            ],
            parent_ctx=active.get("turn_ctx") if isinstance(active, dict) else None,
        )
        return hits
