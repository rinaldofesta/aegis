"""Memory = an archive of ATTRIBUTED beliefs (the anti-Polsia provenance pillar).

Teeth: (1) a 'grounded' kind with no source fails CLOSED to MODEL_ASSERTED; (2) an
ungrounded belief stays a VISIBLE category on recall (not silently grounded); (3) recall
leaves a grounding trace on the span — harness.provenance.refs + the recalled_ungrounded
signal, which means "an ungrounded belief was RECALLED here" (not "it grounded an action").
"""
from __future__ import annotations

import json
from pathlib import Path  # noqa: F401  (kept for parity with sibling tests)

from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from harness_core import HarnessAttr, MemoryEntry, Provenance, ProvenanceKind
from hermes_adapter import ReferenceMemory, SpanEmitter


def _mem():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    return ReferenceMemory(SpanEmitter(provider.get_tracer("aegis.mem"))), exporter


def test_provenance_fails_closed_without_source():
    # a kind that CLAIMS grounding but cites no source cannot prove it -> downgraded, fail-closed
    p = Provenance.make(ProvenanceKind.TOOL_OUTPUT, source="")
    assert p.kind is ProvenanceKind.MODEL_ASSERTED
    assert p.grounded is False
    # with a concrete source it stays grounded
    grounded = Provenance.make(ProvenanceKind.TOOL_OUTPUT, source="search#1")
    assert grounded.kind is ProvenanceKind.TOOL_OUTPUT
    assert grounded.grounded is True


def test_ungrounded_belief_is_visible_not_silent():
    mem, _ = _mem()
    mem.remember(MemoryEntry("fact", "the sky is blue",
                             Provenance.make(ProvenanceKind.TOOL_OUTPUT, source="vision#1")))
    mem.remember(MemoryEntry("hunch", "the user prefers blue",
                             Provenance.make(ProvenanceKind.MODEL_ASSERTED)))  # invented, no source
    hits = {e.key: e.provenance.grounded for e in mem.recall("blue")}
    assert hits["fact"] is True
    assert hits["hunch"] is False  # the made-up belief stays a VISIBLE ungrounded category


def test_recall_emits_provenance_refs_and_recalled_ungrounded_signal():
    mem, exporter = _mem()
    mem.remember(MemoryEntry("a", "grounded thing",
                             Provenance.make(ProvenanceKind.RETRIEVED, source="doc#7")))
    mem.remember(MemoryEntry("b", "invented thing",
                             Provenance.make(ProvenanceKind.MODEL_ASSERTED)))
    mem.recall("thing")
    span = next(s for s in exporter.get_finished_spans() if s.name == "memory.recall")
    refs = json.loads(span.attributes[HarnessAttr.PROVENANCE_REFS])
    assert len(refs) == 2  # using memory left a grounding trace on the decision-log
    assert any(not r["grounded"] for r in refs)  # the ungrounded belief is visible in the trace
    # the signal means an ungrounded belief was RECALLED here (not "it grounded an action")
    assert span.attributes[HarnessAttr.PROVENANCE_RECALLED_UNGROUNDED] is True


def test_recall_all_grounded_signal_false():
    mem, exporter = _mem()
    mem.remember(MemoryEntry("x", "solid fact",
                             Provenance.make(ProvenanceKind.USER_PROVIDED, source="principal")))
    mem.recall("solid")
    span = next(s for s in exporter.get_finished_spans() if s.name == "memory.recall")
    assert span.attributes[HarnessAttr.PROVENANCE_RECALLED_UNGROUNDED] is False  # nothing ungrounded recalled
