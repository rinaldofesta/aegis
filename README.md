# Aegis

Standardized, domain-swappable **autonomous-agent harness**. The thesis: *specialized
verticalized agents in a standardized harness beat one generalist agent*, and
**every action is observable, gated and provenance-tracked**.

## Architecture

- **`harness-core/`** — the vendor-free standard: contracts (interfaces + dataclasses +
  enums) with **zero engine imports**. Adopts **MCP** (tool/data) and **OpenTelemetry
  GenAI** (observability) at the wire level. This is the durable, publishable asset.
- **`hermes-adapter/`** — implements `harness_core.Engine` by wrapping **Hermes**
  (`../external/hermes-agent`), the reference runtime the standard is *extracted from*.
- **`claude-adapter/`** — future **production** runtime adapter (Claude Agent SDK), behind
  the same contracts. v1: a semantic mapping table proving the contracts survive its
  primitives.
- **`examples/reference_agent.py`** — the tracer-bullet: one real turn through the boundary.

**Hard boundary (dependency inversion):** `harness-core` imports nothing vendor; adapters
import `harness-core`, never the reverse. Enforced by `harness-core/tests/test_import_boundary.py`.

## Dev

```bash
# vendor-free boundary check (clean venv, zero deps):
uv run --project harness-core --extra dev pytest harness-core/tests -q
```
