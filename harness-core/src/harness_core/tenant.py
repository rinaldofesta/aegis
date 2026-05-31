"""Tenant isolation context — a minimal vendor-free dataclass.

Deliberately tiny: just enough to scope a run to a tenant. The org data-model
(ontology/provenance) is a DOMAIN concern and lives outside the engine-abstraction.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class TenantContext:
    tenant_id: str
    root: Path
    env_vars: dict[str, str] = field(default_factory=dict)
