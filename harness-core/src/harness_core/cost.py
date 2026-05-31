"""Cost & usage accounting — vendor-free.

The adapter provides the raw numbers (per engine/provider); the core defines the
canonical shapes so cost is comparable across runtimes.
"""
from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from enum import Enum


class CostStatus(str, Enum):
    ACTUAL = "actual"        # fetched from a provider cost API
    ESTIMATED = "estimated"  # computed from tokens + a pricing snapshot
    INCLUDED = "included"    # covered by a subscription / flat plan
    UNKNOWN = "unknown"      # no pricing data available


@dataclass(frozen=True, slots=True)
class CanonicalUsage:
    """Provider-agnostic token usage. Adapters normalize their engine's usage to this."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    reasoning_tokens: int = 0
    request_count: int = 1


@dataclass(frozen=True, slots=True)
class CostResult:
    amount_usd: Decimal
    status: CostStatus
    source: str = ""

    @classmethod
    def unknown(cls) -> "CostResult":
        return cls(amount_usd=Decimal(0), status=CostStatus.UNKNOWN, source="")
