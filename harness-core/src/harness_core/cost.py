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
    """Provider-agnostic token usage. Adapters normalize their engine's usage to this.

    `output_tokens` counts ALL generated tokens for the turn — reasoning/thinking
    INCLUDED. This is uniform across engines (Claude folds thinking into output and exposes
    no separate reasoning count); there is no portable per-category split, so any breakdown
    belongs in the adapter's `Turn.raw`, not here. Do NOT sum input+output+<reasoning>
    anywhere — there is no separate reasoning field to double-count.

    `request_count` = the number of real model API calls made during this turn.
    `None` means unknown — never default to 1 (that would lie), and never map an agentic
    turn/step count (e.g. Claude's `num_turns`) onto it.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    request_count: int | None = None


@dataclass(frozen=True, slots=True)
class CostResult:
    amount_usd: Decimal
    status: CostStatus
    source: str = ""

    @classmethod
    def unknown(cls) -> "CostResult":
        return cls(amount_usd=Decimal(0), status=CostStatus.UNKNOWN, source="")
