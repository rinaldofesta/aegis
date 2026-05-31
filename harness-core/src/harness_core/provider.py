"""Provider registry contract — resolve a canonical provider name to creds/endpoint.

Multi-provider routing is the embryonic 'provider registry'. Some engines (e.g. a
single-provider SDK) may only support one provider; adapters may stub `resolve` for
names they cannot serve.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class ProviderInfo:
    canonical_name: str
    endpoint: str | None = None
    api_mode: str = "chat_completions"
    supports_tools: bool = True


@runtime_checkable
class ProviderRegistry(Protocol):
    def resolve(self, name: str) -> ProviderInfo: ...
