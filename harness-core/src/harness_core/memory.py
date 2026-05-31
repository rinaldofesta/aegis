"""Memory provider contract.

Hermes' model: memory context is fetched per-turn and injected as text into the system
prompt, then synced post-turn. The core keeps that shape engine-neutral.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class MemoryProvider(Protocol):
    def prefetch(self, query: str) -> str: ...
    def sync(self, user_message: str, assistant_response: str) -> None: ...
    def build_prompt(self) -> str: ...
