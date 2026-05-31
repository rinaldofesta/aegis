"""The non-falsifiable boundary check.

`harness_core` must be 100% vendor-free. Two layers:
  1. Runtime: every submodule imports in an environment where NO engine/vendor package
     is installed (harness-core declares zero deps, so its venv is clean by construction).
  2. Static: AST-scan the source — no import of any engine/vendor/adapter package.

If anyone sneaks `import hermes` into the core, layer 1 explodes at import and layer 2
names the offending file. Better than grep, and grep is subsumed by the AST walk.
"""
from __future__ import annotations

import ast
import importlib
import pkgutil
from pathlib import Path

import harness_core

FORBIDDEN_ROOTS = {
    # engines / adapters
    "hermes", "hermes_cli", "hermes_adapter", "claude_adapter", "agent", "tools", "cron",
    "gateway",
    # vendor SDKs
    "anthropic", "openai", "google", "mcp",  # mcp is wire-level only; not import-time in v1
    # infra that must stay in adapters
    "celery", "redis", "sqlmodel", "sqlalchemy", "fastmcp", "honcho", "logfire",
    "opentelemetry",
}

_SRC = Path(harness_core.__file__).parent


def test_all_submodules_import() -> None:
    for mod in pkgutil.iter_modules(harness_core.__path__):
        importlib.import_module(f"harness_core.{mod.name}")


def test_no_vendor_imports_in_source() -> None:
    offenders: list[str] = []
    for py in _SRC.rglob("*.py"):
        tree = ast.parse(py.read_text(encoding="utf-8"), filename=str(py))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                # ignore relative imports (node.level > 0) — those are intra-package
                names = [node.module or ""] if node.level == 0 else []
            else:
                continue
            for name in names:
                if name.split(".")[0] in FORBIDDEN_ROOTS:
                    offenders.append(f"{py.name}: imports {name}")
    assert not offenders, "vendor imports leaked into harness_core:\n" + "\n".join(offenders)
