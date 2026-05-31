"""The MCP <-> OpenAI normalization — the 🔴 boundary that keeps the standard vendor-free.

dir.1  harness_core.ToolDef (MCP `input_schema`)  ->  Hermes/OpenAI function schema
dir.2  OpenAI tool_call.function.arguments (JSON str)  ->  typed dict

Hermes is OpenAI-native internally; this module is the ONLY place the two shapes meet,
so the OpenAI shape never leaks into harness_core.
"""
from __future__ import annotations

import json
from typing import Any

from harness_core import ToolDef


def mcp_tooldef_to_openai_function(td: ToolDef) -> dict[str, Any]:
    """dir.1 — MCP inputSchema is JSON Schema; OpenAI 'parameters' is JSON Schema.

    The teeth (required, types, additionalProperties) must survive verbatim — that is
    exactly what the contract test asserts.
    """
    return {
        "name": td.name,
        "description": td.description,
        "parameters": td.input_schema,
    }


def parse_openai_tool_arguments(arguments: Any) -> dict[str, Any]:
    """dir.2 — decode the OpenAI `function.arguments` JSON string into a typed dict."""
    if isinstance(arguments, dict):
        return arguments
    if not arguments:
        return {}
    return json.loads(arguments)
