"""The typed `echo` tool — the tracer-bullet's payload.

Typed schema with teeth (required field + integer constraint + additionalProperties) so
the dir.1 (MCP -> OpenAI) normalization is genuinely exercised, not a bare-string no-op.
"""
from __future__ import annotations

import json

from harness_core import ToolDef

ECHO_TOOL = ToolDef(
    name="echo",
    description="Echo a typed message a number of times.",
    input_schema={
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "message": {"type": "string", "description": "text to echo"},
            "times": {"type": "integer", "minimum": 1, "default": 1},
        },
        "required": ["message"],
    },
)


def echo_handler(args: dict) -> str:
    message = str(args.get("message", ""))
    times = int(args.get("times", 1))
    return json.dumps({"echoed": message * times})
