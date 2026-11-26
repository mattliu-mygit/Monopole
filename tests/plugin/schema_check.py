"""A JSON Schema check for the keywords Prompting Time's schemas use, so tests can hold the
plugin's output to them without a schema library. It raises ValueError at the first mismatch."""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any

_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def _fits_type(value: Any, name: str) -> bool:
    if name == "integer":
        return type(value) is int
    if name == "number":
        return type(value) in (int, float)
    return isinstance(value, _TYPES[name]) and not (name != "boolean" and type(value) is bool)


def check_schema(value: Any, schema: dict, root: dict | None = None, where: str = "$") -> None:
    root = root or schema
    if "$ref" in schema:
        ref = schema["$ref"].removeprefix("#/$defs/")
        check_schema(
            value,
            {**root["$defs"][ref], **{k: v for k, v in schema.items() if k != "$ref"}},
            root,
            where,
        )
        return

    def fail(reason: str) -> None:
        raise ValueError(f"{where}: {reason}")

    if "type" in schema:
        names = schema["type"] if isinstance(schema["type"], list) else [schema["type"]]
        if not any(_fits_type(value, name) for name in names):
            fail(f"{value!r} is not {names}")
    if "const" in schema and (value != schema["const"] or type(value) is not type(schema["const"])):
        fail(f"{value!r} is not {schema['const']!r}")
    if "enum" in schema and not any(
        value == option and type(value) is type(option) for option in schema["enum"]
    ):
        fail(f"{value!r} is not one of {schema['enum']}")
    if isinstance(value, str):
        if len(value) < schema.get("minLength", 0):
            fail("is too short")
        if "pattern" in schema and not re.search(schema["pattern"], value):
            fail(f"{value!r} does not match {schema['pattern']}")
        if schema.get("format") == "date-time":
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                fail(f"{value!r} has no offset")
    if type(value) in (int, float) and "minimum" in schema and value < schema["minimum"]:
        fail(f"{value} is below {schema['minimum']}")
    if isinstance(value, list):
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get(
            "maxItems", len(value)
        ):
            fail(f"has {len(value)} items")
        if schema.get("uniqueItems") and len({repr(item) for item in value}) != len(value):
            fail("repeats an item")
        for index, item in enumerate(value):
            if "items" in schema:
                check_schema(item, schema["items"], root, f"{where}[{index}]")
    if isinstance(value, dict):
        for key in schema.get("required", []):
            if key not in value:
                fail(f"lacks {key}")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                check_schema(item, properties[key], root, f"{where}.{key}")
            elif schema.get("additionalProperties") is False:
                fail(f"has the extra field {key}")
