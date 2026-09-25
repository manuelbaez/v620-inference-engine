"""qwen3_coder tool-call parsing (<function=...><parameter=...>), with values
coerced to the tool's JSON schema types, like vLLM."""

import json
import re
import uuid

_PARAM_RE = re.compile(r"<\s*parameter\s*=\s*([^>]*)>(.*?)(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))", re.DOTALL)
_FUNC_RE = re.compile(r"<\s*function\s*=\s*([^>\n]*)>?")


def _trim_nl(v):
    if v.startswith("\n"):
        v = v[1:]
    if v.endswith("\n"):
        v = v[:-1]
    return v


def _schema_types(schema):
    if not isinstance(schema, dict):
        return ["string"]
    types = set()
    t = schema.get("type")
    if isinstance(t, str):
        types.add(t)
    elif isinstance(t, list):
        types.update(x for x in t if isinstance(x, str))
    for key in ("anyOf", "oneOf", "allOf"):
        for sub in schema.get(key, []) or []:
            types.update(_schema_types(sub))
    for v in schema.get("enum", []) or []:
        types.add("null" if v is None else "boolean" if isinstance(v, bool) else
                  "integer" if isinstance(v, int) else "number" if isinstance(v, float) else "string")
    return sorted(types) or ["string"]


def _coerce(value, schema):
    """Best-effort coercion like vLLM's coerce_to_schema_type (null > integer >
    number > boolean > object > array > string)."""
    types = {("integer" if t == "int" else "number" if t in ("float", "double") else t) for t in _schema_types(schema)}
    for t in ("null", "integer", "number", "boolean", "object", "array", "string"):
        if t not in types:
            continue
        if t == "null" and value.strip().lower() == "null":
            return None
        if t == "string":
            return value
        if t == "integer":
            try:
                return int(value.strip())
            except ValueError:
                continue
        if t == "number":
            try:
                f = float(value.strip())
                return int(f) if f.is_integer() and "." not in value and "e" not in value.lower() else f
            except ValueError:
                continue
        if t == "boolean":
            v = value.strip().lower()
            if v in ("true", "false"):
                return v == "true"
            continue
        if t in ("object", "array"):
            try:
                parsed = json.loads(value)
                if isinstance(parsed, dict if t == "object" else list):
                    return parsed
            except ValueError:
                continue
    return value


def parse_tool_call(text, tools):
    m = _FUNC_RE.search(text)
    if not m:
        return None
    name = m.group(1).strip()
    body = text[m.end():]
    end = body.find("</function>")
    if end >= 0:
        body = body[:end]
    props = {}
    for t in tools or []:
        fn = t.get("function", t)
        if fn.get("name") == name:
            props = (fn.get("parameters") or {}).get("properties") or {}
    args = {}
    for pm in _PARAM_RE.finditer(body):
        key = pm.group(1).strip()
        val = _trim_nl(pm.group(2))
        args[key] = _coerce(val, props[key]) if key in props else val
    return {
        "id": "chatcmpl-tool-" + uuid.uuid4().hex[:24],
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }
