"""qwen3_coder tool-call parsing (<function=...><parameter=...>), with values
coerced to the tool's JSON schema types, like vLLM."""

import json
import re
import uuid

_PARAM_OPEN = re.compile(r"<\s*parameter\s*=\s*([^>]*)>")
_PARAM_CLOSE = re.compile(r"<\s*/\s*parameter\s*>")
_AFTER_CLOSE = re.compile(r"\s*(?:<\s*parameter\s*=|\Z)")  # what follows a value's closing tag
_FUNC_END = re.compile(r"<\s*/\s*function\s*>\s*\Z")
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


def _strip_partial_close(value):
    """A value cut off inside its closing tag (max_tokens) loses the fragment of the tag."""
    tag = "</parameter>"
    for k in range(len(tag) - 1, 1, -1):
        if value.endswith(tag[:k]):
            return value[:-k]
    return value


def _params(body, known):
    """[(name, value)] of a function's parameters. A value is the text up to the first closing
    tag that is followed by another parameter or the end of the call, so a value may itself
    contain `</parameter>` or `<parameter=` (a file that documents this format). Two exceptions:
    a declared parameter that has not appeared yet and starts a line ends the value before it
    (the model forgot the closing tag), and a value with no such closing tag runs to the end of
    the call (cut off by max_tokens)."""
    out, seen, pos = [], set(), 0
    while True:
        m = _PARAM_OPEN.search(body, pos)
        if not m:
            return out
        name, start = m.group(1).strip(), m.end()
        seen.add(name)
        close = next((c for c in _PARAM_CLOSE.finditer(body, start) if _AFTER_CLOSE.match(body, c.end())), None)
        limit = close.start() if close else len(body)
        split = next((o for o in _PARAM_OPEN.finditer(body, start, limit)
                      if o.group(1).strip() in known and o.group(1).strip() not in seen
                      and body[start:o.start()].endswith("\n")), None)
        if split:
            value, pos = body[start:split.start()], split.start()
        elif close:
            value, pos = body[start:close.start()], close.end()
        else:
            value, pos = _strip_partial_close(body[start:]), len(body)
        out.append((name, _trim_nl(value)))


def parse_tool_call(text, tools):
    m = _FUNC_RE.search(text)
    if not m:
        return None
    name = m.group(1).strip()
    body = text[m.end():]
    end = _FUNC_END.search(body)  # the closing tag at the end: a value may contain `</function>`
    if end:
        body = body[:end.start()]
    elif "</function>" in body:
        body = body[:body.find("</function>")]
    props = {}
    for t in tools or []:
        fn = t.get("function", t)
        if fn.get("name") == name:
            props = (fn.get("parameters") or {}).get("properties") or {}
    args = {}
    for key, val in _params(body, set(props)):
        args[key] = _coerce(val, props[key]) if key in props else val
    return {
        "id": "chatcmpl-tool-" + uuid.uuid4().hex[:24],
        "type": "function",
        "function": {"name": name, "arguments": json.dumps(args, ensure_ascii=False)},
    }
