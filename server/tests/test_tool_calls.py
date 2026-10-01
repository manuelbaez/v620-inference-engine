#!/usr/bin/env python3
"""The qwen3_coder tool-call parser (no GPU, no checkpoint): values may contain the format's own
tags (an agent editing a file that documents it), a truncated last parameter is kept, and for
well-formed calls the result equals the previous regex implementation's (a differential fuzz).

  python3 server/tests/test_tool_calls.py
"""

import json
import os
import random
import re
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import fixtures  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

from qwserve.output_parser import OutputParser  # noqa: E402
from qwserve.tool_calls import _coerce, _trim_nl, parse_tool_call  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "edit", "parameters": {"properties": {
    "path": {"type": "string"}, "count": {"type": "integer"}, "flag": {"type": "boolean"},
    "opts": {"type": "object"}, "items": {"type": "array"}, "ratio": {"type": "number"},
    "maybe": {"type": ["string", "null"]}, "content": {"type": "string"}}}}}]


def P(k, v):
    return f"<parameter={k}>\n{v}\n</parameter>\n"


def call(body, name="edit", tools=TOOLS):
    c = parse_tool_call(f"<function={name}>\n{body}</function>", tools)
    return None if c is None else (c["function"]["name"], json.loads(c["function"]["arguments"]))


# ---- the previous implementation, for the differential fuzz
_LEGACY_RE = re.compile(r"<\s*parameter\s*=\s*([^>]*)>(.*?)(?:<\s*/\s*parameter\s*>|(?=<\s*parameter\s*=))", re.DOTALL)
_FUNC_RE = re.compile(r"<\s*function\s*=\s*([^>\n]*)>?")


def legacy(text, tools):
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
    for pm in _LEGACY_RE.finditer(body):
        key = pm.group(1).strip()
        val = _trim_nl(pm.group(2))
        args[key] = _coerce(val, props[key]) if key in props else val
    return name, args


def main():
    fails = 0

    def check(what, got, want):
        nonlocal fails
        ok = got == want
        fails += not ok
        print(f"{what}: {'ok' if ok else f'FAIL got {got!r} want {want!r}'}")

    # what worked before and still does
    check("plain call", call(P("path", "/a/b.py") + P("count", "3")), ("edit", {"path": "/a/b.py", "count": 3}))
    check("two trailing newlines keep one", call("<parameter=path>\nx\n\n</parameter>\n"), ("edit", {"path": "x\n"}))
    check("leading spaces", call("<parameter=path>  x</parameter>\n"), ("edit", {"path": "  x"}))
    check("1e3 is not an integer", call(P("count", "1e3")), ("edit", {"count": "1e3"}))
    check("3.0 as number", call(P("ratio", "3.0")), ("edit", {"ratio": 3.0}))
    check("True as boolean", call(P("flag", "True")), ("edit", {"flag": True}))
    check("null in a nullable string", call(P("maybe", "null")), ("edit", {"maybe": None}))
    check("invalid JSON stays a string", call(P("opts", "{'a': 1}")), ("edit", {"opts": "{'a': 1}"}))
    check("JSON object and array", call(P("opts", '{"a": [1, 2]}') + P("items", "[1, 2]")),
          ("edit", {"opts": {"a": [1, 2]}, "items": [1, 2]}))
    check("unknown function", call(P("x", "1"), name="nope"), ("nope", {"x": "1"}))
    check("duplicate parameter: the last wins", call(P("path", "a") + P("path", "b")), ("edit", {"path": "b"}))
    check("< and > in a value", call(P("path", "if a < b and c > d: pass")),
          ("edit", {"path": "if a < b and c > d: pass"}))
    check("html-like value", call(P("path", '<div class="x">hi</div>')), ("edit", {"path": '<div class="x">hi</div>'}))
    check("dotted and dashed name", call(P("a", "1"), name="mcp.server-tool_1"), ("mcp.server-tool_1", {"a": "1"}))
    check("spaces inside the tags", call("< parameter = path >\nx\n< / parameter >\n"), ("edit", {"path": "x"}))

    # values that contain the format's own tags (these split or were cut before)
    check("value contains </parameter>", call(P("path", "foo</parameter>bar")), ("edit", {"path": "foo</parameter>bar"}))
    check("value contains <parameter= mid-line", call(P("path", "see <parameter=count> in docs")),
          ("edit", {"path": "see <parameter=count> in docs"}))
    check("value contains </function>", call(P("path", "a </function> b")), ("edit", {"path": "a </function> b"}))
    sample = "<function=edit>\n<parameter=path>\n/x\n</parameter>\n</function>"
    check("value is a whole sample call (path was already used)",
          call(P("path", "/doc.md") + P("content", sample)), ("edit", {"path": "/doc.md", "content": sample}))

    # a forgotten closing tag: the next declared parameter on a new line ends the value
    check("forgotten </parameter> before a declared parameter",
          call("<parameter=path>\n/a\n<parameter=count>\n2\n</parameter>\n"), ("edit", {"path": "/a", "count": 2}))
    check("no schema: a forgotten closing tag cannot be told from a value",
          call("<parameter=a>\n1\n<parameter=b>\n2\n</parameter>\n", name="nope"), ("nope", {"a": "1\n<parameter=b>\n2"}))

    # cut off by max_tokens
    check("truncated last parameter is kept", call("<parameter=path>\n/a\n"), ("edit", {"path": "/a"}))
    check("truncated after one complete parameter", call(P("path", "/a") + "<parameter=content>\nhello wor"),
          ("edit", {"path": "/a", "content": "hello wor"}))
    check("truncated inside the closing tag", call("<parameter=path>\n/a\n</param"), ("edit", {"path": "/a"}))
    check("truncated inside the opening tag", call(P("path", "/a") + "<parameter=con"), ("edit", {"path": "/a"}))
    check("no </function> at all", parse_tool_call("<function=edit>\n" + P("path", "/a"), TOOLS) is not None, True)

    # differential fuzz: well-formed calls (no tag text in values) parse as before
    rng = random.Random(7)
    alphabet = "abcXYZ 0123456789_-./:;,()[]{}=+*#'\"\\ \n"
    names = ["path", "count", "flag", "opts", "ratio", "maybe", "content", "other", "x1", "unknown_param"]
    diffs = 0
    for i in range(3000):
        fn = rng.choice(["edit", "edit", "edit", "nope", "mcp.tool-1"])
        parts = []
        for _ in range(rng.randrange(0, 6)):
            k = rng.choice(names)
            v = "".join(rng.choice(alphabet) for _ in range(rng.randrange(0, 40)))
            if rng.random() < 0.2:
                v += " a < b "
            style = rng.randrange(3)
            parts.append(P(k, v) if style == 0 else f"<parameter={k}>\n{v}</parameter>\n" if style == 1
                         else f"<parameter={k}>{v}</parameter>\n")
        text = f"<function={fn}>\n" + "".join(parts) + "</function>" + rng.choice(["", "\n", " \n"])
        old, new = legacy(text, TOOLS), parse_tool_call(text, TOOLS)
        new = None if new is None else (new["function"]["name"], json.loads(new["function"]["arguments"]))
        if old != new:
            diffs += 1
            if diffs <= 3:
                print("  differs:", repr(text), old, new)
    check("3000 random well-formed calls parse as before", diffs, 0)

    # through the output parser (tokenizer fixture: <tool_call> is a single token)
    with tempfile.TemporaryDirectory() as d:
        fixtures.make_model_dir(d)
        tok = Tokenizer.from_file(os.path.join(d, "tokenizer.json"))

        def run(text, finish=True):
            parser = OutputParser(tok, False, TOOLS, parse_tools=True)
            for tid in tok.encode(text, add_special_tokens=False).ids:
                parser.feed(tid)
            if finish:
                parser.finish()
            return [(c["function"]["name"], json.loads(c["function"]["arguments"])) for c in parser.tool_calls]

        wrapped = "<tool_call>\n<function=edit>\n" + P("path", "foo</parameter>bar") + "</function>\n</tool_call>"
        check("through the parser: a value with </parameter>", run(wrapped), [("edit", {"path": "foo</parameter>bar"})])
        cut = "<tool_call>\n<function=edit>\n<parameter=path>\n/a\n</parameter>\n<parameter=content>\nhello"
        check("through the parser: a call cut off by max_tokens", run(cut), [("edit", {"path": "/a", "content": "hello"})])
        two = wrapped + "\n" + "<tool_call>\n<function=edit>\n" + P("count", "5") + "</function>\n</tool_call>"
        check("through the parser: two calls", run(two), [("edit", {"path": "foo</parameter>bar"}), ("edit", {"count": 5})])

    print("FAILED" if fails else "all ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
