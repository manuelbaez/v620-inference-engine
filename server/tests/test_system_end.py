#!/usr/bin/env python3
"""The server tells the engine where the system prompt ends (no GPU, no checkpoint). The engine keeps a cache
snapshot there for conversations that share the system prompt; searching for the first <|im_start|> token finds a
quoted one inside a system prompt that mentions the string, so the place comes from the chat template instead
(ChatPrompt.encode_chat) and reaches the engine with the request (Engine.set_system_end before begin_prompt).

  python3 server/tests/test_system_end.py
"""

import asyncio
import glob
import os
import shutil
import sys
import tempfile
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.dirname(__file__))

import fixtures  # noqa: E402
import jinja2  # noqa: E402
from qwserve.api import Server  # noqa: E402
from qwserve.prompt import ChatPrompt  # noqa: E402
from qwserve.scheduler import Request, Scheduler  # noqa: E402


class Engine(fixtures.FakeEngine):
    def __init__(self):
        super().__init__()
        self.calls = []

    def set_system_end(self, slot, pos):
        self.calls.append(("system_end", slot, pos))

    def begin_prompt(self, slot, tokens, media=None):
        self.calls.append(("begin", slot))
        return super().begin_prompt(slot, tokens, media)


def main():
    fails = 0

    def check(what, ok, detail=""):
        nonlocal fails
        fails += not ok
        print(f"{what}: {'ok' if ok else 'FAIL'} {detail}")

    def cases(prompt, label):
        start = prompt.tok.token_to_id("<|im_start|>")
        plain = fixtures.words(200, 1)
        quoting = fixtures.words(80, 2) + " a message starts with <|im_start|> and ends with <|im_end|>\n" + fixtures.words(80, 3)
        user = [{"role": "user", "content": "hello <|im_start|> there"}]
        for name, system in (("plain", plain), ("quoting <|im_start|>", quoting)):
            for role in ("system", "developer"):
                body = {"messages": [{"role": role, "content": system}] + user}
                try:
                    text = prompt.render(body)[0]
                except jinja2.exceptions.TemplateError:
                    print(f"{label}: no {role} role in this template")
                    continue
                ids, end = prompt.encode_chat(body, text)
                want = len(prompt.encode(text[:text.index("<|im_start|>user")]))
                first = ids.index(start, 1)
                check(f"{label}: {name} {role} prompt ends at token {want}", end == want and ids[end] == start,
                      f"(got {end}; the first <|im_start|> after the start is at {first})")
                check(f"{label}: same ids as encode", ids == prompt.encode(text))
                longer = dict(body, messages=body["messages"] + [{"role": "assistant", "content": "hi"}, {"role": "user", "content": "more"}])
                check(f"{label}: {name} {role}, a later turn gives the same place",
                      prompt.encode_chat(longer, prompt.render(longer)[0])[1] == want)
        body = {"messages": [{"role": "system", "content": [{"type": "text", "text": quoting}]}] + user}
        text = prompt.render(body)[0]
        check(f"{label}: a system prompt given in parts", prompt.encode_chat(body, text)[1] == len(prompt.encode(text[:text.index("<|im_start|>user")])))
        for name, msgs in (("no system message", user), ("only a system message", [{"role": "system", "content": plain}])):
            body = {"messages": msgs}
            try:
                text = prompt.render(body)[0]
            except Exception:
                continue  # (not every template accepts it)
            check(f"{label}: {name} gives -1", prompt.encode_chat(body, text)[1] == -1)

    with tempfile.TemporaryDirectory() as d:
        fixtures.make_model_dir(d)
        prompt = ChatPrompt(d)
        cases(prompt, "test template")

        # the checkpoint's chat template, or a Qwen one from the Hugging Face cache, when there is one (tools go into the system block)
        real = sorted(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--*Qwen3*/snapshots/*/chat_template.jinja")))
        model = os.path.join(os.environ.get("QW_MODEL_DIR", "/mnt/llms/qwen3.8-flash-next-awq"), "chat_template.jinja")
        real += [model] if os.path.exists(model) else []  # the checkpoint's own, when it is here
        if real:
            shutil.copy(real[-1], os.path.join(d, "chat_template.jinja"))
            qwen = ChatPrompt(d)
            cases(qwen, "Qwen template")
            tools = [{"type": "function", "function": {"name": "f", "description": "uses <|im_start|>", "parameters": {"type": "object", "properties": {}}}}]
            body = {"messages": [{"role": "system", "content": "be brief"}, {"role": "user", "content": "x"}], "tools": tools}
            text = qwen.render(body)[0]
            ids, end = qwen.encode_chat(body, text)
            check("Qwen template: with tools the end is after the tool list",
                  end == len(qwen.encode(text[:text.index("<|im_start|>user")])), f"(got {end})")
        else:
            print("no Qwen chat template here: skipped")

        srv = Server.__new__(Server)
        srv.prompt, srv.tok = prompt, prompt.tok
        body = {"messages": [{"role": "system", "content": "quote <|im_start|> here"}, {"role": "user", "content": "x"}]}
        out = asyncio.run(srv.prepare(body))
        check("prepare() returns the place after the tokens", len(out) == 5 and out[3][out[4]] == prompt.tok.token_to_id("<|im_start|>")
              and out[4] > out[3].index(prompt.tok.token_to_id("<|im_start|>"), 1))

    # the scheduler passes it on before the prompt begins, and only when it is known
    e = Engine()
    s = Scheduler(e)
    a = Request([7] * 50, {"max_tokens": 4}, lambda *x: None, set())
    a.sys_end = 20
    b = Request([9] * 50, {"max_tokens": 4}, lambda *x: None, set())
    s.submit(a)
    s.submit(b)
    for _ in range(200):
        if len(e.calls) >= 3:
            break
        time.sleep(0.01)
    s.shutdown(1)
    check("the scheduler sets it before begin_prompt, and not without one",
          e.calls == [("system_end", 0, 20), ("begin", 0), ("begin", 1)], str(e.calls))

    print("FAILED" if fails else "all ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
