"""Chat prompts: the model's own chat template (sandboxed Jinja2) and tokenizer."""

import json
import os

import jinja2
from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

EOS_TEXT = ("<|im_end|>", "<|endoftext|>")


class ChatPrompt:
    def __init__(self, model_dir):
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        with open(os.path.join(model_dir, "chat_template.jinja")) as f:
            src = f.read()
        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True, extensions=["jinja2.ext.loopcontrols"])

        def raise_exception(msg):
            raise jinja2.exceptions.TemplateError(msg)

        env.globals["raise_exception"] = raise_exception
        env.filters["tojson"] = lambda x, indent=None, separators=None, sort_keys=False, ensure_ascii=False: json.dumps(
            x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)
        self.template = env.from_string(src)
        self.eos_ids = {self.tok.token_to_id(t) for t in EOS_TEXT}

    def render(self, body):
        messages = []
        for m in body.get("messages", []):
            m = dict(m)
            rc = m.get("reasoning_content", m.get("reasoning"))
            if rc is not None:
                m["reasoning_content"] = rc
            if m.get("tool_calls"):
                calls = []
                for c in m["tool_calls"]:
                    c = json.loads(json.dumps(c))
                    fn = c.get("function", c)
                    if isinstance(fn.get("arguments"), str):
                        try:
                            fn["arguments"] = json.loads(fn["arguments"]) if fn["arguments"].strip() else {}
                        except ValueError:
                            pass
                    calls.append(c)
                m["tool_calls"] = calls
            messages.append(m)
        kw = dict(body.get("chat_template_kwargs") or {})
        tools = body.get("tools") if body.get("tool_choice") != "none" else None
        if "reasoning_effort" in body and "reasoning_effort" not in kw:
            kw["reasoning_effort"] = body["reasoning_effort"]
        text = self.template.render(messages=messages, tools=tools, add_generation_prompt=body.get("add_generation_prompt", True), **kw)
        thinking = kw.get("enable_thinking", True) is not False
        return text, thinking, tools

    def encode(self, text):
        return self.tok.encode(text, add_special_tokens=False).ids

