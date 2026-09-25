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

    @staticmethod
    def _normalize_parts(content):
        """OpenAI part shapes the template does not know -> its own: video_url/input_video become
        {"type": "video", "video": url}, input_image becomes {"type": "image_url", ...}."""
        if not isinstance(content, list):
            return content
        out = []
        for part in content:
            kind = part.get("type") if isinstance(part, dict) else None
            if kind in ("video_url", "input_video"):
                src = part.get(kind) or part.get("video_url") or {}
                part = {"type": "video", "video": src.get("url") if isinstance(src, dict) else src}
            elif kind == "input_image":
                src = part.get("image_url") or part.get("input_image") or {}
                part = {"type": "image_url", "image_url": src if isinstance(src, dict) else {"url": src}}
            out.append(part)
        return out

    def media_parts(self, body):
        """The image and video parts of the messages, in the order the template renders them."""
        parts = []
        for m in body.get("messages", []):
            content = self._normalize_parts(m.get("content"))
            if isinstance(content, list):
                for p in content:
                    if isinstance(p, dict) and ("image" in p or "image_url" in p or p.get("type") == "image" or
                                                "video" in p or p.get("type") == "video"):
                        parts.append(p)
        return parts

    def render(self, body):
        messages = []
        for m in body.get("messages", []):
            m = dict(m)
            m["content"] = self._normalize_parts(m.get("content"))
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

