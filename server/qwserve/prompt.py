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
        enable, effort, budget = resolve_thinking(body, self.default_effort, self.default_budget)
        kw["enable_thinking"] = enable
        if enable:
            kw["reasoning_effort"] = effort
        else:
            kw.pop("reasoning_effort", None)
        text = self.template.render(messages=messages, tools=tools, add_generation_prompt=body.get("add_generation_prompt", True), **kw)
        return text, enable, tools, (budget if enable else None)

    default_effort = "xhigh"  # set by the server (--reasoning-effort)
    default_budget = -1       # thinking tokens, -1 unlimited (--thinking-budget)

    def encode(self, text):
        # encode_batch gives the same ids but releases the GIL for the whole call; encode does not,
        # so on a worker thread only this keeps the event loop free (a 100k-token prompt takes ~0.1 s)
        return self.tok.encode_batch([text], add_special_tokens=False)[0].ids


# reasoning_effort values of the OpenAI / vLLM APIs onto this template's levels (low, medium, xhigh)
EFFORT_LEVELS = {"minimal": "low", "low": "low", "medium": "medium", "high": "xhigh", "xhigh": "xhigh", "max": "xhigh"}


def resolve_thinking(body, default_effort="xhigh", default_budget=-1):
    """Whether to think, at which template effort level, and the thinking-token budget (None:
    unlimited) of a chat request. Accepts what clients send: chat_template_kwargs.enable_thinking /
    .reasoning_effort (these win, as in vLLM), reasoning_effort (OpenAI / vLLM: none, minimal,
    low, medium, high, xhigh, max), thinking_token_budget (vLLM, -1 unlimited),
    thinking_budget_tokens (opencode provider option) and Anthropic's
    thinking: {type: enabled|disabled, budget_tokens}. A budget of 0 means no thinking."""
    kw = body.get("chat_template_kwargs") or {}
    anth = body.get("thinking") if isinstance(body.get("thinking"), dict) else {}
    effort = kw.get("reasoning_effort", body.get("reasoning_effort"))
    if "enable_thinking" in kw:
        enable = kw["enable_thinking"] is not False
    elif anth.get("type") == "disabled" or effort == "none":
        enable = False
    elif effort is not None or anth.get("type") == "enabled":
        enable = True  # asked for explicitly, whatever the server default
    else:
        enable = default_effort != "none"
    if effort in (None, "none"):
        effort = default_effort if default_effort != "none" else "xhigh"
    if effort not in EFFORT_LEVELS:
        raise ValueError(f"unsupported reasoning_effort {effort!r} (none, minimal, low, medium, high, xhigh, max)")
    budget = default_budget
    for v in (anth.get("budget_tokens"), body.get("thinking_budget_tokens"), body.get("thinking_token_budget")):
        if v is not None:
            budget = v
    if isinstance(budget, bool) or not isinstance(budget, int) or budget < -1:
        raise ValueError("thinking budget must be a non-negative integer, or -1 for unlimited")
    if budget == 0:
        enable = False
    return enable, EFFORT_LEVELS[effort], (None if budget < 0 or not enable else budget)

