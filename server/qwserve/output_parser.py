"""Splits generated tokens into reasoning, content and tool calls (vLLM's
qwen3 reasoning parser and qwen3_coder tool parser)."""

from .detok import Detok
from .tool_calls import parse_tool_call

THINK_START, THINK_END = "<think>", "</think>"
TOOL_START, TOOL_END = "<tool_call>", "</tool_call>"


class OutputParser:
    """Token-level router: think/tool tags are single tokens in this vocabulary."""

    def __init__(self, tok, thinking, tools, parse_tools):
        self.tok = tok
        self.ids = {s: tok.token_to_id(s) for s in (THINK_START, THINK_END, TOOL_START, TOOL_END)}
        self.state = "reasoning" if thinking else "content"
        self.tools = tools
        self.parse_tools = parse_tools
        self.seg = {"reasoning": Detok(tok), "content": Detok(tok)}
        self.tool_ids = []
        self.tool_calls = []
        self.reasoning = ""
        self.content = ""

    def feed(self, tid):
        """Returns a delta dict (reasoning / content / tool_calls) or None."""
        if tid == self.ids[THINK_START] and self.state == "reasoning":
            return None
        if tid == self.ids[THINK_END] and self.state in ("reasoning", "content"):
            self.state = "content"
            return None
        if self.parse_tools and tid == self.ids[TOOL_START] and self.state in ("reasoning", "content"):
            self.state = "tool"
            self.tool_ids = []
            return None
        if self.state == "tool":
            if tid == self.ids[TOOL_END]:
                self.state = "content"
                call = parse_tool_call(self.tok.decode(self.tool_ids, skip_special_tokens=False), self.tools)
                if call is None:
                    return None
                call["index"] = len(self.tool_calls)
                self.tool_calls.append(call)
                return {"tool_calls": [call]}
            self.tool_ids.append(tid)
            return None
        text = self.seg[self.state].add(tid)
        if not text:
            return None
        if self.state == "reasoning":
            self.reasoning += text
            return {"reasoning": text}
        self.content += text
        return {"content": text}

    def finish(self):
        """Unterminated tool call at the end: try to parse it anyway."""
        if self.state == "tool" and self.tool_ids:
            call = parse_tool_call(self.tok.decode(self.tool_ids, skip_special_tokens=False), self.tools)
            if call:
                call["index"] = len(self.tool_calls)
                self.tool_calls.append(call)
                return {"tool_calls": [call]}
        return None
