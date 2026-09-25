"""Incremental detokenization."""


class Detok:
    """Incremental detokenizer for one output segment (holds back partial UTF-8)."""

    def __init__(self, tok):
        self.tok = tok
        self.ids = []
        self.prefix = 0  # ids[prefix:read] decode to text already emitted (context window)
        self.read = 0

    def add(self, tid):
        self.ids.append(tid)
        prefix_text = self.tok.decode(self.ids[self.prefix:self.read], skip_special_tokens=False)
        new_text = self.tok.decode(self.ids[self.prefix:], skip_special_tokens=False)
        if new_text.endswith("�") or len(new_text) <= len(prefix_text):
            return ""
        self.prefix, self.read = self.read, len(self.ids)
        return new_text[len(prefix_text):]
