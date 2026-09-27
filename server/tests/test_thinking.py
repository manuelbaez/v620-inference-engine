#!/usr/bin/env python3
"""Thinking controls (no GPU): the request fields clients send, mapped onto this chat
template's enable_thinking / reasoning_effort and a thinking-token budget.

  python3 server/tests/test_thinking.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from qwserve.prompt import resolve_thinking  # noqa: E402


def main():
    fails = 0
    cases = [
        # (request body, server default effort, default budget) -> (enable, effort, budget)
        ({}, "xhigh", -1, (True, "xhigh", None)),
        ({}, "low", 2000, (True, "low", 2000)),
        ({}, "none", -1, (False, "xhigh", None)),
        ({"reasoning_effort": "none"}, "xhigh", -1, (False, "xhigh", None)),
        ({"reasoning_effort": "minimal"}, "xhigh", -1, (True, "low", None)),
        ({"reasoning_effort": "high"}, "low", -1, (True, "xhigh", None)),
        ({"reasoning_effort": "medium"}, "xhigh", -1, (True, "medium", None)),
        ({"reasoning_effort": "max"}, "xhigh", -1, (True, "xhigh", None)),
        ({"reasoning_effort": "low"}, "none", -1, (True, "low", None)),  # an explicit effort turns thinking on
        ({"chat_template_kwargs": {"enable_thinking": False}, "reasoning_effort": "high"}, "xhigh", -1,
         (False, "xhigh", None)),  # chat_template_kwargs win, as in vLLM
        ({"chat_template_kwargs": {"reasoning_effort": "medium"}, "reasoning_effort": "low"}, "xhigh", -1,
         (True, "medium", None)),
        ({"thinking_token_budget": 512}, "xhigh", -1, (True, "xhigh", 512)),  # vLLM
        ({"thinking_token_budget": -1}, "xhigh", 300, (True, "xhigh", None)),
        ({"thinking_budget_tokens": 256}, "xhigh", -1, (True, "xhigh", 256)),  # opencode provider option
        ({"thinking": {"type": "enabled", "budget_tokens": 4000}}, "xhigh", -1, (True, "xhigh", 4000)),  # Anthropic
        ({"thinking": {"type": "disabled"}}, "xhigh", -1, (False, "xhigh", None)),
        ({"thinking_token_budget": 0}, "xhigh", -1, (False, "xhigh", None)),  # 0: no thinking
    ]
    for body, de, db, want in cases:
        got = resolve_thinking(body, de, db)
        ok = got == want
        fails += not ok
        print(f"{body} (default {de}, {db}): {got} {'ok' if ok else f'FAIL (want {want})'}")
    for bad in ({"reasoning_effort": "extreme"}, {"thinking_token_budget": -5}, {"thinking_token_budget": "10"}):
        try:
            resolve_thinking(bad)
            print(f"{bad}: accepted  FAIL (want an error)")
            fails += 1
        except ValueError as e:
            print(f"{bad}: rejected ({e})  ok")
    print("FAILED" if fails else "all ok")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
