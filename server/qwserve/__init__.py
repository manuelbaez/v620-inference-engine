"""OpenAI-compatible HTTP server for the qw engine (Qwen3.8-Flash-Next).

The engine (libqw_engine.so, C API in include/qw/capi.h) does all per-token
work on the 4 GPUs. This process does the per-request text work: the model's
own chat template (Jinja2), HF `tokenizers`, and output parsing that matches
production vLLM's `--reasoning-parser qwen3 --tool-call-parser qwen3_coder`:

  * output starts inside <think> (the generation prompt ends with it) unless
    enable_thinking is false; </think> switches to content; text before it is
    `reasoning` (also sent as `reasoning_content` for older clients);
  * <tool_call> ... </tool_call> blocks are tool calls: <function=NAME> with
    <parameter=KEY>VALUE</parameter> pairs, one leading/trailing newline
    trimmed, values coerced to the tool's JSON schema type.

Requests run concurrently, one per engine slot (continuous batching: every
active request advances one token per batched step; others queue). Prompt
state is reused: a request goes to the slot holding the longest prefix of its
prompt and only prefills the rest (usage.prompt_tokens_details.cached_tokens).
"""
