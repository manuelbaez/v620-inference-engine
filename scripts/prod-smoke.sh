#!/usr/bin/env bash
# Smoke test of the production engine through llama-swap (starts it if unloaded):
# a greedy request, and a sampled one with penalties and top_logprobs. Runs inside llm-backend-amd:
#   ssh main-srv.local.net 'incus exec llm-backend-amd --project llms -- /home/server/qw-tools/prod-smoke.sh'
set -euo pipefail
URL=localhost:8080/v1/chat/completions
curl -s -m 900 $URL -H content-type:application/json -d '{"model":"qw/qwen3.8-flash-next","max_tokens":20,"temperature":0,"messages":[{"role":"user","content":"Say hello in five words."}],"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); print("greedy:", d["choices"][0]["message"]["content"])'
curl -s -m 900 $URL -H content-type:application/json -d '{"model":"qw/qwen3.8-flash-next","max_tokens":60,"temperature":0.7,"top_p":0.8,"top_k":20,"presence_penalty":1.0,"logprobs":true,"top_logprobs":2,"messages":[{"role":"user","content":"Write one sentence about the sea."}],"chat_template_kwargs":{"enable_thinking":false}}' \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); c=d["choices"][0]; print("sampled:", c["message"]["content"]); print("top_logprobs ok:", len(c["logprobs"]["content"][0]["top_logprobs"])==2)'
