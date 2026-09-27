#!/usr/bin/env bash
# Unloads the model llama-swap is running once it has had no request for 60 s
# (checks every 5 s, gives up after an hour). Runs inside llm-backend-amd:
#   ssh main-srv.local.net 'incus exec llm-backend-amd --project llms -- /home/server/qw-tools/prod-idle-unload.sh'
# Exits 0 after unloading, 1 if it was never idle.
set -u
for i in $(seq 1 720); do
  last=$(curl -s localhost:8080/api/metrics | python3 -c "import json,sys; m=json.load(sys.stdin); print(m[-1]['timestamp'][:19] if m else '2000-01-01T00:00:00')")
  age=$(( $(date -u +%s) - $(date -u -d "$last" +%s) ))
  if [ "$age" -ge 60 ]; then curl -s -X POST localhost:8080/api/models/unload; echo " unloaded after ${age}s idle"; exit 0; fi
  sleep 5
done
echo "never idle"; exit 1
