#!/usr/bin/env bash
# Unloads the model llama-swap is running once it has had no request for 60 s
# (checks every 5 s, gives up after an hour). Runs inside llm-backend-amd:
#   ssh main-srv.local.net 'incus exec llm-backend-amd --project llms -- /home/server/qw-tools/prod-idle-unload.sh'
# llama-swap's /api/metrics lists finished requests only, so a long request
# still in flight looks idle there: the qw engine's own /metrics.json must also
# show nothing running, prefilling, waiting or loading (asked at the proxy
# address /running gives, which never starts a model).
# Exits 0 after unloading, 1 if it was never idle.
set -u
busy_engines() {  # in-flight requests over the running models that report them
  curl -s localhost:8080/running | python3 -c '
import json, sys, urllib.request
n = 0
for m in json.load(sys.stdin).get("running", []):
    try:
        live = json.load(urllib.request.urlopen(m["proxy"] + "/metrics.json", timeout=5))["live"]
        n += live.get("running", 0) + live.get("prefilling", 0) + live.get("waiting", 0)
    except Exception:
        pass  # not the qw engine, or still starting
print(n)'
}
for i in $(seq 1 720); do
  last=$(curl -s localhost:8080/api/metrics | python3 -c "import json,sys; m=json.load(sys.stdin); print(m[-1]['timestamp'][:19] if m else '2000-01-01T00:00:00')")
  age=$(( $(date -u +%s) - $(date -u -d "$last" +%s) ))
  if [ "$age" -ge 60 ] && [ "$(busy_engines)" = 0 ]; then
    curl -s -X POST localhost:8080/api/models/unload; echo " unloaded after ${age}s idle"; exit 0
  fi
  sleep 5
done
echo "never idle"; exit 1
