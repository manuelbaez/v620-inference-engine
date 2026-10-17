// The dashboard page: polls the metrics snapshot and renders cards, charts, memory and requests.
import {fetchMetrics} from "./api.js";
import {lineChart} from "./charts.js";
import * as ui from "./components.js";
import * as fmt from "./format.js";

const POLL_MS = 3000;
const $ = id => document.getElementById(id);
const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
let last = null;

function renderHeader(d) {
  const live = d.live;
  const busy = (live.running || 0) + (live.prefilling || 0) + (live.waiting || 0);
  $("model").textContent = d.model;
  $("uptime").textContent = "up " + fmt.duration(d.uptime_s);
  $("state").textContent = busy
    ? `${live.running} running, ${live.prefilling} prefilling, ${live.waiting} waiting` : "idle";
  $("dot").style.background = cssVar(busy ? "--ok" : "--muted");
}

function renderCards(d) {
  const t = d.totals, a = d.averages, h = d.history;
  const lastOf = key => {  // the latest sample that measured `key`, and how it is described
    for (let i = h.length - 1; i >= 0; i--)
      if (h[i][key] != null) {
        const ago = Date.now() / 1000 - h[i].t;
        return [h[i][key], ago < 12 ? "last 5 s" : "last busy " + fmt.seconds(ago) + " ago"];
      }
    return [null, "no work yet"];
  };
  const [pp, ppWhen] = lastOf("prompt_tps"), [tg, tgWhen] = lastOf("gen_tps");
  const fromCache = t.prompt_tokens ? fmt.percent(100 * t.cached_tokens / t.prompt_tokens) + " from cache" : "";
  $("cards").innerHTML = [
    ui.card("Prompt processing speed", fmt.rate(pp), "all prefilling requests, " + ppWhen),
    ui.card("Generation speed", fmt.rate(tg), "all running requests, " + tgWhen),
    ui.card("Avg time to first token", fmt.seconds(a.ttft_s), `last ${a.requests} requests`),
    ui.card("Avg request time", fmt.seconds(a.total_s), a.queue_s != null ? "queue " + fmt.seconds(a.queue_s) : ""),
    ui.card("Avg decode speed", fmt.rate(a.decode_tps), "per request"),
    ui.card("Requests", fmt.count(t.requests), t.errors ? `<span class="err">${t.errors} errors</span>` : "since start"),
    ui.card("Prompt tokens", fmt.count(t.prompt_tokens), fromCache),
    ui.card("Prompt tokens computed", fmt.count(t.computed_prompt_tokens), "prefilled on the GPUs"),
    ui.card("Tokens generated", fmt.count(t.generated_tokens), "since start"),
  ].join("");
}

function renderCharts(d) {
  const win = +$("win").value, h = d.history;
  const tps = v => v.toFixed(1) + " tok/s", n = v => String(Math.round(v));
  lineChart($("chart-prompt"), h, [{key: "prompt_tps", color: "--accent", label: "prompt processing", fmt: tps,
                                     none: "no prefill", marks: true}], win, 10);
  lineChart($("chart-gen"), h, [{key: "gen_tps", color: "--accent2", label: "generation", fmt: tps,
                                  none: "no decoding", marks: true}], win, 10);
  lineChart($("chart-demand"), h, [
    {key: "prompt_demand_tps", color: "--accent", label: "prompt tokens computed", fmt: tps},
    {key: "gen_demand_tps", color: "--accent2", label: "tokens generated", fmt: tps}], win, 10);
  lineChart($("chart-queue"), h, [
    {key: "running", color: "--accent", label: "running", fmt: n},
    {key: "waiting", color: "--accent2", label: "waiting / prefilling / loading", fmt: n}], win, 4, true);
  lineChart($("chart-kv"), h, [
    {key: "kv_pct", color: "--accent", label: "KV used", fmt: v => v.toFixed(1) + "%"}], win, 10);
  lineChart($("chart-pcie"), h, PCIE_COLORS.map((color, r) =>
    ({key: "pcie" + r, color, label: "GPU " + r, fmt: fmt.gbps})), win, 1);
  lineChart($("chart-spill"), h, [
    {key: "kv_total", color: "--accent", label: "total", fmt: fmt.gbps},
    {key: "kv_read", color: "--accent2", label: "decode reads", fmt: fmt.gbps},
    {key: "kv_write", color: "--ok", label: "writes", fmt: fmt.gbps},
    {key: "kv_stage_in", color: "--violet", label: "prefill staging in", fmt: fmt.gbps},
    {key: "kv_stage_out", color: "--warn", label: "prefill staging back", fmt: fmt.gbps},
    {key: "kv_prefill_read", color: "--bad", label: "prefill direct reads", fmt: fmt.gbps}], win, 0.1);
}

const PCIE_COLORS = ["--accent", "--accent2", "--ok", "--violet"];  // one per card, distinct in both themes

// Per card over the last sample: in (host copies, peers, spilled KV) and out (host copies, peers, spilled KV).
function pcieLines(pcie) {
  return (pcie || []).map(([h2d, d2h, fromPeers, toPeers, kvRead = 0, kvWrite = 0, stageIn = 0, stageOut = 0,
                           pfRead = 0], r) => {
    const kvIn = kvRead + stageIn + pfRead, kvOut = kvWrite + stageOut;
    return ui.line(`GPU ${r} PCIe (last 5 s)`,
      `in ${fmt.gbps(h2d + fromPeers + kvIn)} (host ${fmt.gbps(h2d)}, peers ${fmt.gbps(fromPeers)}, ` +
      `spilled KV ${fmt.gbps(kvIn)}) · ` +
      `out ${fmt.gbps(d2h + toPeers + kvOut)} (host ${fmt.gbps(d2h)}, peers ${fmt.gbps(toPeers)}, ` +
      `spilled KV ${fmt.gbps(kvOut)})`);
  }).join("");
}

function renderSlots(m) {
  const st = m.slot_state || [], vram = m.slot_vram || [];
  if (!st.length) { $("slots").innerHTML = ""; return; }
  // every bar is the slot's whole capacity wide, so its fill is the share the label states ("x of y tokens")
  const scale = Math.max(...m.slots, 1);
  $("slots").innerHTML = st.map((s, i) => ui.slotBar(i, s.tokens, vram[i] ?? 0, m.slots[i], s.busy, scale)).join("");
  $("slots-scale").textContent = `bar width ${fmt.count(scale)} tokens (a slot's capacity); tick: end of the VRAM part`;
}

function renderMemory(m, pcie) {
  renderSlots(m);
  let gpu = ui.bar("KV of the slots, VRAM + host RAM (in use / capacity)", m.kv_gpu_used, m.kv_gpu_allocated,
                   "slots " + m.slots.map(fmt.count).join(" / "));
  for (const c of m.gpus || [])
    gpu += ui.bar(`${c.card} VRAM`, c.vram_used, c.vram_total, c.power_w != null ? c.power_w.toFixed(1) + " W" : "");
  $("gpu").innerHTML = gpu + pcieLines(pcie);

  let host = ui.bar("Prefix cache in RAM", m.host_cache_used, m.host_cache_budget,
                    `${fmt.count(m.cache_blocks)} blocks, ${fmt.count(m.cache_snapshots)} snapshots`);
  host += ui.bar("Prefix cache on disk", m.disk_cache_used, m.disk_cache_budget);
  if (m.ple_table) host += ui.line("PLE n-gram table (pinned)", `${fmt.bytes(m.ple_table)} · ${m.ple_dir}`);
  if (m.host_total) host += ui.bar("Container RAM in use", m.host_total - m.host_available, m.host_total);
  $("host").innerHTML = host;
}

function render(d) {
  if (d.error || !d.totals) return;
  renderHeader(d);
  renderCards(d);
  renderCharts(d);
  renderMemory(d.memory, d.pcie);
  $("requests").innerHTML = d.recent.map(ui.requestRow).join("");
}

async function refresh() {
  try {
    last = await fetchMetrics();
    render(last);
  } catch (e) {
    $("state").textContent = "unreachable";
    $("dot").style.background = cssVar("--bad");
  }
}

$("win").addEventListener("change", () => last && render(last));
window.addEventListener("resize", () => last && render(last));
refresh();
setInterval(refresh, POLL_MS);
