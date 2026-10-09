// HTML fragments of the dashboard.
import * as fmt from "./format.js";

export function card(label, value, note) {
  return `<div class="card"><div class="k">${label}</div><div class="v">${value}</div>` +
         `<div class="s">${note || "&nbsp;"}</div></div>`;
}

export function bar(label, used, total, extra) {
  const pct = total ? Math.min(100, 100 * used / total) : 0;
  return `<div class="mem-item"><div class="row"><span>${label}</span>` +
         `<span class="m">${fmt.bytes(used)} / ${fmt.bytes(total)}${extra ? " · " + extra : ""}</span></div>` +
         `<div class="bar"><div style="width:${pct.toFixed(1)}%"></div></div></div>`;
}

// One slot: the tokens it holds, the part in VRAM and the part spilled to host RAM, on a bar `scale` tokens wide;
// the tick is where the slot's VRAM part ends.
export function slotBar(i, tokens, vram, capacity, busy, scale) {
  const hot = Math.min(tokens, vram), cold = Math.max(0, tokens - vram), pct = v => (100 * Math.min(1, v / scale)).toFixed(2);
  const state = tokens ? (busy ? "in use" : "idle, kept") : "empty";
  const split = cold ? ` (${fmt.count(hot)} + ${fmt.count(cold)})` : "";
  return `<div class="mem-item"><div class="row"><span>slot ${i} <span class="m">${state}</span></span>` +
         `<span class="m">${fmt.count(tokens)}${split} of ${fmt.count(capacity)} tokens</span></div>` +
         `<div class="bar split"><div class="hot" style="width:${pct(hot)}%"></div>` +
         `<div class="cold" style="width:${pct(cold)}%"></div>` +
         (vram < scale ? `<i style="left:${pct(vram)}%"></i>` : "") + `</div></div>`;
}

export function line(label, value) {
  return `<div class="mem-item"><div class="row"><span>${label}</span><span class="m">${value}</span></div></div>`;
}

export function requestRow(q) {
  const bad = q.finish === "error" || q.finish === "abort" ? "err" : "";
  return `<tr><td>${new Date(q.end * 1000).toLocaleTimeString()}</td><td>${q.id}</td>` +
         `<td><span class="pill ${bad}">${q.finish}</span></td>` +
         `<td>${fmt.count(q.prompt)}</td><td>${fmt.count(q.cached)}</td><td>${fmt.count(q.generated)}</td>` +
         `<td>${fmt.seconds(q.queue_s)}</td><td>${fmt.seconds(q.ttft_s)}</td><td>${fmt.seconds(q.total_s)}</td>` +
         `<td>${fmt.rate(q.decode_tps)}</td><td>${q.slot}</td></tr>`;
}
