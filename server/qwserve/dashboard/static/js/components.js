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
