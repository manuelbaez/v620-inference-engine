// A line chart on a canvas: time on x (the last windowS seconds), one line per series.
// Samples come every 5 s; a gap of more than 3 samples breaks the line.
// Hovering (or touching) a chart marks the nearest sample and shows its exact values.

const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const L = 48, R = 8, T = 8, B = 20;

export function lineChart(canvas, history, series, windowS, minMax) {
  canvas._chart = {history, series, windowS, minMax};
  if (!canvas._tip) attachTooltip(canvas);
  draw(canvas);
}

function draw(canvas) {
  const {history, series, windowS, minMax} = canvas._chart;
  const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);

  const now = Date.now() / 1000, t0 = now - windowS;
  const points = history.filter(p => p.t >= t0);
  let ymax = minMax || 1;
  for (const p of points) for (const s of series) ymax = Math.max(ymax, p[s.key] || 0);
  ymax *= 1.1;

  const pw = w - L - R, ph = h - T - B;
  g.strokeStyle = cssVar("--line");
  g.fillStyle = cssVar("--muted");
  g.font = "11px system-ui";
  g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = T + ph * i / 4;
    g.beginPath(); g.moveTo(L, y); g.lineTo(w - R, y); g.stroke();
    g.fillText((ymax * (1 - i / 4)).toFixed(1), 4, y + 4);
  }
  g.fillText(`-${Math.round(windowS / 60)} min`, L, h - 4);
  g.fillText("now", w - R - 22, h - 4);

  const X = t => L + pw * (t - t0) / windowS, Y = v => T + ph * (1 - v / ymax);
  for (const s of series) {
    g.strokeStyle = cssVar(s.color);
    g.lineWidth = 1.8;
    g.beginPath();
    let prev = null;
    for (const p of points) {
      const x = X(p.t), y = Y(p[s.key] || 0);
      if (prev == null || p.t - prev > 16) g.moveTo(x, y); else g.lineTo(x, y);
      prev = p.t;
    }
    g.stroke();
  }

  const p = canvas._hover != null ? nearest(points, t0 + (canvas._hover - L) / pw * windowS) : null;
  if (!p) { canvas._tip.hidden = true; return; }
  const x = X(p.t);
  g.strokeStyle = cssVar("--muted");
  g.lineWidth = 1;
  g.setLineDash([3, 3]);
  g.beginPath(); g.moveTo(x, T); g.lineTo(x, T + ph); g.stroke();
  g.setLineDash([]);
  for (const s of series) {
    g.fillStyle = cssVar(s.color);
    g.beginPath(); g.arc(x, Y(p[s.key] || 0), 3.5, 0, 2 * Math.PI); g.fill();
  }
  showTip(canvas, p, x);
}

// The sample closest in time to t, or null when none is within 10 s.
function nearest(points, t) {
  let best = null;
  for (const p of points) if (!best || Math.abs(p.t - t) < Math.abs(best.t - t)) best = p;
  return best && Math.abs(best.t - t) <= 10 ? best : null;
}

function showTip(canvas, p, x) {
  const tip = canvas._tip, {series} = canvas._chart;
  const time = new Date(p.t * 1000).toLocaleTimeString();
  tip.innerHTML = `<div class="tip-t">${time}</div>` + series.map(s =>
    `<div><span class="tip-c" style="--c:var(${s.color})"></span>${s.label}: <b>${(s.fmt || fixed1)(p[s.key] || 0)}</b></div>`
  ).join("");
  tip.hidden = false;
  // Keep the box inside the chart: right of the marker, or left of it near the right edge.
  const left = canvas.offsetLeft + x, right = canvas.offsetLeft + canvas.clientWidth;
  tip.style.top = canvas.offsetTop + 4 + "px";
  tip.style.left = (left + 12 + tip.offsetWidth > right ? left - 12 - tip.offsetWidth : left + 12) + "px";
}

const fixed1 = v => v.toFixed(1);

function attachTooltip(canvas) {
  const tip = document.createElement("div");
  tip.className = "chart-tip";
  tip.hidden = true;
  canvas.parentElement.appendChild(tip);
  canvas._tip = tip;
  const move = e => { canvas._hover = e.clientX - canvas.getBoundingClientRect().left; draw(canvas); };
  canvas.addEventListener("pointermove", move);
  canvas.addEventListener("pointerdown", move);
  canvas.addEventListener("pointerleave", () => { canvas._hover = null; draw(canvas); });
}
