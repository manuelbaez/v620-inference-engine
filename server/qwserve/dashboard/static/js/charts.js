// A line chart on a canvas: time on x (the last windowS seconds), one line per series.
// Samples come every 5 s (none while the server is idle). The line joins the samples with a
// value: solid between consecutive ones, dotted across inactivity (more than 3 samples'
// time without any, or samples whose value is null: no work of that kind). A series with
// `marks` (the speeds, which have no value while idle) gets a dot on every sample and a
// straight dotted line between measurements; the others keep their last value (0 once the
// server went idle) across the gap and step to the next one.
// Hovering (or touching) a chart marks the nearest sample and shows its exact values.

const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const L = 48, R = 8, T = 8, B = 20;

// `whole`: the values are counts, so the axis shows whole numbers.
export function lineChart(canvas, history, series, windowS, minMax, whole = false) {
  canvas._chart = {history, series, windowS, minMax, whole};
  if (!canvas._tip) attachTooltip(canvas);
  draw(canvas);
}

function draw(canvas) {
  const {history, series, windowS, minMax, whole} = canvas._chart;
  const dpr = window.devicePixelRatio || 1, w = canvas.clientWidth, h = canvas.clientHeight;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const g = canvas.getContext("2d");
  g.scale(dpr, dpr);
  g.clearRect(0, 0, w, h);

  const now = Date.now() / 1000, t0 = now - windowS;
  const points = history.filter(p => p.t >= t0);
  let ymax = minMax || 1;
  for (const p of points) for (const s of series) ymax = Math.max(ymax, p[s.key] ?? 0);
  // counts get whole numbers on the axis: the top is the next multiple of the 4 grid steps above the largest value
  ymax = whole ? 4 * Math.ceil((ymax + 1) / 4) : ymax * 1.1;

  const pw = w - L - R, ph = h - T - B;
  g.strokeStyle = cssVar("--line");
  g.fillStyle = cssVar("--muted");
  g.font = "11px system-ui";
  g.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = T + ph * i / 4;
    g.beginPath(); g.moveTo(L, y); g.lineTo(w - R, y); g.stroke();
    g.fillText((ymax * (1 - i / 4)).toFixed(whole ? 0 : 1), 4, y + 4);
  }
  g.fillText(`-${Math.round(windowS / 60)} min`, L, h - 4);
  g.fillText("now", w - R - 22, h - 4);

  const X = t => L + pw * (t - t0) / windowS, Y = v => T + ph * (1 - v / ymax);
  for (const s of series) {
    g.strokeStyle = g.fillStyle = cssVar(s.color);
    g.lineWidth = 1.8;
    g.lineCap = g.lineJoin = "round";
    drawSeries(g, points, s.key, X, Y, s.marks);
  }

  // the tooltip snaps to the nearest sample with a value, wherever it is on the line
  const candidates = points.filter(p => series.some(s => p[s.key] != null));
  const p = canvas._hover != null ? nearest(candidates, t0 + (canvas._hover - L) / pw * windowS) : null;
  if (!p) { canvas._tip.hidden = true; return; }
  const x = X(p.t);
  g.strokeStyle = cssVar("--muted");
  g.lineWidth = 1;
  g.setLineDash([3, 3]);
  g.beginPath(); g.moveTo(x, T); g.lineTo(x, T + ph); g.stroke();
  g.setLineDash([]);
  for (const s of series) {
    if (p[s.key] == null) continue;
    g.fillStyle = cssVar(s.color);
    g.beginPath(); g.arc(x, Y(p[s.key]), 3.5, 0, 2 * Math.PI); g.fill();
  }
  showTip(canvas, p, x);
}

function drawSeries(g, points, key, X, Y, marks) {
  const solid = new Path2D(), dotted = new Path2D();
  let prev = null, idle = false, n = 0;
  for (const p of points) {
    if (p[key] == null) { idle = true; continue; }
    ++n;
    if (prev) {
      const gap = idle || p.t - prev.t > 16, path = gap ? dotted : solid;
      path.moveTo(X(prev.t), Y(prev[key]));
      if (gap && !marks) path.lineTo(X(p.t), Y(prev[key]));  // held across the gap, then a step
      path.lineTo(X(p.t), Y(p[key]));
    }
    prev = p;
    idle = false;
  }
  g.stroke(solid);
  g.save();
  g.lineWidth = 1.2;
  g.globalAlpha = 0.6;
  g.setLineDash([2, 4]);
  g.stroke(dotted);
  g.restore();
  if (n === 1) {  // a single sample: show it as a dot
    g.beginPath(); g.arc(X(prev.t), Y(prev[key]), 2, 0, 2 * Math.PI); g.fill();
  }
  if (marks) for (const p of points)
    if (p[key] != null) { g.beginPath(); g.arc(X(p.t), Y(p[key]), 2, 0, 2 * Math.PI); g.fill(); }
}

// The sample closest in time to t (null when there are none).
function nearest(points, t) {
  let best = null;
  for (const p of points) if (!best || Math.abs(p.t - t) < Math.abs(best.t - t)) best = p;
  return best;
}

function showTip(canvas, p, x) {
  const tip = canvas._tip, {series} = canvas._chart;
  const time = new Date(p.t * 1000).toLocaleTimeString();
  tip.innerHTML = `<div class="tip-t">${time}</div>` + series.map(s =>
    `<div><span class="tip-c" style="--c:var(${s.color})"></span>${s.label}: <b>${p[s.key] == null ? (s.none || "-") : (s.fmt || fixed1)(p[s.key])}</b></div>`
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
