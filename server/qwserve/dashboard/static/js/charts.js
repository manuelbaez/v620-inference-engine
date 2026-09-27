// A line chart on a canvas: time on x (the last windowS seconds), one line per series.
// Samples come every 5 s; a gap of more than 3 samples breaks the line.

const cssVar = name => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function lineChart(canvas, history, series, windowS, minMax) {
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

  const L = 48, R = 8, T = 8, B = 20, pw = w - L - R, ph = h - T - B;
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
}
