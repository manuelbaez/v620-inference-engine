// Number formatting for the dashboard: one decimal place at most.

export const count = n => n == null ? "-"
  : n >= 1e9 ? (n / 1e9).toFixed(1) + " B"
  : n >= 1e6 ? (n / 1e6).toFixed(1) + " M"
  : n >= 1e4 ? (n / 1e3).toFixed(1) + " k"
  : String(Math.round(n));

export const bytes = b => b == null ? "-"
  : b >= 1e12 ? (b / 1e12).toFixed(1) + " TB"
  : b >= 1e9 ? (b / 1e9).toFixed(1) + " GB"
  : (b / 1e6).toFixed(1) + " MB";

export const seconds = s => s == null ? "-"
  : s < 120 ? s.toFixed(1) + " s"
  : (s / 60).toFixed(1) + " min";

export const rate = r => r == null ? "-" : r.toFixed(1) + " tok/s";

export const percent = p => p == null ? "-" : p.toFixed(1) + "%";

export const duration = s => {
  const h = Math.floor(s / 3600), m = Math.floor(s % 3600 / 60);
  return h ? `${h} h ${m} min` : `${m} min`;
};
