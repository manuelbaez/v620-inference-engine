// The dashboard's data: the snapshot the server's collector thread refreshes.

export async function fetchMetrics() {
  const r = await fetch("metrics.json", {cache: "no-store"});
  if (!r.ok) throw new Error(`HTTP ${r.status}`);
  return r.json();
}
