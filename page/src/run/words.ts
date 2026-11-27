// Small formatters the run view's stage sections share, after the web UI's
// (frontend/src/features/runs/JudgingProgress.tsx and ScoringProgress.tsx).

/** "not_evaluable" as "not evaluable". */
export const words = (value: string) => value.replaceAll("_", " ");

/** "1 turn", "3 turns". */
export const plural = (count: number, one: string, many = `${one}s`) => `${count.toLocaleString()} ${count === 1 ? one : many}`;

/** Turn positions as readable ranges, such as "turns 1–39, 41" or "turn 114". */
export function turnRanges(positions: number[]): string {
  const sorted = [...positions].sort((left, right) => left - right);
  const ranges: string[] = [];
  for (let start = 0; start < sorted.length; ) {
    let end = start;
    while (end + 1 < sorted.length && sorted[end + 1] === sorted[end] + 1) end += 1;
    ranges.push(end === start ? `${sorted[start]}` : `${sorted[start]}–${sorted[end]}`);
    start = end + 1;
  }
  return `${sorted.length === 1 ? "turn" : "turns"} ${ranges.join(", ")}`;
}

/** Token usage, such as "input tokens 1200 · output tokens 300", or "—". */
export const usageText = (usage: Record<string, unknown>) =>
  Object.entries(usage)
    .map(([key, value]) => `${words(key)} ${String(value)}`)
    .join(" · ") || "—";

/** How long from `start` to `endMs`: "42s", "3m 5s", "1h 2m"; null when either is unreadable. */
export function duration(start: string, endMs: number): string | null {
  const startMs = Date.parse(start);
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) return null;
  const total = Math.max(0, Math.floor((endMs - startMs) / 1_000));
  const hours = Math.floor(total / 3_600);
  const minutes = Math.floor((total % 3_600) / 60);
  const seconds = total % 60;
  if (hours > 0) return `${hours}h ${minutes}m`;
  if (minutes > 0) return `${minutes}m ${seconds}s`;
  return `${seconds}s`;
}

const CLOCK = new Intl.DateTimeFormat("en-US", { hour: "numeric", minute: "2-digit", second: "2-digit" });

/** "3:04:05 PM", in the Mac's time zone. */
export const clock = (iso: string) => (Number.isFinite(Date.parse(iso)) ? CLOCK.format(new Date(iso)) : "");

/** "0.75", or "—" for no score. */
export const fixed = (value: number | null | undefined, digits = 2) => (value === null || value === undefined ? "—" : value.toFixed(digits));
