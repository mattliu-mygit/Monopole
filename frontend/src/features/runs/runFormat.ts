// Formatters for a run's stage activity that the web UI and the Prompting Time plugin's page
// (page/src/run) both use, so the two show a run the same way.

/** Judging events the activity hides unless debug events are shown. */
export const DEBUG_EVENT_PHASES: ReadonlySet<string> = new Set([
  'judging_started',
  'session_started',
  'transport_attempt_started',
  'transport_attempt_completed',
  'schema_recovery_reused',
])

/** "not_evaluable" as "not evaluable". */
export function words(value: string): string {
  return value.replaceAll('_', ' ')
}

/** Turn positions as readable ranges, such as "turns 1–39, 41" or "turn 114". */
export function turnRanges(positions: number[]): string {
  const sorted = [...positions].sort((left, right) => left - right)
  const ranges: string[] = []
  for (let start = 0; start < sorted.length;) {
    let end = start
    while (end + 1 < sorted.length && sorted[end + 1] === sorted[end] + 1) end += 1
    ranges.push(end === start ? `${sorted[start]}` : `${sorted[start]}–${sorted[end]}`)
    start = end + 1
  }
  return `${sorted.length === 1 ? 'turn' : 'turns'} ${ranges.join(', ')}`
}

/** Token usage, such as "input tokens 1200 · output tokens 300", or "—". */
export function usageText(usage: Record<string, unknown>): string {
  return Object.entries(usage).map(([key, value]) => `${words(key)} ${String(value)}`).join(' · ') || '—'
}

/** How long from `start` to `endMs`: "42s", "3m 5s", "1h 2m"; null when either is unreadable. */
export function durationLabel(start: string, endMs: number): string | null {
  const startMs = Date.parse(start)
  if (!Number.isFinite(startMs) || !Number.isFinite(endMs)) return null
  const totalSeconds = Math.max(0, Math.floor((endMs - startMs) / 1_000))
  const hours = Math.floor(totalSeconds / 3_600)
  const minutes = Math.floor((totalSeconds % 3_600) / 60)
  const seconds = totalSeconds % 60
  if (hours > 0) return `${hours}h ${minutes}m`
  if (minutes > 0) return `${minutes}m ${seconds}s`
  return `${seconds}s`
}
