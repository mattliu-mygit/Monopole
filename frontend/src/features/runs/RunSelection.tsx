import type { SessionSummary } from '../../types'
import SignalCallout from '../../components/SignalCallout'

export interface RunSelectionProps {
  since: string
  until: string
  sessions: SessionSummary[]
  selectedSessionIds: string[]
  totalSessions: number
  loading?: boolean
  error?: string | null
  truncated?: boolean
  disabled?: boolean
  onSinceChange: (value: string) => void
  onUntilChange: (value: string) => void
  onSessionIdsChange: (sessionIds: string[]) => void
  onRetry?: () => void
}

const inputClass =
  'block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500 disabled:bg-gray-50 disabled:text-gray-500'

function formatDateTime(value: string | null): string {
  if (!value) return '—'
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}

export default function RunSelection({
  since,
  until,
  sessions,
  selectedSessionIds,
  totalSessions,
  loading = false,
  error = null,
  truncated = false,
  disabled = false,
  onSinceChange,
  onUntilChange,
  onSessionIdsChange,
  onRetry,
}: RunSelectionProps) {
  const selected = new Set(selectedSessionIds)

  function toggleSession(sessionId: string) {
    const next = new Set(selected)
    if (next.has(sessionId)) next.delete(sessionId)
    else next.add(sessionId)
    onSessionIdsChange([...next])
  }

  return (
    <section aria-label="Run data selection" className="space-y-4">
      <div className="grid max-w-lg grid-cols-1 gap-4 sm:grid-cols-2">
        <div>
          <label htmlFor="run-since" className="mb-1 block text-sm font-medium text-gray-700">
            Since
          </label>
          <input
            id="run-since"
            type="date"
            value={since}
            disabled={disabled}
            onChange={(event) => onSinceChange(event.target.value)}
            className={inputClass}
          />
        </div>
        <div>
          <label htmlFor="run-until" className="mb-1 block text-sm font-medium text-gray-700">
            Until
          </label>
          <input
            id="run-until"
            type="date"
            value={until}
            disabled={disabled}
            onChange={(event) => onUntilChange(event.target.value)}
            className={inputClass}
          />
        </div>
      </div>

      <div>
        <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
          <h4 className="text-sm font-medium text-gray-700">
            Sessions in range ({sessions.length})
          </h4>
          <span className="text-xs text-gray-500">
            {selectedSessionIds.length} selected
          </span>
        </div>

        {truncated && (
          <p className="mb-2 rounded-lg border border-amber-200 bg-amber-50 p-2 text-xs text-amber-900">
            Showing {sessions.length} of {totalSessions} sessions. Only displayed sessions are
            available to select.
          </p>
        )}
        {loading && (
          <p role="status" className="text-sm text-gray-500">
            Loading sessions...
          </p>
        )}
        {error && (
          <div className="flex flex-wrap items-center gap-3 rounded-lg border border-red-200 bg-red-50 p-2 text-sm text-red-700">
            <p role="alert">{error}</p>
            {onRetry && (
              <button type="button" className="font-semibold text-blue-700 hover:underline" onClick={onRetry}>
                Retry session discovery
              </button>
            )}
          </div>
        )}

        {!loading && sessions.length > 0 && (
          <div className="max-h-64 space-y-1.5 overflow-y-auto">
            {sessions.map((session) => (
              <label
                key={session.conversation_id}
                className="flex items-center gap-3 rounded-lg border p-2.5 text-sm hover:bg-gray-50"
              >
                <input
                  type="checkbox"
                  aria-label={`Select session ${session.conversation_id}`}
                  checked={selected.has(session.conversation_id)}
                  disabled={disabled}
                  onChange={() => toggleSession(session.conversation_id)}
                />
                <span className="min-w-0">
                  <span className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-gray-500">
                    <span className="font-mono text-gray-700">
                      {session.conversation_id.slice(0, 12)}
                    </span>
                    <span>
                      {session.turn_count} turn{session.turn_count === 1 ? '' : 's'}
                    </span>
                    <span>{formatDateTime(session.started_at)}</span>
                  </span>
                  {session.input_preview && (
                    <span className="mt-0.5 block truncate text-xs text-gray-600">
                      {session.input_preview}
                    </span>
                  )}
                  <SignalCallout evidence={session.signal_evidence} />
                </span>
              </label>
            ))}
          </div>
        )}
        {!loading && !error && sessions.length === 0 && (
          <p className="text-sm text-gray-500">No sessions found in this range.</p>
        )}
      </div>
    </section>
  )
}
