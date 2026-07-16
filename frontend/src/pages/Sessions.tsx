import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import PageHeader from '../components/PageHeader'
import SignalCallout from '../components/SignalCallout'
import { getSessions } from '../api'
import type { SessionSummary } from '../types'

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}k`
  return n.toLocaleString()
}

function formatDate(iso: string | null): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleString()
}

function duration(start: string | null, end: string | null): string {
  if (!start || !end) return '—'
  const ms = new Date(end).getTime() - new Date(start).getTime()
  const mins = Math.floor(ms / 60_000)
  if (mins < 1) return '<1m'
  if (mins < 60) return `${mins}m`
  const hours = Math.floor(mins / 60)
  return `${hours}h ${mins % 60}m`
}

export default function Sessions() {
  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['sessions'],
    queryFn: () => getSessions({ limit: 100 }),
  })

  const sessions = data?.sessions ?? []
  const total = data?.total ?? sessions.length
  const omitted = Math.max(total - sessions.length, 0)

  return (
    <div>
      <PageHeader title="Sessions" />

      {isLoading && <p className="text-gray-500 text-sm">Loading...</p>}
      {error && (
        <div className="flex items-center gap-3 text-sm">
          <p className="text-red-600" role="alert">{(error as Error).message}</p>
          <button
            type="button"
            className="font-medium text-blue-600 hover:underline"
            onClick={() => void refetch()}
          >
            Retry
          </button>
        </div>
      )}

      {data && (sessions.length > 0 || data.truncated) && (
        <p className="mb-3 text-sm text-gray-500">
          {data.truncated ? (
            <>
              Showing the newest {sessions.length} of {total} sessions.{' '}
              {omitted} older session{omitted === 1 ? ' is' : 's are'} omitted.
            </>
          ) : (
            <>Showing {sessions.length} of {total} sessions.</>
          )}
        </p>
      )}

      {sessions.length > 0 && (
        <div className="space-y-2">
          {sessions.map((session: SessionSummary) => (
            <Link
              key={session.conversation_id}
              to={`/sessions/${encodeURIComponent(session.conversation_id)}`}
              className="block rounded-lg border p-4 hover:bg-gray-50 focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <div className="flex items-start justify-between gap-4">
                <div className="flex-1 min-w-0">
                  {session.input_preview ? (
                    <p className="text-sm text-gray-900 line-clamp-2">
                      {session.input_preview}
                    </p>
                  ) : (
                    <p className="text-sm text-gray-400 italic">No input recorded</p>
                  )}
                  <div className="flex flex-wrap items-center gap-x-4 gap-y-1 mt-1.5 text-xs text-gray-500">
                    <span>{formatDate(session.last_activity ?? session.started_at)}</span>
                    <span>{duration(session.started_at, session.ended_at)}</span>
                    <span>
                      {session.turn_count} turn{session.turn_count !== 1 ? 's' : ''}
                    </span>
                    <span>{formatTokens(session.total_tokens)} tokens</span>
                    {session.model && <span>{session.model}</span>}
                    {session.effort_level && (
                      <span className="bg-gray-100 px-1.5 py-0.5 rounded">
                        {session.effort_level}
                      </span>
                    )}
                    {session.config_version && (
                      <span className="font-mono bg-gray-100 px-1.5 py-0.5 rounded">
                        {session.config_version.slice(0, 8)}
                      </span>
                    )}
                    {session.git_branch && <span>{session.git_branch}</span>}
                  </div>
                  <SignalCallout evidence={session.signal_evidence} />
                </div>
                <span className="font-mono text-xs text-gray-400 shrink-0">
                  {session.conversation_id.slice(0, 8)}
                </span>
              </div>
            </Link>
          ))}
        </div>
      )}

      {!isLoading && !error && sessions.length === 0 && (
        <p className="text-gray-500 text-sm">No sessions found.</p>
      )}
    </div>
  )
}
