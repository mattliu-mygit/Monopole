import type { DataSelection } from '../../types'
import {
  formatSelectionRange,
  selectionTimezoneLabel,
} from './selectionSummary'

type Cohort = Record<string, unknown>

function stringField(cohort: Cohort | null, key: string): string | null {
  const value = cohort?.[key]
  return typeof value === 'string' && value.length > 0 ? value : null
}

function countField(cohort: Cohort | null, key: string): number | null {
  const value = cohort?.[key]
  return typeof value === 'number' && Number.isInteger(value) && value >= 0 ? value : null
}

function formatPinnedAt(value: string, timezone: string | null | undefined): string {
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return value
  try {
    return new Intl.DateTimeFormat('en-US', {
      dateStyle: 'medium',
      timeStyle: 'short',
      timeZone: timezone || undefined,
    }).format(date)
  } catch {
    return date.toLocaleString()
  }
}

export default function PinnedSelectionAudit({
  selection,
  cohort,
}: {
  selection: DataSelection | null
  cohort: Cohort | null
}) {
  if (!selection && !cohort) return null

  const timezone = selection?.timezone
  const cohortId = stringField(cohort, 'cohort_id')
  const pinnedAt = stringField(cohort, 'pinned_at')
  const turnCount = countField(cohort, 'turn_count')
  const sessionCount = countField(cohort, 'session_count')
  const sessionIds = selection?.session_ids ?? []

  return (
    <section
      aria-label="Pinned selection and cohort"
      className="space-y-3 rounded-lg border border-gray-200 bg-gray-50/60 p-4"
    >
      <div>
        <h3 className="text-sm font-semibold text-gray-900">Pinned selection and cohort</h3>
        <p className="mt-0.5 text-xs text-gray-500">
          The run evaluates this exact session and turn snapshot.
        </p>
      </div>
      <dl className="grid gap-3 text-sm sm:grid-cols-2 lg:grid-cols-3">
        <div>
          <dt className="text-xs font-medium uppercase tracking-wide text-gray-500">Date range</dt>
          <dd className="mt-1 text-gray-900">
            {selection ? formatSelectionRange(selection) : 'Unavailable'}
          </dd>
        </div>
        <div>
          <dt className="text-xs font-medium uppercase tracking-wide text-gray-500">Timezone</dt>
          <dd className="mt-1 font-mono text-xs text-gray-900">
            {selectionTimezoneLabel(timezone)}
          </dd>
        </div>
        <div>
          <dt className="text-xs font-medium uppercase tracking-wide text-gray-500">Exact evidence</dt>
          <dd className="mt-1 text-gray-900">
            {turnCount !== null && sessionCount !== null
              ? `${turnCount} turn${turnCount === 1 ? '' : 's'} across ${sessionCount} session${sessionCount === 1 ? '' : 's'}`
              : 'Not pinned yet'}
          </dd>
        </div>
        <div className="sm:col-span-2">
          <dt className="text-xs font-medium uppercase tracking-wide text-gray-500">Cohort ID</dt>
          <dd className="mt-1 break-all font-mono text-xs text-gray-900">
            {cohortId ?? 'Not pinned yet'}
          </dd>
        </div>
        <div>
          <dt className="text-xs font-medium uppercase tracking-wide text-gray-500">Pinned at</dt>
          <dd className="mt-1 text-gray-900">
            {pinnedAt ? formatPinnedAt(pinnedAt, timezone) : 'Not pinned yet'}
          </dd>
        </div>
      </dl>
      <details className="rounded-lg border border-gray-200 bg-white p-3">
        <summary className="cursor-pointer text-xs font-semibold text-gray-800">
          {sessionIds.length} selected session ID{sessionIds.length === 1 ? '' : 's'}
        </summary>
        {sessionIds.length > 0 ? (
          <ul className="mt-2 max-h-48 space-y-1 overflow-auto">
            {sessionIds.map((sessionId) => (
              <li key={sessionId} className="break-all font-mono text-xs text-gray-700">
                {sessionId}
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-2 text-xs text-gray-500">No session IDs were recorded.</p>
        )}
      </details>
    </section>
  )
}
