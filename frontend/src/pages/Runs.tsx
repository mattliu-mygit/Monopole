import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router-dom'
import PageHeader from '../components/PageHeader'
import RunStatusBadge from '../features/runs/RunStatusBadge'
import { getRuns } from '../api'
import type { RunReviewState, RunSummary } from '../types'
import { shouldPollRuns } from '../features/runs/runPolling'
import {
  formatSelectionRange,
  selectionTimezoneLabel,
} from '../features/runs/selectionSummary'

function sessionSummary(run: RunSummary): string {
  const sel = run.selection
  if (!sel) return '—'
  return `${sel.session_count} selected`
}

function rangeSummary(run: RunSummary): string {
  const selection = run.selection
  if (!selection) return 'All dates'
  return formatSelectionRange(selection)
}

function reviewLabel(state: RunReviewState): string {
  if (state === 'review-needed') return 'Review needed'
  if (state === 'promoted') return 'Promoted'
  if (state === 'partial') return 'Partially applied'
  if (state === 'dismissed') return 'Dismissed'
  if (state === 'no-change') return 'No change'
  if (state === 'no-valid-proposal') return 'No valid proposal'
  return '—'
}

function ReviewBadge({ run }: { run: RunSummary }) {
  const label = reviewLabel(run.review_state)
  if (label === '—') return <span className="text-gray-400">—</span>
  const colors =
    label === 'Review needed'
      ? 'bg-indigo-100 text-indigo-800'
      : label === 'Promoted'
        ? 'bg-green-100 text-green-800'
        : label === 'Partially applied'
          ? 'bg-amber-100 text-amber-900'
        : label === 'Dismissed'
          ? 'bg-gray-100 text-gray-700'
          : label === 'No change'
            ? 'bg-green-50 text-green-800'
            : label === 'No valid proposal'
              ? 'bg-amber-50 text-amber-900'
            : 'bg-amber-100 text-amber-900'
  return <span className={`rounded-full px-2.5 py-0.5 text-xs font-medium ${colors}`}>{label}</span>
}

export default function Runs() {
  const runsQuery = useQuery({
    queryKey: ['runs'],
    queryFn: getRuns,
    refetchInterval: (query) => {
      const runs = query.state.data?.runs ?? []
      return shouldPollRuns(runs) ? 2000 : false
    },
  })

  const runs = runsQuery.data?.runs ?? []

  return (
    <div>
      <PageHeader title="Runs">
        <Link
          to="/runs/new"
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium disabled:opacity-50"
        >
          New Run
        </Link>
      </PageHeader>

      {runsQuery.isLoading && <p className="text-gray-500 text-sm">Loading...</p>}
      {runsQuery.error && (
        <div className="flex items-center gap-3 text-sm">
          <p role="alert" className="text-red-600">
            {(runsQuery.error as Error).message}
          </p>
          <button
            type="button"
            className="font-medium text-blue-600 hover:underline"
            onClick={() => void runsQuery.refetch()}
          >
            Retry
          </button>
        </div>
      )}

      {runs.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead>
              <tr className="border-b text-gray-500 text-xs uppercase tracking-wider">
                <th className="px-3 py-2 font-medium">Run ID</th>
                <th className="px-3 py-2 font-medium">Status</th>
                <th className="px-3 py-2 font-medium">Review</th>
                <th className="hidden px-3 py-2 font-medium md:table-cell">Created</th>
                <th className="hidden px-3 py-2 font-medium md:table-cell">Selection</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr key={run.run_id} className="border-b hover:bg-gray-50">
                  <td className="px-3 py-2 font-mono text-xs">
                    <Link
                      to={`/runs/${run.run_id}`}
                      className="font-semibold text-blue-700 hover:text-blue-900 hover:underline focus:outline-none focus:ring-2 focus:ring-blue-500"
                    >
                      {run.run_id}
                    </Link>
                  </td>
                  <td className="px-3 py-2">
                    <RunStatusBadge status={run.status} />
                  </td>
                  <td className="px-3 py-2">
                    <ReviewBadge run={run} />
                  </td>
                  <td className="hidden px-3 py-2 text-gray-500 md:table-cell">
                    {new Date(run.created_at).toLocaleString()}
                  </td>
                  <td className="hidden px-3 py-2 text-gray-600 md:table-cell">
                    <div>{sessionSummary(run)}</div>
                    <div className="text-xs text-gray-400">{rangeSummary(run)}</div>
                    {run.selection && (
                      <div className="font-mono text-[0.6875rem] text-gray-400">
                        {selectionTimezoneLabel(run.selection.timezone)}
                      </div>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!runsQuery.isLoading && !runsQuery.error && runs.length === 0 && (
        <p className="text-gray-500 text-sm">
          No runs yet. Click "New Run" to start an evaluation pipeline.
        </p>
      )}
    </div>
  )
}
