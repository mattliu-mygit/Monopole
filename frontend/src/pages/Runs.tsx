import { useQuery, useMutation } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import PageHeader from '../components/PageHeader'
import { getRuns, createRun } from '../api'
import type { Run } from '../types'

const STATUS_COLORS: Record<string, string> = {
  created: 'bg-gray-100 text-gray-700',
  scoring: 'bg-blue-100 text-blue-800',
  judging: 'bg-purple-100 text-purple-800',
  reflecting: 'bg-indigo-100 text-indigo-800',
  complete: 'bg-green-100 text-green-800',
  failed: 'bg-red-100 text-red-800',
}

function StatusBadge({ status }: { status: string }) {
  return (
    <span
      className={`inline-block rounded-full px-2.5 py-0.5 text-xs font-medium ${STATUS_COLORS[status] ?? 'bg-gray-100 text-gray-700'}`}
    >
      {status}
    </span>
  )
}

function sessionSummary(run: Run): string {
  const sel = run.data_selection
  if (!sel) return '—'
  const included = sel.session_ids.length
  const excluded = sel.excluded_session_ids.length
  const base = included > 0 ? `${included} selected` : 'all in range'
  return excluded > 0 ? `${base}, ${excluded} excluded` : base
}

const TERMINAL_STATUSES = new Set(['complete', 'failed'])

export default function Runs() {
  const navigate = useNavigate()

  const runsQuery = useQuery({
    queryKey: ['runs'],
    queryFn: getRuns,
    refetchInterval: (query) => {
      const runs = query.state.data?.runs ?? []
      const hasActive = runs.some((r) => !TERMINAL_STATUSES.has(r.status))
      return hasActive ? 2000 : false
    },
  })

  const createMutation = useMutation({
    mutationFn: createRun,
    onSuccess: (run) => {
      navigate(`/runs/${run.run_id}`)
    },
  })

  const runs = runsQuery.data?.runs ?? []

  return (
    <div>
      <PageHeader title="Runs">
        <button
          type="button"
          onClick={() => createMutation.mutate()}
          disabled={createMutation.isPending}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium disabled:opacity-50"
        >
          {createMutation.isPending ? 'Creating...' : 'New Run'}
        </button>
      </PageHeader>

      {createMutation.error && (
        <p className="text-red-600 text-sm mb-4">
          {(createMutation.error as Error).message}
        </p>
      )}

      {runsQuery.isLoading && <p className="text-gray-500 text-sm">Loading...</p>}
      {runsQuery.error && (
        <p className="text-red-600 text-sm">{(runsQuery.error as Error).message}</p>
      )}

      {runs.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead>
              <tr className="border-b text-gray-500 text-xs uppercase tracking-wider">
                <th className="px-3 py-2 font-medium">Run ID</th>
                <th className="px-3 py-2 font-medium">Status</th>
                <th className="px-3 py-2 font-medium">Created</th>
                <th className="px-3 py-2 font-medium">Sessions</th>
                <th className="px-3 py-2 font-medium">Config</th>
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => (
                <tr
                  key={run.run_id}
                  className="border-b hover:bg-gray-50 cursor-pointer"
                  onClick={() => navigate(`/runs/${run.run_id}`)}
                >
                  <td className="px-3 py-2 font-mono text-xs">{run.run_id}</td>
                  <td className="px-3 py-2">
                    <StatusBadge status={run.status} />
                  </td>
                  <td className="px-3 py-2 text-gray-500">
                    {new Date(run.created_at).toLocaleString()}
                  </td>
                  <td className="px-3 py-2 text-gray-600">{sessionSummary(run)}</td>
                  <td className="px-3 py-2 font-mono text-xs text-gray-500">
                    {run.config_version ? run.config_version.slice(0, 8) : '—'}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {!runsQuery.isLoading && runs.length === 0 && (
        <p className="text-gray-500 text-sm">
          No runs yet. Click "New Run" to start an evaluation pipeline.
        </p>
      )}
    </div>
  )
}
