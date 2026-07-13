import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import PageHeader from '../components/PageHeader'
import JobProgress from '../components/JobProgress'
import { getJobs, submitJob } from '../api'
import type { Job } from '../types'

// Monitor state is fetched directly since there's no dedicated api.ts helper yet
async function getMonitorState(): Promise<{ alert_keys: string[] }> {
  const base = import.meta.env.VITE_API_URL || ''
  const res = await fetch(`${base}/api/monitor/state`)
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`API ${res.status}: ${body}`)
  }
  return res.json()
}

function AlertList({ job }: { job: Job }) {
  const alerts = job.result?.alerts ?? []
  if (alerts.length === 0) {
    return <p className="text-gray-500 text-sm">No alerts triggered.</p>
  }
  return (
    <ul className="space-y-2">
      {alerts.map((alert: any, i: number) => (
        <li key={i} className="rounded-lg border border-amber-200 bg-amber-50 p-3">
          <div className="text-sm font-medium text-amber-800">
            {alert.key ?? alert.type ?? `Alert ${i + 1}`}
          </div>
          {alert.message && (
            <p className="text-sm text-amber-700 mt-1">{alert.message}</p>
          )}
        </li>
      ))}
    </ul>
  )
}

export default function Monitor() {
  const [showForm, setShowForm] = useState(false)
  const [limit, setLimit] = useState(1000)
  const [alertWebhook, setAlertWebhook] = useState('')
  const [dryRun, setDryRun] = useState(false)

  const stateQuery = useQuery({
    queryKey: ['monitor-state'],
    queryFn: getMonitorState,
  })

  const jobsQuery = useQuery({
    queryKey: ['jobs'],
    queryFn: getJobs,
    refetchInterval: (query) => {
      const jobs = query.state.data?.jobs ?? []
      const hasRunning = jobs.some(
        (j) => j.type === 'monitor' && j.status === 'running',
      )
      return hasRunning ? 2000 : false
    },
  })

  const monitorJobs = (jobsQuery.data?.jobs ?? []).filter(
    (j) => j.type === 'monitor',
  )
  const monitorMutation = useMutation({
    mutationFn: () =>
      submitJob('monitor', {
        limit,
        alert_webhook: alertWebhook || undefined,
        dry_run: dryRun,
      }),
    onSuccess: () => {
      jobsQuery.refetch()
      stateQuery.refetch()
    },
  })

  const alertKeys = stateQuery.data?.alert_keys ?? []

  return (
    <div>
      <PageHeader title="Monitor">
        <button
          type="button"
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium"
          onClick={() => setShowForm(!showForm)}
        >
          Run Check
        </button>
      </PageHeader>

      {showForm && (
        <div className="rounded-lg border p-4 mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Monitor Configuration
          </h3>
          <div className="space-y-4 max-w-lg">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Limit
              </label>
              <input
                type="number"
                value={limit}
                onChange={(e) => setLimit(Number(e.target.value))}
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Alert Webhook (optional)
              </label>
              <input
                type="text"
                value={alertWebhook}
                onChange={(e) => setAlertWebhook(e.target.value)}
                placeholder="https://hooks.example.com/..."
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={dryRun}
                onChange={(e) => setDryRun(e.target.checked)}
              />
              Dry run
            </label>
          </div>
          <div className="mt-4">
            <button
              type="button"
              onClick={() => monitorMutation.mutate()}
              disabled={monitorMutation.isPending}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium disabled:opacity-50"
            >
              {monitorMutation.isPending ? 'Submitting...' : 'Submit'}
            </button>
            {monitorMutation.error && (
              <p className="text-red-600 text-sm mt-2">
                {(monitorMutation.error as Error).message}
              </p>
            )}
          </div>
        </div>
      )}

      <div className="mb-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Current State
        </h2>
        {stateQuery.isLoading && (
          <p className="text-gray-500 text-sm">Loading...</p>
        )}
        {stateQuery.error && (
          <p className="text-red-600 text-sm">
            {(stateQuery.error as Error).message}
          </p>
        )}
        {stateQuery.data && (
          <div className="rounded-lg border p-4">
            <span className="text-sm text-gray-700">
              Seen alert keys:{' '}
              <span className="font-medium">{alertKeys.length}</span>
            </span>
            {alertKeys.length > 0 && (
              <div className="flex flex-wrap gap-1 mt-2">
                {alertKeys.map((key) => (
                  <span
                    key={key}
                    className="inline-block rounded-full bg-gray-100 px-2 py-0.5 text-xs text-gray-600"
                  >
                    {key}
                  </span>
                ))}
              </div>
            )}
          </div>
        )}
      </div>

      <div className="mb-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Recent Monitor Jobs
        </h2>
        {jobsQuery.isLoading && (
          <p className="text-gray-500 text-sm">Loading...</p>
        )}
        {monitorJobs.length > 0 ? (
          <div className="space-y-3">
            {monitorJobs.map((job) => (
              <div key={job.job_id}>
                <JobProgress job={job} />
                {job.status === 'complete' && (
                  <div className="mt-2 ml-4">
                    <AlertList job={job} />
                  </div>
                )}
              </div>
            ))}
          </div>
        ) : (
          !jobsQuery.isLoading && (
            <p className="text-gray-500 text-sm">No monitor jobs yet.</p>
          )
        )}
      </div>
    </div>
  )
}
