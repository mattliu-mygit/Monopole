import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import PageHeader from '../components/PageHeader'
import JobProgress from '../components/JobProgress'
import { getJobs, submitJob } from '../api'

export default function Scoring() {
  const [showScoreForm, setShowScoreForm] = useState(false)
  const [showBackfillForm, setShowBackfillForm] = useState(false)

  const [scoreSince, setScoreSince] = useState('')
  const [scoreLimit, setScoreLimit] = useState(100)
  const [scoreDryRun, setScoreDryRun] = useState(false)
  const [scoreForce, setScoreForce] = useState(false)

  const [backfillStart, setBackfillStart] = useState('')
  const [backfillEnd, setBackfillEnd] = useState('')
  const [backfillDryRun, setBackfillDryRun] = useState(false)
  const [backfillForce, setBackfillForce] = useState(false)

  const jobsQuery = useQuery({
    queryKey: ['jobs'],
    queryFn: getJobs,
    refetchInterval: (query) => {
      const jobs = query.state.data?.jobs ?? []
      const hasRunning = jobs.some(
        (j) =>
          (j.type === 'score' || j.type === 'backfill') &&
          j.status === 'running',
      )
      return hasRunning ? 2000 : false
    },
  })

  const relevantJobs = (jobsQuery.data?.jobs ?? []).filter(
    (j) => j.type === 'score' || j.type === 'backfill',
  )

  const scoreMutation = useMutation({
    mutationFn: () =>
      submitJob('score', {
        since: scoreSince || undefined,
        limit: scoreLimit,
        dry_run: scoreDryRun,
        force: scoreForce,
      }),
    onSuccess: () => {
      setShowScoreForm(false)
      jobsQuery.refetch()
    },
  })

  const backfillMutation = useMutation({
    mutationFn: () =>
      submitJob('backfill', {
        start: backfillStart,
        end: backfillEnd || undefined,
        dry_run: backfillDryRun,
        force: backfillForce,
      }),
    onSuccess: () => {
      setShowBackfillForm(false)
      jobsQuery.refetch()
    },
  })

  return (
    <div>
      <PageHeader title="Scoring">
        <button
          type="button"
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium"
          onClick={() => {
            setShowScoreForm(!showScoreForm)
            setShowBackfillForm(false)
          }}
        >
          Score Recent
        </button>
        <button
          type="button"
          className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 text-sm"
          onClick={() => {
            setShowBackfillForm(!showBackfillForm)
            setShowScoreForm(false)
          }}
        >
          Backfill
        </button>
      </PageHeader>

      {showScoreForm && (
        <div className="rounded-lg border p-4 mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Score Recent Turns
          </h3>
          <div className="grid grid-cols-2 gap-4 max-w-lg">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Since
              </label>
              <input
                type="date"
                value={scoreSince}
                onChange={(e) => setScoreSince(e.target.value)}
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Limit
              </label>
              <input
                type="number"
                value={scoreLimit}
                onChange={(e) => setScoreLimit(Number(e.target.value))}
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={scoreDryRun}
                onChange={(e) => setScoreDryRun(e.target.checked)}
              />
              Dry run
            </label>
            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={scoreForce}
                onChange={(e) => setScoreForce(e.target.checked)}
              />
              Force
            </label>
          </div>
          <div className="mt-4">
            <button
              type="button"
              onClick={() => scoreMutation.mutate()}
              disabled={scoreMutation.isPending}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium disabled:opacity-50"
            >
              {scoreMutation.isPending ? 'Submitting...' : 'Submit'}
            </button>
            {scoreMutation.error && (
              <p className="text-red-600 text-sm mt-2">
                {(scoreMutation.error as Error).message}
              </p>
            )}
          </div>
        </div>
      )}

      {showBackfillForm && (
        <div className="rounded-lg border p-4 mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Backfill Scores
          </h3>
          <div className="grid grid-cols-2 gap-4 max-w-lg">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Start <span className="text-red-500">*</span>
              </label>
              <input
                type="date"
                value={backfillStart}
                onChange={(e) => setBackfillStart(e.target.value)}
                required
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                End
              </label>
              <input
                type="date"
                value={backfillEnd}
                onChange={(e) => setBackfillEnd(e.target.value)}
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={backfillDryRun}
                onChange={(e) => setBackfillDryRun(e.target.checked)}
              />
              Dry run
            </label>
            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={backfillForce}
                onChange={(e) => setBackfillForce(e.target.checked)}
              />
              Force
            </label>
          </div>
          <div className="mt-4">
            <button
              type="button"
              onClick={() => backfillMutation.mutate()}
              disabled={backfillMutation.isPending || !backfillStart}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium disabled:opacity-50"
            >
              {backfillMutation.isPending ? 'Submitting...' : 'Submit'}
            </button>
            {backfillMutation.error && (
              <p className="text-red-600 text-sm mt-2">
                {(backfillMutation.error as Error).message}
              </p>
            )}
          </div>
        </div>
      )}

      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Recent Score Jobs
        </h2>
        {jobsQuery.isLoading && (
          <p className="text-gray-500 text-sm">Loading...</p>
        )}
        {relevantJobs.length > 0 ? (
          <div className="space-y-3">
            {relevantJobs.map((job) => (
              <JobProgress key={job.job_id} job={job} />
            ))}
          </div>
        ) : (
          !jobsQuery.isLoading && (
            <p className="text-gray-500 text-sm">No score jobs yet.</p>
          )
        )}
      </div>
    </div>
  )
}
