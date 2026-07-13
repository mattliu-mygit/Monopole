import { useEffect, useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import PageHeader from '../components/PageHeader'
import JobProgress from '../components/JobProgress'
import { getArtifacts, getJobs, getModels, submitJob, applyReflection } from '../api'
import type { Job } from '../types'

function ConfirmDialog({
  open,
  message,
  onConfirm,
  onCancel,
}: {
  open: boolean
  message: string
  onConfirm: () => void
  onCancel: () => void
}) {
  if (!open) return null
  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg p-6 max-w-sm shadow-lg">
        <p className="text-sm text-gray-700 mb-4">{message}</p>
        <div className="flex gap-3 justify-end">
          <button
            type="button"
            onClick={onCancel}
            className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 text-sm"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium"
          >
            Apply
          </button>
        </div>
      </div>
    </div>
  )
}

function CompletedReflection({ job }: { job: Job }) {
  const [confirmOpen, setConfirmOpen] = useState(false)
  const result = job.result ?? {}

  const applyMutation = useMutation({
    mutationFn: () => applyReflection(job.job_id),
  })

  return (
    <div className="rounded-lg border p-4 space-y-3">
      <div className="flex items-center justify-between">
        <span className="font-mono text-xs text-gray-500">{job.job_id}</span>
        <span className="inline-block rounded-full bg-green-100 text-green-800 px-2 py-0.5 text-xs font-medium">
          complete
        </span>
      </div>

      {result.rationale && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-1">Rationale</h4>
          <p className="text-sm text-gray-600">{result.rationale}</p>
        </div>
      )}

      {result.score_delta !== undefined && (
        <div className="text-sm">
          <span className="text-gray-500">Score delta: </span>
          <span
            className={`font-mono font-medium ${result.score_delta >= 0 ? 'text-green-600' : 'text-red-600'}`}
          >
            {result.score_delta >= 0 ? '+' : ''}
            {result.score_delta.toFixed(3)}
          </span>
        </div>
      )}

      {result.diff && (
        <div>
          <h4 className="text-sm font-medium text-gray-700 mb-1">Diff</h4>
          <pre className="bg-gray-50 rounded-lg border p-3 text-xs font-mono overflow-auto max-h-64 whitespace-pre-wrap">
            {result.diff}
          </pre>
        </div>
      )}

      <div>
        <button
          type="button"
          onClick={() => setConfirmOpen(true)}
          disabled={applyMutation.isPending || applyMutation.isSuccess}
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium disabled:opacity-50"
        >
          {applyMutation.isPending
            ? 'Applying...'
            : applyMutation.isSuccess
              ? 'Applied'
              : 'Apply Changes'}
        </button>
        {applyMutation.error && (
          <p className="text-red-600 text-sm mt-2">
            {(applyMutation.error as Error).message}
          </p>
        )}
        {applyMutation.isSuccess && (
          <p className="text-green-600 text-sm mt-2">Changes applied.</p>
        )}
      </div>

      <ConfirmDialog
        open={confirmOpen}
        message="Apply the reflected changes to your config artifacts? This will overwrite the current versions."
        onConfirm={() => {
          setConfirmOpen(false)
          applyMutation.mutate()
        }}
        onCancel={() => setConfirmOpen(false)}
      />
    </div>
  )
}

export default function Reflect() {
  const [showForm, setShowForm] = useState(false)
  const [model, setModel] = useState('')
  const [judgeBackend, setJudgeBackend] = useState('cli')
  const [iterations, setIterations] = useState(3)
  const [dryRun, setDryRun] = useState(false)

  const modelsQuery = useQuery({
    queryKey: ['models'],
    queryFn: getModels,
  })

  const backendModels = (() => {
    const rosters = modelsQuery.data
    if (!rosters || !rosters[judgeBackend]) return []
    const r = rosters[judgeBackend]
    const seen = new Set<string>()
    const all: string[] = []
    for (const m of [...r.default, ...r.poll, r.escalation]) {
      if (!seen.has(m)) {
        seen.add(m)
        all.push(m)
      }
    }
    return all
  })()

  useEffect(() => {
    if (backendModels.length > 0 && !backendModels.includes(model)) {
      setModel(backendModels[0])
    }
  }, [judgeBackend, backendModels.length])

  const artifactsQuery = useQuery({
    queryKey: ['artifacts'],
    queryFn: getArtifacts,
  })

  const jobsQuery = useQuery({
    queryKey: ['jobs'],
    queryFn: getJobs,
    refetchInterval: (query) => {
      const jobs = query.state.data?.jobs ?? []
      const hasRunning = jobs.some(
        (j) => j.type === 'reflect' && j.status === 'running',
      )
      return hasRunning ? 2000 : false
    },
  })

  const reflectJobs = (jobsQuery.data?.jobs ?? []).filter(
    (j) => j.type === 'reflect',
  )
  const completedReflectJobs = reflectJobs.filter(
    (j) => j.status === 'complete',
  )

  const reflectMutation = useMutation({
    mutationFn: () =>
      submitJob('reflect', {
        model,
        judge_backend: judgeBackend,
        iterations,
        dry_run: dryRun,
      }),
    onSuccess: () => {
      setShowForm(false)
      jobsQuery.refetch()
    },
  })

  const artifacts = artifactsQuery.data?.artifacts ?? []

  return (
    <div>
      <PageHeader title="Reflect">
        <button
          type="button"
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium"
          onClick={() => setShowForm(!showForm)}
        >
          Run Reflection
        </button>
      </PageHeader>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div>
          <h2 className="text-lg font-semibold text-gray-900 mb-4">
            Current Artifacts
          </h2>
          {artifactsQuery.isLoading && (
            <p className="text-gray-500 text-sm">Loading...</p>
          )}
          {artifacts.length > 0 ? (
            <div className="space-y-3">
              {artifacts.map((artifact) => (
                <div key={artifact.name} className="rounded-lg border">
                  <div className="flex items-center justify-between bg-gray-50 px-3 py-2 border-b rounded-t-lg">
                    <span className="text-sm font-medium text-gray-700">
                      {artifact.name}
                    </span>
                    <span className="text-xs text-gray-500 font-mono">
                      {artifact.path}
                    </span>
                  </div>
                  <pre className="p-3 text-xs font-mono whitespace-pre-wrap">
                    {artifact.content}
                  </pre>
                </div>
              ))}
            </div>
          ) : (
            !artifactsQuery.isLoading && (
              <p className="text-gray-500 text-sm">No artifacts found.</p>
            )
          )}
        </div>

        <div>
          {showForm && (
            <div className="rounded-lg border p-4 mb-6">
              <h3 className="text-lg font-semibold text-gray-900 mb-4">
                Reflection Configuration
              </h3>
              <div className="space-y-4 max-w-sm">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Model
                  </label>
                  <select
                    value={model}
                    onChange={(e) => setModel(e.target.value)}
                    className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    {backendModels.map((m) => (
                      <option key={m} value={m}>
                        {m}
                      </option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Judge Backend
                  </label>
                  <select
                    value={judgeBackend}
                    onChange={(e) => setJudgeBackend(e.target.value)}
                    className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
                  >
                    <option value="cli">cli</option>
                    <option value="openai">openai</option>
                    <option value="wandb">wandb</option>
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Iterations
                  </label>
                  <input
                    type="number"
                    value={iterations}
                    onChange={(e) => setIterations(Number(e.target.value))}
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
                  onClick={() => reflectMutation.mutate()}
                  disabled={reflectMutation.isPending}
                  className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium disabled:opacity-50"
                >
                  {reflectMutation.isPending ? 'Submitting...' : 'Submit'}
                </button>
                {reflectMutation.error && (
                  <p className="text-red-600 text-sm mt-2">
                    {(reflectMutation.error as Error).message}
                  </p>
                )}
              </div>
            </div>
          )}

          {completedReflectJobs.length > 0 && (
            <div className="mb-6">
              <h2 className="text-lg font-semibold text-gray-900 mb-4">
                Completed Reflections
              </h2>
              <div className="space-y-3">
                {completedReflectJobs.map((job) => (
                  <CompletedReflection key={job.job_id} job={job} />
                ))}
              </div>
            </div>
          )}

          <div>
            <h2 className="text-lg font-semibold text-gray-900 mb-4">
              Recent Reflect Jobs
            </h2>
            {jobsQuery.isLoading && (
              <p className="text-gray-500 text-sm">Loading...</p>
            )}
            {reflectJobs.length > 0 ? (
              <div className="space-y-3">
                {reflectJobs.map((job) => (
                  <JobProgress key={job.job_id} job={job} />
                ))}
              </div>
            ) : (
              !jobsQuery.isLoading && (
                <p className="text-gray-500 text-sm">No reflect jobs yet.</p>
              )
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
