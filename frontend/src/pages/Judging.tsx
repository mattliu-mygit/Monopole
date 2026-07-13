import { useState } from 'react'
import { useQuery, useMutation } from '@tanstack/react-query'
import PageHeader from '../components/PageHeader'
import JobProgress from '../components/JobProgress'
import { getJobs, getModels, getRubrics, submitJob } from '../api'

export default function Judging() {
  const [showForm, setShowForm] = useState(false)
  const [showRubricsRef, setShowRubricsRef] = useState(false)

  const [since, setSince] = useState('')
  const [limit, setLimit] = useState(10)
  const [judgeBackend, setJudgeBackend] = useState('cli')
  const [panelSize, setPanelSize] = useState(1)
  const [dryRun, setDryRun] = useState(false)
  const [force, setForce] = useState(false)
  const [selectedRubrics, setSelectedRubrics] = useState<Set<string>>(new Set())

  const modelsQuery = useQuery({
    queryKey: ['models'],
    queryFn: getModels,
  })

  const rubricsQuery = useQuery({
    queryKey: ['rubrics'],
    queryFn: getRubrics,
  })

  const jobsQuery = useQuery({
    queryKey: ['jobs'],
    queryFn: getJobs,
    refetchInterval: (query) => {
      const jobs = query.state.data?.jobs ?? []
      const hasRunning = jobs.some(
        (j) => j.type === 'judge' && j.status === 'running',
      )
      return hasRunning ? 2000 : false
    },
  })

  const judgeJobs = (jobsQuery.data?.jobs ?? []).filter(
    (j) => j.type === 'judge',
  )

  const rubrics = rubricsQuery.data?.rubrics ?? []

  function toggleRubric(name: string) {
    setSelectedRubrics((prev) => {
      const next = new Set(prev)
      if (next.has(name)) {
        next.delete(name)
      } else {
        next.add(name)
      }
      return next
    })
  }

  const judgeMutation = useMutation({
    mutationFn: () =>
      submitJob('judge', {
        since: since || undefined,
        limit,
        judge_backend: judgeBackend,
        panel_size: panelSize,
        dry_run: dryRun,
        force,
        rubrics:
          selectedRubrics.size > 0 ? Array.from(selectedRubrics) : undefined,
      }),
    onSuccess: () => {
      setShowForm(false)
      jobsQuery.refetch()
    },
  })

  return (
    <div>
      <PageHeader title="Judging">
        <button
          type="button"
          className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium"
          onClick={() => setShowForm(!showForm)}
        >
          Run Judges
        </button>
      </PageHeader>

      {showForm && (
        <div className="rounded-lg border p-4 mb-6">
          <h3 className="text-lg font-semibold text-gray-900 mb-4">
            Judge Configuration
          </h3>
          <div className="grid grid-cols-2 gap-4 max-w-lg">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Since
              </label>
              <input
                type="date"
                value={since}
                onChange={(e) => setSince(e.target.value)}
                className="block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
            </div>
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
                Panel Size
              </label>
              <input
                type="number"
                value={panelSize}
                onChange={(e) => setPanelSize(Number(e.target.value))}
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
            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input
                type="checkbox"
                checked={force}
                onChange={(e) => setForce(e.target.checked)}
              />
              Force
            </label>
          </div>

          {modelsQuery.data?.[judgeBackend] && (
            <div className="mt-4">
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Judge Models ({panelSize > 1 ? 'PoLL panel' : 'default'})
              </label>
              <div className="flex flex-wrap gap-1.5">
                {(panelSize > 1
                  ? modelsQuery.data[judgeBackend].poll
                  : modelsQuery.data[judgeBackend].default
                ).map((m) => (
                  <span
                    key={m}
                    className="inline-block rounded-full bg-gray-100 text-gray-700 px-2.5 py-0.5 text-xs font-medium"
                  >
                    {m}
                  </span>
                ))}
              </div>
            </div>
          )}

          {rubrics.length > 0 && (
            <div className="mt-4">
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Rubrics
              </label>
              <div className="grid grid-cols-2 gap-1">
                {rubrics.map((r) => (
                  <label
                    key={r.name}
                    className="flex items-center gap-2 text-sm text-gray-700"
                  >
                    <input
                      type="checkbox"
                      checked={selectedRubrics.has(r.name)}
                      onChange={() => toggleRubric(r.name)}
                    />
                    {r.name}
                  </label>
                ))}
              </div>
            </div>
          )}

          <div className="mt-4">
            <button
              type="button"
              onClick={() => judgeMutation.mutate()}
              disabled={judgeMutation.isPending}
              className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium disabled:opacity-50"
            >
              {judgeMutation.isPending ? 'Submitting...' : 'Submit'}
            </button>
            {judgeMutation.error && (
              <p className="text-red-600 text-sm mt-2">
                {(judgeMutation.error as Error).message}
              </p>
            )}
          </div>
        </div>
      )}

      <div className="mb-6">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Recent Judge Jobs
        </h2>
        {jobsQuery.isLoading && (
          <p className="text-gray-500 text-sm">Loading...</p>
        )}
        {judgeJobs.length > 0 ? (
          <div className="space-y-3">
            {judgeJobs.map((job) => (
              <JobProgress key={job.job_id} job={job} />
            ))}
          </div>
        ) : (
          !jobsQuery.isLoading && (
            <p className="text-gray-500 text-sm">No judge jobs yet.</p>
          )
        )}
      </div>

      <div>
        <button
          type="button"
          className="flex items-center gap-2 text-sm font-medium text-gray-700 mb-4"
          onClick={() => setShowRubricsRef(!showRubricsRef)}
        >
          <span>{showRubricsRef ? '▼' : '▶'}</span>
          Available Rubrics
        </button>

        {showRubricsRef && (
          <div className="space-y-3">
            {rubricsQuery.isLoading && (
              <p className="text-gray-500 text-sm">Loading...</p>
            )}
            {rubrics.map((r) => (
              <div key={r.name} className="rounded-lg border p-4">
                <div className="font-medium text-sm text-gray-900">
                  {r.name}
                </div>
                <p className="text-sm text-gray-600 mt-1">{r.description}</p>
                {Object.keys(r.criteria).length > 0 && (
                  <div className="mt-2">
                    <span className="text-xs font-medium text-gray-500 uppercase tracking-wider">
                      Criteria
                    </span>
                    <ul className="mt-1 space-y-1">
                      {Object.entries(r.criteria).map(([k, v]) => (
                        <li key={k} className="text-xs text-gray-600">
                          <span className="font-medium">{k}:</span> {v}
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
