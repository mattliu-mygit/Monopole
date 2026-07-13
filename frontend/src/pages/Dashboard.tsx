import { useQuery } from '@tanstack/react-query'
import { useNavigate } from 'react-router-dom'
import PageHeader from '../components/PageHeader'
import ScoreCard from '../components/ScoreCard'
import { getAnalysis, getJobs } from '../api'

function statusColor(status: string): string {
  if (status === 'complete') return 'bg-green-500'
  if (status === 'running') return 'bg-yellow-500'
  return 'bg-red-500'
}

export default function Dashboard() {
  const navigate = useNavigate()

  const analysisQuery = useQuery({
    queryKey: ['analysis'],
    queryFn: () => getAnalysis(),
  })

  const jobsQuery = useQuery({
    queryKey: ['jobs'],
    queryFn: getJobs,
  })

  const analysis = analysisQuery.data
  const recentJobs = (jobsQuery.data?.jobs ?? []).slice(0, 5)

  return (
    <div>
      <PageHeader title="Dashboard" />

      {analysisQuery.isLoading && (
        <p className="text-gray-500 text-sm">Loading...</p>
      )}
      {analysisQuery.error && (
        <p className="text-red-600 text-sm">
          {(analysisQuery.error as Error).message}
        </p>
      )}

      {analysis && (
        <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
          {analysis.summary.map((s) => {
            const trend = analysis.trends.find((t) => t.scorer === s.scorer)
            return (
              <ScoreCard
                key={s.scorer}
                scorer={s.scorer}
                mean={s.mean}
                count={s.count}
                ci={s.ci}
                passRate={s.pass_rate}
                confident={s.confident}
                trend={
                  trend
                    ? { direction: trend.direction, delta: trend.delta }
                    : undefined
                }
              />
            )
          })}
        </div>
      )}

      <div className="mt-8">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">
          Recent Jobs
        </h2>

        {jobsQuery.isLoading && (
          <p className="text-gray-500 text-sm">Loading...</p>
        )}
        {jobsQuery.error && (
          <p className="text-red-600 text-sm">
            {(jobsQuery.error as Error).message}
          </p>
        )}

        {recentJobs.length > 0 && (
          <ul className="space-y-2">
            {recentJobs.map((job) => (
              <li
                key={job.job_id}
                className="flex items-center gap-3 rounded-lg border p-3 hover:bg-gray-50 cursor-pointer"
                onClick={() => navigate('/jobs')}
              >
                <span
                  className={`inline-block rounded-full w-2 h-2 ${statusColor(job.status)}`}
                />
                <span className="font-mono text-xs text-gray-600">
                  {job.job_id}
                </span>
                <span className="text-sm text-gray-700">{job.type}</span>
                <span className="ml-auto text-xs text-gray-500">
                  {new Date(job.started_at).toLocaleString()}
                </span>
              </li>
            ))}
          </ul>
        )}

        {!jobsQuery.isLoading && recentJobs.length === 0 && (
          <p className="text-gray-500 text-sm">No jobs yet.</p>
        )}
      </div>
    </div>
  )
}
