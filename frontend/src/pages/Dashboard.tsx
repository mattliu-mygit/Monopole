import { useQuery } from '@tanstack/react-query'
import PageHeader from '../components/PageHeader'
import ScoreCard from '../components/ScoreCard'
import { getAnalysis } from '../api'

export default function Dashboard() {
  const analysisQuery = useQuery({
    queryKey: ['analysis'],
    queryFn: () => getAnalysis(),
  })

  const analysis = analysisQuery.data

  return (
    <div>
      <PageHeader title="Dashboard" />

      {analysisQuery.isLoading && (
        <p className="text-gray-500 text-sm">Loading...</p>
      )}
      {analysisQuery.error && (
        <div className="flex items-center gap-3 text-sm">
          <p className="text-red-600" role="alert">
            {(analysisQuery.error as Error).message}
          </p>
          <button
            type="button"
            className="font-medium text-blue-600 hover:underline"
            onClick={() => void analysisQuery.refetch()}
          >
            Retry
          </button>
        </div>
      )}

      {analysis && analysis.summary.length === 0 && (
        <div className="rounded-lg border border-dashed p-6 text-sm text-gray-600">
          <p className="font-medium text-gray-900">No scores yet.</p>
          <p className="mt-1">Complete an evaluation run to populate this dashboard.</p>
        </div>
      )}

      {analysis && analysis.summary.length > 0 && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4">
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
    </div>
  )
}
