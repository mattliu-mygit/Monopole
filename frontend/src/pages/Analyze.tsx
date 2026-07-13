import { useState } from 'react'
import { useQuery } from '@tanstack/react-query'
import PageHeader from '../components/PageHeader'
import { getAnalysis } from '../api'
import type { ScoreSummary, ABEntry, TrendEntry } from '../types'

type Tab = 'summary' | 'ab' | 'trends' | 'coaching'

const tabs: { key: Tab; label: string }[] = [
  { key: 'summary', label: 'Summary' },
  { key: 'ab', label: 'A/B Comparison' },
  { key: 'trends', label: 'Trends' },
  { key: 'coaching', label: 'Coaching' },
]

function directionArrow(dir: string): string {
  if (dir === 'up') return '↑'
  if (dir === 'down') return '↓'
  return '—'
}

function bestMeanPerScorer(entries: ABEntry[]): Record<string, string> {
  const best: Record<string, { version: string; mean: number }> = {}
  for (const entry of entries) {
    for (const [scorer, data] of Object.entries(entry.scores)) {
      if (!best[scorer] || data.mean > best[scorer].mean) {
        best[scorer] = { version: entry.config_version, mean: data.mean }
      }
    }
  }
  const result: Record<string, string> = {}
  for (const [scorer, b] of Object.entries(best)) {
    result[scorer] = b.version
  }
  return result
}

function SummaryTable({ data }: { data: ScoreSummary[] }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm text-left">
        <thead>
          <tr className="border-b text-gray-500 text-xs uppercase tracking-wider">
            <th className="px-3 py-2 font-medium">Scorer</th>
            <th className="px-3 py-2 font-medium">Count</th>
            <th className="px-3 py-2 font-medium">Mean</th>
            <th className="px-3 py-2 font-medium">Pass Rate</th>
            <th className="px-3 py-2 font-medium">CI</th>
            <th className="px-3 py-2 font-medium">Confident</th>
          </tr>
        </thead>
        <tbody>
          {data.map((s) => (
            <tr key={s.scorer} className="border-b">
              <td className="px-3 py-2 font-medium">{s.scorer}</td>
              <td className="px-3 py-2">{s.count}</td>
              <td className="px-3 py-2">{s.mean.toFixed(2)}</td>
              <td className="px-3 py-2">
                {s.pass_rate !== null
                  ? `${(s.pass_rate * 100).toFixed(1)}%`
                  : '—'}
              </td>
              <td className="px-3 py-2 font-mono text-xs">
                [{s.ci[0].toFixed(2)}, {s.ci[1].toFixed(2)}]
              </td>
              <td className="px-3 py-2">
                {s.confident ? (
                  <span className="text-green-600">{'✓'}</span>
                ) : (
                  <span className="text-amber-500">{'⚠'}</span>
                )}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ABComparison({ data }: { data: ABEntry[] }) {
  const best = bestMeanPerScorer(data)
  const allScorers = Array.from(
    new Set(data.flatMap((e) => Object.keys(e.scores))),
  )

  return (
    <div className="space-y-4">
      {data.map((entry) => (
        <div key={entry.config_version} className="rounded-lg border p-4">
          <div className="flex items-center gap-3 mb-3">
            <span className="font-mono text-sm font-medium bg-gray-100 px-2 py-0.5 rounded">
              {entry.config_version.slice(0, 8)}
            </span>
            <span className="text-xs text-gray-500">
              {entry.turn_count} turns
            </span>
          </div>
          <table className="w-full text-sm text-left">
            <thead>
              <tr className="border-b text-gray-500 text-xs uppercase tracking-wider">
                <th className="px-2 py-1 font-medium">Scorer</th>
                <th className="px-2 py-1 font-medium">Mean</th>
                <th className="px-2 py-1 font-medium">Count</th>
              </tr>
            </thead>
            <tbody>
              {allScorers.map((scorer) => {
                const scoreData = entry.scores[scorer]
                if (!scoreData) return null
                const isBest = best[scorer] === entry.config_version
                return (
                  <tr key={scorer} className="border-b">
                    <td className="px-2 py-1">{scorer}</td>
                    <td
                      className={`px-2 py-1 ${isBest ? 'text-green-700 font-medium' : ''}`}
                    >
                      {scoreData.mean.toFixed(2)}
                    </td>
                    <td className="px-2 py-1 text-gray-500">
                      {scoreData.count}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        </div>
      ))}
    </div>
  )
}

function TrendsList({ data }: { data: TrendEntry[] }) {
  return (
    <div className="space-y-2">
      {data.map((t, i) => (
        <div
          key={`${t.scorer}-${i}`}
          className="flex items-center gap-3 rounded-lg border p-3"
        >
          <span
            className={`text-lg ${
              t.direction === 'up'
                ? 'text-green-600'
                : t.direction === 'down'
                  ? 'text-red-600'
                  : 'text-gray-500'
            }`}
          >
            {directionArrow(t.direction)}
          </span>
          <span className="font-medium text-sm text-gray-900">{t.scorer}</span>
          <span className="text-sm text-gray-500">
            {t.older_mean.toFixed(2)} {'→'} {t.recent_mean.toFixed(2)}
          </span>
          <span
            className={`text-sm font-mono ${t.delta >= 0 ? 'text-green-600' : 'text-red-600'}`}
          >
            {t.delta >= 0 ? '+' : ''}
            {t.delta.toFixed(3)}
          </span>
          {t.significant ? (
            <span className="inline-block rounded-full bg-blue-100 text-blue-800 px-2 py-0.5 text-xs font-medium">
              significant
            </span>
          ) : (
            <span className="inline-block rounded-full bg-gray-100 text-gray-600 px-2 py-0.5 text-xs font-medium">
              not significant
            </span>
          )}
          <span className="ml-auto text-xs text-gray-400">
            n={t.sample_count}
          </span>
        </div>
      ))}
      {data.length === 0 && (
        <p className="text-gray-500 text-sm">No trend data available.</p>
      )}
    </div>
  )
}

export default function Analyze() {
  const [activeTab, setActiveTab] = useState<Tab>('summary')

  const { data, isLoading, error } = useQuery({
    queryKey: ['analysis'],
    queryFn: () => getAnalysis(),
  })

  return (
    <div>
      <PageHeader title="Analysis" />

      <div className="flex gap-0 border-b mb-6">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            type="button"
            className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px ${
              activeTab === tab.key
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
            onClick={() => setActiveTab(tab.key)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {isLoading && <p className="text-gray-500 text-sm">Loading...</p>}
      {error && (
        <p className="text-red-600 text-sm">{(error as Error).message}</p>
      )}

      {data && activeTab === 'summary' && <SummaryTable data={data.summary} />}

      {data && activeTab === 'ab' && (
        <ABComparison data={data.ab_leaderboard} />
      )}

      {data && activeTab === 'trends' && <TrendsList data={data.trends} />}

      {data && activeTab === 'coaching' && (
        <div className="bg-gray-50 rounded-lg p-6 font-mono text-sm whitespace-pre-wrap">
          {data.coaching_markdown || 'No coaching recommendations available.'}
        </div>
      )}
    </div>
  )
}
