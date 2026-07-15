import { useId, useRef, useState, type KeyboardEvent } from 'react'
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

function directionArrow(dir: TrendEntry['direction']): string {
  return dir === 'regression' ? '↓' : '↑'
}

function directionLabel(dir: TrendEntry['direction']): string {
  return dir === 'regression' ? 'Regression' : 'Improvement'
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
  if (data.length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-6 text-sm text-gray-600">
        <p className="font-medium text-gray-900">No score summaries yet.</p>
        <p className="mt-1">Complete an evaluation run to populate this view.</p>
      </div>
    )
  }

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
  if (data.length === 0) {
    return (
      <div className="rounded-lg border border-dashed p-6 text-sm text-gray-600">
        <p className="font-medium text-gray-900">No A/B comparisons yet.</p>
        <p className="mt-1">
          Evaluate traces from more than one configuration version to compare them.
        </p>
      </div>
    )
  }

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
              {entry.evaluated_target_count} evaluated targets
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
            role="img"
            aria-label={directionLabel(t.direction)}
            className={`text-lg ${t.direction === 'regression' ? 'text-red-600' : 'text-green-600'}`}
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
  const tabRefs = useRef(new Map<Tab, HTMLButtonElement>())
  const tabsId = useId().replaceAll(':', '')

  const { data, isLoading, error, refetch } = useQuery({
    queryKey: ['analysis'],
    queryFn: () => getAnalysis(),
  })

  function tabId(tab: Tab): string {
    return `${tabsId}-tab-${tab}`
  }

  function panelId(tab: Tab): string {
    return `${tabsId}-panel-${tab}`
  }

  function handleTabKeyDown(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    let nextIndex: number | null = null
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      nextIndex = (index + 1) % tabs.length
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
      nextIndex = (index - 1 + tabs.length) % tabs.length
    } else if (event.key === 'Home') {
      nextIndex = 0
    } else if (event.key === 'End') {
      nextIndex = tabs.length - 1
    }
    if (nextIndex === null) return
    event.preventDefault()
    const nextTab = tabs[nextIndex].key
    setActiveTab(nextTab)
    tabRefs.current.get(nextTab)?.focus()
  }

  return (
    <div>
      <PageHeader title="Analysis" />

      <div
        className="flex gap-0 border-b mb-6"
        role="tablist"
        aria-label="Analysis views"
      >
        {tabs.map((tab, index) => (
          <button
            key={tab.key}
            ref={(element) => {
              if (element) tabRefs.current.set(tab.key, element)
              else tabRefs.current.delete(tab.key)
            }}
            id={tabId(tab.key)}
            type="button"
            role="tab"
            aria-selected={activeTab === tab.key}
            aria-controls={panelId(tab.key)}
            tabIndex={activeTab === tab.key ? 0 : -1}
            className={`px-4 py-2 text-sm font-medium border-b-2 -mb-px ${
              activeTab === tab.key
                ? 'border-blue-500 text-blue-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
            onClick={() => setActiveTab(tab.key)}
            onKeyDown={(event) => handleTabKeyDown(event, index)}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {isLoading && <p className="text-gray-500 text-sm">Loading...</p>}
      {error && (
        <div className="flex items-center gap-3 text-sm">
          <p className="text-red-600" role="alert">{(error as Error).message}</p>
          <button
            type="button"
            className="font-medium text-blue-600 hover:underline"
            onClick={() => void refetch()}
          >
            Retry
          </button>
        </div>
      )}

      <div
        id={panelId('summary')}
        role="tabpanel"
        aria-labelledby={tabId('summary')}
        hidden={activeTab !== 'summary'}
        tabIndex={0}
      >
        {data && <SummaryTable data={data.summary} />}
      </div>

      <div
        id={panelId('ab')}
        role="tabpanel"
        aria-labelledby={tabId('ab')}
        hidden={activeTab !== 'ab'}
        tabIndex={0}
      >
        {data && <ABComparison data={data.ab_leaderboard} />}
      </div>

      <div
        id={panelId('trends')}
        role="tabpanel"
        aria-labelledby={tabId('trends')}
        hidden={activeTab !== 'trends'}
        tabIndex={0}
      >
        {data && <TrendsList data={data.trends} />}
      </div>

      <div
        id={panelId('coaching')}
        role="tabpanel"
        aria-labelledby={tabId('coaching')}
        hidden={activeTab !== 'coaching'}
        tabIndex={0}
      >
        {data && (
          <div className="bg-gray-50 rounded-lg p-6 font-mono text-sm whitespace-pre-wrap">
            {data.coaching_markdown || 'No coaching recommendations available.'}
          </div>
        )}
      </div>
    </div>
  )
}
