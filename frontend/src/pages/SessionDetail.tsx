import { useState } from 'react'
import { useParams } from 'react-router-dom'
import { useQuery } from '@tanstack/react-query'
import PageHeader from '../components/PageHeader'
import ScoreBadge from '../components/ScoreBadge'
import { getSession, getRubrics } from '../api'
import type { TurnDetail, FeedbackItem, Rubric } from '../types'

function formatDuration(start: string | null, end: string | null): string {
  if (!start) return '—'
  const s = new Date(start).getTime()
  const e = end ? new Date(end).getTime() : Date.now()
  const sec = Math.round((e - s) / 1000)
  if (sec < 60) return `${sec}s`
  const min = Math.floor(sec / 60)
  const rem = sec % 60
  return `${min}m ${rem}s`
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  if (n >= 1_000) return `${(n / 1_000).toFixed(0)}k`
  return n.toLocaleString()
}

function feedbackScore(fb: FeedbackItem): number {
  return fb.payload.rating
}

function StatPill({ label, value, warn }: { label: string; value: string | number; warn?: boolean }) {
  return (
    <span className={`text-xs px-2 py-0.5 rounded ${warn ? 'bg-red-50 text-red-700' : 'bg-gray-100 text-gray-600'}`}>
      {label} {value}
    </span>
  )
}

function scorerLabel(feedbackType: string): string {
  return feedbackType.replace(/^weave_agent_signals\./, '')
}

function FeedbackDetail({ fb, rubrics }: { fb: FeedbackItem; rubrics: Rubric[] }) {
  const [showCriteria, setShowCriteria] = useState(false)
  const scorer = scorerLabel(fb.feedback_type)
  const rubric = rubrics.find(r => r.scorer_name === scorer)
  const reason = fb.payload.reason
  const details = fb.payload.details ?? {}
  const rationale = details.rationale as string | undefined
  const tags = fb.payload.tags ?? []

  const errorLoops = details.error_loops as Array<{ tool_name: string; attempt_count: number; similarity: number }> | undefined
  const repeatedReads = details.repeated_reads as Array<{ path: string; count: number }> | undefined
  const wasteRatio = details.waste_ratio as number | undefined
  const outcomes = details.outcomes as Array<Record<string, any>> | undefined

  const judgeModel = details.judge_model as string | undefined
  const panelModels = details.panel_models as string[] | undefined

  return (
    <div className="bg-gray-50 rounded p-3 space-y-2">
      <div className="flex items-start justify-between gap-2">
        <div className="flex items-center gap-2">
          <ScoreBadge scorer={fb.feedback_type} value={fb.payload.rating} />
          {tags.map(tag => (
            <span key={tag} className="text-xs px-1.5 py-0.5 rounded bg-gray-200 text-gray-600">{tag}</span>
          ))}
        </div>
        {judgeModel && !panelModels && <span className="text-xs text-gray-400">judged by {judgeModel}</span>}
        {panelModels && <span className="text-xs text-gray-400">panel: {panelModels.join(', ')}</span>}
      </div>

      {rubric && (
        <p className="text-xs text-gray-500 italic">{rubric.description}</p>
      )}

      {(reason || rationale) && (
        <p className="text-sm text-gray-700">{reason || rationale}</p>
      )}

      {errorLoops && errorLoops.length > 0 && (
        <div className="text-xs space-y-0.5">
          <span className="text-gray-500 font-medium">Error loops:</span>
          {errorLoops.map((loop, i) => (
            <div key={i} className="ml-2 text-gray-600">
              {loop.tool_name}: {loop.attempt_count} attempts ({(loop.similarity * 100).toFixed(0)}% similar)
            </div>
          ))}
        </div>
      )}

      {repeatedReads && repeatedReads.length > 0 && (
        <div className="text-xs space-y-0.5">
          <span className="text-gray-500 font-medium">Repeated reads:</span>
          {repeatedReads.map((rr, i) => (
            <div key={i} className="ml-2 text-gray-600 font-mono">{rr.path} ({rr.count}x)</div>
          ))}
        </div>
      )}

      {wasteRatio != null && (
        <div className="text-xs text-gray-500">Waste ratio: {(wasteRatio * 100).toFixed(0)}%</div>
      )}

      {outcomes && outcomes.length > 0 && (
        <div className="text-xs space-y-0.5">
          <span className="text-gray-500 font-medium">Outcomes:</span>
          {outcomes.map((o, i) => (
            <div key={i} className="ml-2 text-gray-600">
              {o.framework || o.tool || o.operation || 'command'}
              {o.passed != null && `: ${o.passed} passed, ${o.failed ?? 0} failed`}
              {o.issue_count != null && `: ${o.issue_count} issues`}
              {o.error_count != null && `: ${o.error_count} errors, ${o.warning_count ?? 0} warnings`}
              {o.files_changed != null && `: ${o.files_changed} files (+${o.insertions ?? 0} -${o.deletions ?? 0})`}
              {o.success != null && o.passed == null && o.issue_count == null && o.error_count == null && o.files_changed == null && `: ${o.success ? 'success' : 'failed'}`}
            </div>
          ))}
        </div>
      )}

      {rubric && (
        <div>
          <button
            type="button"
            className="text-xs text-blue-600 hover:underline"
            onClick={() => setShowCriteria(!showCriteria)}
          >
            {showCriteria ? 'Hide' : 'Show'} scoring criteria
          </button>
          {showCriteria && (
            <div className="mt-1 text-xs space-y-0.5 text-gray-600 border-l-2 border-gray-200 pl-2">
              {Object.entries(rubric.criteria)
                .sort(([a], [b]) => parseFloat(b) - parseFloat(a))
                .map(([level, desc]) => (
                  <div key={level} className={parseFloat(level) === fb.payload.rating ? 'font-medium text-gray-900' : ''}>
                    <span className="font-mono">{level}</span>: {desc}
                  </div>
                ))}
            </div>
          )}
        </div>
      )}
    </div>
  )
}

function TurnCard({ turn, index, feedback, rubrics }: { turn: TurnDetail; index: number; feedback: FeedbackItem[]; rubrics: Rubric[] }) {
  const [expanded, setExpanded] = useState(false)
  const totalTokens = turn.input_tokens + turn.output_tokens + turn.cache_read_tokens

  return (
    <div className="rounded-lg border shadow-sm">
      <button
        type="button"
        className="w-full text-left p-4 hover:bg-gray-50"
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-start justify-between gap-4">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-1">
              <span className="text-sm font-medium text-gray-900">Turn {index + 1}</span>
              <span className="font-mono text-xs text-gray-400">{turn.trace_id.slice(0, 12)}</span>
              <span className="text-xs text-gray-500">{new Date(turn.started_at).toLocaleString()}</span>
              <span className="text-xs text-gray-400">{formatDuration(turn.started_at, turn.ended_at)}</span>
            </div>
            {turn.user_input && (
              <p className="text-sm text-gray-700 mb-1.5">{turn.user_input}</p>
            )}
            <div className="flex flex-wrap gap-1.5">
              {turn.model && <StatPill label="" value={turn.model} />}
              {turn.effort_level && <StatPill label="" value={turn.effort_level} />}
              <StatPill label="" value={`${formatTokens(totalTokens)} tok`} />
              <StatPill label="" value={`${turn.tool_call_count} tools`} />
              {turn.subagent_count > 0 && <StatPill label="" value={`${turn.subagent_count} subagents`} />}
              {turn.steering_count > 0 && <StatPill label="steers" value={turn.steering_count} warn />}
              {turn.denial_count > 0 && <StatPill label="denials" value={turn.denial_count} warn />}
              {turn.tool_error_count > 0 && <StatPill label="errors" value={turn.tool_error_count} warn />}
            </div>
          </div>
          <span className="text-xs text-gray-400 shrink-0">{expanded ? '▲' : '▼'}</span>
        </div>

        {feedback.length > 0 && (
          <div className="flex flex-wrap gap-1 mt-2">
            {feedback.map((fb) => (
              <ScoreBadge key={fb.id} scorer={fb.feedback_type} value={feedbackScore(fb)} />
            ))}
          </div>
        )}
      </button>

      {expanded && (
        <div className="border-t px-4 py-3 space-y-4">
          {turn.user_input && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-1">User Input</h4>
              <p className="text-sm text-gray-600 whitespace-pre-wrap bg-gray-50 rounded p-3">
                {turn.user_input}
              </p>
            </div>
          )}

          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-1">Span Details</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-2 text-xs">
              <div className="bg-gray-50 rounded p-2">
                <span className="text-gray-500">Input tokens</span>
                <p className="font-mono">{turn.input_tokens.toLocaleString()}</p>
              </div>
              <div className="bg-gray-50 rounded p-2">
                <span className="text-gray-500">Output tokens</span>
                <p className="font-mono">{turn.output_tokens.toLocaleString()}</p>
              </div>
              <div className="bg-gray-50 rounded p-2">
                <span className="text-gray-500">Cache read</span>
                <p className="font-mono">{turn.cache_read_tokens.toLocaleString()}</p>
              </div>
              <div className="bg-gray-50 rounded p-2">
                <span className="text-gray-500">Status</span>
                <p className="font-mono">{turn.status_code}</p>
              </div>
              {turn.config_version && (
                <div className="bg-gray-50 rounded p-2">
                  <span className="text-gray-500">Config</span>
                  <p className="font-mono">{turn.config_version.slice(0, 12)}</p>
                </div>
              )}
              {turn.git_branch && (
                <div className="bg-gray-50 rounded p-2">
                  <span className="text-gray-500">Branch</span>
                  <p className="font-mono">{turn.git_branch}</p>
                </div>
              )}
              {turn.session_id && (
                <div className="bg-gray-50 rounded p-2">
                  <span className="text-gray-500">Session ID</span>
                  <p className="font-mono">{turn.session_id.slice(0, 12)}</p>
                </div>
              )}
            </div>
          </div>

          {turn.chat_spans.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-1">
                Chat Spans ({turn.chat_spans.length})
              </h4>
              <div className="space-y-1">
                {turn.chat_spans.map((cs) => (
                  <div key={cs.span_id} className="flex items-center gap-3 text-xs bg-gray-50 rounded p-2">
                    <span className="font-mono text-gray-800">{cs.model}</span>
                    <span className="text-gray-500">
                      {cs.input_tokens.toLocaleString()} in / {cs.output_tokens.toLocaleString()} out
                    </span>
                    {cs.cache_read_tokens > 0 && (
                      <span className="text-gray-400">{cs.cache_read_tokens.toLocaleString()} cached</span>
                    )}
                    {cs.finish_reason && <span className="ml-auto text-gray-400">{cs.finish_reason}</span>}
                  </div>
                ))}
              </div>
            </div>
          )}

          {turn.tool_calls.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-1">
                Tool Calls ({turn.tool_calls.length})
              </h4>
              <div className="space-y-2">
                {turn.tool_calls.map((tc) => (
                  <div key={tc.span_id} className="text-xs bg-gray-50 rounded p-2 space-y-1">
                    <div className="flex items-center gap-2">
                      <span className="font-mono font-medium text-gray-800">{tc.tool_name}</span>
                      <span className={tc.status_code === 'OK' || tc.status_code === 'UNSET' ? 'text-green-600' : 'text-red-600'}>
                        {tc.status_code}
                      </span>
                      {tc.started_at && tc.ended_at && (
                        <span className="ml-auto text-gray-400">
                          {formatDuration(tc.started_at, tc.ended_at)}
                        </span>
                      )}
                    </div>
                    {tc.arguments && (
                      <div>
                        <span className="text-gray-400">args:</span>
                        <pre className="text-gray-600 font-mono whitespace-pre-wrap mt-0.5">{tc.arguments}</pre>
                      </div>
                    )}
                    {tc.result && (
                      <div>
                        <span className="text-gray-400">result:</span>
                        <pre className="text-gray-600 font-mono whitespace-pre-wrap mt-0.5">{tc.result}</pre>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {turn.subagents.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-1">
                Subagents ({turn.subagents.length})
              </h4>
              <div className="space-y-1">
                {turn.subagents.map((sa) => (
                  <div key={sa.span_id} className="flex items-center gap-3 text-xs bg-gray-50 rounded p-2">
                    <span className="font-mono text-gray-800">{sa.agent_type ?? 'unknown'}</span>
                    <span className="text-gray-500">{sa.tool_call_count} tool calls</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {feedback.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 mb-1">Feedback</h4>
              <div className="space-y-2">
                {feedback.map((fb) => (
                  <FeedbackDetail key={fb.id} fb={fb} rubrics={rubrics} />
                ))}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  )
}

export default function SessionDetail() {
  const { id } = useParams<{ id: string }>()

  const { data: session, isLoading, error } = useQuery({
    queryKey: ['session', id],
    queryFn: () => getSession(id!),
    enabled: !!id,
  })

  const { data: rubricsData } = useQuery({
    queryKey: ['rubrics'],
    queryFn: getRubrics,
  })
  const rubrics = rubricsData?.rubrics ?? []

  if (isLoading) {
    return (
      <div>
        <PageHeader title={`Session ${id?.slice(0, 12) ?? ''}`} />
        <p className="text-gray-500 text-sm">Loading...</p>
      </div>
    )
  }

  if (error) {
    return (
      <div>
        <PageHeader title={`Session ${id?.slice(0, 12) ?? ''}`} />
        <p className="text-red-600 text-sm">{(error as Error).message}</p>
      </div>
    )
  }

  if (!session) return null

  const firstTurn = session.turns[0]
  const lastTurn = session.turns[session.turns.length - 1]
  const startedAt = firstTurn?.started_at ?? null
  const endedAt = lastTurn?.ended_at ?? null

  return (
    <div>
      <PageHeader title={`Session ${id?.slice(0, 12) ?? ''}`} />

      <div className="flex flex-wrap gap-3 mb-6">
        <div className="rounded-lg bg-gray-100 px-3 py-2 text-sm">
          <span className="text-gray-500">Turns</span> <span>{session.turn_count}</span>
        </div>
        <div className="rounded-lg bg-gray-100 px-3 py-2 text-sm">
          <span className="text-gray-500">Duration</span>{' '}
          <span>{formatDuration(startedAt, endedAt)}</span>
        </div>
        <div className="rounded-lg bg-gray-100 px-3 py-2 text-sm">
          <span className="text-gray-500">Tokens</span>{' '}
          <span className="font-mono">{formatTokens(session.total_tokens)}</span>
        </div>
        {session.config_version && (
          <div className="rounded-lg bg-gray-100 px-3 py-2 text-sm">
            <span className="text-gray-500">Config</span>{' '}
            <span className="font-mono">{session.config_version.slice(0, 8)}</span>
          </div>
        )}
        {session.git_branch && (
          <div className="rounded-lg bg-gray-100 px-3 py-2 text-sm">
            <span className="text-gray-500">Branch</span>{' '}
            <span className="font-mono">{session.git_branch}</span>
          </div>
        )}
      </div>

      {session.session_feedback.length > 0 && (
        <div className="mb-6">
          <h2 className="text-lg font-semibold text-gray-900 mb-3">Session Scores</h2>
          <div className="space-y-2">
            {session.session_feedback.map((fb) => (
              <FeedbackDetail key={fb.id} fb={fb} rubrics={rubrics} />
            ))}
          </div>
        </div>
      )}

      <div>
        <h2 className="text-lg font-semibold text-gray-900 mb-3">Turns</h2>
        <div className="space-y-3">
          {session.turns.map((turn, i) => (
            <TurnCard
              key={turn.trace_id}
              turn={turn}
              index={i}
              feedback={session.turn_feedback[turn.trace_id] ?? []}
              rubrics={rubrics}
            />
          ))}
        </div>
      </div>
    </div>
  )
}
