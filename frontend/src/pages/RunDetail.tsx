import { useState, type ReactNode } from 'react'
import { useParams } from 'react-router-dom'
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query'
import PageHeader from '../components/PageHeader'
import ConfirmDialog from '../components/ConfirmDialog'
import DiffViewer from '../components/DiffViewer'
import {
  getRun,
  setRunSelection,
  advanceRun,
  applyRunReflection,
  getSessions,
  getModels,
  getRubrics,
} from '../api'
import type { DataSelection, Run } from '../types'

// ---------------------------------------------------------------------------
// Pipeline step bookkeeping
//
// `run.status` names the step currently in flight (or the terminal state).
// Each card below derives its own past/current/future state from that single
// value instead of tracking separate booleans per card.
// ---------------------------------------------------------------------------

type StepKey = 'selection' | 'scoring' | 'judging' | 'reflecting'
type StepState = 'past' | 'current' | 'future'

const STEP_ORDER: StepKey[] = ['selection', 'scoring', 'judging', 'reflecting']

function statusStepIndex(status: Run['status']): number {
  switch (status) {
    case 'created':
      return 0
    case 'scoring':
      return 1
    case 'judging':
      return 2
    case 'reflecting':
      return 3
    case 'complete':
      return 4
    default:
      return 4
  }
}

// FAILED loses track of which step was running (runs.py's state machine
// collapses it to one terminal value), so we infer it: the pipeline is
// linear and each step always writes its `*_result` before the next one can
// start, so the first missing result is the one that failed.
function failedStep(run: Run): StepKey {
  if (!run.scoring_result) return 'scoring'
  if (!run.judging_result) return 'judging'
  return 'reflecting'
}

function stepState(run: Run, step: StepKey): StepState {
  const stepIdx = STEP_ORDER.indexOf(step)
  if (run.status === 'failed') {
    const failedIdx = STEP_ORDER.indexOf(failedStep(run))
    if (stepIdx < failedIdx) return 'past'
    return stepIdx === failedIdx ? 'current' : 'future'
  }
  const curIdx = statusStepIndex(run.status)
  if (stepIdx < curIdx) return 'past'
  return stepIdx === curIdx ? 'current' : 'future'
}

function isRunInProgress(run: Run): boolean {
  return (
    (run.status === 'scoring' && !run.scoring_result) ||
    (run.status === 'judging' && !run.judging_result) ||
    (run.status === 'reflecting' && !run.reflecting_result)
  )
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

function formatDateTime(iso: string | null): string {
  if (!iso) return '—'
  return new Date(iso).toLocaleString()
}

function toDateInputValue(iso: string | null | undefined): string {
  if (!iso) return ''
  return iso.slice(0, 10)
}

function defaultSinceInput(): string {
  return new Date(Date.now() - 7 * 24 * 60 * 60 * 1000).toISOString().slice(0, 10)
}

function selectionSummary(sel: DataSelection | null): string {
  if (!sel) return 'no selection saved'
  const range = [sel.since, sel.until].filter(Boolean).join(' → ') || 'all time'
  const included = sel.session_ids.length
  const excluded = sel.excluded_session_ids.length
  const countLabel =
    included > 0 ? `${included} sessions` : excluded > 0 ? `all minus ${excluded}` : 'all sessions'
  return `${range} · ${countLabel}`
}

// ---------------------------------------------------------------------------
// Result shapes (Run's `*_progress`/`*_result` fields are typed as
// `Record<string, unknown>` in types.ts since they're opaque JSON blobs from
// the backend; these narrow them to what api.py actually writes).
// ---------------------------------------------------------------------------

interface ScoringProgress {
  total: number
  scored: number
  written: number
}

interface ScoringResult {
  turns_scored: number
  sessions_scored: number
  scores_written: number
  errors: number
  dry_run: boolean
}

interface JudgingProgress {
  total: number
  judged: number
  written: number
}

interface JudgingResult {
  turns_judged: number
  scores_written: number
  errors: number
  dry_run: boolean
}

interface ReflectingProgress {
  phase: string
  iterations: number
}

interface ReflectingProposal {
  diff: string
  rationale: string
  score_delta: number
  artifacts: { name: string; path: string; content: string }[]
  coaching_markdown: string
}

interface ReflectingNoProposal {
  proposal: null
  reason: string
  coaching_markdown?: string
  artifact_count?: number
  feedback_count?: number
  dry_run?: boolean
}

type ReflectingResult = ReflectingProposal | ReflectingNoProposal

function hasProposal(r: ReflectingResult): r is ReflectingProposal {
  return 'diff' in r
}

// ---------------------------------------------------------------------------
// Shared presentational pieces
// ---------------------------------------------------------------------------

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

function ProgressBar({ done, total, label }: { done: number; total: number; label: string }) {
  const pct = total > 0 ? Math.round((done / total) * 100) : 0
  return (
    <div>
      <div className="bg-gray-200 rounded-full h-2">
        <div
          className="bg-blue-500 rounded-full h-2 transition-all"
          style={{ width: `${pct}%` }}
        />
      </div>
      <div className="text-xs text-gray-500 mt-1">
        {label} {done} / {total} ({pct}%)
      </div>
    </div>
  )
}

function StepShell({
  title,
  state,
  expanded,
  onToggle,
  summary,
  children,
}: {
  title: string
  state: StepState
  expanded: boolean
  onToggle: () => void
  summary?: string
  children?: ReactNode
}) {
  if (state === 'future') {
    return (
      <div className="rounded-lg border p-4 opacity-50">
        <div className="flex items-center justify-between">
          <h3 className="text-base font-semibold text-gray-400">{title}</h3>
          <span className="text-xs text-gray-400">Not started</span>
        </div>
      </div>
    )
  }

  const showBody = state === 'current' || expanded

  return (
    <div
      className={`rounded-lg border p-4 ${state === 'current' ? 'border-blue-300 shadow-sm' : ''}`}
    >
      <button
        type="button"
        className="w-full flex items-center justify-between text-left"
        onClick={state === 'past' ? onToggle : undefined}
      >
        <h3 className="text-base font-semibold text-gray-900">{title}</h3>
        <div className="flex items-center gap-3">
          {state === 'past' && summary && (
            <span className="text-xs text-gray-500">{summary}</span>
          )}
          {state === 'past' && (
            <span className="text-xs text-gray-400">{expanded ? '▲' : '▼'}</span>
          )}
        </div>
      </button>
      {showBody && <div className="mt-3">{children}</div>}
    </div>
  )
}

function ErrorBox({ message }: { message: string }) {
  return (
    <div className="rounded-lg bg-red-50 border border-red-200 p-3 text-sm text-red-700">
      {message}
    </div>
  )
}

const inputClass =
  'block w-full rounded-lg border border-gray-300 px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500'
const primaryBtn =
  'px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm font-medium disabled:opacity-50'
const secondaryBtn =
  'px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 text-sm disabled:opacity-50'

// ---------------------------------------------------------------------------
// Data Selection card
// ---------------------------------------------------------------------------

function SelectionSection({
  run,
  state,
  expanded,
  onToggle,
  onSave,
  onStart,
  saving,
  saveError,
  starting,
  startError,
}: {
  run: Run
  state: StepState
  expanded: boolean
  onToggle: () => void
  onSave: (selection: DataSelection) => void
  onStart: () => void
  saving: boolean
  saveError: Error | null
  starting: boolean
  startError: Error | null
}) {
  const sel = run.data_selection
  const [since, setSince] = useState(toDateInputValue(sel?.since) || defaultSinceInput())
  const [until, setUntil] = useState(toDateInputValue(sel?.until))
  const [excluded, setExcluded] = useState<Set<string>>(new Set(sel?.excluded_session_ids ?? []))

  const sessionsQuery = useQuery({
    queryKey: ['sessions-for-run', since],
    queryFn: () => getSessions({ since: since || undefined, limit: 100 }),
    enabled: state === 'current',
  })

  const sessions = (sessionsQuery.data?.sessions ?? []).filter((s) => {
    if (!until || !s.started_at) return true
    return new Date(s.started_at).getTime() <= new Date(until).getTime()
  })

  function toggleExcluded(id: string) {
    setExcluded((prev) => {
      const next = new Set(prev)
      if (next.has(id)) next.delete(id)
      else next.add(id)
      return next
    })
  }

  return (
    <StepShell
      title="Data Selection"
      state={state}
      expanded={expanded}
      onToggle={onToggle}
      summary={selectionSummary(sel)}
    >
      {state === 'current' ? (
        <div>
          <div className="grid grid-cols-2 gap-4 max-w-lg">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Since</label>
              <input
                type="date"
                value={since}
                onChange={(e) => setSince(e.target.value)}
                className={inputClass}
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Until</label>
              <input
                type="date"
                value={until}
                onChange={(e) => setUntil(e.target.value)}
                className={inputClass}
              />
            </div>
          </div>

          <div className="mt-4">
            <h4 className="text-sm font-medium text-gray-700 mb-2">
              Sessions in range ({sessions.length})
            </h4>
            {sessionsQuery.isLoading && <p className="text-gray-500 text-sm">Loading...</p>}
            {sessionsQuery.error && (
              <p className="text-red-600 text-sm">{(sessionsQuery.error as Error).message}</p>
            )}
            {sessions.length > 0 && (
              <div className="space-y-1.5 max-h-96 overflow-y-auto">
                {sessions.map((s) => (
                  <label
                    key={s.conversation_id}
                    className="flex items-center gap-3 rounded-lg border p-2.5 text-sm hover:bg-gray-50"
                  >
                    <input
                      type="checkbox"
                      checked={!excluded.has(s.conversation_id)}
                      onChange={() => toggleExcluded(s.conversation_id)}
                    />
                    <div className="flex flex-wrap items-center gap-x-3 gap-y-1 text-xs text-gray-500">
                      <span className="font-mono text-gray-700">
                        {s.conversation_id.slice(0, 8)}
                      </span>
                      <span>
                        {s.turn_count} turn{s.turn_count !== 1 ? 's' : ''}
                      </span>
                      <span>{formatDateTime(s.started_at)}</span>
                      {s.config_version && (
                        <span className="font-mono bg-gray-100 px-1.5 py-0.5 rounded">
                          {s.config_version.slice(0, 8)}
                        </span>
                      )}
                    </div>
                  </label>
                ))}
              </div>
            )}
            {!sessionsQuery.isLoading && sessions.length === 0 && (
              <p className="text-gray-500 text-sm">No sessions found in this range.</p>
            )}
          </div>

          <div className="mt-4 flex items-center gap-3">
            <button
              type="button"
              className={secondaryBtn}
              disabled={saving}
              onClick={() =>
                onSave({
                  since: since || null,
                  until: until || null,
                  session_ids: [],
                  excluded_session_ids: Array.from(excluded),
                })
              }
            >
              {saving ? 'Saving...' : 'Save Selection'}
            </button>
            <button
              type="button"
              className={primaryBtn}
              disabled={starting || !run.data_selection}
              title={!run.data_selection ? 'Save a selection first' : undefined}
              onClick={onStart}
            >
              {starting ? 'Starting...' : 'Start Scoring'}
            </button>
          </div>
          {saveError && <p className="text-red-600 text-sm mt-2">{saveError.message}</p>}
          {startError && <p className="text-red-600 text-sm mt-2">{startError.message}</p>}
        </div>
      ) : (
        <div className="text-sm text-gray-600">
          Since {sel?.since ?? '—'}, until {sel?.until ?? '—'}.{' '}
          {sel && sel.session_ids.length > 0 && (
            <span>Included: {sel.session_ids.join(', ')}. </span>
          )}
          {sel && sel.excluded_session_ids.length > 0 && (
            <span>Excluded: {sel.excluded_session_ids.join(', ')}.</span>
          )}
        </div>
      )}
    </StepShell>
  )
}

// ---------------------------------------------------------------------------
// Scoring card
// ---------------------------------------------------------------------------

function ScoringSection({
  run,
  state,
  expanded,
  onToggle,
  judgeBackend,
  onJudgeBackendChange,
  onStartJudging,
  starting,
  startError,
}: {
  run: Run
  state: StepState
  expanded: boolean
  onToggle: () => void
  judgeBackend: string
  onJudgeBackendChange: (v: string) => void
  onStartJudging: (params: Record<string, unknown>) => void
  starting: boolean
  startError: Error | null
}) {
  const progress = run.scoring_progress as ScoringProgress | null
  const result = run.scoring_result as ScoringResult | null

  const [panelSize, setPanelSize] = useState(1)
  const [selectedRubrics, setSelectedRubrics] = useState<Set<string>>(new Set())
  const [dryRun, setDryRun] = useState(false)
  const [force, setForce] = useState(false)

  const modelsQuery = useQuery({
    queryKey: ['models'],
    queryFn: getModels,
    enabled: state === 'current' && !!result,
  })
  const rubricsQuery = useQuery({
    queryKey: ['rubrics'],
    queryFn: getRubrics,
    enabled: state === 'current' && !!result,
  })
  const rubrics = rubricsQuery.data?.rubrics ?? []

  function toggleRubric(name: string) {
    setSelectedRubrics((prev) => {
      const next = new Set(prev)
      if (next.has(name)) next.delete(name)
      else next.add(name)
      return next
    })
  }

  const summary = result
    ? `${result.turns_scored} turns, ${result.sessions_scored} sessions, ${result.scores_written} written${result.errors ? `, ${result.errors} errors` : ''}`
    : 'no result'

  return (
    <StepShell title="Scoring" state={state} expanded={expanded} onToggle={onToggle} summary={summary}>
      {state === 'current' && run.status === 'failed' && run.error && (
        <ErrorBox message={run.error} />
      )}

      {!result && run.status !== 'failed' && (
        <>
          {progress ? (
            <ProgressBar done={progress.scored} total={progress.total} label="Scored" />
          ) : (
            <p className="text-gray-500 text-sm">Starting...</p>
          )}
        </>
      )}

      {result && (
        <div className="text-sm text-gray-700 space-y-1">
          <div>Turns scored: {result.turns_scored}</div>
          <div>Sessions scored: {result.sessions_scored}</div>
          <div>Scores written: {result.scores_written}</div>
          {result.errors > 0 && <div className="text-red-600">Errors: {result.errors}</div>}
          {result.dry_run && <div className="text-amber-600">Dry run — nothing written</div>}
        </div>
      )}

      {state === 'current' && result && (
        <div className="mt-4 border-t pt-4">
          <h4 className="text-sm font-medium text-gray-700 mb-3">Judging configuration</h4>
          <div className="grid grid-cols-2 gap-4 max-w-lg">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">
                Judge Backend
              </label>
              <select
                value={judgeBackend}
                onChange={(e) => onJudgeBackendChange(e.target.value)}
                className={inputClass}
              >
                <option value="cli">cli</option>
                <option value="openai">openai</option>
                <option value="wandb">wandb</option>
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Panel Size</label>
              <input
                type="number"
                min={1}
                value={panelSize}
                onChange={(e) => setPanelSize(Number(e.target.value))}
                className={inputClass}
              />
            </div>
            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input type="checkbox" checked={dryRun} onChange={(e) => setDryRun(e.target.checked)} />
              Dry run
            </label>
            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input type="checkbox" checked={force} onChange={(e) => setForce(e.target.checked)} />
              Force
            </label>
          </div>

          {modelsQuery.data?.[judgeBackend] && (
            <div className="mt-3">
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
            <div className="mt-3">
              <label className="block text-sm font-medium text-gray-700 mb-2">Rubrics</label>
              <div className="grid grid-cols-2 gap-1">
                {rubrics.map((r) => (
                  <label key={r.name} className="flex items-center gap-2 text-sm text-gray-700">
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
              className={primaryBtn}
              disabled={starting}
              onClick={() =>
                onStartJudging({
                  judge_backend: judgeBackend,
                  panel_size: panelSize,
                  rubrics: selectedRubrics.size > 0 ? Array.from(selectedRubrics) : undefined,
                  dry_run: dryRun,
                  force,
                })
              }
            >
              {starting ? 'Starting...' : 'Start Judging'}
            </button>
            {startError && <p className="text-red-600 text-sm mt-2">{startError.message}</p>}
          </div>
        </div>
      )}
    </StepShell>
  )
}

// ---------------------------------------------------------------------------
// Judging card
// ---------------------------------------------------------------------------

function JudgingSection({
  run,
  state,
  expanded,
  onToggle,
  judgeBackend,
  onStartReflecting,
  starting,
  startError,
}: {
  run: Run
  state: StepState
  expanded: boolean
  onToggle: () => void
  judgeBackend: string
  onStartReflecting: (params: Record<string, unknown>) => void
  starting: boolean
  startError: Error | null
}) {
  const progress = run.judging_progress as JudgingProgress | null
  const result = run.judging_result as JudgingResult | null

  const [model, setModel] = useState('gpt-4o')
  const [iterations, setIterations] = useState(3)
  const [dryRun, setDryRun] = useState(false)

  const modelsQuery = useQuery({
    queryKey: ['models'],
    queryFn: getModels,
    enabled: state === 'current' && !!result,
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

  const summary = result
    ? `${result.turns_judged} turns, ${result.scores_written} written${result.errors ? `, ${result.errors} errors` : ''}`
    : 'no result'

  return (
    <StepShell title="Judging" state={state} expanded={expanded} onToggle={onToggle} summary={summary}>
      {state === 'current' && run.status === 'failed' && run.error && (
        <ErrorBox message={run.error} />
      )}

      {!result && run.status !== 'failed' && (
        <>
          {progress ? (
            <ProgressBar done={progress.judged} total={progress.total} label="Judged" />
          ) : (
            <p className="text-gray-500 text-sm">Starting...</p>
          )}
        </>
      )}

      {result && (
        <div className="text-sm text-gray-700 space-y-1">
          <div>Turns judged: {result.turns_judged}</div>
          <div>Scores written: {result.scores_written}</div>
          {result.errors > 0 && <div className="text-red-600">Errors: {result.errors}</div>}
          {result.dry_run && <div className="text-amber-600">Dry run — nothing written</div>}
        </div>
      )}

      {state === 'current' && result && (
        <div className="mt-4 border-t pt-4">
          <h4 className="text-sm font-medium text-gray-700 mb-3">Reflecting configuration</h4>
          <p className="text-xs text-gray-500 mb-3">
            Reusing <span className="font-mono">{judgeBackend}</span> as the judge backend
            (chosen on the Scoring step).
          </p>
          <div className="grid grid-cols-2 gap-4 max-w-lg">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Model</label>
              <select
                value={model}
                onChange={(e) => setModel(e.target.value)}
                className={inputClass}
              >
                {backendModels.length === 0 && <option value={model}>{model}</option>}
                {backendModels.map((m) => (
                  <option key={m} value={m}>
                    {m}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Iterations</label>
              <input
                type="number"
                min={1}
                value={iterations}
                onChange={(e) => setIterations(Number(e.target.value))}
                className={inputClass}
              />
            </div>
            <label className="flex items-center gap-2 text-sm text-gray-700">
              <input type="checkbox" checked={dryRun} onChange={(e) => setDryRun(e.target.checked)} />
              Dry run
            </label>
          </div>

          <div className="mt-4">
            <button
              type="button"
              className={primaryBtn}
              disabled={starting}
              onClick={() =>
                onStartReflecting({
                  model,
                  iterations,
                  judge_backend: judgeBackend,
                  dry_run: dryRun,
                })
              }
            >
              {starting ? 'Starting...' : 'Start Reflecting'}
            </button>
            {startError && <p className="text-red-600 text-sm mt-2">{startError.message}</p>}
          </div>
        </div>
      )}
    </StepShell>
  )
}

// ---------------------------------------------------------------------------
// Reflecting card
// ---------------------------------------------------------------------------

function ReflectingSection({
  run,
  state,
  expanded,
  onToggle,
  onComplete,
  completing,
  completeError,
}: {
  run: Run
  state: StepState
  expanded: boolean
  onToggle: () => void
  onComplete: () => void
  completing: boolean
  completeError: Error | null
}) {
  const [confirmOpen, setConfirmOpen] = useState(false)
  const progress = run.reflecting_progress as ReflectingProgress | null
  const result = run.reflecting_result as ReflectingResult | null

  const applyMutation = useMutation({
    mutationFn: () => applyRunReflection(run.run_id),
  })

  const summary = result
    ? hasProposal(result)
      ? `score Δ ${result.score_delta >= 0 ? '+' : ''}${result.score_delta.toFixed(3)}`
      : result.reason
    : 'no result'

  return (
    <StepShell
      title="Reflecting"
      state={state}
      expanded={expanded}
      onToggle={onToggle}
      summary={summary}
    >
      {state === 'current' && run.status === 'failed' && run.error && (
        <ErrorBox message={run.error} />
      )}

      {!result && run.status !== 'failed' && (
        <p className="text-gray-500 text-sm">
          {progress ? `Reflecting (${progress.phase})...` : 'Starting...'}
        </p>
      )}

      {result && !hasProposal(result) && (
        <p className="text-gray-600 text-sm">{result.reason}</p>
      )}

      {result && hasProposal(result) && (
        <div className="space-y-3">
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-1">Rationale</h4>
            <p className="text-sm text-gray-600">{result.rationale}</p>
          </div>
          <div className="text-sm">
            <span className="text-gray-500">Score delta: </span>
            <span
              className={`font-mono font-medium ${result.score_delta >= 0 ? 'text-green-600' : 'text-red-600'}`}
            >
              {result.score_delta >= 0 ? '+' : ''}
              {result.score_delta.toFixed(3)}
            </span>
          </div>
          <div>
            <h4 className="text-sm font-medium text-gray-700 mb-1">Diff</h4>
            <DiffViewer diff={result.diff} />
          </div>
        </div>
      )}

      {state === 'current' && result && (
        <div className="mt-4 border-t pt-4 flex items-center gap-3">
          {hasProposal(result) && (
            <button
              type="button"
              className={secondaryBtn}
              disabled={applyMutation.isPending || applyMutation.isSuccess}
              onClick={() => setConfirmOpen(true)}
            >
              {applyMutation.isPending
                ? 'Applying...'
                : applyMutation.isSuccess
                  ? 'Applied'
                  : 'Apply Changes'}
            </button>
          )}
          <button type="button" className={primaryBtn} disabled={completing} onClick={onComplete}>
            {completing ? 'Completing...' : 'Complete'}
          </button>
        </div>
      )}
      {applyMutation.error && (
        <p className="text-red-600 text-sm mt-2">{(applyMutation.error as Error).message}</p>
      )}
      {applyMutation.isSuccess && (
        <p className="text-green-600 text-sm mt-2">Changes applied.</p>
      )}
      {completeError && <p className="text-red-600 text-sm mt-2">{completeError.message}</p>}

      <ConfirmDialog
        open={confirmOpen}
        title="Apply reflected changes"
        message="Apply the reflected changes to your config artifacts? This will overwrite the current versions."
        onConfirm={() => {
          setConfirmOpen(false)
          applyMutation.mutate()
        }}
        onCancel={() => setConfirmOpen(false)}
      />
    </StepShell>
  )
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function RunDetail() {
  const { runId } = useParams<{ runId: string }>()
  const queryClient = useQueryClient()
  const [expandedSteps, setExpandedSteps] = useState<Partial<Record<StepKey, boolean>>>({})
  const [judgeBackend, setJudgeBackend] = useState('cli')

  const runQuery = useQuery({
    queryKey: ['run', runId],
    queryFn: () => getRun(runId!),
    enabled: !!runId,
    refetchInterval: (query) => {
      const run = query.state.data
      return run && isRunInProgress(run) ? 2000 : false
    },
  })

  const run = runQuery.data

  const selectionMutation = useMutation({
    mutationFn: (selection: DataSelection) => setRunSelection(runId!, selection),
    onSuccess: (data) => queryClient.setQueryData(['run', runId], data),
  })

  const advanceMutation = useMutation({
    mutationFn: (params: Record<string, unknown>) => advanceRun(runId!, params),
    onSuccess: (data) => queryClient.setQueryData(['run', runId], data),
  })

  function isExpanded(step: StepKey, state: StepState): boolean {
    if (step in expandedSteps) return !!expandedSteps[step]
    // Auto-expand the step that most recently produced the run's final
    // outcome, so landing on a finished run shows something useful by
    // default instead of four collapsed rows.
    if (run?.status === 'complete' && step === 'reflecting') return true
    if (run?.status === 'failed' && state === 'current') return true
    return false
  }

  function toggleExpanded(step: StepKey, state: StepState) {
    setExpandedSteps((prev) => ({ ...prev, [step]: !isExpanded(step, state) }))
  }

  if (!runId) return null

  if (runQuery.isLoading) {
    return (
      <div>
        <PageHeader title="Run" />
        <p className="text-gray-500 text-sm">Loading...</p>
      </div>
    )
  }

  if (runQuery.error) {
    return (
      <div>
        <PageHeader title="Run" />
        <p className="text-red-600 text-sm">{(runQuery.error as Error).message}</p>
      </div>
    )
  }

  if (!run) return null

  const selectionState = stepState(run, 'selection')
  const scoringState = stepState(run, 'scoring')
  const judgingState = stepState(run, 'judging')
  const reflectingState = stepState(run, 'reflecting')

  return (
    <div>
      <PageHeader title={run.run_id}>
        <StatusBadge status={run.status} />
      </PageHeader>

      <div className="text-sm text-gray-500 mb-4">
        Created {formatDateTime(run.created_at)}
      </div>

      {run.status === 'failed' && run.error && (
        <div className="mb-4">
          <ErrorBox message={`Run failed: ${run.error}`} />
        </div>
      )}

      <div className="space-y-4">
        <SelectionSection
          run={run}
          state={selectionState}
          expanded={isExpanded('selection', selectionState)}
          onToggle={() => toggleExpanded('selection', selectionState)}
          onSave={(selection) => selectionMutation.mutate(selection)}
          onStart={() => advanceMutation.mutate({})}
          saving={selectionMutation.isPending}
          saveError={selectionMutation.error as Error | null}
          starting={advanceMutation.isPending}
          startError={run.status === 'created' ? (advanceMutation.error as Error | null) : null}
        />

        <ScoringSection
          run={run}
          state={scoringState}
          expanded={isExpanded('scoring', scoringState)}
          onToggle={() => toggleExpanded('scoring', scoringState)}
          judgeBackend={judgeBackend}
          onJudgeBackendChange={setJudgeBackend}
          onStartJudging={(params) => advanceMutation.mutate(params)}
          starting={advanceMutation.isPending}
          startError={run.status === 'scoring' ? (advanceMutation.error as Error | null) : null}
        />

        <JudgingSection
          run={run}
          state={judgingState}
          expanded={isExpanded('judging', judgingState)}
          onToggle={() => toggleExpanded('judging', judgingState)}
          judgeBackend={judgeBackend}
          onStartReflecting={(params) => advanceMutation.mutate(params)}
          starting={advanceMutation.isPending}
          startError={run.status === 'judging' ? (advanceMutation.error as Error | null) : null}
        />

        <ReflectingSection
          run={run}
          state={reflectingState}
          expanded={isExpanded('reflecting', reflectingState)}
          onToggle={() => toggleExpanded('reflecting', reflectingState)}
          onComplete={() => advanceMutation.mutate({})}
          completing={advanceMutation.isPending}
          completeError={run.status === 'reflecting' ? (advanceMutation.error as Error | null) : null}
        />
      </div>
    </div>
  )
}
