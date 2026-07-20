import { useState } from 'react'
import { useQueries, useQuery } from '@tanstack/react-query'
import { getSession, getSessions } from '../../api'
import type { DataSelection, ModelCatalog, RubricCatalog, RunConfig } from '../../types'
import RunConfiguration from './RunConfiguration'
import RunSelection from './RunSelection'
import {
  assessRunConfig,
  initializeRunConfigState,
  toRunConfig,
  transitionRunConfig,
  type RunConfigAction,
} from './runConfigState'

const primaryButton =
  'rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50'

function dateInputValue(value: string | null | undefined, timezone?: string | null): string {
  if (!value) return ''
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return ''
  try {
    const parts = new Intl.DateTimeFormat('en-US', {
      timeZone: timezone || undefined,
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
    }).formatToParts(date)
    const values = Object.fromEntries(parts.map((part) => [part.type, part.value]))
    return `${values.year}-${values.month}-${values.day}`
  } catch {
    return value.slice(0, 10)
  }
}

function defaultSince(): string {
  const date = new Date(Date.now() - 7 * 24 * 60 * 60 * 1000)
  date.setMinutes(date.getMinutes() - date.getTimezoneOffset())
  return date.toISOString().slice(0, 10)
}

function dayBoundary(value: string, end: boolean, timezone?: string): string | null {
  if (!value) return null
  const [year, month, day] = value.split('-').map(Number)
  if (![year, month, day].every(Number.isFinite)) return null
  const hour = end ? 23 : 0
  const minute = end ? 59 : 0
  const second = end ? 59 : 0
  const millisecond = end ? 999 : 0
  const serialize = (date: Date) =>
    end
      ? date.toISOString().replace(/\.999Z$/, '.999999Z')
      : date.toISOString()

  if (!timezone) {
    return serialize(new Date(year, month - 1, day, hour, minute, second, millisecond))
  }
  try {
    const formatter = new Intl.DateTimeFormat('en-US', {
      timeZone: timezone,
      hourCycle: 'h23',
      year: 'numeric',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
    const desired = Date.UTC(year, month - 1, day, hour, minute, second)
    let instant = desired
    for (let iteration = 0; iteration < 2; iteration += 1) {
      const parts = Object.fromEntries(
        formatter.formatToParts(new Date(instant)).map((part) => [part.type, part.value]),
      )
      const represented = Date.UTC(
        Number(parts.year),
        Number(parts.month) - 1,
        Number(parts.day),
        Number(parts.hour),
        Number(parts.minute),
        Number(parts.second),
      )
      instant += desired - represented
    }
    return serialize(new Date(instant + millisecond))
  } catch {
    return serialize(new Date(year, month - 1, day, hour, minute, second, millisecond))
  }
}

function ErrorNotice({ message }: { message: string }) {
  return (
    <div role="alert" className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
      {message}
    </div>
  )
}

export default function RunSetup({
  initialSelection,
  initialConfig,
  initialAutoRun,
  models,
  rubrics,
  pending,
  onStart,
}: {
  initialSelection: DataSelection | null
  initialConfig: RunConfig | null
  initialAutoRun: boolean
  models: ModelCatalog
  rubrics: RubricCatalog
  pending: boolean
  onStart: (selection: DataSelection, config: RunConfig, autoRun: boolean) => void
}) {
  const timezone = initialSelection?.timezone ?? Intl.DateTimeFormat().resolvedOptions().timeZone
  const [since, setSince] = useState(
    () => dateInputValue(initialSelection?.since, timezone) || defaultSince(),
  )
  const [until, setUntil] = useState(() => dateInputValue(initialSelection?.until, timezone))
  const [selectedIds, setSelectedIds] = useState<string[]>(
    () => initialSelection?.session_ids ?? [],
  )
  const [autoRun, setAutoRunChoice] = useState(initialAutoRun)
  const [config, setConfig] = useState(() =>
    initializeRunConfigState(models, rubrics, initialConfig),
  )
  const sinceInstant = dayBoundary(since, false, timezone) ?? undefined
  const untilInstant = dayBoundary(until, true, timezone) ?? undefined
  const sessionsQuery = useQuery({
    queryKey: ['sessions-for-run', sinceInstant, untilInstant, timezone],
    queryFn: () => getSessions({ since: sinceInstant, until: untilInstant, timezone }),
  })
  const sessions = sessionsQuery.data?.sessions ?? []
  const selectedSessions = sessions.filter((session) =>
    selectedIds.includes(session.conversation_id))
  const selectedDetailQueries = useQueries({
    queries: selectedIds.map((sessionId) => ({
      queryKey: ['session-capacity', sessionId],
      queryFn: () => getSession(sessionId),
    })),
  })
  const capacitySessions = selectedDetailQueries.flatMap((query) =>
    query.data ? [{ judging_token_estimates: query.data.judging_token_estimates }] : [])
  const capacityLoading = selectedDetailQueries.some((query) => query.isPending)
  const capacityError = selectedDetailQueries.find((query) => query.error)?.error
  const assessment = assessRunConfig(config, models, rubrics, capacitySessions)
  const selectionError =
    selectedIds.length === 0
      ? 'Select at least one session.'
      : selectedSessions.length !== selectedIds.length
        ? 'Load every selected session before starting.'
        : capacityError instanceof Error
          ? `Could not load selected session evidence: ${capacityError.message}`
          : null
  const cannotStart =
    pending || sessionsQuery.isLoading || Boolean(sessionsQuery.error) ||
    capacityLoading || Boolean(selectionError) || assessment.errors.length > 0

  function changeDate(setter: (value: string) => void, value: string) {
    setter(value)
    setSelectedIds([])
  }

  function dispatch(action: RunConfigAction) {
    setConfig((current) => transitionRunConfig(current, action, models, capacitySessions))
  }

  function start() {
    if (cannotStart) return
    onStart(
      {
        since: sinceInstant ?? null,
        until: untilInstant ?? null,
        timezone,
        session_ids: selectedIds,
      },
      toRunConfig(config, models, rubrics, capacitySessions),
      autoRun,
    )
  }

  return (
    <div className="space-y-6">
      <RunSelection
        since={since}
        until={until}
        sessions={sessions}
        selectedSessionIds={selectedIds}
        totalSessions={sessionsQuery.data?.total ?? sessions.length}
        truncated={Boolean(sessionsQuery.data?.truncated)}
        loading={sessionsQuery.isLoading}
        error={sessionsQuery.error instanceof Error ? sessionsQuery.error.message : null}
        disabled={pending}
        onSinceChange={(value) => changeDate(setSince, value)}
        onUntilChange={(value) => changeDate(setUntil, value)}
        onSessionIdsChange={setSelectedIds}
        onRetry={() => void sessionsQuery.refetch()}
      />
      <RunConfiguration
        state={config}
        models={models}
        rubrics={rubrics}
        sessions={capacitySessions}
        disabled={pending}
        onAction={dispatch}
      />
      {selectionError && <ErrorNotice message={selectionError} />}
      <div className="flex flex-wrap items-center gap-4 border-t border-gray-200 pt-4">
        <label className="flex items-center gap-2 text-sm text-gray-700">
          <input
            type="checkbox"
            checked={autoRun}
            disabled={pending}
            onChange={(event) => setAutoRunChoice(event.target.checked)}
          />
          Continue automatically
        </label>
        <button type="button" className={primaryButton} disabled={cannotStart} onClick={start}>
          {pending ? 'Starting…' : 'Start Scoring'}
        </button>
      </div>
    </div>
  )
}
