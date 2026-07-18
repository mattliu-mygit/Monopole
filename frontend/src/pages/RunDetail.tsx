import { useCallback, useEffect, useState, type ReactNode } from 'react'
import { useMutation, useQueries, useQuery, useQueryClient } from '@tanstack/react-query'
import { useBeforeUnload, useBlocker, useParams } from 'react-router-dom'
import {
  advanceRun,
  ApiError,
  cancelRun,
  dismissRunReflection,
  getModels,
  getRubrics,
  getRun,
  getSession,
  getSessions,
  promoteRunReflection,
  resetReflectionDraft,
  saveReflectionDraft,
  setAutoRun,
  setReflectionSelection,
  setRunConfig,
  setRunSelection,
} from '../api'
import Dialog from '../components/Dialog'
import PageHeader from '../components/PageHeader'
import ReflectionActivity from '../features/reflection/ReflectionActivity'
import ReflectionReview from '../features/reflection/ReflectionReview'
import RunStatusBadge from '../features/runs/RunStatusBadge'
import JudgingProgress from '../features/runs/JudgingProgress'
import RunConfigAudit from '../features/runs/RunConfigAudit'
import RunConfiguration from '../features/runs/RunConfiguration'
import PinnedSelectionAudit from '../features/runs/PinnedSelectionAudit'
import RunSelection from '../features/runs/RunSelection'
import ScoringProgress from '../features/runs/ScoringProgress'
import {
  assessRunConfig,
  initializeRunConfigState,
  toRunConfig,
  transitionRunConfig,
  type RunConfigAction,
} from '../features/runs/runConfigState'
import type {
  DataSelection,
  ModelCatalog,
  ReflectionBundleSnapshot,
  RubricCatalog,
  Run,
  RunConfig,
} from '../types'
import { shouldPollRun } from '../features/runs/runPolling'

const primaryButton =
  'rounded-lg bg-blue-600 px-4 py-2 text-sm font-semibold text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-50'
const dangerButton =
  'rounded-lg bg-red-600 px-4 py-2 text-sm font-semibold text-white hover:bg-red-700 disabled:cursor-not-allowed disabled:opacity-50'

function formatDateTime(value: string): string {
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? value : date.toLocaleString()
}

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

function StageCard({
  title,
  children,
  expandedByDefault,
}: {
  title: string
  children: ReactNode
  expandedByDefault: boolean
}) {
  const slug = title.toLowerCase().replaceAll(/[^a-z0-9]+/g, '-')
  const titleId = `run-stage-${slug}`
  const panelId = `${titleId}-content`
  const [expanded, setExpanded] = useState(expandedByDefault)

  useEffect(() => setExpanded(expandedByDefault), [expandedByDefault])

  return (
    <section aria-labelledby={titleId} className="rounded-xl border border-gray-200 bg-white p-5">
      <h2 id={titleId} className="text-base font-semibold text-gray-900">
        <button
          type="button"
          aria-label={title}
          aria-expanded={expanded}
          aria-controls={panelId}
          className="flex w-full items-center justify-between gap-3 text-left"
          onClick={() => setExpanded((current) => !current)}
        >
          <span>{title}</span>
          <span aria-hidden="true" className="text-sm text-gray-400">
            {expanded ? '−' : '+'}
          </span>
        </button>
      </h2>
      <div id={panelId} hidden={!expanded} className="mt-4">
        {children}
      </div>
    </section>
  )
}

function ErrorNotice({ message }: { message: string }) {
  return (
    <div role="alert" className="rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-700">
      {message}
    </div>
  )
}

function CreatedRunSetup({
  run,
  models,
  rubrics,
  pending,
  onStart,
}: {
  run: Run
  models: ModelCatalog
  rubrics: RubricCatalog
  pending: boolean
  onStart: (selection: DataSelection, config: RunConfig, autoRun: boolean) => void
}) {
  const timezone = run.data_selection?.timezone ?? Intl.DateTimeFormat().resolvedOptions().timeZone
  const [since, setSince] = useState(
    () => dateInputValue(run.data_selection?.since, timezone) || defaultSince(),
  )
  const [until, setUntil] = useState(() => dateInputValue(run.data_selection?.until, timezone))
  const [selectedIds, setSelectedIds] = useState<string[]>(
    () => run.data_selection?.session_ids ?? [],
  )
  const [autoRun, setAutoRunChoice] = useState(run.auto_run)
  const [config, setConfig] = useState(() =>
    initializeRunConfigState(models, rubrics, run.run_config),
  )
  const sinceInstant = dayBoundary(since, false, timezone) ?? undefined
  const untilInstant = dayBoundary(until, true, timezone) ?? undefined
  const sessionsQuery = useQuery({
    queryKey: ['sessions-for-run', sinceInstant, untilInstant, timezone],
    queryFn: () =>
      getSessions({ since: sinceInstant, until: untilInstant, timezone }),
  })
  const sessions = sessionsQuery.data?.sessions ?? []
  const selectedSessionIds = selectedIds
  const selectedSessions = sessions.filter((session) =>
    selectedSessionIds.includes(session.conversation_id))
  const selectedDetailQueries = useQueries({
    queries: selectedSessionIds.map((sessionId) => ({
      queryKey: ['session-capacity', sessionId],
      queryFn: () => getSession(sessionId),
    })),
  })
  const capacitySessions = selectedDetailQueries.flatMap((query) =>
    query.data ? [{ judging_token_estimates: query.data.judging_token_estimates }] : [])
  const capacityLoading = selectedDetailQueries.some((query) => query.isPending)
  const capacityError = selectedDetailQueries.find((query) => query.error)?.error
  const truncated = Boolean(sessionsQuery.data?.truncated)
  const assessment = assessRunConfig(config, models, rubrics, capacitySessions)
  const selectionError =
    selectedSessionIds.length === 0
      ? 'Select at least one session.'
      : selectedSessions.length !== selectedSessionIds.length
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
    setConfig((current) => transitionRunConfig(
      current,
      action,
      models,
      capacitySessions,
    ))
  }

  function start() {
    if (cannotStart) return
    onStart(
      {
        since: sinceInstant ?? null,
        until: untilInstant ?? null,
        timezone,
        session_ids: selectedSessionIds,
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
        selectedSessionIds={selectedSessionIds}
        totalSessions={sessionsQuery.data?.total ?? sessions.length}
        truncated={truncated}
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

function isCancellable(run: Run): boolean {
  if (!['created', 'scoring', 'judging', 'reflecting'].includes(run.status)) return false
  return !(
    run.status === 'reflecting' &&
    (run.reflecting_result !== null || run.reflection_review !== null)
  )
}

export default function RunDetail() {
  const { runId } = useParams<{ runId: string }>()
  const queryClient = useQueryClient()
  const [cancelDialogOpen, setCancelDialogOpen] = useState(false)
  const [actionError, setActionError] = useState<string | null>(null)
  const [reviewDirty, setReviewDirty] = useState(false)
  const blocker = useBlocker(useCallback(
    ({ currentLocation, nextLocation }) =>
      reviewDirty && currentLocation.pathname !== nextLocation.pathname,
    [reviewDirty],
  ))
  useBeforeUnload(useCallback((event) => {
    if (!reviewDirty) return
    event.preventDefault()
    event.returnValue = ''
  }, [reviewDirty]), { capture: true })
  const runQuery = useQuery({
    queryKey: ['run', runId],
    queryFn: () => getRun(runId!),
    enabled: Boolean(runId),
    refetchInterval: (query) => {
      const current = query.state.data
      return current && shouldPollRun(current) ? 2_000 : false
    },
  })
  const run = runQuery.data
  const modelsQuery = useQuery({
    queryKey: ['models'],
    queryFn: getModels,
    enabled: run?.status === 'created',
  })
  const rubricsQuery = useQuery({
    queryKey: ['rubrics'],
    queryFn: getRubrics,
    enabled: run?.status === 'created',
  })

  function updateRun(next: Run) {
    queryClient.setQueryData(['run', runId], next)
    void queryClient.invalidateQueries({ queryKey: ['runs'] })
  }

  function reconcileError(error: unknown) {
    setActionError(error instanceof Error ? error.message : String(error))
    if (run && error instanceof ApiError && error.status === 409) {
      const detail = error.detail
      if (detail && typeof detail === 'object') {
        const conflict = detail as {
          code?: string
          current_revision?: number
          current_review?: Run['reflection_review']
          changed_targets?: string[]
          current?: ReflectionBundleSnapshot | null
          message?: string
        }
        if (
          typeof conflict.current_revision === 'number' &&
          conflict.current_review && typeof conflict.current_review === 'object'
        ) {
          queryClient.setQueryData<Run>(['run', runId], {
            ...run,
            reflection_review: conflict.current_review,
            reflection_review_revision: conflict.current_revision,
          })
        } else if (conflict.code === 'baseline_stale' && run.reflection_review) {
          queryClient.setQueryData<Run>(['run', runId], {
            ...run,
            reflection_review: {
              ...run.reflection_review,
              stale: true,
              changed_targets: conflict.changed_targets ?? [],
              current: conflict.current ?? null,
              stale_reason: conflict.message ?? 'baseline_changed',
            },
          })
        }
      }
    }
    void queryClient.invalidateQueries({ queryKey: ['run', runId] })
  }

  const startMutation = useMutation({
    mutationFn: async ({
      selection,
      config,
      autoRun,
    }: {
      selection: DataSelection
      config: RunConfig
      autoRun: boolean
    }) => {
      await setRunSelection(runId!, selection)
      await setRunConfig(runId!, config)
      await setAutoRun(runId!, autoRun)
      return advanceRun(runId!)
    },
    onMutate: () => setActionError(null),
    onSuccess: updateRun,
    onError: reconcileError,
  })
  const advanceMutation = useMutation({
    mutationFn: () => advanceRun(runId!),
    onMutate: () => setActionError(null),
    onSuccess: updateRun,
    onError: reconcileError,
  })
  const cancelMutation = useMutation({
    mutationFn: () => cancelRun(runId!),
    onMutate: () => setActionError(null),
    onSuccess: (next) => {
      updateRun(next)
      setCancelDialogOpen(false)
    },
    onError: reconcileError,
  })

  if (!runId) return null
  if (runQuery.isLoading) {
    return <><PageHeader title="Run" /><p role="status">Loading run…</p></>
  }
  if (runQuery.error && !run) {
    return (
      <>
        <PageHeader title="Run" />
        <ErrorNotice message={(runQuery.error as Error).message} />
        <button type="button" className="mt-3 text-sm font-medium text-blue-700" onClick={() => void runQuery.refetch()}>
          Retry
        </button>
      </>
    )
  }
  if (!run) return null

  const reviewError = (error: unknown): never => {
    reconcileError(error)
    throw error
  }
  const selectCandidate = async (candidateId: string, discardDraft: boolean) => {
    try {
      updateRun(await setReflectionSelection(
        run.run_id, candidateId, run.reflection_review_revision, discardDraft,
      ))
    } catch (error) { reviewError(error) }
  }
  const saveDraft = async (contents: Record<string, string | null>) => {
    try {
      updateRun(await saveReflectionDraft(
        run.run_id,
        contents,
        run.reflection_review_revision,
        run.reflection_review?.draft?.revision ?? null,
      ))
    } catch (error) { reviewError(error) }
  }
  const resetDraft = async () => {
    try {
      updateRun(await resetReflectionDraft(
        run.run_id,
        run.reflection_review_revision,
        run.reflection_review?.draft?.revision ?? null,
      ))
    } catch (error) { reviewError(error) }
  }
  const promote = async (acknowledgeUnevaluated: boolean) => {
    try {
      updateRun(await promoteRunReflection(run.run_id, {
        expectedRevision: run.reflection_review_revision,
        expectedDraftRevision: run.reflection_review?.draft?.revision ?? null,
        acknowledgeUnevaluated,
        idempotencyKey: [
          run.run_id,
          run.reflection_review_revision,
          run.reflection_review?.draft?.revision ?? 'evaluated',
        ].join(':'),
      }))
    } catch (error) { reviewError(error) }
  }
  const dismiss = async () => {
    try {
      updateRun(await dismissRunReflection(run.run_id, run.reflection_review_revision))
    } catch (error) { reviewError(error) }
  }

  const showJudging = Boolean(
    run.judging_plan || run.judging_progress || run.judging_result ||
    ['judging', 'reflecting', 'complete'].includes(run.status),
  )
  const showReflecting = Boolean(
    run.reflecting_progress || run.reflecting_result || run.reflection_review ||
    ['reflecting', 'complete'].includes(run.status),
  )

  return (
    <div>
      <PageHeader title={run.run_id}>
        <RunStatusBadge status={run.status} />
        {isCancellable(run) && (
          <button type="button" className={dangerButton} disabled={cancelMutation.isPending} onClick={() => setCancelDialogOpen(true)}>
            Cancel Run
          </button>
        )}
      </PageHeader>

      {cancelDialogOpen && (
        <Dialog title="Cancel this run?" closeLabel="Keep running" busy={cancelMutation.isPending} onClose={() => setCancelDialogOpen(false)}>
          <p className="mt-2 text-sm text-gray-600">Completed evidence remains available.</p>
          <button type="button" className={`${dangerButton} mt-4`} disabled={cancelMutation.isPending} onClick={() => cancelMutation.mutate()}>
            {cancelMutation.isPending ? 'Cancelling…' : 'Cancel run'}
          </button>
        </Dialog>
      )}

      {blocker.state === 'blocked' && (
        <Dialog
          title="Leave without saving edited D?"
          closeLabel="Keep editing"
          onClose={() => blocker.reset()}
        >
          <p className="mt-2 text-sm text-gray-600">
            Your inline edits have not been saved. Leaving this page will discard them.
          </p>
          <button
            type="button"
            className={`${dangerButton} mt-4`}
            onClick={() => blocker.proceed()}
          >
            Leave without saving
          </button>
        </Dialog>
      )}

      <p className="mb-4 text-sm text-gray-500">Created {formatDateTime(run.created_at)}</p>
      {runQuery.error && (
        <div className="mb-4 space-y-2">
          <ErrorNotice message={`Could not refresh run: ${(runQuery.error as Error).message}`} />
          <button
            type="button"
            className="text-sm font-semibold text-blue-700 hover:underline"
            onClick={() => void runQuery.refetch()}
          >
            Retry run refresh
          </button>
        </div>
      )}
      {run.status === 'failed' && run.error && <ErrorNotice message={`Run failed: ${run.error}`} />}
      {run.status === 'cancelled' && (
        <div className="rounded-lg border border-yellow-200 bg-yellow-50 p-3 text-sm text-yellow-800">Run was cancelled.</div>
      )}
      {actionError && <div className="mt-4"><ErrorNotice message={actionError} /></div>}

      <div className="mt-4 space-y-4">
        <StageCard
          title="Data selection & configuration"
          expandedByDefault={run.status === 'created'}
        >
          {run.status === 'created' ? (
            modelsQuery.data && rubricsQuery.data ? (
              <CreatedRunSetup
                key={run.run_id}
                run={run}
                models={modelsQuery.data}
                rubrics={rubricsQuery.data}
                pending={startMutation.isPending}
                onStart={(selection, config, autoRun) =>
                  startMutation.mutate({ selection, config, autoRun })}
              />
            ) : modelsQuery.error || rubricsQuery.error ? (
              <div>
                <ErrorNotice message={((modelsQuery.error ?? rubricsQuery.error) as Error).message} />
                <button
                  type="button"
                  className="mt-3 text-sm font-semibold text-blue-700 hover:underline"
                  onClick={() => {
                    void modelsQuery.refetch()
                    void rubricsQuery.refetch()
                  }}
                >
                  Retry configuration catalogs
                </button>
              </div>
            ) : (
              <p role="status" className="text-sm text-gray-500">Loading configuration catalogs…</p>
            )
          ) : (
            <div className="space-y-4">
              <PinnedSelectionAudit selection={run.data_selection} cohort={run.turn_cohort} />
              {run.effective_config ? (
                <RunConfigAudit config={run.effective_config} />
              ) : (
                <p className="text-sm text-gray-500">Pinned configuration unavailable.</p>
              )}
            </div>
          )}
        </StageCard>

        {run.status !== 'created' && (
          <StageCard title="Scoring" expandedByDefault={run.status === 'scoring'}>
            <ScoringProgress progress={run.scoring_progress} result={run.scoring_result} />
            {run.status === 'scoring' && run.current_stage_succeeded && run.scoring_result && !run.auto_run && (
              <button type="button" className={`${primaryButton} mt-4`} disabled={advanceMutation.isPending} onClick={() => advanceMutation.mutate()}>
                {advanceMutation.isPending ? 'Starting…' : 'Continue to judging'}
              </button>
            )}
          </StageCard>
        )}

        {showJudging && (
          <StageCard title="Judging" expandedByDefault={run.status === 'judging'}>
            <JudgingProgress
              plan={run.judging_plan ?? null}
              progress={run.judging_progress}
              result={run.judging_result}
            />
            {run.status === 'judging' && run.current_stage_succeeded && run.judging_result?.coverage_complete && !run.auto_run && (
              <button type="button" className={`${primaryButton} mt-4`} disabled={advanceMutation.isPending} onClick={() => advanceMutation.mutate()}>
                {advanceMutation.isPending ? 'Starting…' : 'Continue to reflection'}
              </button>
            )}
          </StageCard>
        )}

        {showReflecting && (
          <StageCard
            title="Reflecting"
            expandedByDefault={
              run.status === 'reflecting' || run.reflection_review?.status === 'pending'
            }
          >
            {run.reflecting_progress && (
              <ReflectionActivity progress={run.reflecting_progress} active={run.status === 'reflecting'} />
            )}
            {run.reflecting_result && (
              <ReflectionReview
                run={run}
                onSelect={selectCandidate}
                onSaveDraft={saveDraft}
                onResetDraft={resetDraft}
                onPromote={promote}
                onDismiss={dismiss}
                onDirtyChange={setReviewDirty}
              />
            )}
          </StageCard>
        )}
      </div>
    </div>
  )
}
