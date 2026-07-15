import type {
  JudgingAttemptSummary,
  JudgingPlan,
  JudgingProgress as JudgingProgressData,
  JudgingResult,
  ReviewAttempt,
  InferenceStepAudit,
} from '../../types'

export interface JudgingProgressProps {
  plan: JudgingPlan | null
  progress: JudgingProgressData | null
  result: JudgingResult | null
}

function words(value: string): string {
  return value.replaceAll('_', ' ')
}

function usageText(usage: Record<string, unknown>): string {
  return Object.entries(usage).map(([key, value]) => `${words(key)} ${String(value)}`).join(' · ') || '—'
}

function Step({ step }: { step: InferenceStepAudit }) {
  return (
    <li className="rounded border border-gray-200 bg-white p-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="font-medium text-gray-800">{step.phase} · {step.reused ? 'reused' : 'current'}</span>
        <span className="font-mono text-gray-500">{step.schema_name ?? '—'}</span>
      </div>
      <dl className="mt-1 grid gap-1 sm:grid-cols-2">
        <div><dt className="inline text-gray-400">Model: </dt><dd className="inline">{step.resolved_model ?? step.requested_model}</dd></div>
        <div><dt className="inline text-gray-400">Transport requests: </dt><dd className="inline">{step.transport_request_count}</dd></div>
        <div><dt className="inline text-gray-400">Output mode: </dt><dd className="inline">{step.output_mode ? words(step.output_mode) : '—'}</dd></div>
        <div><dt className="inline text-gray-400">Usage: </dt><dd className="inline">{usageText(step.usage)}</dd></div>
        <div className="sm:col-span-2"><dt className="inline text-gray-400">Artifact: </dt><dd className="inline break-all font-mono">{step.artifact_id}</dd></div>
        {step.schema_fallback_reason && <div className="sm:col-span-2"><dt className="inline text-gray-400">Schema fallback: </dt><dd className="inline">{step.schema_fallback_reason}</dd></div>}
        {step.raw_output_digest && <div className="sm:col-span-2"><dt className="inline text-gray-400">Output digest: </dt><dd className="inline break-all font-mono">{step.raw_output_digest}</dd></div>}
      </dl>
    </li>
  )
}

function Attempt({ attempt }: { attempt: ReviewAttempt }) {
  const statusClass = attempt.status === 'failed'
    ? 'text-red-700'
    : attempt.status === 'abstained'
      ? 'text-amber-700'
      : 'text-green-700'
  return (
    <li className="rounded-md border border-gray-200 bg-white p-2">
      <div className="flex flex-wrap items-center justify-between gap-2">
        <span className="font-medium text-gray-800">
          Judge {attempt.position} · {attempt.requested_model}
        </span>
        <span className={statusClass}>
          {words(attempt.trigger)} · {attempt.status}
        </span>
      </div>
      {attempt.resolved_model && attempt.resolved_model !== attempt.requested_model && (
        <div className="mt-1 text-gray-500">Resolved as {attempt.resolved_model}</div>
      )}
      {attempt.score !== null && (
        <div className="mt-1 font-mono tabular-nums text-gray-700">Score {attempt.score.toFixed(2)}</div>
      )}
      {attempt.rationale && <p className="mt-1 text-gray-600">{attempt.rationale}</p>}
      {attempt.behavioral_feedback && (
        <dl className="mt-2 grid gap-1 rounded border border-purple-100 bg-purple-50 p-2 text-gray-700">
          {attempt.behavioral_feedback.success && <div><dt className="inline font-medium text-green-700">Success: </dt><dd className="inline">{attempt.behavioral_feedback.success}</dd></div>}
          {attempt.behavioral_feedback.problem && <div><dt className="inline font-medium text-red-700">Problem: </dt><dd className="inline">{attempt.behavioral_feedback.problem}</dd></div>}
          {attempt.behavioral_feedback.desired_behavior && <div><dt className="inline font-medium text-purple-700">Desired behavior: </dt><dd className="inline">{attempt.behavioral_feedback.desired_behavior}</dd></div>}
        </dl>
      )}
      {attempt.status === 'failed' && (
        <p className="mt-1 text-red-700">
          {attempt.error_type ?? 'Error'}: {attempt.message ?? 'No error detail returned'}
        </p>
      )}
      <details className="mt-2 rounded border border-gray-100 bg-gray-50 p-2 text-[0.6875rem] text-gray-600">
        <summary className="cursor-pointer font-medium text-gray-700">Transport and schema audit</summary>
        <dl className="mt-2 grid gap-1 sm:grid-cols-2">
          <div><dt className="inline text-gray-400">Requested: </dt><dd className="inline">{attempt.requested_family} · {attempt.requested_backend}</dd></div>
          <div><dt className="inline text-gray-400">Resolved family: </dt><dd className="inline">{attempt.resolved_family ?? '—'}</dd></div>
          <div><dt className="inline text-gray-400">Output mode: </dt><dd className="inline">{attempt.output_mode ? words(attempt.output_mode) : '—'}</dd></div>
          <div><dt className="inline text-gray-400">Schema: </dt><dd className="inline">{attempt.schema_name ?? '—'}{attempt.verdict_schema_version ? ` v${attempt.verdict_schema_version}` : ''}</dd></div>
          <div><dt className="inline text-gray-400">Transport requests: </dt><dd className="inline">{attempt.transport_request_count}</dd></div>
          <div><dt className="inline text-gray-400">Usage: </dt><dd className="inline">{usageText(attempt.usage)}</dd></div>
          {attempt.evidence_ids.length > 0 && <div className="sm:col-span-2"><dt className="inline text-gray-400">Evidence IDs: </dt><dd className="inline font-mono">{attempt.evidence_ids.join(', ')}</dd></div>}
          {attempt.schema_fallback_reason && <div className="sm:col-span-2"><dt className="inline text-gray-400">Schema fallback: </dt><dd className="inline">{attempt.schema_fallback_reason}</dd></div>}
          {attempt.raw_output_digest && <div className="sm:col-span-2"><dt className="inline text-gray-400">Output digest: </dt><dd className="inline break-all font-mono">{attempt.raw_output_digest}</dd></div>}
        </dl>
      </details>
      {attempt.steps.length > 0 && (
        <details className="mt-2 rounded border border-gray-100 bg-gray-50 p-2 text-[0.6875rem] text-gray-600">
          <summary className="cursor-pointer font-medium text-gray-700">Ordered inference steps</summary>
          <ol className="mt-2 space-y-2">
            {attempt.steps.map((step, index) => (
              <Step key={`${step.artifact_id}-${index}`} step={step} />
            ))}
          </ol>
        </details>
      )}
    </li>
  )
}

function ReviewRecord({ record }: { record: JudgingAttemptSummary }) {
  return (
    <details open className="rounded-lg border border-gray-200 bg-gray-50 text-xs">
      <summary className="cursor-pointer px-3 py-2">
        <span className="font-medium text-gray-900">{record.rubric}</span>
        <span className="ml-2 text-gray-500">
          {record.scope} · {words(record.review_status)}
          {record.rating === null ? '' : ` · ${record.rating.toFixed(2)}`}
        </span>
      </summary>
      <div className="space-y-2 border-t border-gray-200 px-3 py-2">
        <div className="text-gray-500">
          session {record.conversation_id} ·{' '}
          {record.attempt_count} reviewer attempt{record.attempt_count === 1 ? '' : 's'}
        </div>
        <ol className="space-y-2">
          {record.attempts.map((attempt, index) => (
            <Attempt key={`${attempt.position}-${attempt.trigger}-${index}`} attempt={attempt} />
          ))}
        </ol>
      </div>
    </details>
  )
}

export default function JudgingProgress({
  plan,
  progress,
  result,
}: JudgingProgressProps) {
  const state = result ?? progress
  if (!plan && !state) {
    return <p role="status" className="text-sm text-gray-500">Waiting for judging to start.</p>
  }

  const planned = state?.planned_rubrics ?? plan?.totals.planned_rubrics ?? 0
  const completed = state?.rubrics_completed ?? 0
  const percent = planned === 0 ? 0 : Math.round(Math.min(1, completed / planned) * 100)
  const attempts = state?.reviewer_attempts_completed ?? 0
  const minimum = state?.minimum_reviewer_attempts ?? plan?.totals.minimum_reviewer_attempts ?? 0
  const maximum = state?.maximum_reviewer_attempts ?? plan?.totals.maximum_reviewer_attempts ?? 0
  const reviewRecords = state?.attempt_summaries ?? []
  const failures = state?.failure_details ?? []

  return (
    <section aria-label="Judging progress" className="space-y-4">
      <div className="space-y-2">
        <div className="flex flex-wrap items-center justify-between gap-2">
          <p role="status" aria-live="polite" className="text-sm font-medium text-gray-800">
            {state?.status_message ?? `Planned ${planned} rubric reviews`}
          </p>
          {plan && (
            <span className="text-xs text-gray-500">
              {plan.totals.sessions_planned} session{plan.totals.sessions_planned === 1 ? '' : 's'} ·{' '}
              {plan.totals.turns_considered} turns · {plan.totals.windows_planned} raw windows
            </span>
          )}
        </div>
        <div
          role="progressbar"
          aria-label="Rubrics reviewed"
          aria-valuemin={0}
          aria-valuemax={planned}
          aria-valuenow={completed}
          className="h-2 overflow-hidden rounded-full bg-gray-200"
        >
          <div className="h-full rounded-full bg-purple-500 transition-all" style={{ width: `${percent}%` }} />
        </div>
      </div>

      <div className="grid gap-2 text-sm sm:grid-cols-2 lg:grid-cols-4">
        <div className="rounded-lg bg-gray-50 p-3">
          <div className="font-medium text-gray-900">{completed} of {planned} rubrics reviewed</div>
          <div className="mt-1 text-xs text-gray-500">{state?.rated_rubrics ?? 0} received a rating</div>
        </div>
        <div className="rounded-lg bg-gray-50 p-3">
          <div className="font-medium text-gray-900">{attempts} reviewer attempt{attempts === 1 ? '' : 's'} so far</div>
          <div className="mt-1 text-xs text-gray-500">{minimum} minimum · {maximum} maximum</div>
        </div>
        <div className="rounded-lg bg-gray-50 p-3">
          <div className="font-medium text-gray-900">{state?.scores_written ?? 0} scores written</div>
          <div className="mt-1 text-xs text-gray-500">Written only after complete coverage</div>
        </div>
        <div className={`rounded-lg p-3 ${state && (state.failure_count || state.write_failure_count) ? 'bg-red-50 text-red-800' : 'bg-gray-50'}`}>
          <div className="font-medium">{state?.failure_count ?? 0} review failures</div>
          <div className="mt-1 text-xs">{state?.write_failure_count ?? 0} write failures</div>
        </div>
      </div>

      <div className="grid gap-2 text-sm sm:grid-cols-3">
        <div className="rounded-lg bg-purple-50 p-3 font-medium text-purple-900">
          {state?.digest_steps_completed ?? 0} of {state?.maximum_digest_steps ?? plan?.totals.maximum_digest_calls ?? 0} digests
        </div>
        <div className="rounded-lg bg-purple-50 p-3 font-medium text-purple-900">
          {state?.window_steps_completed ?? 0} of {state?.maximum_window_steps ?? plan?.totals.maximum_window_calls ?? 0} windows
        </div>
        <div className="rounded-lg bg-purple-50 p-3 font-medium text-purple-900">
          {state?.merge_steps_completed ?? 0} of {state?.maximum_merge_steps ?? plan?.totals.maximum_merge_calls ?? 0} merges
        </div>
      </div>

      {plan && plan.sessions.length > 0 && (
        <div>
          <h4 className="mb-2 text-xs font-medium uppercase tracking-wide text-gray-500">Pinned sliding windows</h4>
          <div className="space-y-2">
            {plan.sessions.map((session) => (
              <details key={session.conversation_id} className="rounded-lg border border-gray-200 bg-gray-50 text-xs">
                <summary className="cursor-pointer px-3 py-2 font-medium text-gray-900">
                  {session.conversation_id} · {session.turn_count} turns · {session.reviewers.length} reviewers
                </summary>
                <div className="space-y-3 border-t border-gray-200 px-3 py-2">
                  {session.reviewers.map((reviewer) => (
                    <div key={reviewer.ordinal}>
                      <div className="font-medium text-gray-800">Judge {reviewer.ordinal} · {reviewer.judge.label}</div>
                      <div className="mt-0.5 text-gray-500">
                        maximum {reviewer.work_bounds.digest_calls} digests ·{' '}
                        {reviewer.work_bounds.window_calls_per_rubric} windows per rubric ·{' '}
                        {reviewer.work_bounds.merge_calls_per_rubric} merge per rubric
                      </div>
                      <ol className="mt-2 space-y-1">
                        {reviewer.window_plan.windows.map((window) => (
                          <li key={window.window_id} className="rounded border border-gray-200 bg-white p-2">
                            <span className="font-medium">Window {window.index}</span>{' · '}
                            core {window.core_trace_ids.join(', ')} · raw {window.raw_trace_ids.join(', ')} ·{' '}
                            {window.raw_tokens.toLocaleString()} estimated tokens
                          </li>
                        ))}
                      </ol>
                    </div>
                  ))}
                </div>
              </details>
            ))}
          </div>
        </div>
      )}

      {reviewRecords.length > 0 && (
        <div>
          <h4 className="mb-2 text-xs font-medium uppercase tracking-wide text-gray-500">Reviewer audit</h4>
          <div className="max-h-[32rem] space-y-2 overflow-y-auto">
            {reviewRecords.map((record, index) => (
              <ReviewRecord key={`${record.scope}-${record.rubric}-${record.conversation_id}-${index}`} record={record} />
            ))}
          </div>
          {state?.attempt_summaries_truncated && (
            <p className="mt-2 text-xs text-amber-700">
              Showing {reviewRecords.length} of {state.attempt_summary_count ?? reviewRecords.length} review records.
            </p>
          )}
        </div>
      )}

      {failures.length > 0 && (
        <div className="rounded-lg border border-red-200 bg-red-50 p-3">
          <h4 className="text-sm font-medium text-red-900">Incomplete coverage</h4>
          <ul className="mt-2 space-y-2 text-xs text-red-800">
            {failures.map((failure, index) => (
              <li key={`${failure.scope}-${failure.rubric}-${index}`}>
                <div className="font-medium">{failure.rubric}</div>
                <div>{failure.error_type}: {failure.message ?? 'No error detail returned'}</div>
                {failure.attempts.length > 0 && (
                  <details className="mt-2 rounded border border-red-200 bg-white/70 p-2 text-gray-800">
                    <summary className="cursor-pointer font-medium text-red-900">Full failed-attempt audit</summary>
                    <ol className="mt-2 space-y-2">
                      {failure.attempts.map((attempt, attemptIndex) => (
                        <Attempt key={`${attempt.position}-${attempt.trigger}-${attemptIndex}`} attempt={attempt} />
                      ))}
                    </ol>
                  </details>
                )}
              </li>
            ))}
          </ul>
          {state?.failure_details_truncated && (
            <p className="mt-2 text-xs text-amber-800">
              Showing {failures.length} of {state.failure_detail_count ?? failures.length} failures.
            </p>
          )}
        </div>
      )}
    </section>
  )
}
