import type { PromotionReceipt, ReflectionTargetSnapshot } from '../../types'
import { bundleActions, receiptBundles } from './reflectionState'
import { lineDiff } from './lineDiff'

function mapTargets(targets: ReflectionTargetSnapshot[]) {
  return new Map(targets.map((target) => [target.locator, target]))
}

function content(target: ReflectionTargetSnapshot | undefined): string {
  return target?.exists ? target.content ?? '' : 'This file did not exist.'
}

export default function PromotionReceiptView({ receipt }: { receipt: PromotionReceipt }) {
  const snapshots = receiptBundles(receipt)
  const past = mapTargets(snapshots.past.targets)
  const evaluated = mapTargets(snapshots.evaluated.targets)
  const requested = mapTargets(snapshots.promoted.targets)
  const actions = new Map(
    bundleActions(snapshots.past, snapshots.promoted).map((action) => [action.locator, action]),
  )
  const applied = receipt.outcomes.filter((outcome) => outcome.status === 'applied').length
  const partial = applied > 0 && applied < receipt.outcomes.length
  const containerClass = partial
    ? 'space-y-3 rounded-lg border border-amber-200 bg-amber-50/40 p-4'
    : 'space-y-3 rounded-lg border border-green-200 bg-green-50/40 p-4'
  const headingClass = partial ? 'text-sm font-semibold text-amber-950' : 'text-sm font-semibold text-green-950'
  const textClass = partial ? 'mt-1 text-xs text-amber-900' : 'mt-1 text-xs text-green-900'
  const timeClass = partial ? 'text-xs text-amber-900' : 'text-xs text-green-900'
  const detailsClass = partial
    ? 'grid gap-2 text-xs text-amber-950 sm:grid-cols-3'
    : 'grid gap-2 text-xs text-green-950 sm:grid-cols-3'

  return (
    <section
      aria-label="Promotion receipt"
      className={containerClass}
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h4 className={headingClass}>
            {partial ? 'Partial promotion receipt' : 'Promotion receipt'}
          </h4>
          <p className={textClass}>
            {partial
              ? `${applied} of ${receipt.outcomes.length} complete files were applied. Earlier applied files were not rolled back.`
              : receipt.promoted_was_evaluated
                ? 'Evaluated B was promoted.'
                : 'Unevaluated edited C was promoted.'}
          </p>
        </div>
        <time className={timeClass} dateTime={receipt.decided_at}>
          {new Date(receipt.decided_at).toLocaleString()}
        </time>
      </div>

      <dl className={detailsClass}>
        <div><dt className="font-medium">Promotion ID</dt><dd className="font-mono break-all">{receipt.promotion_id}</dd></div>
        <div><dt className="font-medium">Candidate ID</dt><dd className="font-mono break-all">{receipt.candidate_id}</dd></div>
        <div><dt className="font-medium">Review revision</dt><dd className="font-mono">{receipt.review_revision}</dd></div>
        <div><dt className="font-medium">Past A</dt><dd className="font-mono break-all">{receipt.past.revision}</dd></div>
        <div><dt className="font-medium">Evaluated B</dt><dd className="font-mono break-all">{receipt.evaluated_candidate.revision}</dd></div>
        <div><dt className="font-medium">Requested {receipt.promoted_was_evaluated ? 'B' : 'C'}</dt><dd className="font-mono break-all">{receipt.promoted.revision}</dd></div>
        <div>
          <dt className="font-medium">Sandbox verification</dt>
          <dd>{receipt.sandbox_verified ? 'Passed' : 'Did not pass'}</dd>
        </div>
        {receipt.challenge_id && (
          <div><dt className="font-medium">Challenge ID</dt><dd className="font-mono break-all">{receipt.challenge_id}</dd></div>
        )}
        {receipt.challenge_status && (
          <div><dt className="font-medium">Challenge status</dt><dd>{receipt.challenge_status}</dd></div>
        )}
      </dl>

      <p className={`rounded border p-2 text-xs ${
        receipt.sandbox_verified
          ? 'border-green-200 bg-green-50 text-green-900'
          : 'border-amber-200 bg-amber-50 text-amber-900'
      }`}>
        {receipt.sandbox_verified
          ? 'Paired sandbox verification passed.'
          : `Paired sandbox verification did not pass.${receipt.challenge_reason ? ` ${receipt.challenge_reason}` : ''}`}
      </p>

      {!receipt.sandbox_verified && receipt.unverified_b_acknowledged && (
        <p className="rounded border border-amber-200 bg-amber-50 p-2 text-xs text-amber-900">
          Unverified B acknowledgement recorded.
        </p>
      )}

      {!receipt.promoted_was_evaluated && (
        <p className="rounded border border-amber-200 bg-amber-50 p-2 text-xs text-amber-900">
          Unevaluated C acknowledgement recorded.
        </p>
      )}

      <div className="space-y-2">
        {receipt.outcomes.map((outcome) => {
          const action = actions.get(outcome.locator)
          return (
            <details key={outcome.locator} className="rounded-lg border border-gray-200 bg-white p-3">
              <summary className="cursor-pointer text-xs font-semibold text-gray-950">
                {outcome.status === 'applied' ? 'APPLIED' : 'NOT APPLIED'} · {outcome.action.toUpperCase()} {outcome.locator}
              </summary>
              {outcome.message && <p className="mt-2 text-xs text-red-800">{outcome.message}</p>}
              <div className="mt-3 grid gap-3 lg:grid-cols-3">
                <div><div className="mb-1 text-xs font-medium text-gray-500">Past A</div><pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded bg-gray-950 p-2 font-mono text-xs text-gray-100">{content(past.get(outcome.locator))}</pre></div>
                <div><div className="mb-1 text-xs font-medium text-indigo-700">Evaluated B</div><pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded bg-gray-950 p-2 font-mono text-xs text-gray-100">{content(evaluated.get(outcome.locator))}</pre></div>
                <div><div className="mb-1 text-xs font-medium text-green-700">Requested {receipt.promoted_was_evaluated ? 'B' : 'C'}</div><pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded bg-gray-950 p-2 font-mono text-xs text-gray-100">{content(requested.get(outcome.locator))}</pre></div>
              </div>
              {action && (
                <pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap rounded bg-gray-950 p-2 font-mono text-xs text-gray-100">
                  {lineDiff(action.locator, action.before, action.after, 'Past A', 'Requested contents')}
                </pre>
              )}
            </details>
          )
        })}
      </div>
    </section>
  )
}
