import type { PromotionReceipt, ReflectionTargetSnapshot } from '../../types'
import { receiptBundles } from './reflectionState'
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
  const promoted = mapTargets(snapshots.promoted.targets)

  return (
    <section aria-label="Promotion receipt" className="space-y-3 rounded-lg border border-green-200 bg-green-50/40 p-4">
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h4 className="text-sm font-semibold text-green-950">Promotion receipt</h4>
          <p className="mt-1 text-xs text-green-900">
            {receipt.promoted_was_evaluated ? 'Evaluated C was promoted.' : 'Unevaluated edited D was promoted.'}
          </p>
        </div>
        <time className="text-xs text-green-900" dateTime={receipt.decided_at}>
          {new Date(receipt.decided_at).toLocaleString()}
        </time>
      </div>

      <dl className="grid gap-2 text-xs text-green-950 sm:grid-cols-3">
        <div><dt className="font-medium">Promotion ID</dt><dd className="font-mono break-all">{receipt.promotion_id}</dd></div>
        <div><dt className="font-medium">Candidate ID</dt><dd className="font-mono break-all">{receipt.candidate_id}</dd></div>
        <div><dt className="font-medium">Review revision</dt><dd className="font-mono">{receipt.review_revision}</dd></div>
        <div><dt className="font-medium">Target kind</dt><dd>{receipt.target_kind}</dd></div>
        <div className="sm:col-span-2"><dt className="font-medium">Target identity</dt><dd className="font-mono break-all">{receipt.target_id}</dd></div>
        <div><dt className="font-medium">Past B</dt><dd className="font-mono break-all">{receipt.past.revision}</dd></div>
        <div><dt className="font-medium">Evaluated C</dt><dd className="font-mono break-all">{receipt.evaluated_candidate.revision}</dd></div>
        <div><dt className="font-medium">Promoted {receipt.promoted_was_evaluated ? 'C' : 'D'}</dt><dd className="font-mono break-all">{receipt.promoted.revision}</dd></div>
      </dl>

      {!receipt.promoted_was_evaluated && (
        <p className="rounded border border-amber-200 bg-amber-50 p-2 text-xs text-amber-900">
          {receipt.unevaluated_d_acknowledged
            ? 'Unevaluated D acknowledgement recorded.'
            : 'Unevaluated D acknowledgement was not recorded.'}
        </p>
      )}

      <div className="space-y-2">
        {receipt.actions.map((action) => (
          <details key={action.locator} className="rounded-lg border border-green-200 bg-white p-3">
            <summary className="cursor-pointer text-xs font-semibold text-green-950">
              {action.action.toUpperCase()} {action.locator}
            </summary>
            <div className="mt-3 grid gap-3 lg:grid-cols-3">
              <div><div className="mb-1 text-xs font-medium text-gray-500">Past B</div><pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded bg-gray-950 p-2 font-mono text-xs text-gray-100">{content(past.get(action.locator))}</pre></div>
              <div><div className="mb-1 text-xs font-medium text-indigo-700">Evaluated C</div><pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded bg-gray-950 p-2 font-mono text-xs text-gray-100">{content(evaluated.get(action.locator))}</pre></div>
              <div><div className="mb-1 text-xs font-medium text-green-700">Promoted {receipt.promoted_was_evaluated ? 'C' : 'D'}</div><pre className="max-h-48 overflow-auto whitespace-pre-wrap rounded bg-gray-950 p-2 font-mono text-xs text-gray-100">{content(promoted.get(action.locator))}</pre></div>
            </div>
            <pre className="mt-3 max-h-64 overflow-auto whitespace-pre-wrap rounded bg-gray-950 p-2 font-mono text-xs text-gray-100">
              {lineDiff(action.locator, action.before, action.after, 'Past B', receipt.promoted_was_evaluated ? 'Promoted C' : 'Promoted D')}
            </pre>
            {!receipt.promoted_was_evaluated && (
              <div className="mt-3">
                <div className="mb-1 text-xs font-medium text-gray-600">Diff C → D</div>
                <pre className="max-h-64 overflow-auto whitespace-pre-wrap rounded bg-gray-950 p-2 font-mono text-xs text-gray-100">
                  {lineDiff(
                    action.locator,
                    evaluated.get(action.locator) ?? action.before,
                    promoted.get(action.locator) ?? action.after,
                    'Evaluated C',
                    'Promoted D',
                  )}
                </pre>
              </div>
            )}
          </details>
        ))}
      </div>

      {receipt.git_metadata && Object.keys(receipt.git_metadata).length > 0 && (
        <details>
          <summary className="cursor-pointer text-xs font-medium text-green-950">Git metadata</summary>
          <pre className="mt-2 overflow-auto rounded bg-gray-950 p-2 font-mono text-xs text-gray-100">
            {JSON.stringify(receipt.git_metadata, null, 2)}
          </pre>
        </details>
      )}
    </section>
  )
}
