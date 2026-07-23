import type { ReflectionBundleSnapshot } from '../../types'
import { bundleActions } from './reflectionState'
import { lineDiff } from './lineDiff'

export default function StaleEvidence({
  past,
  current,
  reason,
  changedTargets,
}: {
  past: ReflectionBundleSnapshot
  current: ReflectionBundleSnapshot | null | undefined
  reason: string | null | undefined
  changedTargets: string[] | undefined
}) {
  const actions = current ? bundleActions(past, current) : []
  return (
    <section aria-label="Stale reflection evidence" className="space-y-3 rounded-lg border border-amber-200 bg-amber-50 p-4">
      <h4 className="text-sm font-semibold text-amber-950">The evaluated baseline is stale</h4>
      <p className="text-sm text-amber-900">
        {reason ?? 'Managed instructions changed after A was evaluated.'}
      </p>
      {actions.map((action) => (
        <details key={action.locator} className="rounded border border-amber-200 bg-white p-3">
          <summary className="cursor-pointer font-mono text-xs font-semibold text-amber-950">
            {action.locator} · Diff A → Current
          </summary>
          <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap rounded bg-gray-950 p-3 font-mono text-xs text-gray-100">
            {lineDiff(action.locator, action.before, action.after, 'Past (A)', 'Current')}
          </pre>
        </details>
      ))}
      {actions.length === 0 && changedTargets && changedTargets.length > 0 && (
        <p className="font-mono text-xs text-amber-900">{changedTargets.join(', ')}</p>
      )}
      <a href="/runs" className="inline-block text-sm font-semibold text-blue-700 hover:underline">
        Start a new run
      </a>
    </section>
  )
}
