import { useEffect, useMemo, useState } from 'react'
import type {
  ReflectionBundleSnapshot,
  ReflectionTargetSnapshot,
} from '../../types'
import { bundleActions } from './reflectionState'
import UnifiedDiff from './UnifiedDiff'

function targetContent(target: ReflectionTargetSnapshot): string {
  return target.exists ? target.content ?? '' : 'This file does not exist in this snapshot.'
}

function lineCount(content: string): number {
  return content.split('\n').length
}

export default function BundleComparison({
  past,
  proposed,
  editedTargets = null,
  onEditedTargetsChange,
}: {
  past: ReflectionBundleSnapshot
  proposed: ReflectionBundleSnapshot
  editedTargets?: ReflectionTargetSnapshot[] | null
  onEditedTargetsChange?: (targets: ReflectionTargetSnapshot[]) => void
}) {
  const actions = useMemo(() => bundleActions(past, proposed), [past, proposed])
  const hasEdited = editedTargets != null
  const editable = hasEdited && onEditedTargetsChange != null
  const preferredLocator = (editable
    ? actions.find((action) => action.action === 'update') ?? actions[0]
    : actions[0])?.locator ?? ''
  const [selectedLocator, setSelectedLocator] = useState(preferredLocator)

  useEffect(() => {
    if (!actions.some((action) => action.locator === selectedLocator)) {
      setSelectedLocator(preferredLocator)
    }
  }, [actions, preferredLocator, selectedLocator])

  useEffect(() => {
    if (editable && preferredLocator) setSelectedLocator(preferredLocator)
  }, [editable, preferredLocator])

  const action = actions.find((item) => item.locator === selectedLocator) ??
    actions.find((item) => item.locator === preferredLocator)
  if (!action) {
    return <p className="text-sm text-gray-500">The proposed bundle has no file changes.</p>
  }
  const edited = editedTargets?.find((target) => target.locator === action.locator) ?? action.after
  const editedContent = edited.content ?? ''
  const lines = lineCount(editedContent)

  function updateContent(content: string) {
    if (!onEditedTargetsChange || !editedTargets) return
    onEditedTargetsChange(editedTargets.map((target) => (
      target.locator === action.locator
        ? { ...target, content, revision: `unsaved:${target.locator}` }
        : target
    )))
  }

  return (
    <section aria-label="Bundle comparison" className="rounded-lg border border-gray-200">
      <div className="border-b border-gray-200 p-3">
        <h4 className="mb-2 text-xs font-semibold uppercase tracking-wide text-gray-500">
          Affected files
        </h4>
        <div className="flex flex-wrap gap-2" role="group" aria-label="Affected files">
          {actions.map((item) => (
            <button
              key={item.locator}
              type="button"
              aria-pressed={item.locator === action.locator}
              onClick={() => setSelectedLocator(item.locator)}
              className={`rounded-lg border px-3 py-2 text-xs ${
                item.locator === action.locator
                  ? 'border-indigo-400 bg-indigo-50 text-indigo-900'
                  : 'border-gray-200 text-gray-700 hover:bg-gray-50'
              }`}
            >
              <span className="font-mono">{item.locator}</span>
              <span className="ml-2 rounded bg-gray-100 px-1.5 py-0.5 font-semibold uppercase text-gray-600">
                {item.action}
              </span>
            </button>
          ))}
        </div>
      </div>

      <div className="grid min-w-0 gap-3 p-3 xl:grid-cols-2">
        <section className="min-w-0 overflow-hidden rounded-lg border border-gray-200">
          <div className="flex items-center justify-between border-b border-gray-200 px-3 py-2">
            <h4 className="text-xs font-semibold text-gray-800">
              {editable ? 'Editable proposal C' : hasEdited ? 'Edited C' : 'Proposed B'}
            </h4>
            {editable && <span className="text-xs text-gray-500">{lines} {lines === 1 ? 'line' : 'lines'}</span>}
          </div>
          {editable ? (
            <div className="flex min-h-80 max-h-[32rem] overflow-auto bg-white font-mono text-xs leading-5 text-gray-700">
              <pre
                aria-label={`Line numbers for ${action.locator}`}
                className="select-none border-r border-gray-200 bg-gray-50 px-3 py-3 text-right text-gray-400"
              >
                {Array.from({ length: lines }, (_, index) => index + 1).join('\n')}
              </pre>
              <textarea
                aria-label={`Edit ${action.locator}`}
                value={editedContent}
                rows={lines}
                onChange={(event) => updateContent(event.target.value)}
                className="min-h-80 min-w-0 flex-1 resize-none overflow-hidden whitespace-pre bg-white p-3 font-mono text-xs leading-5 text-gray-700 outline-none focus:ring-2 focus:ring-inset focus:ring-blue-500"
                spellCheck={false}
              />
            </div>
          ) : (
            <pre className="max-h-[32rem] overflow-auto whitespace-pre-wrap bg-gray-950 p-3 font-mono text-xs text-gray-100">
              {targetContent(hasEdited ? edited : action.after)}
            </pre>
          )}
        </section>

        <section className="min-w-0 overflow-hidden rounded-lg border border-gray-200">
          <h4 className="border-b border-gray-200 px-3 py-2 text-xs font-semibold text-gray-800">
            {hasEdited ? 'Live diff A → C' : 'Diff A → B'}
          </h4>
          <div className="overflow-auto p-3">
            <UnifiedDiff
              locator={action.locator}
              before={action.before}
              after={hasEdited ? edited : action.after}
              beforeLabel="Past (A)"
              afterLabel={hasEdited ? 'Edited (C)' : 'Proposed (B)'}
            />
          </div>
        </section>
      </div>
    </section>
  )
}
