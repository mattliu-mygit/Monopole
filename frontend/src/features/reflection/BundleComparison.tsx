import { useEffect, useId, useMemo, useRef, useState, type KeyboardEvent } from 'react'
import type {
  ReflectionBundleSnapshot,
  ReflectionTargetSnapshot,
} from '../../types'
import { bundleActions } from './reflectionState'
import { lineDiff } from './lineDiff'

type View = 'past' | 'proposed' | 'diff' | 'edited' | 'edited-diff'

function targetContent(target: ReflectionTargetSnapshot): string {
  return target.exists ? target.content ?? '' : 'This file does not exist in this snapshot.'
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
  const [selectedLocator, setSelectedLocator] = useState(actions[0]?.locator ?? '')
  const [view, setView] = useState<View>('diff')
  const editorRef = useRef<HTMLTextAreaElement>(null)
  const tabRefs = useRef(new Map<View, HTMLButtonElement>())
  const tabsId = useId().replaceAll(':', '')

  useEffect(() => {
    if (!actions.some((action) => action.locator === selectedLocator)) {
      setSelectedLocator(actions[0]?.locator ?? '')
    }
  }, [actions, selectedLocator])

  useEffect(() => {
    if (onEditedTargetsChange) {
      const firstEditable = actions.find((candidate) => candidate.action === 'update')
        ?? actions.find((candidate) => candidate.action !== 'delete')
      if (firstEditable) setSelectedLocator(firstEditable.locator)
      setView('edited')
    }
  }, [actions, onEditedTargetsChange])

  useEffect(() => {
    if (!editedTargets) {
      setView((current) => current === 'edited' || current === 'edited-diff' ? 'diff' : current)
    }
  }, [editedTargets])

  useEffect(() => {
    if (view === 'edited') editorRef.current?.focus()
  }, [selectedLocator, view])

  const action = actions.find((item) => item.locator === selectedLocator) ?? actions[0]
  if (!action) {
    return <p className="text-sm text-gray-500">The proposed bundle has no file changes.</p>
  }
  const edited = editedTargets?.find((target) => target.locator === action.locator) ?? action.after
  const editable = Boolean(onEditedTargetsChange && action.action !== 'delete')

  function updateContent(content: string) {
    if (!onEditedTargetsChange || !editedTargets) return
    onEditedTargetsChange(editedTargets.map((target) => (
      target.locator === action.locator
        ? { ...target, content, revision: `unsaved:${target.locator}` }
        : target
    )))
  }

  const views: { value: View; label: string }[] = [
    { value: 'past', label: 'Past (B)' },
    { value: 'proposed', label: 'Proposed (C)' },
    { value: 'diff', label: 'Diff B → C' },
  ]
  if (editedTargets) views.push({ value: 'edited', label: 'Edited (D)' })
  if (editedTargets) views.push({ value: 'edited-diff', label: 'Diff C → D' })

  function tabId(value: View): string {
    return `${tabsId}-tab-${value}`
  }

  function panelId(value: View): string {
    return `${tabsId}-panel-${value}`
  }

  function handleTabKeyDown(event: KeyboardEvent<HTMLButtonElement>, index: number) {
    let nextIndex: number | null = null
    if (event.key === 'ArrowRight' || event.key === 'ArrowDown') {
      nextIndex = (index + 1) % views.length
    } else if (event.key === 'ArrowLeft' || event.key === 'ArrowUp') {
      nextIndex = (index - 1 + views.length) % views.length
    } else if (event.key === 'Home') {
      nextIndex = 0
    } else if (event.key === 'End') {
      nextIndex = views.length - 1
    }
    if (nextIndex === null) return
    event.preventDefault()
    const nextView = views[nextIndex].value
    setView(nextView)
    tabRefs.current.get(nextView)?.focus()
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

      <div
        className="flex flex-wrap gap-1 border-b border-gray-200 px-3 pt-3"
        role="tablist"
        aria-label="Bundle views"
      >
        {views.map((item, index) => (
          <button
            key={item.value}
            ref={(element) => {
              if (element) tabRefs.current.set(item.value, element)
              else tabRefs.current.delete(item.value)
            }}
            id={tabId(item.value)}
            type="button"
            role="tab"
            aria-selected={view === item.value}
            aria-controls={panelId(item.value)}
            tabIndex={view === item.value ? 0 : -1}
            onClick={() => setView(item.value)}
            onKeyDown={(event) => handleTabKeyDown(event, index)}
            className={`rounded-t px-3 py-2 text-xs font-medium ${
              view === item.value ? 'bg-gray-900 text-white' : 'text-gray-600 hover:bg-gray-100'
            }`}
          >
            {item.label}
          </button>
        ))}
      </div>

      {views.map((item) => (
        <div
          key={item.value}
          id={panelId(item.value)}
          role="tabpanel"
          aria-labelledby={tabId(item.value)}
          hidden={view !== item.value}
          tabIndex={0}
          className="p-3"
        >
          {item.value === 'edited' && editable ? (
            <textarea
              ref={editorRef}
              aria-label={`Edit ${action.locator}`}
              value={edited.content ?? ''}
              onChange={(event) => updateContent(event.target.value)}
              className="min-h-72 w-full rounded bg-gray-950 p-3 font-mono text-xs text-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          ) : (
            <pre className="max-h-[32rem] overflow-auto whitespace-pre-wrap rounded bg-gray-950 p-3 font-mono text-xs text-gray-100">
              {item.value === 'past' && targetContent(action.before)}
              {item.value === 'proposed' && targetContent(action.after)}
              {item.value === 'edited' && targetContent(edited)}
              {item.value === 'edited-diff' && lineDiff(
                action.locator,
                action.after,
                edited,
                'Proposed (C)',
                'Edited (D)',
              )}
              {item.value === 'diff' && lineDiff(
                action.locator,
                action.before,
                action.after,
                'Past (B)',
                'Proposed (C)',
              )}
            </pre>
          )}
        </div>
      ))}
    </section>
  )
}
