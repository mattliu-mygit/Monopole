import type { PromotionAvailability } from './reflectionState'

export default function ReflectionDecisionControls({
  availability,
  editing,
  hasSavedDraft,
  hasUnsavedChanges,
  actionMismatchPaths,
  busy,
  canDismiss,
  acknowledged,
  onAcknowledgedChange,
  onStartEditing,
  onSaveDraft,
  onResetDraft,
  onPromote,
  onDismiss,
}: {
  availability: PromotionAvailability
  editing: boolean
  hasSavedDraft: boolean
  hasUnsavedChanges: boolean
  actionMismatchPaths: string[]
  busy: boolean
  canDismiss: boolean
  acknowledged: boolean
  onAcknowledgedChange: (value: boolean) => void
  onStartEditing: () => void
  onSaveDraft: () => void
  onResetDraft: () => void
  onPromote: () => void
  onDismiss: () => void
}) {
  const isD = availability.target === 'edited-candidate'
  const invalidActionSet = actionMismatchPaths.length > 0
  const promoteDisabled = !availability.enabled || busy || hasUnsavedChanges || (isD && !acknowledged)

  return (
    <section id="reflection-decision" aria-label="Reflection decision" className="space-y-3 rounded-lg border border-gray-200 p-4">
      {availability.reason && <p className="text-sm text-gray-600">{availability.reason}</p>}

      {isD && (
        <label className="flex items-start gap-2 rounded-lg border border-amber-200 bg-amber-50 p-3 text-sm text-amber-900">
          <input
            type="checkbox"
            checked={acknowledged}
            disabled={busy}
            onChange={(event) => onAcknowledgedChange(event.target.checked)}
          />
          <span>I understand edited proposal D differs from evaluated C and has not been evaluated.</span>
        </label>
      )}

      <div className="flex flex-wrap gap-2">
        {!editing && availability.enabled && (
          <button type="button" disabled={busy} onClick={onStartEditing} className="rounded-lg border border-gray-300 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50">
            Edit proposal inline
          </button>
        )}
        {editing && (
          <button type="button" disabled={busy || !availability.enabled || !hasUnsavedChanges || invalidActionSet} onClick={onSaveDraft} className="rounded-lg border border-blue-300 bg-blue-50 px-3 py-2 text-sm font-medium text-blue-800 hover:bg-blue-100 disabled:opacity-50">
            Save edited D
          </button>
        )}
        {(editing || hasSavedDraft) && (
          <button type="button" disabled={busy || (!availability.enabled && hasSavedDraft)} onClick={onResetDraft} className="rounded-lg border border-gray-300 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50">
            Reset to evaluated C
          </button>
        )}
        <button type="button" disabled={promoteDisabled} onClick={onPromote} className="rounded-lg bg-green-700 px-3 py-2 text-sm font-semibold text-white hover:bg-green-800 disabled:cursor-not-allowed disabled:bg-gray-300">
          {isD ? 'Promote unevaluated D' : 'Promote evaluated C'}
        </button>
        <button type="button" disabled={!canDismiss || busy} onClick={onDismiss} className="rounded-lg border border-gray-300 px-3 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50 disabled:opacity-50">
          Dismiss proposal
        </button>
      </div>
      {hasUnsavedChanges && (
        <p className="text-xs text-amber-700">Save or reset your inline edits before promotion.</p>
      )}
      {invalidActionSet && (
        <p role="alert" className="text-xs text-red-700">
          D changes the evaluated action set for {actionMismatchPaths.join(', ')}. Reset to evaluated C before saving.
        </p>
      )}
    </section>
  )
}
